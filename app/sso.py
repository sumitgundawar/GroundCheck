"""Single sign-on with OpenID Connect.

Works with any OpenID Connect provider: Microsoft Entra ID, Okta, Google
Workspace, Auth0, Keycloak, Ping and ADFS among them. Providers that only
speak SAML can be connected through one that bridges to OpenID Connect,
such as Keycloak or Entra ID.

The flow is the authorisation code flow with PKCE:

1. `start()` records a pending sign-in (a random state, a nonce and a PKCE
   verifier) and returns the provider's sign-in address. The state is also
   set in a cookie, so the return can only complete in the same browser.
2. The provider sends the person back with a code. `finish()` checks the
   state, exchanges the code for tokens, and verifies the ID token: its
   signature against the provider's published keys (RSA or EC only), issuer,
   audience, expiry and nonce.
3. The person is matched by provider and subject, or linked to an existing
   account by verified email, or created. Their role comes from the
   provider's role or group claims when those are mapped, so the provider
   stays the source of truth.

Multi-factor authentication happens at the provider. OIDC_REQUIRE_MFA
refuses sign-ins whose token doesn't say MFA was used."""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlencode

import httpx
from sqlalchemy import delete, select

from . import auth, config, db
from .db import AuthSession, SsoLogin, User, utcnow

LOGIN_MINUTES = 10
_ALGORITHMS = ["RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"]
_MFA_METHODS = {"mfa", "otp", "hwk", "swk", "fido", "sms", "tel", "face", "fpt", "iris", "retina", "vbm", "sc", "pop"}
_cache: dict[str, tuple[float, object]] = {}
_lock = threading.Lock()


class SsoError(Exception):
    """Single sign-on failed. The message is safe to show."""


def enabled() -> bool:
    return bool(config.OIDC_ISSUER and config.OIDC_CLIENT_ID)


def _client() -> httpx.Client:
    return httpx.Client(timeout=10.0, follow_redirects=False)


def _cached(key: str, seconds: float, load):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    value = load()
    with _lock:
        _cache[key] = (now + seconds, value)
    return value


def forget_provider() -> None:
    with _lock:
        _cache.clear()


def discovery() -> dict:
    def load():
        url = f"{config.OIDC_ISSUER}/.well-known/openid-configuration"
        try:
            with _client() as client:
                response = client.get(url)
                response.raise_for_status()
                document = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SsoError(f"Couldn't reach {config.OIDC_PROVIDER_NAME}'s sign-in service.") from exc
        if document.get("issuer", "").rstrip("/") != config.OIDC_ISSUER:
            raise SsoError("The sign-in service's issuer doesn't match OIDC_ISSUER.")
        for field in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            if not str(document.get(field, "")).startswith(("https://", "http://localhost", "http://127.0.0.1")):
                raise SsoError(f"The sign-in service's {field} must use HTTPS.")
        return document
    return _cached("discovery", 3600, load)


def _jwks(force: bool = False) -> dict:
    def load():
        try:
            with _client() as client:
                response = client.get(discovery()["jwks_uri"])
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SsoError("Couldn't load the sign-in service's keys.") from exc
    if force:
        with _lock:
            _cache.pop("jwks", None)
    return _cached("jwks", 3600, load)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_next(path: str | None) -> str:
    """Only a path on this site, never another site."""
    if not path or not path.startswith("/") or path.startswith("//") or "\\" in path:
        return "/"
    return path[:300]


def redirect_url(base_url: str) -> str:
    return config.OIDC_REDIRECT_URL or f"{base_url.rstrip('/')}/api/auth/sso/callback"


def start(base_url: str, next_path: str | None = None) -> tuple[str, str]:
    """Returns (the provider's sign-in address, the state for the cookie)."""
    if not enabled():
        raise SsoError("Single sign-on isn't set up.")
    document = discovery()
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    now = utcnow()
    with db.session() as s:
        s.execute(delete(SsoLogin).where(SsoLogin.expires_at < now))
        s.add(SsoLogin(state_hash=_hash(state), nonce=nonce, code_verifier=verifier,
                       next_path=_safe_next(next_path), created_at=now,
                       expires_at=now + timedelta(minutes=LOGIN_MINUTES)))
    query = urlencode({
        "response_type": "code", "client_id": config.OIDC_CLIENT_ID, "redirect_uri": redirect_url(base_url),
        "scope": config.OIDC_SCOPES, "state": state, "nonce": nonce,
        "code_challenge": challenge, "code_challenge_method": "S256",
    })
    separator = "&" if "?" in document["authorization_endpoint"] else "?"
    return f"{document['authorization_endpoint']}{separator}{query}", state


def _verify_id_token(token: str, nonce: str) -> dict:
    import jwt

    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise SsoError("The sign-in service returned an invalid token.") from exc
    if header.get("alg") not in _ALGORITHMS:
        raise SsoError("The sign-in service's token uses an algorithm that isn't allowed.")

    def key_for(jwks: dict):
        for jwk in jwks.get("keys", []):
            if jwk.get("kid") == header.get("kid") or (header.get("kid") is None and len(jwks["keys"]) == 1):
                return jwt.PyJWK(jwk, algorithm=header["alg"]).key
        return None

    key = key_for(_jwks()) or key_for(_jwks(force=True))  # the provider may have rotated keys
    if key is None:
        raise SsoError("The sign-in service's token was signed with an unknown key.")
    try:
        claims = jwt.decode(token, key=key, algorithms=[header["alg"]], audience=config.OIDC_CLIENT_ID,
                            issuer=discovery()["issuer"], leeway=60,
                            options={"require": ["exp", "iat", "iss", "aud", "sub"]})
    except jwt.PyJWTError as exc:
        raise SsoError("The sign-in service's token couldn't be verified.") from exc
    audiences = claims["aud"] if isinstance(claims["aud"], list) else [claims["aud"]]
    if len(audiences) > 1 and claims.get("azp") != config.OIDC_CLIENT_ID:
        raise SsoError("The sign-in service's token was issued to another application.")
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise SsoError("The sign-in didn't match. Try again.")
    return claims


def role_from_claims(claims: dict) -> str | None:
    values = claims.get(config.OIDC_ROLES_CLAIM) or []
    values = {values} if isinstance(values, str) else {str(v) for v in values}
    if values & config.OIDC_ADMIN_VALUES:
        return "admin"
    if values & config.OIDC_REVIEWER_VALUES:
        return "reviewer"
    if values & config.OIDC_CLINICIAN_VALUES:
        return "clinician"
    return None


def _mapping_configured() -> bool:
    return bool(config.OIDC_ADMIN_VALUES or config.OIDC_REVIEWER_VALUES or config.OIDC_CLINICIAN_VALUES)


@dataclass
class Finished:
    token: str
    principal: auth.Principal
    next_path: str


def finish(code: str, state: str, cookie_state: str | None, base_url: str, ip: str = "",
           user_agent: str = "") -> Finished:
    if not enabled():
        raise SsoError("Single sign-on isn't set up.")
    if not state or not cookie_state or not secrets.compare_digest(state, cookie_state):
        raise SsoError("The sign-in didn't start in this browser, or took too long. Try again.")
    now = utcnow()
    with db.session() as s:
        pending = s.scalar(select(SsoLogin).where(SsoLogin.state_hash == _hash(state)))
        if pending is None or pending.expires_at < now:
            raise SsoError("The sign-in didn't start in this browser, or took too long. Try again.")
        nonce, verifier, next_path = pending.nonce, pending.code_verifier, pending.next_path
        s.delete(pending)  # one use only
    if not code:
        raise SsoError("The sign-in service didn't return a sign-in code.")

    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_url(base_url),
            "code_verifier": verifier, "client_id": config.OIDC_CLIENT_ID}
    auth_header = None
    if config.OIDC_CLIENT_SECRET:
        auth_header = (config.OIDC_CLIENT_ID, config.OIDC_CLIENT_SECRET)
    try:
        with _client() as client:
            response = client.post(discovery()["token_endpoint"], data=form, auth=auth_header,
                                   headers={"Accept": "application/json"})
        tokens = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SsoError(f"Couldn't complete sign-in with {config.OIDC_PROVIDER_NAME}.") from exc
    if response.status_code != 200 or "id_token" not in tokens:
        raise SsoError(f"{config.OIDC_PROVIDER_NAME} didn't accept the sign-in. Try again.")

    claims = _verify_id_token(tokens["id_token"], nonce)
    if config.OIDC_REQUIRE_MFA:
        methods = claims.get("amr") or []
        if not (set(methods) & _MFA_METHODS or claims.get("acr") in ("mfa", "http://schemas.openid.net/pape/policies/2007/06/multi-factor")):
            raise SsoError("Sign in with multi-factor authentication to use GroundCheck.")
    email = str(claims.get("email") or claims.get("preferred_username") or claims.get("upn") or "").strip().lower()
    if "@" not in email:
        raise SsoError("Your account didn't share an email address. Ask your administrator to allow the email scope.")
    if config.OIDC_ALLOWED_DOMAINS and email.rsplit("@", 1)[1] not in config.OIDC_ALLOWED_DOMAINS:
        raise SsoError("Accounts from that email domain can't sign in here.")
    mapped_role = role_from_claims(claims)
    issuer, subject = discovery()["issuer"], str(claims["sub"])

    with db.session() as s:
        user = s.scalar(select(User).where(User.sso_issuer == issuer, User.sso_subject == subject))
        if user is None:
            existing = s.scalar(select(User).where(User.email == email))
            if existing is not None:
                if claims.get("email_verified") is not True:
                    raise SsoError("An account with your email already exists. Ask an administrator to link it; "
                                   "your sign-in service didn't confirm the email address.")
                if existing.sso_subject and existing.sso_issuer == issuer:
                    raise SsoError("That email is already linked to a different sign-in.")
                user = existing
                user.sso_issuer, user.sso_subject = issuer, subject
            elif config.OIDC_AUTO_CREATE:
                role = mapped_role or (config.OIDC_DEFAULT_ROLE if config.OIDC_DEFAULT_ROLE != "none" else None)
                if role is None:
                    raise SsoError("Your account doesn't have a GroundCheck role. Ask your administrator for access.")
                user = User(email=email, name=str(claims.get("name") or "")[:200], password_hash="!",
                            role=auth.check_role(role), sso_issuer=issuer, sso_subject=subject)
                s.add(user)
                s.flush()
            else:
                raise SsoError("You don't have a GroundCheck account yet. Ask your administrator for access.")
        if _mapping_configured():
            if mapped_role is None and config.OIDC_DEFAULT_ROLE == "none":
                raise SsoError("Your account no longer has a GroundCheck role. Ask your administrator for access.")
            new_role = mapped_role or config.OIDC_DEFAULT_ROLE
            if new_role != user.role:
                user.role = auth.check_role(new_role)
        if not user.is_active:
            raise SsoError("Your GroundCheck account is deactivated.")
        if claims.get("name") and not user.name:
            user.name = str(claims["name"])[:200]
        user.last_login_at = now
        token = secrets.token_urlsafe(32)
        s.add(AuthSession(token_hash=auth._hash_token(token), user_id=user.id, created_at=now, last_seen_at=now,
                          expires_at=now + timedelta(hours=config.SESSION_HOURS), mfa_pending=False,
                          ip_address=(ip or "")[:45], user_agent=(user_agent or "")[:300]))
        s.flush()
        return Finished(token=token, principal=auth._principal(user), next_path=next_path)
