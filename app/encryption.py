"""Encryption at rest for what people type and what they're shown.

With DATA_ENCRYPTION_KEYS set, the database stores these encrypted with
AES-256-GCM: questions and the full audit record (answers, sources, trace),
review cases, their comments and tests, and two-factor secrets. Someone with
a copy of the database, a backup or the disk sees ciphertext.

Keys are 32 random bytes, base64-encoded (`python -m app.cli generate-key`).
The first key in DATA_ENCRYPTION_KEYS encrypts. Every key in it, and in
DATA_ENCRYPTION_RETIRED_KEYS, decrypts. To rotate, put the new key first, run
`python -m app.cli reencrypt`, then remove the old key. To turn encryption
off, move every key to the retired list and run `reencrypt`.

Stored values look like `gcenc:v1:<key id>:<base64>`. The key id is derived
from the key and reveals nothing about it. The column's name is bound to the
ciphertext as associated data, so a value moved to another column fails to
decrypt instead of being accepted. Values written before encryption was
turned on are read as they are and encrypted by `reencrypt`.

Keys never go in the database. Keep them in a secrets manager, and keep a
copy somewhere safe: without the key, the data is unrecoverable."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import threading

from . import config

PREFIX = "gcenc:v1:"
BYTES_PREFIX = b"GCENC1"
KEY_BYTES = 32
NONCE_BYTES = 12


class EncryptionError(RuntimeError):
    """A value couldn't be decrypted, or a key is invalid. Safe to show."""


def generate_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(KEY_BYTES)).decode("ascii")


def _decode_key(text: str) -> bytes:
    text = text.strip()
    try:
        key = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        key = b""
    if len(key) != KEY_BYTES:
        raise EncryptionError(
            "Each encryption key must be 32 random bytes, base64-encoded. "
            "Generate one with: python -m app.cli generate-key")
    return key


def key_id(key: bytes) -> str:
    return hashlib.sha256(b"groundcheck key id\x00" + key).hexdigest()[:8]


class Keyring:
    def __init__(self, keys: list[bytes], retired: list[bytes] | None = None):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self._ciphers = {key_id(k): AESGCM(k) for k in [*keys, *(retired or [])]}
        self.primary = key_id(keys[0]) if keys else None
        # A separate key, derived from the primary, for lookup hashes.
        self._lookup_key = hmac.new(keys[0], b"groundcheck lookup", hashlib.sha256).digest() if keys else None

    @property
    def enabled(self) -> bool:
        return self.primary is not None

    @property
    def key_ids(self) -> list[str]:
        return list(self._ciphers)

    def summary(self) -> dict:
        return {"enabled": self.enabled, "primary_key_id": self.primary,
                "decrypt_key_ids": [k for k in self._ciphers if k != self.primary]}

    def encrypt(self, plaintext: str, context: str) -> str:
        if not self.enabled:
            return plaintext
        nonce = os.urandom(NONCE_BYTES)
        sealed = self._ciphers[self.primary].encrypt(nonce, plaintext.encode("utf-8"), context.encode("utf-8"))
        return f"{PREFIX}{self.primary}:{base64.urlsafe_b64encode(nonce + sealed).decode('ascii')}"

    def decrypt(self, value: str, context: str) -> str:
        if not is_encrypted(value):
            return value
        try:
            kid, payload = value[len(PREFIX):].split(":", 1)
            raw = base64.urlsafe_b64decode(payload)
        except (ValueError, binascii.Error) as exc:
            raise EncryptionError("An encrypted value is malformed.") from exc
        cipher = self._ciphers.get(kid)
        if cipher is None:
            raise EncryptionError(
                f"Data was encrypted with key {kid}, which isn't in DATA_ENCRYPTION_KEYS.")
        from cryptography.exceptions import InvalidTag

        try:
            return cipher.decrypt(raw[:NONCE_BYTES], raw[NONCE_BYTES:], context.encode("utf-8")).decode("utf-8")
        except InvalidTag as exc:
            raise EncryptionError(f"An encrypted value in {context} failed its integrity check.") from exc

    def encrypt_bytes(self, data: bytes, context: str) -> bytes:
        """Files, such as stored images: the same scheme, in binary."""
        if not self.enabled:
            return data
        nonce = os.urandom(NONCE_BYTES)
        sealed = self._ciphers[self.primary].encrypt(nonce, data, context.encode("utf-8"))
        return BYTES_PREFIX + self.primary.encode("ascii") + nonce + sealed

    def decrypt_bytes(self, data: bytes, context: str) -> bytes:
        if not data.startswith(BYTES_PREFIX):
            return data
        start = len(BYTES_PREFIX)
        kid = data[start:start + 8].decode("ascii", "replace")
        cipher = self._ciphers.get(kid)
        if cipher is None:
            raise EncryptionError(f"A file was encrypted with key {kid}, which isn't in DATA_ENCRYPTION_KEYS.")
        from cryptography.exceptions import InvalidTag

        nonce = data[start + 8:start + 8 + NONCE_BYTES]
        try:
            return cipher.decrypt(nonce, data[start + 8 + NONCE_BYTES:], context.encode("utf-8"))
        except InvalidTag as exc:
            raise EncryptionError(f"The encrypted file {context} failed its integrity check.") from exc

    def lookup_hash(self, text: str) -> str:
        """A hash for finding equal values, such as a repeated question. Keyed
        when encryption is on, so it can't be used to confirm a guess."""
        data = text.encode("utf-8")
        if self._lookup_key is None:
            return hashlib.sha256(data).hexdigest()
        return hmac.new(self._lookup_key, data, hashlib.sha256).hexdigest()

    def is_current(self, value: str) -> bool:
        """True when a stored value needs no re-encryption."""
        if not self.enabled:
            return not is_encrypted(value)
        return value.startswith(f"{PREFIX}{self.primary}:")


def is_encrypted(value) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


_keyring: Keyring | None = None
_lock = threading.Lock()


def keyring() -> Keyring:
    global _keyring
    with _lock:
        if _keyring is None:
            keys = [_decode_key(t) for t in config.DATA_ENCRYPTION_KEYS.split(",") if t.strip()]
            retired = [_decode_key(t) for t in config.DATA_ENCRYPTION_RETIRED_KEYS.split(",") if t.strip()]
            if len({key_id(k) for k in [*keys, *retired]}) != len(keys) + len(retired):
                raise EncryptionError("The same encryption key is listed twice.")
            _keyring = Keyring(keys, retired)
        return _keyring


def reset() -> None:
    """Forget the keyring, so the next use reads DATA_ENCRYPTION_KEYS again."""
    global _keyring
    with _lock:
        _keyring = None
