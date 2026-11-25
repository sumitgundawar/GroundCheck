"""Language detection for the input guards.

A question in a language the installation has no pack for must be refused, not
answered from English sources by accident. The detector is pure Python: script
ranges decide the non-Latin languages, and for Latin-script text a count of
distinctive function words combined with a character-trigram score decides
between English, Spanish, French, German, Portuguese, Italian and Dutch.

It errs towards "supported". Short text, text with no function words, drug
names, abbreviations (bd, tds, mg/kg), Latin terms and numbers never count as
evidence of another language: only two or more distinctive words of one
language with no matching English evidence do."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from . import config

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "hi": "Hindi", "ar": "Arabic", "ru": "Russian", "el": "Greek", "he": "Hebrew",
    "th": "Thai", "bn": "Bengali", "ta": "Tamil", "fa": "Persian", "uk": "Ukrainian",
}

# Distinctive function words. Words that are also ordinary English words, or
# that appear in drug names and Latin phrases, are left out on purpose.
_FUNCTION = {
    "en": "the of and is are what how does do can should with for this that my you your if when which from "
          "has have be it to not or by on was were been will would could there their they them these those "
          "an any about after before during into than then also only just give take taking per".split(),
    "es": "de en y el la los las del que es un una dime podría puedo quisiera necesito hola gracias favor unos unas por para con se su sus al como más pero muy sin sobre "
          "también hasta hay donde quien desde todo durante todos entre cuál cuáles qué cómo cuándo cuánto "
          "cuánta cuántos dónde puede pueden debe deben tiene tienen está están son fue ser estar mi tu "
          "esto esta este esos esas ese cada porque cuando si ya hay lo les nos ella ellos".split(),
    "fr": "le la les des du de en un une et est sont pour avec dans sur par que qui quel quelle quels quelles "
          "comment quand où combien peut peuvent doit doivent ce cette ces mon ma mes son sa ses nous vous "
          "ils elles pas ne plus très aussi mais ou donc être avoir fait faire au aux chez sans sous "
          "entre depuis pendant avant après à où été on quelle".split(),
    "de": "der die das den dem des ein eine einen einem einer und ist sind für mit von zu zum zur auf bei "
          "nach aus über unter vor wie was wann wo wer welche welcher welches wieviel kann können soll "
          "sollen muss müssen nicht auch aber oder wenn dass ich du er sie wir ihr mein dein sein "
          "hat haben hatte wird werden wurde wofür weshalb warum durch gegen ohne um bis seit während ob".split(),
    "pt": "o as um uma uns umas de do da dos das em no na nos nas por para com sem sobre entre até "
          "como quando onde quem qual quais quanto quanta quantos é são foi ser estar está estão "
          "pode podem deve devem tem têm não também mais muito meu minha seu sua isso isto esse essa "
          "este esta ele ela eles elas que durante ao à se pode".split(),
    "it": "il lo la i gli le un uno una di del della dei delle degli che è sono per con su tra fra "
          "come quando dove chi quale quali quanto quanta quanti non anche più molto mio mia suo sua "
          "questo questa quello quella essere avere può possono deve devono si ma o se nel nella "
          "nei nelle sul sulla dal dalla cosa negli serve puo può".split(),
    "nl": "de het een en van is zijn voor met op aan bij naar uit over onder tussen hoe wat wanneer "
          "waar wie welke hoeveel kan kunnen moet moeten niet ook maar of als dat dit die deze mijn "
          "jouw zijn haar ons hun ik jij hij zij wij jullie".split(),
}

# Words shared by several languages, or with English, that don't count.
_SHARED = {
    "a", "e", "o", "i", "y", "no", "me", "he", "son", "in", "as", "an", "on", "at", "die", "was",
    "per", "os", "do", "dose", "post", "pro", "re", "ad", "ex", "vivo", "vitro",
}

# A seed vocabulary per language for the trigram profile: the function words
# plus common words, medical and everyday. Small on purpose; it only breaks ties.
_SEED = {
    "en": "dose daily treatment symptoms side effects safe patient children pregnancy kidney liver tablets "
          "morning night weeks months should maximum minimum often given taken blood pressure pain fever "
          "infection allergy monitoring guidance question answer information medicine drug condition "
          "management adults review hours milligrams strength warning problem normal usual starting "
          "stopping missed overdose reaction rash".split(),
    "es": "dosis diaria tratamiento síntomas efectos secundarios seguro paciente niños embarazo riñón hígado "
          "comprimidos mañana noche semanas meses máxima mínima frecuencia administra toma sangre presión "
          "dolor fiebre infección alergia control información medicamento enfermedad adultos revisión horas "
          "miligramos fuerza advertencia problema normal habitual inicio suspender olvidada sobredosis "
          "reacción erupción pregunta respuesta".split(),
    "fr": "dose quotidienne traitement symptômes effets secondaires sûr patient enfants grossesse rein foie "
          "comprimés matin soir semaines mois maximale minimale fréquence administré pris sang pression "
          "douleur fièvre infection allergie surveillance information médicament maladie adultes révision "
          "heures milligrammes force avertissement problème normal habituel début arrêt oubliée surdosage "
          "réaction éruption question réponse".split(),
    "de": "dosis täglich behandlung symptome nebenwirkungen sicher patient kinder schwangerschaft niere leber "
          "tabletten morgen nacht wochen monate maximal minimal häufigkeit verabreicht eingenommen blut druck "
          "schmerz fieber infektion allergie überwachung information medikament krankheit erwachsene "
          "überprüfung stunden milligramm stärke warnung problem normal üblich beginn absetzen vergessen "
          "überdosis reaktion ausschlag frage antwort".split(),
    "pt": "dose diária tratamento sintomas efeitos secundários seguro paciente crianças gravidez rim fígado "
          "comprimidos manhã noite semanas meses máxima mínima frequência administrada tomada sangue pressão "
          "dor febre infeção alergia controlo informação medicamento doença adultos revisão horas "
          "miligramas força aviso problema normal habitual início suspender esquecida sobredosagem "
          "reação erupção pergunta resposta".split(),
    "it": "dose giornaliera trattamento sintomi effetti collaterali sicuro paziente bambini gravidanza rene "
          "fegato compresse mattina notte settimane mesi massima minima frequenza somministrata assunta "
          "sangue pressione dolore febbre infezione allergia controllo informazione farmaco malattia adulti "
          "revisione ore milligrammi forza avvertenza problema normale abituale inizio sospendere dimenticata "
          "sovradosaggio reazione eruzione domanda risposta".split(),
    "nl": "dosis dagelijks behandeling symptomen bijwerkingen veilig patiënt kinderen zwangerschap nier lever "
          "tabletten ochtend nacht weken maanden maximaal minimaal frequentie toegediend ingenomen bloed "
          "druk pijn koorts infectie allergie controle informatie medicijn ziekte volwassenen herziening "
          "uur milligram sterkte waarschuwing probleem normaal gebruikelijk begin stoppen vergeten "
          "overdosis reactie uitslag vraag antwoord".split(),
}

_SCRIPTS = [
    ("zh", re.compile(r"[一-鿿㐀-䶿]")),
    ("ja", re.compile(r"[぀-ヿ]")),
    ("ko", re.compile(r"[가-힯ᄀ-ᇿ]")),
    ("hi", re.compile(r"[ऀ-ॿ]")),
    ("bn", re.compile(r"[ঀ-৿]")),
    ("ta", re.compile(r"[஀-௿]")),
    ("th", re.compile(r"[฀-๿]")),
    ("he", re.compile(r"[֐-׿]")),
    ("ar", re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
    ("el", re.compile(r"[Ͱ-Ͽἀ-῿]")),
]
_NOISE = re.compile(r"https?://\S+|\S+@\S+|\[[A-Z_ ]+\]|\d+(?:[.,]\d+)*")
_TOKEN = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
_LATIN_CODES = ("en", "es", "fr", "de", "pt", "it", "nl")
# Distinctive letters, a weak hint on top of the words.
_DIACRITIC_HINT = {"es": "ñ¿¡", "fr": "çœ", "de": "ß", "pt": "ãõ", "it": "", "nl": ""}


@dataclass(frozen=True)
class Detection:
    lang: str            # an ISO code, or "unknown" when there is no safe call
    confidence: float
    reason: str


def _has_letters(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


@lru_cache(maxsize=1)
def _profiles() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for lang in _LATIN_CODES:
        counts: Counter = Counter()
        for word in _SEED[lang] + _FUNCTION[lang]:
            padded = f"  {word.lower()} "
            for i in range(len(padded) - 2):
                counts[padded[i:i + 3]] += 1
        total = sum(counts.values())
        vocab = len(counts)
        out[lang] = {"_total": total, "_vocab": vocab, **counts}
    return out


def _trigram_scores(words: list[str]) -> dict[str, float]:
    grams: list[str] = []
    for word in words:
        padded = f"  {word} "
        grams += [padded[i:i + 3] for i in range(len(padded) - 2)]
    scores = {}
    for lang, prof in _profiles().items():
        total, vocab = prof["_total"], prof["_vocab"] + 1
        scores[lang] = sum(math.log((prof.get(g, 0) + 0.1) / (total + 0.1 * vocab)) for g in grams)
    return scores


def detect(text: str) -> Detection:
    """The language of `text`, or "unknown" when it is short, mixed or has no
    distinctive evidence. Never raises."""
    text = unicodedata.normalize("NFKC", text or "")
    cleaned = _NOISE.sub(" ", text)
    letters = _has_letters(cleaned)
    if letters == 0:
        return Detection("unknown", 0.0, "no letters")

    by_script = {code: len(rx.findall(cleaned)) for code, rx in _SCRIPTS}
    if by_script["ja"] >= 1 and by_script["zh"] + by_script["ja"] >= 2:
        by_script["ja"] += by_script["zh"]
        by_script["zh"] = 0
    code, count = max(by_script.items(), key=lambda kv: kv[1])
    if count >= 2 and count / letters >= 0.4:
        return Detection(code, min(1.0, count / letters), "script")

    words = [w.lower() for w in _TOKEN.findall(cleaned)]
    words = [w.replace("’", "'") for w in words if any(ch.isalpha() for ch in w)]
    # Words with an apostrophe or hyphen contribute their parts.
    parts: list[str] = []
    for w in words:
        parts += [p for p in re.split(r"['-]", w) if p]
    if len(parts) < 3:
        return Detection("unknown", 0.0, "too short to tell")

    hits = {lang: [w for w in parts if w in set(_FUNCTION[lang]) and w not in _SHARED] for lang in _LATIN_CODES}
    # English evidence includes shared words that are plainly English.
    en_hits = len([w for w in parts if w in set(_FUNCTION["en"])])
    scores = _trigram_scores(parts)
    best_foreign = max((lang for lang in _LATIN_CODES if lang != "en"), key=lambda lang: (len(set(hits[lang])), scores[lang]))
    distinct = len(set(hits[best_foreign]))
    hint = sum(text.count(ch) for ch in _DIACRITIC_HINT.get(best_foreign, ""))
    evidence = distinct + (1 if hint else 0)

    margin = scores[best_foreign] - scores["en"]
    # One distinctive word is evidence only when it is a long one and the
    # trigrams agree: "os" in "per os" or "dose" must never be enough.
    lone = evidence == 1 and any(len(w) >= 5 for w in hits[best_foreign])
    if evidence < 2 and not (lone and en_hits == 0 and margin > 6.0):
        return Detection("unknown", 0.0, "no distinctive evidence")
    foreign_count = len(hits[best_foreign])
    if en_hits and (en_hits >= foreign_count or scores["en"] > scores[best_foreign]):
        return Detection("en", 0.5, "english evidence")
    confidence = min(1.0, (foreign_count - en_hits) / max(1, len(parts)) + 0.3)
    return Detection(best_foreign, confidence, "function words and trigrams")


def supported() -> set[str]:
    """Configured languages that have a pack; English always does."""
    from . import langpack

    configured = {c.strip().lower() for c in config.SUPPORTED_LANGUAGES.split(",") if c.strip()}
    return ({"en"} | configured) & langpack.available() or {"en"}


@dataclass(frozen=True)
class LanguageCheck:
    ok: bool
    lang: str
    detail: str


def check(text: str) -> LanguageCheck:
    """Whether the question is in a language this installation has a pack for."""
    d = detect(text)
    langs = supported()
    if d.lang == "unknown" or d.lang in langs:
        shown = d.lang if d.lang != "unknown" else "en"
        return LanguageCheck(True, shown, f"language {shown}" + (" (not detected)" if d.lang == "unknown" else ""))
    name = LANGUAGE_NAMES.get(d.lang, d.lang)
    return LanguageCheck(False, d.lang, f"{name} is not a supported language")
