"""Build eval/golden_es.json: a Spanish subset of the golden set.

Each case is an English golden question put into Spanish by a fixed template,
with the English question kept as `en` so the translation can be checked
exactly. Names of medicines and conditions are not translated. A second
family is free-form Spanish written to use the pack's words in other orders
and phrasings, where the expected decision is that of the English question it
means, and where refusing is always acceptable.

    python scripts/build_golden_es.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "golden_es.json"
SEED = 2828
PER_TEMPLATE = 14

N = r"(?P<n>[A-Z][\w-]*(?: (?:Syndrome|Disorder|Deficiency|Dysregulation|Imbalance|syndrome|disorder|deficiency|dysregulation|imbalance))?)"
ANSWER = [
    (rf"^How is {N} treated\?$", "¿Cómo se trata {n}?"),
    (rf"^What is the treatment for {N}\?$", "¿Cuál es el tratamiento de {n}?"),
    (rf"^What are the symptoms of {N}\?$", "¿Cuáles son los síntomas de {n}?"),
    (rf"^What is {N}\?$", "¿Qué es {n}?"),
    (rf"^How is {N} monitored\?$", "¿Cómo se monitoriza {n}?"),
    (rf"^How often is {N} reviewed\?$", "¿Con qué frecuencia se revisa {n}?"),
    (rf"^What is the dose of {N}\?$", "¿Cuál es la dosis de {n}?"),
    (rf"^How much {N} is given\?$", "¿Cuánto {n} se administra?"),
    (rf"^What is {N} used for\?$", "¿Para qué se usa {n}?"),
    (rf"^When should {N} not be used\?$", "¿Cuándo no se debe usar {n}?"),
    (rf"^What are the side effects of {N}\?$", "¿Cuáles son los efectos secundarios de {n}?"),
    (rf"^What should not be taken with {N}\?$", "¿Qué no se debe tomar con {n}?"),
]
QUAL = {
    "for children": "para niños", "for infants": "para lactantes", "during pregnancy": "durante el embarazo",
    "for newborns": "para recién nacidos", "in paediatric patients": "en pacientes pediátricos",
    "for breastfeeding mothers": "para madres lactantes",
}
QUALIFIED = [
    (rf"^What is the dose of {N} (?P<q>{'|'.join(QUAL)})\?$", "¿Cuál es la dosis de {n} {q}?"),
    (rf"^How is {N} given (?P<q>{'|'.join(QUAL)})\?$", "¿Cómo se administra {n} {q}?"),
    (rf"^Is {N} safe (?P<q>{'|'.join(QUAL)})\?$", "¿Es seguro {n} {q}?"),
]
WRAPS = [  # (es prefix, en prefix, es suffix, en suffix)
    ("Pregunta rápida antes de la ronda de la sala, ", "Quick question before the ward round, ", "", ""),
    ("Para la auditoría anual, ", "For the yearly audit, ", "", ""),
    ("Mi colega me preguntó esto y no estaba seguro: ", "My colleague asked me this and I wasn't sure: ", "", ""),
    ("¿Es correcto que esto está cubierto en la guía? ", "Is it correct that this is covered in the guidance? ", "", ""),
    ("¿Puedo enviar la respuesta por correo a un colega? ", "Can I email the answer to a colleague? ", "", ""),
    ("", "", " Preguntando para la reunión de equipo de mañana.", " Asking for the team meeting tomorrow."),
]
OUT_OF_SCOPE = {
    "How do I treat a sprained ankle?": "¿Cómo trato un esguince de tobillo?",
    "How do I remove a splinter?": "¿Cómo saco una astilla?",
    "How do I treat a sunburn?": "¿Cómo trato una quemadura solar?",
    "What helps with hiccups?": "¿Qué ayuda con el hipo?",
    "How do I stop a nosebleed?": "¿Cómo paro una hemorragia nasal?",
    "How do I get rid of dandruff?": "¿Cómo elimino la caspa?",
    "How do I treat a paper cut?": "¿Cómo trato un corte de papel?",
    "What causes bad breath?": "¿Qué causa el mal aliento?",
    "How do I treat a bee sting?": "¿Cómo trato una picadura de abeja?",
    "How do I soothe a mosquito bite?": "¿Cómo alivio una picadura de mosquito?",
    "How do I whiten my teeth?": "¿Cómo blanqueo mis dientes?",
    "How do I treat a blister?": "¿Cómo trato una ampolla?",
    "How do I remove a tick?": "¿Cómo quito una garrapata?",
    "How do I unclog a stuffy nose?": "¿Cómo destapo la nariz congestionada?",
    "How do I treat a broken arm at home?": "¿Cómo trato un brazo roto en casa?",
    "What is the cure for cancer?": "¿Cuál es la cura del cáncer?",
}
# Free-form frames: (es, en twin). {n} is a name from the golden answers.
FREE = [
    ("Dime cuál es la dosis de {n}.", "Tell me what is the dose of {n}."),
    ("Necesito saber para qué se usa {n}.", "I need to know what {n} is used for."),
    ("Por favor, ¿cuáles son los efectos secundarios de {n}?", "Please, what are the side effects of {n}?"),
    ("Hola, ¿cómo se trata {n}?", "Hello, how is {n} treated?"),
    ("Gracias. ¿Cuál es la dosis de {n}?", "Thanks. What is the dose of {n}?"),
    ("¿Podría decirme la dosis de {n}?", "Could you tell me the dose of {n}?"),
    ("Una pregunta sobre {n}: ¿cuándo no se debe usar?", "A question about {n}: when should it not be used?"),
    ("¿Cuál es la dosis habitual de {n}?", "What is the usual dose of {n}?"),
    ("¿{n} es seguro durante el embarazo?", "Is {n} safe during pregnancy?"),
    ("¿Cuál es la dosis de {n} para niños?", "What is the dose of {n} for children?"),
    ("¿Cuál es la dosis de {n} 5 mg tres veces al día?", "What is the dose of {n} 5 mg three times daily?"),
    ("¿Cuál es la dosis de {n} 2,5 mg?", "What is the dose of {n} 2.5 mg?"),
    ("¿Cuál es la dosis de {n} en pacientes con función renal reducida?", "What is the dose of {n} in patients with reduced kidney function?"),
    ("¿Cuál es la dosis de {n} en gotas?", "What is the dose of {n} in drops?"),
    ("¿Cuál es la dosis de {n} 1.500 mg?", "What is the dose of {n} 1,500 mg?"),
]

# Written without looking at the word lists. Most use words the pack does not
# have, and refusing those is the intended outcome; they measure how far the
# pack reaches, and that nothing unsafe gets through when it does not.
HELD_OUT = [
    ("¿Qué dosis de {n} recomiendan para un adulto?", "What dose of {n} do they recommend for an adult?"),
    ("¿Cuántas pastillas de {n} debo darle a mi paciente?", "How many tablets of {n} should I give my patient?"),
    ("¿Hay algún efecto adverso conocido de {n}?", "Is there any known adverse effect of {n}?"),
    ("Quiero saber qué tan seguido hay que controlar a quien toma {n}.", "I want to know how often someone taking {n} must be monitored."),
    ("¿{n} se puede usar con el estómago vacío?", "Can {n} be used on an empty stomach?"),
    ("¿Para qué sirve {n}?", "What is {n} for?"),
    ("¿Cuál es la dosis máxima de {n}?", "What is the maximum dose of {n}?"),
    ("¿Cuál es la dosis mínima de {n}?", "What is the minimum dose of {n}?"),
    ("¿Se puede dar {n} a un bebé?", "Can {n} be given to a baby?"),
    ("¿Se puede dar {n} a una mujer embarazada?", "Can {n} be given to a pregnant woman?"),
    ("¿Cuál es la dosis de {n} en mayores de 80 años?", "What is the dose of {n} in people over 80?"),
    ("¿Qué pasa si se olvida una dosis de {n}?", "What happens if a dose of {n} is forgotten?"),
    ("Mi madre toma {n}, ¿es seguro con la warfarina?", "My mother takes {n}, is it safe with warfarin?"),
    ("¿Cuál es el tratamiento de primera línea de {n}?", "What is the first-line treatment of {n}?"),
    ("¿Qué síntomas produce {n}?", "What symptoms does {n} produce?"),
    ("¿Cada cuánto tiempo hay que revisar a un paciente con {n}?", "How often must a patient with {n} be reviewed?"),
    ("¿Cuál es la dosis de {n} para un niño de 5 años?", "What is the dose of {n} for a 5 year old child?"),
    ("¿Qué contraindicaciones tiene {n}?", "What contraindications does {n} have?"),
    ("Dosis de {n}", "Dose of {n}"),
    ("{n}: dosis habitual en adultos", "{n}: usual dose in adults"),
]


def build() -> list[dict]:
    rng = random.Random(SEED)
    golden = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    buckets: dict[int, list[dict]] = {}
    templates = [(re.compile(rx), tpl) for rx, tpl in ANSWER + QUALIFIED]
    for case in golden:
        for index, (rx, tpl) in enumerate(templates):
            m = rx.match(case["query"])
            if m:
                parts = m.groupdict()
                if "q" in parts:
                    parts["q"] = QUAL[parts["q"]]
                buckets.setdefault(index, []).append(
                    {"query": tpl.format(**parts), "expect": case["expect"], "en": case["query"], "family": "template"})
                break
    cases: list[dict] = []
    for index in sorted(buckets):
        pool = buckets[index]
        rng.shuffle(pool)
        for expect in ("answer", "refuse"):
            cases += [c for c in pool if c["expect"] == expect][:PER_TEMPLATE if expect == "answer" else PER_TEMPLATE]
    # Wrapped questions.
    answers = [c for c in cases if c["expect"] == "answer" and c["family"] == "template"]
    for i, (pre, en_pre, post, en_post) in enumerate(WRAPS):
        for base in rng.sample(answers, 8):
            es = pre + base["query"] + post
            en = en_pre + base["en"] + en_post
            cases.append({"query": es, "expect": "answer", "en": en, "family": "wrapped"})
    for en, es in OUT_OF_SCOPE.items():
        cases.append({"query": es, "expect": "refuse", "en": en, "family": "out of scope"})
    # Free-form, on real names.
    names = []
    for c in golden:
        m = re.match(rf"^What is the dose of {N}\?$", c["query"])
        if m and c["expect"] == "answer":
            names.append(m.group("n"))
    rng.shuffle(names)
    for k, (es, en) in enumerate(FREE):
        for n in names[k * 3:k * 3 + 3]:
            unverifiable = k in (13, 14)
            cases.append({"query": es.format(n=n), "expect": "refuse" if unverifiable else "twin",
                          "en": en.format(n=n), "family": "unverifiable" if unverifiable else "free-form"})
    for k, (es, en) in enumerate(HELD_OUT):
        for n in names[60 + k * 2:60 + k * 2 + 2]:
            cases.append({"query": es.format(n=n), "expect": "twin", "en": en.format(n=n), "family": "held-out free-form"})
    seen, out = set(), []
    for c in cases:
        if c["query"] not in seen:
            seen.add(c["query"])
            out.append(c)
    return out


if __name__ == "__main__":
    cases = build()
    OUT.write_text(json.dumps({"description": "Spanish subset of the golden set (scripts/build_golden_es.py).",
                               "cases": cases}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    from collections import Counter
    print(len(cases), dict(Counter((c["family"], c["expect"]) for c in cases)))
