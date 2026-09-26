# GroundCheckHealth — launch copy

The canonical caption is `share-copy.txt`. These are the same claim cut for
different rooms. Every number below is from a real run and can be reproduced
with `python scripts/stress_eval.py`.

Attach a recording of the product if you have one; LinkedIn and X both favour
native video over a link preview.

---

## X / Twitter

**Short (fits comfortably):**

> Most AI demos show you what the model answered.
>
> GroundCheckHealth shows you what it refused — and why.
>
> 265,778 test questions. All 222,210 that had to be refused, were.

**Alternative opener, if you want the number first:**

> 222,210 questions that had no grounded answer.
>
> GroundCheckHealth answered none of them.
>
> Open source, runs on a laptop, refuses out loud.

**Thread continuation (optional, 4 posts):**

1. Most AI demos show you what the model answered. GroundCheckHealth shows you what it refused — and why. 265,778 test questions. All 222,210 that had to be refused, were.
2. Every question runs eleven checks before a model is allowed to write a word: PII redaction, injection scope, retrieval, source coverage, then schema, grounding and a dosage guard on the way out. A question about a medicine that's in no source stops at check six, in 6 ms, with the reason recorded.
3. Answers are the same machine running to the end: every claim carries a citation back to the passage it came from, and a claim that isn't grounded doesn't ship. With a patient loaded it also checks allergies, interactions, kidney and liver function, weight, age and pregnancy against the formulary.
4. Self-hosted, open source, Docker Compose or Helm, encrypted at rest, and a hash-chained audit record for every question — including the ones it refused. github.com/sumitgundawar/GroundCheck

---

## LinkedIn

> **GroundCheckHealth refuses to guess.**
>
> Clinical software that answers from documents has one failure mode that
> matters: answering when it shouldn't. So I built the refusal first.
>
> Every question runs eleven checks before a model is allowed to write a word.
> Ask about a medicine that appears in no trusted source and it stops at check
> six — in 6 milliseconds — records the reason, routes the question to a
> clinician for review, and never calls a model at all. Ask something the
> documents do cover and every claim in the answer carries a citation back to
> the passage it came from.
>
> It was tested on 265,778 generated questions and patient scenarios, with the
> expected outcome taken from the source documents and the formulary rather
> than from its own code. Of those, 222,210 had to be refused. It refused all
> of them.
>
> It runs on your own servers — Docker Compose or Kubernetes — with encryption
> at rest, a hash-chained audit trail, multi-site separation, DICOM imaging,
> SMART on FHIR and CDS Hooks. It is open source.
>
> Not yet cleared as a medical device, and the demo corpus is synthetic: every
> medicine and dose in the video is invented, on purpose, so the project is
> safe to share and fork.
>
> github.com/sumitgundawar/GroundCheck

---

## Hacker News / Reddit (r/MachineLearning, r/healthIT)

**Title:** GroundCheckHealth – a clinical RAG system built around refusing to answer

**Body:**

> The interesting part of retrieval-augmented generation in medicine isn't the
> generation, it's the refusal. GroundCheckHealth runs eleven checks per question and
> stops at the first one that fails — before any model is called — then records
> the reason, the retrieval scores and the stage timings in a hash-chained
> audit record.
>
> The evaluation is the part I'd most like torn apart: 265,778 generated
> questions and patient scenarios, expectations derived from the corpus and the
> formulary rather than from the code, 222,210 of which must be refused
> (medicines that don't exist, one-letter misspellings of ones that do,
> children and pregnancy where the sources only cover adults, doses the sources
> never state, prompt injections, and patients the formulary rules out). It
> answers none of them, and over-refuses 0.5% of the answerable ones.
>
> Finding real bugs at that scale is what the number is for — a 5,000-case
> sample caught misspelled names being answered and a dose accepted because an
> unrelated medicine's page happened to contain that number.
>
> Self-hosted, offline-capable (there's an extractive mode with no model at
> all), MIT licensed. The demo corpus is synthetic on purpose.

---

## Discord / Slack

> Built a clinical Q&A system whose party trick is saying no. 11 checks per
> question, refuses in 6 ms when a medicine isn't in any source, and every
> answer cites the passage it came from. 265,778 test questions, 222,210 of
> them had to be refused, it refused all of them. 🎥 below

---

## Video alt text (use it — this project is about legibility)

> Screen recording of GroundCheckHealth. A clinician's question — "What is the
> recommended dose of Zalortin for a patient with Veltris syndrome?" — is
> refused. An audit record fills in stage by stage: five checks pass, "Source
> coverage" fails because "zalortin" is in no trusted source, and four later
> stages read "not reached". The verdict says "Refused. Routed for review. No
> model was called." A second record shows the same pipeline answering a
> question it can support, with the citation [VELT-002] beside its source. The
> film ends on 222,210 questions that had to be refused, 0 answered, and the
> line "It refuses to guess."

---

## Notes before posting

- The video is 23 seconds, 1920×1080, with a quiet music bed. Most feeds
  autoplay muted, and it reads fine silent.
- Every figure shown is reproducible: `python scripts/run_eval.py` for the
  2,008-case gate, `python scripts/stress_eval.py` for the 265,778.
- Say the corpus is synthetic if the post might reach clinicians. The medicines
  in the film — Zalortin, Caloradine, Veltris syndrome — do not exist, which is
  the point of the demo, but it should never read as real dosing advice.
