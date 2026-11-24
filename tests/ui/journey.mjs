// End-to-end checks of the browser app: the parts the Python tests can't see.
// It drives a real browser against a running GroundCheckHealth, the way a clinician
// would, in both themes, and at phone and laptop widths.
//
//   pip install -r requirements.txt && python scripts/build_index.py
//   uvicorn app.main:app --port 8000
//   cd tests/ui && npm install && npm test
//
// Point it somewhere else with GROUNDCHECK_URL. It expects the open demo
// (AUTH_REQUIRED unset), which is what `uvicorn app.main:app` gives you.

import { chromium } from "playwright";

const BASE = process.env.GROUNDCHECK_URL || "http://127.0.0.1:8000";
const HEADLESS = process.env.HEADED !== "1";

const results = [];
let browser;

async function check(name, fn) {
  try {
    await fn();
    results.push({ name, ok: true });
    console.log(`  ok    ${name}`);
  } catch (error) {
    results.push({ name, ok: false, error: error.message });
    console.log(`  FAIL  ${name}\n        ${error.message}`);
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function equal(actual, expected, message) {
  assert(actual === expected, `${message}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

/** Ask a question and wait for this run — not the last one — to finish. */
async function ask(page, question) {
  await page.fill("#query", question);
  await page.click("#ask-btn");
  // The button says "Asking" and is disabled while the request is in flight.
  await page.waitForFunction(() => document.querySelector("#ask-btn").disabled, undefined, { timeout: 15000 });
  await page.waitForFunction(() => !document.querySelector("#ask-btn").disabled, undefined, { timeout: 60000 });
  await page.waitForSelector("#decision-panel:not([hidden])");
  return (await page.textContent("#decision-chip")).trim();
}

async function go(page, view) {
  await page.evaluate((v) => { location.hash = `#/${v}`; }, view);
  await page.waitForFunction((v) => !document.querySelector(`#view-${v}`)?.hidden, view, { timeout: 15000 });
  await page.waitForTimeout(500);
}

async function main() {
  browser = await chromium.launch({ headless: HEADLESS });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  const page = await context.newPage();
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") pageErrors.push(`console: ${m.text()}`); });

  console.log(`GroundCheckHealth browser journey against ${BASE}\n`);
  await page.goto(BASE + "/", { waitUntil: "networkidle" });

  console.log("Asking questions");

  await check("a question the sources answer is answered, with citations", async () => {
    const decision = await ask(page, "What is the first-line medication for Veltris syndrome?");
    equal(decision, "ANSWER", "the decision");
    const text = (await page.textContent("#answer-body")) || "";
    assert(text.trim().length > 20, "the answer is empty");
    const citations = await page.locator("#answer-body a.citation").count();
    assert(citations > 0, "the answer cites nothing");
    const cited = await page.locator("#sources .source-card.cited").count();
    assert(cited > 0, "no retrieved source is marked as cited");
    const id = await page.locator("#answer-body a.citation").first().textContent();
    const card = await page.locator(`#sources #src-${id.replace(/[[\]]/g, "")}`).count();
    equal(card, 1, `the source card for ${id}`);
  });

  await check("a medicine in no source is refused, and the reason says so", async () => {
    const decision = await ask(page, "What is the recommended dose of Zalortin for a patient with Veltris syndrome?");
    equal(decision, "REFUSED", "the decision");
    const reason = (await page.textContent("#decision-reason")) || "";
    assert(/zalortin/i.test(reason), `the reason doesn't name the medicine: ${reason}`);
    assert(await page.locator("#refuse-callout").isVisible(), "no refusal callout");
    equal((await page.textContent("#answer-body")).trim(), "", "the answer body on a refusal");
  });

  await check("the trace shows which check stopped the answer", async () => {
    await page.waitForSelector("#trace .trace-item");
    const stages = await page.locator("#trace .trace-item").count();
    assert(stages >= 10, `only ${stages} stages in the trace`);
    // The last row is the decision itself; exactly one check before it failed.
    const failed = await page.$$eval("#trace .trace-item.fail", (items) => items.map((i) => i.textContent));
    equal(failed.length, 2, `failed rows in the trace (${failed.join(" / ")})`);
    assert(/coverage/i.test(failed[0]), `the failing check was ${failed[0]}`);
    assert(/refuse/i.test(failed[1]), `the last row should be the decision, was ${failed[1]}`);
    const skipped = await page.locator("#trace .trace-item.skip").count();
    equal(skipped, 4, "stages not reached after the refusal");
  });

  await check("the audit record for the run can be opened", async () => {
    assert(!(await page.locator("#audit-panel").isHidden()), "no audit panel after a question");
    await page.click("#audit-toggle");
    await page.waitForSelector("#audit-json:not([hidden])");
    await page.waitForFunction(() => document.querySelector("#audit-json").textContent.includes("audit_id"));
    const text = (await page.textContent("#audit-json")) || "";
    for (const key of ["audit_id", "decision", "trace", "refused_reason"]) {
      assert(text.includes(key), `the audit record has no ${key}`);
    }
    assert(/refuse/.test(text), "the audit record doesn't record the refusal");
    assert(!(await page.locator("#audit-controls").isHidden()), "the expand and copy controls stayed hidden");
    await page.click("#audit-toggle");
  });

  await check("a child's age refuses a dose the formulary doesn't cover", async () => {
    const box = page.locator("#patient-box");
    if ((await box.getAttribute("open")) === null) await page.click("#patient-box > summary");
    await page.fill("#pt-age", "5");
    await page.fill("#pt-weight", "18");
    const decision = await ask(page, "What is the dose of Caloradine?");
    equal(decision, "REFUSED", "the decision for a five-year-old");
    const findings = (await page.textContent("#patient-findings")) || "";
    assert(/aged 5/.test(findings), `no patient finding explains the refusal: ${findings}`);
  });

  await check("an answer about adults says so when the patient is a child", async () => {
    const decision = await ask(page, "What is the first-line medication for Veltris syndrome?");
    equal(decision, "ANSWER", "the decision for an informational question");
    const findings = (await page.textContent("#patient-findings")) || "";
    assert(/about adults/i.test(findings), `the answer isn't qualified for a child: ${findings}`);
    await page.click("#pt-clear");
  });

  console.log("\nMoving around the app");

  await check("every page in the sidebar opens", async () => {
    // Hidden links are the ones this signed-in person can't use, such as Users
    // when the deployment has no accounts.
    const views = await page.$$eval(".nav-link[data-view]:not([hidden])", (links) => links.map((l) => l.dataset.view));
    assert(views.length >= 10, `only ${views.length} pages in the sidebar`);
    const stuck = [];
    for (const view of views) {
      try {
        await go(page, view);
        const title = await page.textContent("#topbar-title");
        assert((title || "").trim().length > 0, `${view} has no title in the top bar`);
      } catch (error) {
        stuck.push(`${view}: ${error.message.split("\n")[0]}`);
      }
    }
    assert(stuck.length === 0, stuck.join("; "));
  });

  await check("usage shows counts for the questions just asked", async () => {
    await go(page, "usage");
    await page.waitForSelector("#usage-stats .stat");
    const text = (await page.textContent("#usage-stats")) || "";
    assert(/\d/.test(text), "no numbers on the usage page");
  });

  await check("the refusal is waiting in the review queue", async () => {
    await go(page, "review");
    await page.waitForSelector("#review-rows tr, #review-empty:not([hidden])");
    const rows = await page.locator("#review-rows tr").count();
    assert(rows > 0, "the refusal didn't reach the review queue");
    const text = (await page.textContent("#review-rows")) || "";
    assert(/zalortin/i.test(text), "the refused question isn't in the queue");
  });

  await check("a medicine's rules open from the formulary", async () => {
    await go(page, "medicines");
    await page.fill("#medicine-search", "Caloradine");
    await page.waitForFunction(() => document.querySelectorAll("#medicine-rows tr").length === 1);
    await page.click("#medicine-rows tr button");
    await page.waitForSelector("#medicine-dialog[open]");
    const title = await page.textContent("#medicine-dialog-title");
    equal((title || "").trim(), "Caloradine", "the dialog's medicine");
    const rules = (await page.textContent("#medicine-rules")) || "";
    assert(rules.trim().length > 0, "the dialog shows no rules");
    await page.keyboard.press("Escape");
    await page.waitForFunction(() => !document.querySelector("#medicine-dialog").open);
  });

  await check("monitoring shows the alert rules and what this instance runs on", async () => {
    await go(page, "monitoring");
    await page.waitForSelector("#instance-details dt");
    const details = (await page.textContent("#instance-details")) || "";
    for (const label of ["Name", "Hardware", "Accelerator", "Batch sizes", "Answers remembered"]) {
      assert(details.includes(label), `the hardware panel has no ${label}`);
    }
    assert(/version \d/.test(details), "the panel doesn't say which version is running");
    assert(/\d+ cores, [\d.]+ GB memory/.test(details), `the panel's hardware line is odd: ${details}`);
    const rules = await page.locator("#alert-list li, #alert-empty:not([hidden])").count();
    assert(rules > 0, "no alert rules are listed");
  });

  await check("the audit trail verifies from the data protection page", async () => {
    await go(page, "data-protection");
    await page.click("#audit-verify");
    await page.waitForSelector("#verify-result:not([hidden])");
    const result = page.locator("#verify-result");
    const classes = (await result.getAttribute("class")) || "";
    const text = (await result.textContent()) || "";
    assert(classes.includes("ok"), `verification failed: ${text}`);
    assert(/\d/.test(text), "the result doesn't say how many records were checked");
  });

  await check("the evaluation page shows the results that gate the build", async () => {
    await go(page, "evaluation");
    await page.waitForSelector("#eval-summary-inline, #eval-body");
    const text = (await page.textContent("#eval-body")) || "";
    assert(/\d/.test(text), "no evaluation figures");
  });

  console.log("\nThemes, keyboard and small screens");

  await check("choosing a theme applies it, and it survives a reload", async () => {
    await go(page, "ask");
    await page.click('.app [data-theme-choice="dark"]');
    equal(await page.evaluate(() => document.documentElement.dataset.theme), "dark", "the theme after choosing dark");
    await page.reload({ waitUntil: "networkidle" });
    equal(await page.evaluate(() => document.documentElement.dataset.theme), "dark", "the theme after a reload");
    const pressed = await page.getAttribute('.app [data-theme-choice="dark"]', "aria-checked");
    equal(pressed, "true", "the dark button's aria-checked after a reload");
    await page.click('.app [data-theme-choice="light"]');
    equal(await page.evaluate(() => document.documentElement.dataset.theme), "light", "the theme after choosing light");
    await page.click('.app [data-theme-choice="system"]');
    equal(await page.evaluate(() => document.documentElement.dataset.theme || ""), "", "the theme after choosing system");
  });

  await check("the question can be asked from the keyboard alone", async () => {
    await go(page, "ask");
    await page.focus("#query");
    await page.keyboard.type("What is the first-line medication for Veltris syndrome?");
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => /ANSWER|REFUSED/i.test(document.querySelector("#decision-chip").textContent));
    const focusVisible = await page.evaluate(() => {
      const el = document.querySelector("#ask-btn");
      el.focus();
      return getComputedStyle(el, ":focus-visible").outlineStyle !== "none" || el.matches(":focus-visible");
    });
    assert(focusVisible, "the ask button has no visible focus");
  });

  console.log("Feedback, adoption, controls and telemetry");

  await check("feedback: a thumbs up is saved and shown as chosen", async () => {
    equal(await ask(page, "What is the standard dose of Caloradine?"), "ANSWER", "the decision");
    assert(await page.locator("#feedback-row").isVisible(), "no feedback row under the answer");
    await page.click("#feedback-up");
    await page.waitForSelector("#feedback-message:not([hidden])");
    assert(/saved/i.test(await page.textContent("#feedback-message")), "no confirmation");
    equal(await page.getAttribute("#feedback-up", "aria-pressed"), "true", "the up button's state");
  });

  await check("feedback: 'wrong' with a comment goes to a reviewer, with identifiers removed", async () => {
    equal(await ask(page, "What is the standard dose of Caloradine?"), "ANSWER", "the decision");
    await page.click("#feedback-down");
    assert(await page.locator("#feedback-form").isVisible(), "no form after thumbs down");
    await page.selectOption("#feedback-reason", "wrong");
    await page.fill("#feedback-comment", "Mr John Smith NHS 943 476 5919 was given this and it was wrong.");
    await page.click("#feedback-send");
    await page.waitForSelector("#feedback-message:not([hidden])");
    assert(/reviewer/i.test(await page.textContent("#feedback-message")), "the message doesn't say it went to a reviewer");
    const queue = await page.evaluate(async () => (await fetch("/api/reviews?status=open")).json());
    const flagged = queue.cases.filter((c) => c.kind === "flagged");
    assert(flagged.length >= 1 && flagged[0].priority === "high", "no high-priority flagged case");
    assert(!JSON.stringify(queue).includes("943 476 5919") && !JSON.stringify(queue).includes("John Smith"), "an identifier reached the queue");
  });

  await check("the usage page shows adoption numbers and the pilot report downloads", async () => {
    await go(page, "usage");
    await page.waitForSelector("#adoption-panel:not([hidden])");
    const tiles = await page.locator("#adoption-stats .stat").count();
    equal(tiles, 6, "adoption tiles");
    const href = await page.getAttribute("#pilot-report-download", "href");
    const report = await page.evaluate(async (u) => { const r = await fetch(u); return { ok: r.ok, text: await r.text() }; }, href);
    assert(report.ok && report.text.startsWith("# GroundCheckHealth pilot report"), "the pilot report did not download");
    assert(!report.text.includes("John Smith") && !report.text.includes("943 476"), "the report holds an identifier");
  });

  await check("the controls tab shows the matrix with its banner and filters it", async () => {
    await go(page, "review");
    await page.click("#tab-controls");
    await page.waitForSelector("#controls-frameworks table");
    assert(/no certification/i.test(await page.textContent("#controls-banner")), "the no-certification banner is missing");
    const all = await page.locator("#controls-frameworks tbody tr").count();
    assert(all > 90, `only ${all} controls`);
    await page.selectOption("#controls-status", "not_addressed");
    const some = await page.locator("#controls-frameworks tbody tr").count();
    assert(some > 0 && some < all, `filtering left ${some} of ${all}`);
    const gone = await page.locator("#controls-frameworks tbody tr td:nth-child(3)", { hasText: "Supported by product" }).count();
    equal(gone, 0, "supported rows still shown under the not-addressed filter");
  });

  await check("telemetry is off and documented, and the evidence pack downloads", async () => {
    await go(page, "data-protection");
    await page.waitForSelector("#protection-admin:not([hidden])");
    await page.waitForFunction(() => document.querySelector("#telemetry-summary").textContent.length > 10);
    assert(/^Off\. Nothing is sent\./.test(await page.textContent("#telemetry-summary")), "telemetry is not shown as off");
    equal(await page.locator("#telemetry-fields dt").count(), 9, "documented telemetry fields");
    const pack = await page.evaluate(async () => { const r = await fetch("/api/governance/evidence-pack"); return { type: r.headers.get("content-type"), n: (await r.arrayBuffer()).byteLength }; });
    assert(pack.type === "application/zip" && pack.n > 1000, `the evidence pack was ${pack.type}, ${pack.n} bytes`);
  });

  await check("feedback and the controls tab fit a phone screen", async () => {
    const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });
    const small = await phone.newPage();
    await small.goto(BASE + "/", { waitUntil: "networkidle" });
    equal(await ask(small, "What is the standard dose of Caloradine?"), "ANSWER", "the decision");
    await small.click("#feedback-down");
    const overflow = () => small.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert((await overflow()) <= 1, "the feedback form makes the page scroll sideways");
    const box = await small.locator("#feedback-send").boundingBox();
    assert(box && box.width >= 44 && box.height >= 30, "the send button is too small to tap");
    await go(small, "review");
    await small.click("#tab-controls");
    await small.waitForSelector("#controls-frameworks table");
    assert((await overflow()) <= 1, "the controls table makes the page scroll sideways");
    await phone.close();
  });

  await check("nothing on any page overflows a phone screen", async () => {
    const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });
    const small = await phone.newPage();
    await small.goto(BASE + "/", { waitUntil: "networkidle" });
    const views = await small.$$eval(".nav-link[data-view]", (links) => links.map((l) => l.dataset.view));
    const wide = [];
    for (const view of views) {
      await small.evaluate((v) => { location.hash = `#/${v}`; }, view);
      await small.waitForTimeout(400);
      const over = await small.evaluate(() => {
        const vw = document.documentElement.clientWidth;
        return document.documentElement.scrollWidth > vw + 1
          ? `${document.documentElement.scrollWidth} > ${vw}` : "";
      });
      if (over) wide.push(`${view} (${over})`);
    }
    await phone.close();
    assert(wide.length === 0, `these pages scroll sideways on a phone: ${wide.join(", ")}`);
  });

  await check("the sidebar opens and closes on a phone", async () => {
    const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });
    const small = await phone.newPage();
    await small.goto(BASE + "/", { waitUntil: "networkidle" });
    await small.click("#nav-open");
    await small.waitForFunction(() => document.querySelector("#sidebar").classList.contains("is-open")
      || !document.querySelector("#nav-scrim").hidden);
    await small.keyboard.press("Escape");
    await small.waitForFunction(() => !document.querySelector("#sidebar").classList.contains("is-open"));
    await phone.close();
  });

  await check("no page reported a script error", () => {
    assert(pageErrors.length === 0, pageErrors.slice(0, 3).join(" | "));
  });

  await browser.close();

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  if (failed.length) {
    console.log("\nFailures:");
    for (const f of failed) console.log(`  ${f.name}: ${f.error}`);
    process.exit(1);
  }
}

main().catch(async (error) => {
  if (browser) await browser.close();
  console.error(error);
  process.exit(1);
});
