// Pre-deploy checks for the static site. No dependencies.
//
// Fails (exit code 1) when a page:
//   - links to an in-page anchor that does not exist
//   - references a local file that does not exist (HTML, CSS or og:image)
//   - loads a script, stylesheet, font or image from another origin, which
//     the Content-Security-Policy in _headers would block in production
//   - uses inline scripts or style attributes, also blocked by that policy
//   - is missing lang, a title, a meta description, or exactly one h1
//   - skips a heading level
//   - has an image without alt text, or JSON-LD that does not parse
//   - has unbalanced or misnested tags, or list items outside a list
//
// Usage: node scripts/check.mjs

import { readFile, readdir, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("../public", import.meta.url)));

const problems = [];
const fail = (file, message) => problems.push(`${file}: ${message}`);

const exists = async (path) => {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
};

const SITE_ORIGIN = "https://groundcheckhealth.com";

// Maps a same-site URL, relative ("/x") or absolute on the site origin, to a
// path inside public/. Returns null for anything else.
const localTarget = (url) => {
  if (url.startsWith(`${SITE_ORIGIN}/`)) url = url.slice(SITE_ORIGIN.length);
  if (!url.startsWith("/") || url.startsWith("//")) return null;
  return url.split(/[?#]/)[0];
};

const attrs = (tag) => {
  const out = {};
  for (const m of tag.matchAll(/([a-zA-Z:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'))?/g)) {
    out[m[1].toLowerCase()] = m[2] ?? m[3] ?? "";
  }
  return out;
};

const checkHtml = async (name, html) => {
  const tags = [...html.matchAll(/<([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>/g)].map((m) => ({
    name: m[1].toLowerCase(),
    attrs: attrs(m[2]),
  }));

  // Tag balance and list structure. Comments, scripts and styles are skipped.
  const VOID = new Set(["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr",
    "path", "circle", "line", "rect", "polyline", "polygon", "ellipse", "use", "stop"]);
  const stripped = html.replace(/<!--[\s\S]*?-->/g, "").replace(/<(script|style)\b[\s\S]*?<\/\1>/gi, "");
  const stack = [];
  for (const m of stripped.matchAll(/<(\/?)([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*?(\/?)>/g)) {
    const [, closing, rawName, selfClosing] = m;
    const tag = rawName.toLowerCase();
    if (VOID.has(tag) || selfClosing || tag === "!doctype") continue;
    const line = stripped.slice(0, m.index).split("\n").length;
    if (!closing) {
      if (tag === "li" && !["ul", "ol", "menu"].includes(stack.at(-1)?.tag)) fail(name, `<li> outside a list near line ${line}`);
      stack.push({ tag, line });
    } else if (stack.at(-1)?.tag === tag) {
      stack.pop();
    } else {
      fail(name, `</${tag}> near line ${line} doesn't close <${stack.at(-1)?.tag ?? "nothing"}>`);
      break;
    }
  }
  if (stack.length) fail(name, `unclosed <${stack.at(-1).tag}> from line ${stack.at(-1).line}`);

  // Document basics.
  if (!/<html[^>]*\blang="[a-z-]+"/i.test(html)) fail(name, "missing <html lang>");
  if (!/<title>[^<]+<\/title>/.test(html)) fail(name, "missing <title>");
  if (name !== "404.html" && !tags.some((t) => t.name === "meta" && t.attrs.name === "description")) {
    fail(name, "missing meta description");
  }

  // Headings: one h1, no skipped levels.
  const headings = [...html.matchAll(/<h([1-6])\b/g)].map((m) => Number(m[1]));
  if (headings.filter((h) => h === 1).length !== 1) fail(name, "must have exactly one <h1>");
  headings.reduce((prev, level) => {
    if (level > prev + 1) fail(name, `heading level skips from h${prev} to h${level}`);
    return level;
  }, 0);

  // Anchors.
  const ids = new Set(tags.map((t) => t.attrs.id).filter(Boolean));
  for (const tag of tags) {
    const { href, src, style } = tag.attrs;
    // Social preview images are fetched by crawlers, so they must exist too.
    const content = tag.name === "meta" && /:image$/.test(tag.attrs.property ?? "") ? tag.attrs.content : undefined;

    if (href?.startsWith("#") && href.length > 1 && !ids.has(href.slice(1))) {
      fail(name, `anchor ${href} has no matching id`);
    }

    // Local files.
    for (const url of [href, src, content]) {
      const target = url && localTarget(url);
      if (target && target !== "/" && !(await exists(join(ROOT, target)))) {
        fail(name, `${url} does not exist in public/`);
      }
    }

    // Anything the browser would fetch as a subresource must be same-origin.
    const subresource =
      (tag.name === "script" && src) ||
      (tag.name === "img" && src) ||
      (tag.name === "link" && /stylesheet|preload|icon|manifest/.test(tag.attrs.rel ?? "") && href);
    if (subresource && /^(https?:)?\/\//.test(subresource)) {
      fail(name, `loads ${subresource} from another origin (blocked by CSP)`);
    }

    if (tag.name === "img" && tag.attrs.alt === undefined) fail(name, `<img src="${src}"> has no alt`);
    if (style !== undefined) fail(name, `<${tag.name}> uses a style attribute (blocked by CSP)`);
  }

  // Scripts: external files, or non-executable JSON-LD only.
  for (const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)) {
    const a = attrs(m[1]);
    if (a.type === "application/ld+json") {
      try {
        JSON.parse(m[2]);
      } catch (error) {
        fail(name, `JSON-LD does not parse: ${error.message}`);
      }
    } else if (!a.src && m[2].trim()) {
      fail(name, "inline <script> (blocked by CSP)");
    }
  }
};

const checkCss = async (name, css) => {
  for (const m of css.matchAll(/url\(\s*["']?([^"')]+)["']?\s*\)/g)) {
    const url = m[1];
    if (/^(https?:)?\/\//.test(url)) fail(name, `loads ${url} from another origin (blocked by CSP)`);
    const target = localTarget(url);
    if (target && !(await exists(join(ROOT, target)))) fail(name, `${url} does not exist in public/`);
  }
};

const main = async () => {
  const files = await readdir(ROOT);
  const html = files.filter((f) => f.endsWith(".html"));
  const css = files.filter((f) => f.endsWith(".css"));

  for (const f of html) await checkHtml(f, await readFile(join(ROOT, f), "utf8"));
  for (const f of css) await checkCss(f, await readFile(join(ROOT, f), "utf8"));

  if (problems.length) {
    console.error(`Site check failed with ${problems.length} problem(s):\n`);
    for (const p of problems) console.error(`  - ${p}`);
    process.exitCode = 1;
    return;
  }
  console.log(`Site check passed: ${html.length} page(s), ${css.length} stylesheet(s).`);
};

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
