// Local static server for the site. No dependencies, and it runs on any
// supported Node version, so reviewing a change needs no install.
//
// It mirrors production closely enough to catch real problems: files come
// from ./public, unknown paths get 404.html, and the rules in public/_headers
// (including the Content-Security-Policy) are applied to every response.
// `npm run preview` runs the actual Cloudflare runtime when exact parity matters.
//
// Usage: node scripts/serve.mjs [--port 4321]

import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("../public", import.meta.url)));
const DEFAULT_PORT = 4321;

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".webp": "image/webp",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".txt": "text/plain; charset=utf-8",
  ".xml": "application/xml; charset=utf-8",
};

// Files Cloudflare reads as configuration and never serves.
const CONFIG_FILES = new Set(["/_headers", "/_redirects"]);

const parsePort = (argv) => {
  const i = argv.indexOf("--port");
  const value = i === -1 ? process.env.PORT : argv[i + 1];
  const port = Number.parseInt(value ?? "", 10);
  return Number.isInteger(port) && port > 0 && port < 65536 ? port : DEFAULT_PORT;
};

// Parses the subset of the _headers format this site uses: path patterns
// with a trailing "*", followed by indented "Name: value" lines.
const parseHeaderRules = (source) => {
  const rules = [];
  let current = null;
  for (const raw of source.split("\n")) {
    if (!raw.trim() || raw.trimStart().startsWith("#")) continue;
    if (!/^\s/.test(raw)) {
      current = { pattern: raw.trim(), headers: [] };
      rules.push(current);
      continue;
    }
    const colon = raw.indexOf(":");
    if (current && colon > 0) {
      current.headers.push([raw.slice(0, colon).trim(), raw.slice(colon + 1).trim()]);
    }
  }
  return rules;
};

const matches = (pattern, path) =>
  pattern.endsWith("*") ? path.startsWith(pattern.slice(0, -1)) : pattern === path;

// Resolves a URL path to a file inside ROOT, or null. Never escapes ROOT.
const resolveFile = async (urlPath) => {
  const decoded = decodeURIComponent(urlPath);
  const candidates = decoded.endsWith("/")
    ? [join(decoded, "index.html")]
    : [decoded, `${decoded}.html`, join(decoded, "index.html")];

  for (const candidate of candidates) {
    const full = resolve(ROOT, `.${normalize(candidate)}`);
    if (full !== ROOT && !full.startsWith(ROOT + sep)) return null;
    try {
      if ((await stat(full)).isFile()) return full;
    } catch {
      // Not found: try the next candidate.
    }
  }
  return null;
};

const main = async () => {
  const headerRules = parseHeaderRules(await readFile(join(ROOT, "_headers"), "utf8"));
  const port = parsePort(process.argv.slice(2));

  const server = createServer(async (req, res) => {
    const { pathname } = new URL(req.url ?? "/", "http://localhost");

    for (const rule of headerRules) {
      if (matches(rule.pattern, pathname)) {
        for (const [name, value] of rule.headers) res.setHeader(name, value);
      }
    }
    // Strip HSTS so it is never pinned to the development host.
    res.removeHeader("Strict-Transport-Security");

    if (req.method !== "GET" && req.method !== "HEAD") {
      res.writeHead(405, { Allow: "GET, HEAD" }).end();
      return;
    }

    let file = null;
    try {
      file = CONFIG_FILES.has(pathname) ? null : await resolveFile(pathname);
    } catch {
      file = null; // Malformed percent-encoding.
    }

    const path = file ?? join(ROOT, "404.html");
    const body = await readFile(path);
    const type = MIME_TYPES[extname(path)] ?? "application/octet-stream";

    // Byte ranges, as Cloudflare serves them. Browsers need these to seek
    // within video.
    const range = file && /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
    if (range) {
      const size = body.length;
      let start = range[1] === "" ? size - Number(range[2]) : Number(range[1]);
      let end = range[1] === "" || range[2] === "" ? size - 1 : Math.min(Number(range[2]), size - 1);
      if (!(start >= 0 && start <= end)) {
        res.writeHead(416, { "Content-Range": `bytes */${size}` }).end();
        return;
      }
      res.writeHead(206, {
        "Content-Type": type,
        "Content-Length": end - start + 1,
        "Content-Range": `bytes ${start}-${end}/${size}`,
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
      });
      res.end(req.method === "HEAD" ? undefined : body.subarray(start, end + 1));
      console.log(`206 ${req.method} ${pathname}`);
      return;
    }

    const status = file ? 200 : 404;
    res.writeHead(status, {
      "Content-Type": type,
      "Content-Length": body.length,
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
    });
    res.end(req.method === "HEAD" ? undefined : body);
    console.log(`${status} ${req.method} ${pathname}`);
  });

  server.listen(port, "127.0.0.1", () => {
    console.log(`GroundCheckHealth site: http://localhost:${port}`);
  });
};

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
