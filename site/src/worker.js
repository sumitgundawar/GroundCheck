// Serves /media/* with HTTP byte-range support.
//
// Static assets are answered with the whole file even when a browser asks for
// a range. Chrome then cannot seek within a video, and iOS Safari will not play
// MP4 at all. This Worker runs only for /media/* (see run_worker_first in
// wrangler.jsonc); every other request is served directly from static assets.

const MEDIA_HEADERS = {
  "Accept-Ranges": "bytes",
  "Cache-Control": "public, max-age=86400",
  "X-Content-Type-Options": "nosniff",
};

const RANGE = /^bytes=(\d*)-(\d*)$/;

export default {
  async fetch(request, env) {
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response(null, { status: 405, headers: { Allow: "GET, HEAD" } });
    }

    // Ask for the whole asset; ranges are applied here.
    const assetRequest = new Request(request.url, { method: "GET" });
    const asset = await env.ASSETS.fetch(assetRequest);
    if (asset.status !== 200) return asset;

    const type = asset.headers.get("Content-Type") ?? "application/octet-stream";
    const etag = asset.headers.get("ETag");
    const base = { ...MEDIA_HEADERS, "Content-Type": type, ...(etag ? { ETag: etag } : {}) };

    const match = RANGE.exec(request.headers.get("Range") ?? "");
    if (!match || (!match[1] && !match[2])) {
      const length = asset.headers.get("Content-Length");
      return new Response(request.method === "HEAD" ? null : asset.body, {
        status: 200,
        headers: { ...base, ...(length ? { "Content-Length": length } : {}) },
      });
    }

    const bytes = await asset.arrayBuffer();
    const size = bytes.byteLength;
    let start;
    let end;
    if (match[1] === "") {
      // Suffix range: the last N bytes.
      start = Math.max(0, size - Number(match[2]));
      end = size - 1;
    } else {
      start = Number(match[1]);
      end = match[2] === "" ? size - 1 : Math.min(Number(match[2]), size - 1);
    }

    if (start > end || start >= size) {
      return new Response(null, { status: 416, headers: { ...base, "Content-Range": `bytes */${size}` } });
    }

    return new Response(request.method === "HEAD" ? null : bytes.slice(start, end + 1), {
      status: 206,
      headers: {
        ...base,
        "Content-Range": `bytes ${start}-${end}/${size}`,
        "Content-Length": String(end - start + 1),
      },
    });
  },
};
