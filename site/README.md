# GroundCheck website

The source for [groundcheckhealth.com](https://groundcheckhealth.com): static
HTML, CSS and JavaScript with no framework and no build step, served by
[Cloudflare Workers static assets](https://developers.cloudflare.com/workers/static-assets/).

## Layout

```
site/
  public/                 everything that is served
    index.html            the page
    styles.css            design tokens and all styles
    main.js               progressive enhancement: hero replay, demo player, tabs,
                          charts, pipeline light-up, dot fields, menu, scroll spy
    orbs.js               loading orbs (ES module) on the vendored engine
    icons.svg             icon sprite built from Lucide
    404.html
    _headers              security headers and caching, applied by Cloudflare
    media/                demo recording (WebM and MP4) and its poster
    screenshots/          dashboard screenshots (WebP), also used by the README
    fonts/                Atkinson Hyperlegible Next and Mono, self-hosted (OFL)
    vendor/
      thinking-orbs/      engine.es.js from thinking-orbs 0.3.1, unmodified (MIT)
      lucide/             licence for the icons in icons.svg (ISC)
    favicon.svg, apple-touch-icon.png, og.png, robots.txt, sitemap.xml
  src/worker.js           byte-range responses for /media/* only
  scripts/
    serve.mjs             local server that applies _headers, no dependencies
    check.mjs             pre-deploy checks, no dependencies
  wrangler.jsonc          Cloudflare configuration
```

## Develop locally

Any Node.js 20+ works for local review. Nothing needs installing.

```bash
cd site
npm run serve     # http://localhost:4321
npm run check     # anchors, local files, CSP compatibility, headings, alt text
```

`serve` applies the rules in `public/_headers`, including the
Content-Security-Policy, so a change that would break in production (an inline
style or a third-party script, for example) also breaks locally.

## Rules the page follows

- **No third-party requests.** Fonts, images and scripts are self-hosted. The
  CSP (`default-src 'self'`) enforces this, and `npm run check` catches
  violations before deploy.
- **No inline scripts or style attributes**, for the same reason.
- **Works without JavaScript.** Every section is complete in plain HTML; the
  script only adds the trace replay, the screenshot tabs and copy buttons.
- **Accurate claims only.** Numbers come from `python scripts/run_eval.py`
  and examples are checked against the real pipeline. Anything not built yet
  is labelled as planned.
- **Responsive and accessible.** Test at phone, tablet and laptop widths, in
  light and dark mode, and with reduced motion.

## Third-party code

| What | Where | Licence |
| --- | --- | --- |
| Atkinson Hyperlegible Next and Mono | `public/fonts/` | SIL Open Font License 1.1 |
| Lucide icons (subset, in a sprite) | `public/icons.svg` | ISC, `public/vendor/lucide/LICENSE.txt` |
| thinking-orbs engine 0.3.1 | `public/vendor/thinking-orbs/` | MIT, `public/vendor/thinking-orbs/LICENSE.txt` |

Keep each licence file next to the code it covers. When adding UI code from
elsewhere, check the licence first: this repository is MIT, so code that
forbids redistribution (for example "MIT + Commons Clause" component
libraries) can't be copied in.

## Updating the demo video

`public/media/demo.webm`, `demo.mp4` and `demo-poster.webp` are a screen
recording of the dashboard running locally in extractive mode. They are
recorded with a scripted browser (Playwright, through a DevTools screencast at
2x), cropped to the dashboard column, and encoded with:

```bash
ffmpeg -f concat -i frames.ffconcat -vf "fps=30,crop=2100:1520:230:0,scale=1600:-2,format=yuv420p" \
  -c:v libx264 -preset slow -crf 23 -movflags +faststart -an demo.mp4
ffmpeg -f concat -i frames.ffconcat -vf "fps=30,crop=2100:1520:230:0,scale=1600:-2" \
  -c:v libvpx-vp9 -b:v 0 -crf 36 -row-mt 1 -an demo.webm
```

If the chapters move, update the `data-chapter` start times (in seconds) and
the transcript in `index.html`. Re-record whenever the dashboard changes.

## Updating screenshots

The screenshots in `public/screenshots/` are captured from the dashboard
running locally in extractive mode (`FORCE_EXTRACTIVE=true`), at 2x device
scale, and converted to WebP (`cwebp -q 88 -m 6`). Keep them unedited.

## Deploy

Deployment needs Node.js 22+ (see `.nvmrc`) and a Cloudflare login with
access to the `groundcheckhealth.com` zone.

```bash
cd site
nvm use
npm install
npx wrangler login
npm run deploy    # runs the checks, then wrangler deploy
```

The Worker is served only on the custom domain. `workers.dev` and preview URLs
are disabled in `wrangler.jsonc`.

Almost everything is served straight from static assets. The one exception is
`/media/*`, which runs `src/worker.js` to answer byte-range requests: static
assets always return whole files, and without ranges Chrome can't seek in the
demo video and iOS Safari won't play it.
