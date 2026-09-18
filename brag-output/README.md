# Launch material

A 23-second film about the thing GroundCheck does that nothing else brags
about: refusing. Everything on screen is real product output — the audit ids,
the stage timings, the retrieval scores, the refusal reason and the citation
are all from runs you can reproduce.

| File | What it is |
| --- | --- |
| `brag.mp4` | 1920×1080, 23s, with a quiet music bed. For X, LinkedIn, YouTube, a site embed. |
| `brag.jpg` | The poster frame (the refused audit record), also baked as frame 0 so every platform's thumbnail matches. |
| `brag-vertical.mp4` | 1080×1920 for Reels, TikTok and Shorts. |
| `brag-vertical.jpg` | Its poster frame. |
| `share-copy.txt` | The one caption to post with it. |
| `share-copy-variants.md` | The same claim cut for X, LinkedIn, Hacker News, Discord, plus alt text. |
| `composition/`, `composition-vertical/` | The HyperFrames sources. `index.html` is the whole film. |

## Rebuilding it

The compositions need their assets back first — they are not committed, because
the music and the fonts belong to other people. `<brag>` below is a checkout
of [latent-spaces/brag](https://github.com/latent-spaces/brag), which ships both:

```bash
cp <brag>/skills/brag/assets/music/happy-beats-business-moves-vol-9-by-ende-dot-app.mp3 \
   composition/assets/music/
cp <brag>/skills/brag/assets/sfx/interface/click_00{2,3}.ogg \
   <brag>/skills/brag/assets/sfx/impact/impactSoft_medium_001.ogg \
   <brag>/skills/brag/assets/sfx/interface/bong_001.ogg composition/assets/sfx/
cp ../site/public/fonts/atkinson-hyperlegible-*.woff2 composition/assets/fonts/
```

Then, with Node 22 or later:

```bash
cd composition
npx hyperframes check     # lint, layout, motion, WCAG contrast — must be clean
npx hyperframes render -o ../brag.mp4
```

The film was built with [/brag](https://github.com/latent-spaces/brag) on top of
HyperFrames. Music is *Happy Beats — Business Moves vol. 9* by ende.app and the
sound effects are Kenney's, both shipped with that skill. The typeface is
Atkinson Hyperlegible, from the Braille Institute, the same face the product
and the site use.

## Before posting

The demo corpus is synthetic on purpose. Zalortin, Caloradine and Veltris
syndrome do not exist, which is what makes the project safe to share and fork —
say so if the post might reach clinicians, and never let the film read as real
dosing advice.
