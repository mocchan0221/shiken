[日本語](README.md) | English

# Demo

A page where, just by opening it in a browser, you can try answering pairwise questions and watch the value scale adapt to you. No Python or GPU needed. Your answers are not sent anywhere.

Public preview: https://shiken.mocchan.net/ (also available at `https://shiken-demo.pages.dev/`; not indexed by search engines)

## Data used

All images and music are generated, and all text is fictional. No real people or works are included.

| Data | Contents | Questions |
| :-- | :-- | :-- |
| Dishes | 179 dish photos generated with MOSS-Image | Looks tasty, Looks spicy, Healthy, Looks expensive |
| Rooms | 119 room photos generated with MOSS-Image | Would like to live there, Relaxing, Tidy |
| Scenery | 120 travel-destination landscapes generated with MOSS-Image | Would like to visit, Looks cold, Lively |
| Animals | 100 animal photos generated with MOSS-Image | Cute, Cool, Scary, Would like to keep as a pet |
| Paintings | 120 paintings in various styles generated with MOSS-Image | Paintings I like, Cute, Creepy, Realistic |
| Clothes | 100 outfits (flat lay) generated with MOSS-Image | Would like to wear, Formal, Flashy |
| World music | 110 songs across 55 genres generated with ACE-Step 1.5 (30 s) | Songs I like, Groovy, Relaxing |
| Japanese songs | 100 Japanese-language songs generated with ACE-Step 1.5 (30 s) | Songs I like, Groovy, Nostalgic |
| Novel synopses | 120 synopses of fictional novels written by Claude (12 genres) | Want to read, Looks scary, Looks tear-jerking, Looks funny, Epic |

The images were generated with MOSS-Image (Apache-2.0) and the music with ACE-Step 1.5 (MIT).

(The demo's interface and questions are in Japanese; the question labels above are English translations.)

## What you can do

1. Pick a dataset (a collection of images or sounds) and a question (e.g. the generated dishes and "looks tasty").
2. Of the two items shown at a time, pick the one that better fits the question (you can also answer with the ← → ↓ keys).
3. Each time you answer, the whole ranking reshuffles. Each item shows how far it has moved from the ranking with zero answers.
4. It shows whether its prediction made before your answer was right, and the hit rate over the last 10 questions.
5. "Your value scale in words" shows the words it rates high and the words it rates low.
6. You can save and load value-scale files. Answers are also kept in the browser's storage, so you can pick up where you left off after reopening the page.

## Building one from your own images

1. Write a config file. Copy `sets.example.json` to `sets.json` and change the image folders and questions.

   | Field | Meaning |
   | :-- | :-- |
   | `id`, `label`, `note` | The dataset's name, and the name and description shown on screen |
   | `folder` | The folder of images (or sounds). A path relative to the config file, or an absolute path |
   | `kind` | `image`, `audio`, or `text` (one .txt per item; the first line is shown as the heading) |
   | `concepts` | The concept words (`food` / `art` / `ui` / `photo` / `face` / `music` / `story`). If omitted, the explanation of the correction is not shown |
   | `questions` | For each question: `id`, `label` (button name), `prompt` (the pairwise prompt), and `question` (same as the question definition in the [value-scale file format](../docs/en/profile-format.md)) |

2. Export (requires EmbeddingGemma 2; without a GPU it runs on the CPU, taking about 2 minutes for 100 images).

   ```bash
   pip install -e ".[eg2]"
   python demo/build.py demo/sets.json
   ```

   This produces `demo/data/` (packs), `demo/thumbs/` (thumbnails), and `demo/shiken.js` (a copy of the browser version). Item names become "dataset id / file name", and the original folder paths do not appear on the page.

3. Open it. ES modules cannot be loaded when the file is opened directly, so run a local server.

   ```bash
   python -m http.server -d demo 8000     # open http://localhost:8000
   ```

## How to publish

The exported output (`demo/data/` etc.; about 130 MB for the current demo) is not committed to git; it is uploaded from your machine to Cloudflare Pages.

```bash
npx wrangler@3 login                                   # first time only: log in to Cloudflare in the browser
python demo/deploy.py --deploy --project shiken-demo   # collect into dist/, check, then upload
```

`deploy.py` collects only allowed kinds of files (`index.html`, `shiken.js`, the JSON in `data/`, the images in `thumbs/`, the sounds in `audio/`) into `dist/`. It checks the collected text files for local paths (drive names or user folders) and value-scale file names, and aborts the upload if any are found. It also adds security headers (`_headers`), such as restricting load sources to your own site and Google Fonts, and keeps the site out of search engines. The latest wrangler requires Node.js 22 or later, so on Node.js 20 use `wrangler@3`.
