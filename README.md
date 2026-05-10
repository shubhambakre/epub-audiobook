# epub-audiobook

Convert an EPUB to a single audiobook MP3 (with navigable chapter markers) using
Google Cloud Text-to-Speech Neural2 voices.

## Why

Most EPUB→MP3 scripts re-encode every chunk through `pydub`, run sequentially,
and emit tiny SSML payloads. This one:

- Packs SSML into ~4.8 KB chunks (Google's hard limit), sentence-aligned.
- Synthesizes in parallel (default 8 workers) under a token-bucket rate limiter.
- Concatenates with `ffmpeg -c copy` — no re-encode, fast on long books.
- Writes ID3 `CHAP`/`CTOC` frames so chapter navigation works in Apple Books,
  VLC, and most audiobook players (no need for M4B).
- Resumable: chunk MP3s are cached in `--workdir` keyed by chapter+index.
- Italics/bold map to SSML `<emphasis>` (only when the span doesn't cross a
  sentence boundary, otherwise sentence-splitting would unbalance the tags).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg     # macOS; apt install ffmpeg on Linux
```

## Auth

Pick one. Service-account JSON is recommended; API keys travel in URLs.

```bash
# Option A: service account
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

# Option B: API key
export GOOGLE_TTS_API_KEY=YOUR_KEY
```

Never commit credentials. The `.gitignore` blocks `*.json`, `*.key`, and
common credential filename patterns.

## Usage

```bash
# Dry run — parse EPUB, report chapter/chunk/char counts, no API calls.
python audiobook.py book.epub out.mp3 --dry-run

# Real run
python audiobook.py book.epub out.mp3

# Different voice / rate
python audiobook.py book.epub out.mp3 --voice en-US-Neural2-D --speaking-rate 1.05

# Resumable run with explicit working dir
python audiobook.py book.epub out.mp3 --workdir ./chunks --keep-workdir
```

## Free-tier note

Neural2 voices include 1,000,000 characters/month free, then $0.000016/char.
A typical 300-page non-fiction book is ~500–700k characters — runs free.
Always `--dry-run` first to see the count.

## Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--voice` | `en-US-Neural2-F` | Any Neural2 voice ID |
| `--lang` | `en-US` | Must match the voice's locale |
| `--workers` | `8` | Parallel synthesis workers |
| `--rate-per-minute` | `900` | TTS request cap (Neural2 quota is 1000/min) |
| `--speaking-rate` | `1.0` | 0.25–4.0 |
| `--pitch` | `0.0` | -20.0 to 20.0 semitones |
| `--workdir` | temp | Chunk cache; reuse to resume |
| `--keep-workdir` | off | Don't delete chunks after success |
| `--dry-run` | off | Skip TTS, print stats and exit |

## License

MIT
