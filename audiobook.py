#!/usr/bin/env python3
"""
EPUB -> single audiobook MP3 using Google Cloud Text-to-Speech (Neural2).

Pip:
    pip install ebooklib beautifulsoup4 lxml requests tqdm mutagen \
                google-auth

System (macOS):
    brew install ffmpeg

Auth (pick one):
    --api-key  YOUR_KEY                # Google API key (REST, simplest)
    GOOGLE_TTS_API_KEY env var         # same, via env
    --key /path/to/sa.json             # service-account JSON (recommended)
    GOOGLE_APPLICATION_CREDENTIALS env # same, via env

Usage:
    python audiobook.py input.epub output.mp3 --key /path/to/sa.json
    python audiobook.py input.epub output.mp3 --api-key $GOOGLE_TTS_API_KEY
    python audiobook.py input.epub output.mp3 --dry-run     # just print stats

Notes:
  * Neural2 free tier = 1,000,000 chars / month (then $0.000016/char).
    Run --dry-run first to see total chars.
  * Output is a single MP3 with ID3 tags + CHAP frames so chapter navigation
    works in Apple Books / Audible-compatible players.
  * Resumable: existing chunk MP3s in --workdir are reused.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# EPUB chapter docs are technically XHTML; the lxml HTML parser handles them
# fine but bs4 emits a benign warning. Silence it.
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass
# Suppress urllib3's LibreSSL notice on macOS system Python.
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from ebooklib import epub, ITEM_DOCUMENT
from mutagen.id3 import (
    ID3,
    TIT2,
    TPE1,
    TALB,
    TCON,
    CHAP,
    CTOC,
    CTOCFlags,
    error as ID3Error,
)
from mutagen.mp3 import MP3
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
DEFAULT_VOICE = "en-US-Neural2-F"
DEFAULT_LANG = "en-US"
MAX_SSML_BYTES = 4800           # Google hard limit is 5000 bytes for SSML input
MAX_RETRIES = 6
RETRY_BACKOFF = 1.7
DEFAULT_WORKERS = 8             # Neural2 default quota is 1000 req/min
DEFAULT_RPM = 900               # rate-limit headroom

SKIP_TAGS = {"script", "style", "nav", "header", "footer", "form", "svg"}
EMPH_STRONG = {"strong", "b"}
EMPH_MODERATE = {"em", "i"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Sentence splitter; tolerant of leading SSML tags after punctuation.
SENTENCE_RE = re.compile(r'(?<=[\.\!\?])\s+(?=[A-Z"\(<“‘])')


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class Chapter:
    index: int
    title: str
    paragraphs: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    chapter_index: int
    chunk_index: int
    ssml: str


# --------------------------------------------------------------------------- #
# EPUB -> SSML paragraphs
# --------------------------------------------------------------------------- #

def _node_to_ssml(node) -> str:
    """Recursively convert an HTML node tree into an SSML body fragment."""
    if isinstance(node, NavigableString):
        return html.escape(str(node))
    if not isinstance(node, Tag):
        return ""
    name = node.name.lower()
    if name in SKIP_TAGS:
        return ""
    inner = "".join(_node_to_ssml(c) for c in node.children)
    stripped = inner.strip()
    if not stripped:
        return ""

    # Avoid wrapping <emphasis> across sentence boundaries -- we split on
    # sentences later and unbalanced tags would break the SSML payload.
    spans_sentence = bool(re.search(r"[\.\!\?]\s", inner))

    if name in EMPH_STRONG and not spans_sentence:
        return f'<emphasis level="strong">{inner}</emphasis>'
    if name in EMPH_MODERATE and not spans_sentence:
        return f'<emphasis level="moderate">{inner}</emphasis>'
    if name in HEADING_TAGS:
        return f'{inner}<break time="600ms"/>'
    if name == "br":
        return " "
    return inner


def _html_to_paragraphs(html_text: str) -> Tuple[Optional[str], List[str]]:
    """Return (heading-or-None, [paragraph_ssml_fragments]) for one chapter doc."""
    soup = BeautifulSoup(html_text, "lxml")

    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()

    heading = None
    first_h = soup.find(["h1", "h2", "h3"])
    if first_h:
        heading = first_h.get_text(" ", strip=True) or None
        first_h.decompose()

    body = soup.body or soup
    paragraphs: List[str] = []
    seen_ids = set()
    for el in body.find_all(["p", "li", "blockquote", "h4", "h5", "h6"]):
        if id(el) in seen_ids:
            continue
        seen_ids.add(id(el))
        frag = _node_to_ssml(el).strip()
        if frag:
            # collapse runs of whitespace
            frag = re.sub(r"\s+", " ", frag)
            paragraphs.append(frag)

    # Fallback if doc had no block tags at all (rare)
    if not paragraphs:
        text = body.get_text("\n", strip=True)
        for p in re.split(r"\n\s*\n", text):
            p = p.strip()
            if p:
                paragraphs.append(html.escape(p))

    return heading, paragraphs


def _first_meta(book, name: str) -> Optional[str]:
    md = book.get_metadata("http://purl.org/dc/elements/1.1/", name)
    if md:
        return md[0][0]
    return None


def extract_chapters(epub_path: Path) -> Tuple[dict, List[Chapter]]:
    """Read an EPUB, return (metadata, chapters in spine order)."""
    book = epub.read_epub(str(epub_path))
    meta = {
        "title": _first_meta(book, "title") or epub_path.stem,
        "author": _first_meta(book, "creator") or "Unknown",
        "language": _first_meta(book, "language") or "en",
    }

    # Walk in spine order so chapters are sequential.
    chapters: List[Chapter] = []
    idx = 0
    spine_ids = [s[0] for s in book.spine]
    items_by_id = {it.id: it for it in book.get_items_of_type(ITEM_DOCUMENT)}
    ordered = [items_by_id[i] for i in spine_ids if i in items_by_id]
    if not ordered:
        ordered = list(book.get_items_of_type(ITEM_DOCUMENT))

    for item in ordered:
        try:
            html_text = item.get_content().decode("utf-8", errors="replace")
        except Exception:
            continue
        heading, paragraphs = _html_to_paragraphs(html_text)
        # Skip docs that produced no text (cover image, blank pages, etc.)
        total_chars = sum(len(p) for p in paragraphs)
        if total_chars < 40:
            continue
        idx += 1
        title = heading or f"Chapter {idx}"
        chapters.append(Chapter(index=idx, title=title, paragraphs=paragraphs))

    return meta, chapters


# --------------------------------------------------------------------------- #
# SSML chunking
# --------------------------------------------------------------------------- #

def _ssml_wrap(body: str, lang: str) -> str:
    return (
        f'<speak xml:lang="{lang}">'
        f'<prosody rate="medium">{body}</prosody>'
        f'</speak>'
    )


def _force_split(text: str, max_bytes: int) -> List[str]:
    """Split a single oversized sentence on commas, then on words."""
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]
    out: List[str] = []
    for piece in re.split(r"(?<=,)\s+", text):
        if len(piece.encode("utf-8")) <= max_bytes:
            out.append(piece)
            continue
        cur, cur_bytes = [], 0
        for word in piece.split():
            wb = len(word.encode("utf-8")) + 1
            if cur_bytes + wb > max_bytes and cur:
                out.append(" ".join(cur))
                cur, cur_bytes = [], 0
            cur.append(word)
            cur_bytes += wb
        if cur:
            out.append(" ".join(cur))
    return out


def _split_sentences(text: str) -> List[str]:
    parts = SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_chapters(chapters: List[Chapter], lang: str) -> List[Chunk]:
    """Pack chapters into SSML payloads up to MAX_SSML_BYTES, sentence-aligned."""
    chunks: List[Chunk] = []
    envelope_overhead = len(_ssml_wrap("", lang).encode("utf-8"))
    budget = MAX_SSML_BYTES - envelope_overhead

    for ch in chapters:
        title_ssml = (
            f'<emphasis level="strong">{html.escape(ch.title)}</emphasis>'
            f'<break time="900ms"/>'
        )
        buf: List[str] = [title_ssml]
        buf_bytes = len(title_ssml.encode("utf-8"))
        chap_chunks: List[str] = []

        def flush():
            nonlocal buf, buf_bytes
            if buf:
                chap_chunks.append("".join(buf))
            buf = []
            buf_bytes = 0

        for para in ch.paragraphs:
            sentences = _split_sentences(para) or [para]
            for sentence in sentences:
                # Hard-cap any monster sentences first
                for piece in _force_split(sentence, budget - 64):
                    seg = piece + '<break time="320ms"/>'
                    seg_bytes = len(seg.encode("utf-8"))
                    if buf_bytes + seg_bytes > budget and buf:
                        flush()
                    buf.append(seg)
                    buf_bytes += seg_bytes
            pbreak = '<break time="650ms"/>'
            buf.append(pbreak)
            buf_bytes += len(pbreak.encode("utf-8"))

        flush()

        for i, body in enumerate(chap_chunks):
            chunks.append(
                Chunk(
                    chapter_index=ch.index,
                    chunk_index=i,
                    ssml=_ssml_wrap(body, lang),
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Google TTS client (REST, supports both API key and service account)
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Token-bucket-ish limiter -- enforces minimum spacing between requests."""

    def __init__(self, rate_per_minute: int):
        self.interval = 60.0 / max(1, rate_per_minute)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                self._next = max(now, self._next) + self.interval
            else:
                self._next = now + self.interval


class GoogleTTS:
    def __init__(
        self,
        voice: str,
        lang: str,
        api_key: Optional[str] = None,
        sa_credentials_path: Optional[str] = None,
        rate_per_minute: int = DEFAULT_RPM,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ):
        self.voice = voice
        self.lang = lang
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self._rate = RateLimiter(rate_per_minute)

        if api_key:
            self._session = requests.Session()
            self._url = f"{GOOGLE_TTS_URL}?key={api_key}"
        elif sa_credentials_path:
            from google.oauth2 import service_account
            from google.auth.transport.requests import AuthorizedSession
            creds = service_account.Credentials.from_service_account_file(
                sa_credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self._session = AuthorizedSession(creds)
            self._url = GOOGLE_TTS_URL
        else:
            raise ValueError("Provide api_key or sa_credentials_path")

    def synthesize(self, ssml: str) -> bytes:
        body = {
            "input": {"ssml": ssml},
            "voice": {"languageCode": self.lang, "name": self.voice},
            "audioConfig": {
                "audioEncoding": "MP3",
                "sampleRateHertz": 24000,
                "speakingRate": self.speaking_rate,
                "pitch": self.pitch,
                "effectsProfileId": ["headphone-class-device"],
            },
        }

        last_err: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            self._rate.acquire()
            try:
                resp = self._session.post(self._url, json=body, timeout=180)
            except requests.RequestException as e:
                last_err = e
                time.sleep(RETRY_BACKOFF ** attempt)
                continue

            if resp.status_code == 200:
                audio_b64 = resp.json().get("audioContent")
                if not audio_b64:
                    raise RuntimeError(f"TTS returned no audioContent: {resp.text[:300]}")
                return base64.b64decode(audio_b64)

            text = resp.text[:400]
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"retryable {resp.status_code}: {text}")
                # Honor Retry-After if present
                ra = resp.headers.get("Retry-After")
                sleep_for = float(ra) if ra and ra.isdigit() else RETRY_BACKOFF ** attempt
                time.sleep(sleep_for + 0.1 * attempt)
                continue

            # Non-retryable: bad request, auth, etc. Fail fast.
            raise RuntimeError(f"TTS HTTP {resp.status_code}: {text}")

        raise RuntimeError(f"TTS failed after {MAX_RETRIES} retries: {last_err}")


# --------------------------------------------------------------------------- #
# Synthesis (parallel, resumable)
# --------------------------------------------------------------------------- #

def synthesize_all(
    chunks: List[Chunk],
    tts: GoogleTTS,
    workdir: Path,
    workers: int,
) -> List[Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    paths: List[Optional[Path]] = [None] * len(chunks)

    def task(i: int, chunk: Chunk) -> Tuple[int, Path]:
        out = workdir / f"chunk_{chunk.chapter_index:04d}_{chunk.chunk_index:04d}.mp3"
        if out.exists() and out.stat().st_size > 0:
            return i, out
        audio = tts.synthesize(chunk.ssml)
        tmp = out.with_suffix(".mp3.part")
        with open(tmp, "wb") as f:
            f.write(audio)
        tmp.replace(out)
        return i, out

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, i, c): i for i, c in enumerate(chunks)}
        with tqdm(total=len(chunks), desc="Synthesizing", unit="chunk") as pbar:
            for fut in concurrent.futures.as_completed(futures):
                idx, out = fut.result()
                paths[idx] = out
                pbar.update(1)

    return [p for p in paths if p is not None]


# --------------------------------------------------------------------------- #
# Concat + tags
# --------------------------------------------------------------------------- #

def ffmpeg_concat(parts: List[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", dir=str(output.parent), delete=False
    ) as f:
        list_path = Path(f.name)
        for p in parts:
            # ffmpeg concat demuxer needs single-quoted absolute paths;
            # escape any embedded single quotes.
            esc = str(p.resolve()).replace("'", r"'\''")
            f.write(f"file '{esc}'\n")
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output),
        ]
        subprocess.run(cmd, check=True)
    finally:
        list_path.unlink(missing_ok=True)


def get_mp3_duration_ms(path: Path) -> int:
    try:
        return int(MP3(str(path)).info.length * 1000)
    except Exception:
        return 0


def write_id3(
    path: Path,
    meta: dict,
    chunks: List[Chunk],
    chunk_paths: List[Path],
    chapters: List[Chapter],
) -> None:
    """Write title/artist/album + CHAP/CTOC frames so chapters are navigable."""
    audio = MP3(str(path))
    if audio.tags is None:
        audio.add_tags()
    tags: ID3 = audio.tags  # type: ignore[assignment]

    # Wipe any pre-existing chapter frames so re-runs stay clean.
    for key in list(tags.keys()):
        if key.startswith("CHAP") or key.startswith("CTOC"):
            del tags[key]

    title = meta.get("title", "")
    author = meta.get("author", "")
    tags.add(TIT2(encoding=3, text=title))
    tags.add(TPE1(encoding=3, text=author))
    tags.add(TALB(encoding=3, text=title))
    tags.add(TCON(encoding=3, text="Audiobook"))

    # Compute per-chunk durations and chapter offsets.
    durations = [get_mp3_duration_ms(p) for p in chunk_paths]
    cursor = 0
    chunk_starts: List[int] = []
    for d in durations:
        chunk_starts.append(cursor)
        cursor += d
    total_ms = cursor

    by_chapter: dict[int, List[int]] = {}
    for i, c in enumerate(chunks):
        by_chapter.setdefault(c.chapter_index, []).append(i)

    chap_ids: List[str] = []
    for ch in chapters:
        idxs = by_chapter.get(ch.index)
        if not idxs:
            continue
        start = chunk_starts[idxs[0]]
        last = idxs[-1]
        end = chunk_starts[last] + durations[last]
        cid = f"chp{ch.index:04d}"
        tags.add(
            CHAP(
                element_id=cid,
                start_time=start,
                end_time=end if end > start else total_ms,
                start_offset=0xFFFFFFFF,
                end_offset=0xFFFFFFFF,
                sub_frames=[TIT2(encoding=3, text=ch.title)],
            )
        )
        chap_ids.append(cid)

    if chap_ids:
        tags.add(
            CTOC(
                element_id="toc",
                flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
                child_element_ids=chap_ids,
                sub_frames=[TIT2(encoding=3, text="Chapters")],
            )
        )

    audio.save()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert an EPUB to a single audiobook MP3 via Google Cloud TTS."
    )
    p.add_argument("input", help="Path to .epub")
    p.add_argument("output", help="Path to output .mp3")
    p.add_argument("--key", help="Service-account JSON path (recommended)")
    p.add_argument("--api-key", help="Google API key (alternative to --key)")
    p.add_argument("--voice", default=DEFAULT_VOICE,
                   help=f"Neural2 voice name (default: {DEFAULT_VOICE})")
    p.add_argument("--lang", default=DEFAULT_LANG,
                   help=f"Language code (default: {DEFAULT_LANG})")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="Parallel synthesis workers")
    p.add_argument("--rate-per-minute", type=int, default=DEFAULT_RPM,
                   help="Cap on TTS requests per minute")
    p.add_argument("--speaking-rate", type=float, default=1.0,
                   help="0.25..4.0 (1.0 = normal)")
    p.add_argument("--pitch", type=float, default=0.0,
                   help="-20.0..20.0 semitones")
    p.add_argument("--workdir", default=None,
                   help="Where chunk MP3s land (resumable). Default: temp dir.")
    p.add_argument("--keep-workdir", action="store_true",
                   help="Don't delete chunk dir after success")
    p.add_argument("--dry-run", action="store_true",
                   help="Extract+chunk only; print stats and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    epub_path = Path(args.input).expanduser()
    if not epub_path.is_file():
        print(f"ERROR: EPUB not found: {epub_path}", file=sys.stderr)
        return 2

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("GOOGLE_TTS_API_KEY")
    sa_path = args.key or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not args.dry_run and not (api_key or sa_path):
        print(
            "ERROR: provide --api-key, --key, GOOGLE_TTS_API_KEY env, "
            "or GOOGLE_APPLICATION_CREDENTIALS env",
            file=sys.stderr,
        )
        return 2

    if not args.dry_run and not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH. Install: brew install ffmpeg",
              file=sys.stderr)
        return 2

    print(f"Reading EPUB: {epub_path.name}")
    meta, chapters = extract_chapters(epub_path)
    if not chapters:
        print("ERROR: no readable chapters found in EPUB", file=sys.stderr)
        return 1

    chunks = chunk_chapters(chapters, args.lang)

    total_chars = sum(len(c.ssml) for c in chunks)
    print(f"Title:     {meta['title']}")
    print(f"Author:    {meta['author']}")
    print(f"Chapters:  {len(chapters)}")
    print(f"Chunks:    {len(chunks)}")
    print(f"SSML size: {total_chars:,} chars (~billed chars: "
          f"{sum(len(c.ssml.encode('utf-8')) for c in chunks):,} bytes)")
    print(f"Voice:     {args.voice}  ({args.lang})")

    if args.dry_run:
        # Show first/last chunk previews
        if chunks:
            preview = re.sub(r"<[^>]+>", " ", chunks[0].ssml)[:240]
            print("\nFirst chunk preview:")
            print(re.sub(r"\s+", " ", preview).strip())
        return 0

    workdir = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="audiobook_chunks_")
    )
    print(f"Workdir:   {workdir}")

    tts = GoogleTTS(
        voice=args.voice,
        lang=args.lang,
        api_key=api_key,
        sa_credentials_path=sa_path,
        rate_per_minute=args.rate_per_minute,
        speaking_rate=args.speaking_rate,
        pitch=args.pitch,
    )

    started = time.monotonic()
    chunk_paths = synthesize_all(chunks, tts, workdir, args.workers)
    synth_secs = time.monotonic() - started
    print(f"Synthesis took {synth_secs:.1f}s "
          f"({len(chunk_paths)/max(synth_secs,1):.2f} chunks/s)")

    print(f"Concatenating to {output_path}")
    ffmpeg_concat(chunk_paths, output_path)

    print("Writing ID3 tags + chapter markers")
    write_id3(output_path, meta, chunks, chunk_paths, chapters)

    if not args.keep_workdir and not args.workdir:
        # Only auto-clean temp dirs we created ourselves
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"Done: {output_path} ({output_path.stat().st_size/1_048_576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
