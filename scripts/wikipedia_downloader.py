#!/usr/bin/env python3

# ╭──────────────────────────────────────╮
# │ wikipedia_downloader.py on Nelm      │
# │ Nercone <nercone@diamondgotcat.net>  │
# │ Made by Nercone / MIT License        │
# │ Copyright (c) 2025 DiamondGotCat     │
# ╰──────────────────────────────────────╯

from __future__ import annotations

import argparse
import bz2
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

DEFAULT_DUMP_URL = "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2"
DEFAULT_OUT_DIR = str(Path(__file__).parent.joinpath("pages"))

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAILING_DOTS_SPACES = re.compile(r"[ .]+$")

def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag

def wiki_title_to_url(title: str) -> str:
    t = title.replace(" ", "_")
    return "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(t, safe="()!,'-_.~")

def safe_filename(title: str, max_bytes: int = 200) -> str:
    name = _ILLEGAL_CHARS.sub("_", title)
    name = name.replace("\u200e", "").replace("\u200f", "")
    name = _TRAILING_DOTS_SPACES.sub("", name)
    if not name:
        name = "_"

    b = name.encode("utf-8")
    if len(b) <= max_bytes:
        return name

    h = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    keep = max_bytes - (2 + len(h))
    truncated = b[:keep]
    while True:
        try:
            s = truncated.decode("utf-8")
            break
        except UnicodeDecodeError:
            truncated = truncated[:-1]
            if not truncated:
                s = "_"
                break
    return f"{s}__{h}"

def shard_prefix(title: str, shard: int) -> str:
    if shard <= 0:
        return ""
    hx = hashlib.md5(title.encode("utf-8")).hexdigest()
    return hx[:shard]

@dataclass(frozen=True)
class OutputLayout:
    out_dir: Path
    ext: str
    shard: int

    def relpath_for_title(self, title: str) -> str:
        prefix = shard_prefix(title, self.shard)
        fname = safe_filename(title) + self.ext
        if prefix:
            return str(Path(prefix) / fname)
        return fname

    def abspath_for_title(self, title: str) -> Path:
        return self.out_dir / self.relpath_for_title(title)

def _have_requests() -> bool:
    try:
        import requests  # noqa
        return True
    except Exception:
        return False

def download_file(url: str, dest: Path, resume: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _have_requests():
        import requests

        headers = {}
        mode = "wb"
        downloaded = 0
        if resume and dest.exists():
            downloaded = dest.stat().st_size
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                mode = "ab"

        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            total = int(total) + downloaded if total is not None else None

            last = time.time()
            done = downloaded
            with open(dest, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last >= 1.0:
                        if total:
                            pct = done * 100.0 / total
                            print(f"\rDownloading: {done/1e9:.2f} GB / {total/1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)
                        else:
                            print(f"\rDownloading: {done/1e9:.2f} GB", end="", flush=True)
                        last = now
            print()
        return

    import urllib.request
    print("requests が見つからないため、urllibでダウンロードします（レジュームなし）。")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

_WIKI_HEADING = re.compile(r"^(={1,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)
_WIKI_BOLD_ITALIC = re.compile(r"'''''(.*?)'''''", re.DOTALL)
_WIKI_BOLD = re.compile(r"'''(.*?)'''", re.DOTALL)
_WIKI_ITALIC = re.compile(r"''(.*?)''", re.DOTALL)
_WIKI_LINK = re.compile(r"\[\[([^\]|#]+)(#[^\]|]+)?(?:\|([^\]]+))?\]\]")
_EXT_LINK = re.compile(r"\[([a-z]+://[^\s\]]+)(?:\s+([^\]]+))?\]")
_TAG_REF = re.compile(r"<ref\b[^>/]*?/?>", re.IGNORECASE)
_TAG_REF_BLOCK = re.compile(r"<ref\b[^>]*?>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
_TAG_GENERIC_BLOCKS = re.compile(
    r"<(gallery|math|code|syntaxhighlight|timeline|imagemap|score|hiero|source)\b[^>]*?>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)

def _remove_templates_fallback(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    depth = 0
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if depth > 0 and text.startswith("}}", i):
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)

def _remove_tables(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    depth = 0
    while i < n:
        if text.startswith("{|", i):
            depth += 1
            i += 2
            continue
        if depth > 0 and text.startswith("|}", i):
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)

def convert_simple(wikitext: str, title: str, layout: OutputLayout) -> str:
    if not wikitext:
        wikitext = ""

    text = wikitext.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = _TAG_REF_BLOCK.sub("", text)
    text = _TAG_REF.sub("", text)
    text = _TAG_GENERIC_BLOCKS.sub("", text)
    text = _TAG_BR.sub("\n", text)

    try:
        import mwparserfromhell  # type: ignore

        code = mwparserfromhell.parse(text)
        for tpl in code.filter_templates(recursive=True):
            code.remove(tpl)
        for tag in code.filter_tags(recursive=True):
            t = str(tag.tag).lower()
            if t in {"ref", "gallery", "math", "code", "syntaxhighlight", "timeline", "imagemap", "source", "score", "hiero"}:
                code.remove(tag)
        text = str(code)
    except Exception:
        text = _remove_templates_fallback(text)
        text = _remove_tables(text)

    text = re.sub(r"\[\[(?:File|ファイル|Image|Category|カテゴリ):[^\]]+\]\]", "", text, flags=re.IGNORECASE)

    def _h(m: re.Match) -> str:
        eq = m.group(1)
        body = m.group(2).strip()
        level = min(6, len(eq) + 0)  # keep roughly aligned: == -> ##
        return f"{'#' * level} {body}"
    text = _WIKI_HEADING.sub(_h, text)

    text = _WIKI_BOLD_ITALIC.sub(r"***\1***", text)
    text = _WIKI_BOLD.sub(r"**\1**", text)
    text = _WIKI_ITALIC.sub(r"*\1*", text)

    def _link(m: re.Match) -> str:
        target = m.group(1).strip()
        anchor = m.group(2) or ""
        label = (m.group(3) or target).strip()

        if ":" in target and not target.startswith("カテゴリ") and not target.startswith("Category"):
            return label

        rel = layout.relpath_for_title(target)
        if anchor:
            anc = anchor.lstrip("#")
            anc = anc.replace(" ", "-")
            return f"[{label}]({rel}#{urllib.parse.quote(anc)})"
        return f"[{label}]({rel})"
    text = _WIKI_LINK.sub(_link, text)

    def _ext(m: re.Match) -> str:
        url = m.group(1)
        label = (m.group(2) or url).strip()
        return f"[{label}]({url})"
    text = _EXT_LINK.sub(_ext, text)

    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return f"# {title}\n\n" + text + "\n"

def have_pandoc() -> bool:
    return shutil.which("pandoc") is not None

def convert_pandoc(wikitext: str, title: str) -> str:
    if wikitext is None:
        wikitext = ""
    proc = subprocess.run(
        ["pandoc", "-f", "mediawiki", "-t", "gfm", "--wrap=none"],
        input=wikitext.encode("utf-8", errors="replace"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    out = proc.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n").strip()
    if proc.returncode != 0 and proc.stderr:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        out = f"<!-- pandoc error: {err[:500]} -->\n\n" + out

    return f"# {title}\n\n" + out + "\n"

def open_maybe_bz2(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, "rb")
    return open(path, "rb")

def iter_pages(dump_path: Path) -> Iterable[Tuple[str, int, bool, str, int]]:
    with open_maybe_bz2(dump_path) as f:
        context = ET.iterparse(f, events=("start", "end"))
        _, root = next(context)
        for event, elem in context:
            if event == "end" and strip_ns(elem.tag) == "page":
                title = elem.findtext("./{*}title") or ""
                ns_txt = elem.findtext("./{*}ns") or "-1"
                try:
                    ns = int(ns_txt)
                except ValueError:
                    ns = -1
                page_id_txt = elem.findtext("./{*}id") or "0"
                try:
                    page_id = int(page_id_txt)
                except ValueError:
                    page_id = 0
                is_redirect = elem.find("./{*}redirect") is not None
                text = elem.findtext("./{*}revision/{*}text") or ""
                yield title, page_id, is_redirect, text, ns

                elem.clear()
                root.clear()

def ensure_unique_path(path: Path, title: str) -> Path:
    if not path.exists():
        return path
    h = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    return path.with_name(path.stem + f"__{h}" + path.suffix)

def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    os.replace(tmp, path)

def process_one(title: str, page_id: int, is_redirect: bool, wikitext: str, ns: int, layout: OutputLayout, backend: str, include_redirects: bool) -> Optional[dict]:
    if ns != 0:
        return None
    if is_redirect and not include_redirects:
        return None
    if not title:
        return None

    out_path = layout.abspath_for_title(title)
    out_path = ensure_unique_path(out_path, title)

    if backend == "pandoc":
        md = convert_pandoc(wikitext, title)
    else:
        md = convert_simple(wikitext, title, layout)

    write_text_atomic(out_path, md)
    return {"title": title, "page_id": page_id, "path": str(out_path), "redirect": is_redirect}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_DUMP_URL, help="Dump URL")
    ap.add_argument("--dump", default="jawiki-latest-pages-articles.xml.bz2", help="Local dump path")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output directory (default: pages)")
    ap.add_argument("--ext", default=".md", help="File extension (default: .md). Use '' for no extension.")
    ap.add_argument("--shard", type=int, default=0, help="Hash prefix length for sharding (e.g. 2 => pages/ab/Title.md)")
    ap.add_argument("--download", action="store_true", help="Download dump if not present (or always if --force-download)")
    ap.add_argument("--force-download", action="store_true", help="Force re-download (overwrite local dump)")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume download")
    ap.add_argument("--backend", choices=["auto", "pandoc", "simple"], default="auto", help="Markdown conversion backend")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = disable). Useful esp. with pandoc.")
    ap.add_argument("--include-redirects", action="store_true", help="Include redirect pages (default: skip)")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N pages (for testing)")
    ap.add_argument("--write-index", action="store_true", help="Write index.jsonl (title -> path mapping)")
    args = ap.parse_args()

    dump_path = Path(args.dump)
    if args.force_download and dump_path.exists():
        dump_path.unlink()

    if args.download and not dump_path.exists():
        print(f"Downloading dump from: {args.url}")
        download_file(args.url, dump_path, resume=(not args.no_resume))
    elif not dump_path.exists():
        print(f"Dump file not found: {dump_path}", file=sys.stderr)
        print("Use --download to fetch it, or specify --dump path.", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    layout = OutputLayout(out_dir=out_dir, ext=args.ext, shard=args.shard)

    backend = args.backend
    if backend == "auto":
        backend = "pandoc" if have_pandoc() else "simple"
    if backend == "pandoc" and not have_pandoc():
        print("pandoc が見つかりません。backend を simple に切り替えます。", file=sys.stderr)
        backend = "simple"

    idx_f = None
    if args.write_index:
        out_dir.mkdir(parents=True, exist_ok=True)
        idx_f = open(out_dir / "index.jsonl", "w", encoding="utf-8")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    processed = 0
    written = 0
    skipped = 0
    t0 = time.time()

    def submit_or_run(ex, item):
        title, page_id, is_redirect, wikitext, ns = item
        if args.workers <= 0:
            res = process_one(title, page_id, is_redirect, wikitext, ns, layout, backend, args.include_redirects)
            return res
        else:
            return ex.submit(
                process_one,
                title, page_id, is_redirect, wikitext, ns,
                layout, backend, args.include_redirects
            )

    if args.workers > 0:
        pending = []
        max_pending = max(10, args.workers * 4)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for item in iter_pages(dump_path):
                processed += 1
                fut = submit_or_run(ex, item)
                pending.append(fut)

                if len(pending) >= max_pending:
                    done = []
                    for f in pending:
                        if f.done():
                            done.append(f)
                    if not done:
                        done = [pending[0]]
                        done[0].result(timeout=None)
                    for f in done:
                        pending.remove(f)
                        res = f.result()
                        if res is None:
                            skipped += 1
                        else:
                            written += 1
                            if idx_f:
                                idx_f.write(json.dumps(res, ensure_ascii=False) + "\n")

                if args.limit and processed >= args.limit:
                    break

            for f in as_completed(pending):
                res = f.result()
                if res is None:
                    skipped += 1
                else:
                    written += 1
                    if idx_f:
                        idx_f.write(json.dumps(res, ensure_ascii=False) + "\n")
    else:
        for item in iter_pages(dump_path):
            processed += 1
            res = submit_or_run(None, item)
            if res is None:
                skipped += 1
            else:
                written += 1
                if idx_f:
                    idx_f.write(json.dumps(res, ensure_ascii=False) + "\n")

            if processed % 1000 == 0:
                dt = time.time() - t0
                rate = processed / dt if dt > 0 else 0.0
                print(f"Processed: {processed:,}  Written: {written:,}  Skipped: {skipped:,}  ({rate:.1f} pages/s)")

            if args.limit and processed >= args.limit:
                break

    if idx_f:
        idx_f.close()

    dt = time.time() - t0
    print(f"Done. processed={processed:,} written={written:,} skipped={skipped:,} time={dt/60:.1f} min backend={backend}")
    print(f"Output dir: {layout.out_dir.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
