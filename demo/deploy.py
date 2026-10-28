"""デモを Cloudflare Pages に上げる: 許可したファイルだけを dist/ に集め、混入がないか調べてから上げる。

    python demo/deploy.py                         # dist/ を作って調べるだけ
    python demo/deploy.py --deploy --project shiken-demo

上げる前に Cloudflare にログインしておく (npx wrangler@3 login。wrangler の最新版は Node.js 22 以上が要る)。
上げるのは index.html・shiken.js・_headers・data/*.json・thumbs/ の画像・audio/ の音だけ。
持ち主の価値尺度ファイル (shiken-*.json)、手元のパスが入る sets.json、build.py は上げない。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE.parent / "dist"

ALLOW = [
    ("index.html", None),
    ("shiken.js", None),
    ("data", {".json"}),
    ("thumbs", {".jpg", ".jpeg", ".png", ".webp"}),
    ("audio", {".mp3", ".ogg", ".m4a", ".wav"}),
]
MAX_BYTES = 25 * 1024 * 1024  # Cloudflare Pages の 1 ファイルあたりの上限
MAX_FILES = 20000
# 文のファイルに入っていてはいけないもの (手元のパス・ユーザー名・価値尺度ファイルの名前)
FORBIDDEN = [
    re.compile(r"(?<![A-Za-z])[A-Z]:[\\/]"),  # Windows のドライブ (https:// の s:/ には当たらない)
    *(re.compile(p, re.I) for p in (r"/Users/", r"\\Users\\", r"/home/", r"AppData", r"\b(mekiki|shiken)-[\w-]+\.json\b")),
    re.compile(re.escape(Path.home().name), re.I),  # 実行したマシンのユーザー名
]
TEXT_EXTS = {".html", ".js", ".json", ".css", ".txt"}

HEADERS = """/*
  Content-Security-Policy: default-src 'none'; script-src 'self' {script_hashes}; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; img-src 'self' data: blob:; media-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: no-referrer
  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=(), usb=(), interest-cohort=()
  Cross-Origin-Opener-Policy: same-origin
  X-Robots-Tag: noindex, nofollow
/data/*
  Cache-Control: no-cache
"""


def collect() -> list[Path]:
    files = []
    for name, exts in ALLOW:
        src = HERE / name
        if not src.exists():
            raise SystemExit(f"{src} がありません。先に python demo/build.py <設定ファイル> で書き出してください")
        if src.is_file():
            files.append(src)
            continue
        for p in sorted(src.rglob("*")):
            if p.is_file():
                if p.suffix.lower() not in exts:
                    raise SystemExit(f"許可していない種類のファイルがあります: {p}")
                files.append(p)
    return files


def inline_script_hashes(html: str) -> str:
    """index.html の中に直接書いたスクリプトの sha256 (CSP でそのスクリプトだけを許すため)。"""
    out = []
    for m in re.finditer(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S):
        digest = hashlib.sha256(m.group(1).encode("utf-8")).digest()
        out.append(f"'sha256-{base64.b64encode(digest).decode()}'")
    return " ".join(out)


def check(files: list[Path]) -> None:
    problems = []
    for p in files:
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"大きすぎる ({p.stat().st_size / 2**20:.1f} MB): {p}")
        if p.suffix.lower() in TEXT_EXTS:
            text = p.read_text(encoding="utf-8", errors="replace")
            for pat in FORBIDDEN:
                m = pat.search(text)
                if m:
                    problems.append(f"{p.relative_to(HERE)} に「{m.group(0)}」が入っています")
    if len(files) > MAX_FILES:
        problems.append(f"ファイルが多すぎます ({len(files)} 個)")
    if problems:
        raise SystemExit("上げるのをやめました:\n  " + "\n  ".join(problems))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", action="store_true", help="dist/ を作ったあと Cloudflare Pages に上げる")
    ap.add_argument("--project", default="shiken-demo", help="Cloudflare Pages のプロジェクト名")
    args = ap.parse_args()

    files = collect()
    check(files)
    if DIST.exists():
        shutil.rmtree(DIST)
    for p in files:
        dst = DIST / p.relative_to(HERE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    html = (HERE / "index.html").read_text(encoding="utf-8")
    (DIST / "_headers").write_text(HEADERS.format(script_hashes=inline_script_hashes(html)), encoding="utf-8")
    total = sum(p.stat().st_size for p in files)
    print(f"dist/ を作りました: {len(files) + 1} ファイル、{total / 2**20:.1f} MB")

    if args.deploy:
        npx = shutil.which("npx") or shutil.which("npx.cmd")
        cmd = [npx, "--yes", "wrangler@3", "pages", "deploy", str(DIST), "--project-name", args.project,
               "--branch", "main", "--commit-dirty=true"]
        sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
