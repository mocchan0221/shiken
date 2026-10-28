"""デモのデータを書き出す: 設定ファイル (画像や音のフォルダと質問) → demo/data/ と縮小画像。

    python demo/build.py demo/sets.json          # EmbeddingGemma 2 が要る (pip install "shiken[eg2]")
    python -m http.server -d demo 8000           # http://localhost:8000 で開く

設定ファイルの形は demo/sets.example.json を見てください。データセットの項目の名前は
「<セットの id>/<ファイル名>」にする (元のフォルダのパスはデモに出さない)。
kind は image / audio / text。text は 1 件 1 つの .txt で、1 行目を見出しとしてページに出す。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import shiken  # noqa: E402
from shiken.concepts import BUILTIN  # noqa: E402
from shiken.encoders.eg2 import AUDIO_EXTS, IMAGE_EXTS, EmbeddingGemma2  # noqa: E402
from shiken.questions import question_from_dict  # noqa: E402

THUMB = 400
TEXT_EXTS = {".txt"}


def media_files(folder: Path, kind: str, limit: int | None) -> list[Path]:
    exts = {"audio": AUDIO_EXTS, "text": TEXT_EXTS}.get(kind, IMAGE_EXTS)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in exts and not p.name.startswith("contact_sheet"))
    return files[:limit] if limit else files


def save_media(files: list[Path], set_id: str, kind: str, out: Path) -> list[str]:
    """縮小画像 (音はそのままコピー) を書き出し、ページから見たパスを返す。文はパスの代わりに本文を返す。"""
    if kind == "text":
        return [f.read_text(encoding="utf-8").strip() for f in files]
    d = out / ("audio" if kind == "audio" else "thumbs") / set_id
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, src in enumerate(files):
        if kind == "audio":
            name = f"{k:03d}{src.suffix.lower()}"
            if not (d / name).exists():
                shutil.copyfile(src, d / name)
        else:
            from PIL import Image

            name = f"{k:03d}.jpg"
            if not (d / name).exists():
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im.thumbnail((THUMB, THUMB))
                    im.save(d / name, quality=82)
        paths.append(f"{d.relative_to(out).as_posix()}/{name}")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="設定ファイル (JSON)")
    ap.add_argument("--out", type=Path, default=HERE, help="書き出し先 (既定: demo/)")
    ap.add_argument("--device", default=None, help="cuda / cpu (省くと GPU があれば GPU)")
    args = ap.parse_args()

    conf = json.loads(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent
    out = args.out.resolve()
    (out / "data").mkdir(parents=True, exist_ok=True)
    kinds = {s.get("kind", "image") for s in conf["sets"]}
    enc = EmbeddingGemma2(modalities=tuple(sorted(kinds)), device=args.device)

    index = {"sets": []}
    for s in conf["sets"]:
        kind = s.get("kind", "image")
        folder = (base / s["folder"]).resolve()
        files = media_files(folder, kind, s.get("limit"))
        if len(files) < 2:
            raise SystemExit(f"セット「{s['id']}」のファイルが 2 つ未満です: {folder}")
        print(f"[{s['id']}] {len(files)} 件を埋め込み中…")
        S = enc.embed_files(files, show_progress=True)
        ids = [f"{s['id']}/{f.name}" for f in files]
        media = dict(zip(ids, save_media(files, s["id"], kind, out)))
        concepts = BUILTIN.get(s.get("concepts", "")) if s.get("concepts") else None
        entry = {"id": s["id"], "label": s["label"], "note": s.get("note", ""), "kind": kind, "questions": []}
        for q in s["questions"]:
            taste = shiken.Taste(question_from_dict(q["question"]), S, ids, enc, concepts=concepts)
            pack_file = f"data/{s['id']}-{q['id']}.json"
            taste.export_pack(out / pack_file, include_embeddings=True,
                              meta={"set": s["id"], "question": q["id"], "label": q["label"], "prompt": q["prompt"],
                                    "kind": kind, "media": media})
            entry["questions"].append({"id": q["id"], "label": q["label"], "pack": pack_file})
            print(f"  {q['label']}: {pack_file}")
        index["sets"].append(entry)

    (out / "data" / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copyfile(HERE.parent / "js" / "shiken.js", out / "shiken.js")
    print(f"書き出しました: {out}")


if __name__ == "__main__":
    main()
