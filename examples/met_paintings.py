"""生成した絵で教えた好みを、メトロポリタン美術館の所蔵品に当てはめる。

デモの「絵」で 2 択に答えて保存した価値尺度ファイルを使い、パブリックドメインの所蔵品 (画像は CC0) を並べる。

    python examples/met_paintings.py my_art.json                  # 上位と下位を表示
    python examples/met_paintings.py my_art.json --html out.html  # 画像を並べたページ

作品は美術館の Collection API から実行時に取ってきて、縮小画像を --cache のフォルダに保存する
(2 回目からは取りに行かない)。ヨーロッパの絵画・アジアの絵画と浮世絵・素描・版画などを混ぜる。
"""

import argparse
import html
import io
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

import shiken
from shiken.encoders import EmbeddingGemma2

API = "https://collectionapi.metmuseum.org/public/collection/v1"
UA = "shiken-example/0.1"
# (部門の id, 検索語, 件数)。部門の一覧は API の /departments
PLAN = [(11, "painting", 150), (6, "painting", 50), (6, "woodblock print", 50), (15, "painting", 20),
        (9, "etching", 20), (9, "watercolor", 20), (21, "painting", 10)]
# 検索には器や布も混ざるので、平面の作品だけ残す
KINDS = {"Paintings", "Prints", "Drawings", "Illustrated Books", "Codices", "Miniatures",
         "Pastels & Oil Sketches on Paper"}


def get(url: str) -> bytes:
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or k == 4:  # 検索に出ても記録が消えていて 404 になる作品がある
                raise
            time.sleep(5 * (k + 1))
        except Exception:  # noqa: BLE001  (名前解決の一時的な失敗は、間を空けて取り直す)
            if k == 4:
                raise
            time.sleep(5 * (k + 1))


def collect(folder: Path, seed: int = 0) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    rng, out, seen = random.Random(seed), [], set()
    for dept, q, n in PLAN:
        params = {"hasImages": "true", "isPublicDomain": "true", "departmentId": dept, "q": q, "limit": 100}
        total = json.loads(get(f"{API}.1/search?" + urllib.parse.urlencode({**params, "limit": 1})))["total"]
        ids = []
        for off in rng.sample(range(0, max(total - 100, 1), 100), k=min(6, max(total // 100, 1))):
            ids += json.loads(get(f"{API}.1/search?" + urllib.parse.urlencode({**params, "offset": off}))).get("objectIDs") or []
        rng.shuffle(ids)
        got = 0
        for oid in ids:
            if got >= n or oid in seen:
                continue
            seen.add(oid)
            try:
                o = json.loads(get(f"{API}/objects/{oid}"))
                if not (o.get("isPublicDomain") and o.get("primaryImageSmall") and o.get("classification") in KINDS):
                    continue
                im = Image.open(io.BytesIO(get(o["primaryImageSmall"]))).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            im.thumbnail((640, 640))
            im.save(folder / f"{oid}.jpg", quality=88)
            out.append({"id": oid, "file": f"{oid}.jpg", "title": o.get("title", ""),
                        "artist": o.get("artistDisplayName") or o.get("culture") or "作者不詳",
                        "date": o.get("objectDate", ""), "url": o.get("objectURL", "")})
            got += 1
            time.sleep(0.2)
        print(f"  部門 {dept}「{q}」: {got} 件")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", type=Path, help="絵で作った価値尺度ファイル")
    ap.add_argument("--cache", type=Path, default=Path("met_paintings"))
    ap.add_argument("--html", type=Path)
    args = ap.parse_args()

    index = args.cache / "works.json"
    if index.exists():
        works = json.loads(index.read_text(encoding="utf-8"))
    else:
        print("メトロポリタン美術館から作品を集めています (5〜10 分)…")
        works = collect(args.cache)
        index.write_text(json.dumps(works, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(works)} 作品")

    enc = EmbeddingGemma2(modalities=("image",))
    items = enc.embed_files([args.cache / w["file"] for w in works], show_progress=True)
    ids = [str(w["id"]) for w in works]
    taste = shiken.Taste.load(args.profile, items=items, ids=ids, encoder=enc)
    ranked = taste.rank()
    by_id = dict(zip(ids, works))
    label = lambda i: f"{by_id[i]['title']} — {by_id[i]['artist']}, {by_id[i]['date']}"
    for name, part in (("上位", ranked[:10]), ("下位", ranked[-5:])):
        print(f"\n{name}")
        for i in part:
            print(f"  {ranked.index(i) + 1:3d}. {label(i)}")

    if args.html:
        src = lambda i: (args.cache / by_id[i]["file"]).resolve().as_uri()
        cards = "".join(f'<a href="{html.escape(by_id[i]["url"])}" title="{html.escape(label(i))}">'
                        f'<img src="{src(i)}" loading="lazy" alt=""><span>{r + 1}</span></a>' for r, i in enumerate(ranked))
        args.html.write_text(
            '<!doctype html><meta charset="utf-8"><title>美術館の所蔵品を私の好みで</title><style>'
            'body{font-family:sans-serif;margin:16px}div{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px}'
            'a{position:relative}img{width:100%;aspect-ratio:1;object-fit:cover;display:block;border-radius:6px}'
            'span{position:absolute;left:4px;top:4px;background:#fff;border-radius:4px;padding:0 5px;font-size:12px;color:#000}</style>'
            f'<h1>メトロポリタン美術館の所蔵品を私の好みで並べる</h1><p>画像: The Metropolitan Museum of Art (CC0)</p><div>{cards}</div>',
            encoding="utf-8")
        print(f"\n書き出しました: {args.html}")


if __name__ == "__main__":
    main()
