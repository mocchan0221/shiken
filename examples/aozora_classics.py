"""あらすじで教えた好みを、青空文庫の名作に当てはめる。

デモの「小説のあらすじ」で 2 択に答えて保存した価値尺度ファイルを使い、著作権の切れた日本の小説を
Wikipedia の「あらすじ」の節で並べる。上位の作品は青空文庫でそのまま読める。

    python examples/aozora_classics.py my_novels.json              # 上位と下位を表示
    python examples/aozora_classics.py my_novels.json --html out.html

作品の一覧は青空文庫 (著作権の切れた作品のファイルは自由に使える)、あらすじは Wikipedia (CC BY-SA 4.0) から
実行時に取ってくる。取ったものは --cache のファイルに保存し、2 回目からは取りに行かない。
作品名で記事を探すと同名の別の記事に当たるので、著者の記事から作品の記事へのリンクをたどる。
"""

import argparse
import csv
import html
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import shiken
from shiken.encoders import EmbeddingGemma2

AOZORA = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
WIKI = "https://ja.wikipedia.org/w/api.php?"
UA = "shiken-example/0.1"


def get(url: str) -> bytes:
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001  (名前解決の一時的な失敗や 429 は、間を空けて取り直す)
            if k == 4:
                raise
            time.sleep(5 * (k + 1))


def wikitexts(pages: list[str]) -> dict[str, str]:
    """記事名 → 本文。転送をたどり、50 件ずつまとめて取る。"""
    out = {}
    for s in range(0, len(pages), 50):
        chunk = pages[s: s + 50]
        q = urllib.parse.urlencode({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
                                    "titles": "|".join(chunk), "redirects": 1, "format": "json", "formatversion": 2})
        d = json.loads(get(WIKI + q))["query"]
        alias = {m["from"]: m["to"] for k in ("normalized", "redirects") for m in d.get(k, [])}
        found = {p["title"]: p["revisions"][0]["slots"]["main"]["content"] for p in d.get("pages", []) if "revisions" in p}
        for name in chunk:
            t = name
            while t in alias:
                t = alias[t]
            if t in found:
                out[name] = found[t]
        time.sleep(1)
    return out


def plain(s: str) -> str:
    """ウィキ記法を外して地の文にする。"""
    s = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->", "", s, flags=re.S)
    while re.search(r"\{\{[^{}]*\}\}", s):  # 入れ子のテンプレートを内側から消す
        s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"\[\[(?:ファイル|画像|File|Image):.*?\]\]", "", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"<[^>]+>|'''?", "", s)
    return s.strip()


def synopsis(text: str, limit: int = 400) -> str:
    """「あらすじ」などの節の冒頭。無ければ空。"""
    parts = re.split(r"^(={2,3})\s*(.+?)\s*\1\s*$", text, flags=re.M)
    for k in range(1, len(parts), 3):
        if re.search(r"あらすじ|ストーリー|粗筋", parts[k + 1]):
            # 見出しの直後が空で、下の節に書いてある記事もあるので、続く節も足す
            body = "\n\n".join(parts[j + 2] for j in range(k, min(k + 9, len(parts)), 3))
            paras = [plain(" ".join(l for l in p.splitlines() if l.strip() and l.lstrip()[0] not in "*#:;|!"))
                     for p in re.split(r"\n\s*\n", body)]
            t = "".join(p for p in paras if len(p) >= 30)
            if t:
                end = t.rfind("。", 0, limit)
                return t if len(t) <= limit else (t[: end + 1] if end > limit // 2 else t[:limit] + "…")
    return ""


def collect(per_author: int) -> list[dict]:
    z = zipfile.ZipFile(io.BytesIO(get(AOZORA)))
    rows = csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8-sig"))
    works = defaultdict(dict)  # (著者, 姓) → 作品名 → 図書カードの URL
    for r in rows:
        # 著作権の切れた、日本の小説 (日本十進分類法 913) だけ
        if r["作品著作権フラグ"] == "なし" and r["役割フラグ"] == "著者" and "913" in r["分類番号"]:
            works[(r["姓"] + r["名"], r["姓"])].setdefault(r["作品名"], r["図書カードURL"])
    authors = wikitexts([a for a, _ in works])
    found = {}
    for (author, surname), titles in works.items():
        for target, label in re.findall(r"\[\[([^|\]#]+)(?:\|([^\]]+))?\]\]", authors.get(author, "")):
            name = next((n for n in (label.strip(), re.sub(r"\s*\(.*?\)$", "", target).strip()) if n in titles), None)
            if name and target not in found:
                found[target] = {"title": name, "author": author, "surname": surname, "card": titles[name]}
    texts = wikitexts(list(found))
    out, count = {}, defaultdict(int)
    for art, w in found.items():
        t = texts.get(art, "")
        if w["surname"] not in t[:3000] or (w["title"], w["author"]) in out or count[w["author"]] >= per_author:
            continue  # 同名の別の記事・重複・多すぎる著者 (江戸川乱歩だけで 2 割になる) を除く
        s = synopsis(t)
        if s:
            out[(w["title"], w["author"])] = {**w, "synopsis": s}
            count[w["author"]] += 1
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", type=Path, help="小説のあらすじで作った価値尺度ファイル")
    ap.add_argument("--cache", type=Path, default=Path("aozora_classics.json"))
    ap.add_argument("--per-author", type=int, default=15)
    ap.add_argument("--html", type=Path)
    args = ap.parse_args()

    if args.cache.exists():
        works = json.loads(args.cache.read_text(encoding="utf-8"))
    else:
        print("青空文庫と Wikipedia から作品を集めています (1〜3 分)…")
        works = collect(args.per_author)
        args.cache.write_text(json.dumps(works, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(works)} 作品")

    enc = EmbeddingGemma2(modalities=())  # 文だけなので、画像や音の部分は読み込まない
    # デモのあらすじと同じ形 (1 行目が題、2 行目からがあらすじ) にそろえて埋め込む
    items = enc.embed_documents([f"{w['title']}\n{w['synopsis']}\n" for w in works])
    ids = [f"{w['title']}（{w['author']}）" for w in works]
    taste = shiken.Taste.load(args.profile, items=items, ids=ids, encoder=enc)
    ranked = taste.rank()
    by_id = dict(zip(ids, works))
    for name, part in (("上位", ranked[:15]), ("下位", ranked[-5:])):
        print(f"\n{name}")
        for i in part:
            print(f"  {ranked.index(i) + 1:3d}. {i}  {by_id[i]['card']}")

    if args.html:
        cards = "".join(f'<details><summary><b>{r + 1}. {html.escape(i)}</b></summary><p>{html.escape(by_id[i]["synopsis"])}</p>'
                        f'<a href="{html.escape(by_id[i]["card"])}">青空文庫で読む</a></details>' for r, i in enumerate(ranked))
        args.html.write_text(f'<!doctype html><meta charset="utf-8"><title>青空文庫の名作を私の好みで</title>'
                             f'<body style="max-width:760px;margin:auto;font-family:sans-serif;line-height:1.7">'
                             f'<h1>青空文庫の名作を私の好みで並べる</h1><p>あらすじの出典: Wikipedia (CC BY-SA 4.0)</p>{cards}',
                             encoding="utf-8")
        print(f"\n書き出しました: {args.html}")


if __name__ == "__main__":
    main()
