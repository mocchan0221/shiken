"""ターミナルで 2 択に答えて、フォルダの画像を自分に合わせて並べる。

2 枚を横に並べた画像を開くので、左なら 1、右なら 2、同じくらいなら 0、終わるなら q を入力する。

    python examples/teach_terminal.py photos/ my_taste.json
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

import shiken
from shiken import Score
from shiken.encoders import EmbeddingGemma2

folder, profile = Path(sys.argv[1]), Path(sys.argv[2])
paths = sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

enc = EmbeddingGemma2(modalities=("image",))
items = enc.embed_files(paths, show_progress=True)
if profile.exists():
    taste = shiken.Taste.load(profile, items=items, ids=paths, encoder=enc)
else:
    question = Score([(1, "全然美味しくなさそうな料理の写真"), (5, "最高に美味しそうな料理の写真")])
    taste = shiken.Taste(question, items, paths, enc, concepts=shiken.concepts.FOOD)


def show(a: str, b: str) -> None:
    ims = [Image.open(p).convert("RGB") for p in (a, b)]
    h = 480
    ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
    canvas = Image.new("RGB", (ims[0].width + ims[1].width + 20, h), "white")
    canvas.paste(ims[0], (0, 0))
    canvas.paste(ims[1], (ims[0].width + 20, 0))
    out = Path(tempfile.gettempdir()) / "shiken_pair.jpg"
    canvas.save(out)
    if sys.platform == "win32":
        os.startfile(out)
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(out)], check=False)


while True:
    a, b = taste.next_pair()
    show(a, b)
    key = input("左 1 / 右 2 / 同じくらい 0 / 終わる q > ").strip()
    if key == "q":
        break
    if key not in ("0", "1", "2"):
        continue
    p = taste.answer(a, b, prefer={"1": a, "2": b, "0": None}[key])
    print(f"  答える前の予想: 左 {p:.0%}  回答 {taste.n_answers}  直近の的中率 {taste.hit_rate(10)}")

taste.save(profile)
print("\n上位 10:")
for name in taste.rank()[:10]:
    print("  " + Path(name).name)
print("言葉にすると:", taste.explain(top=5))
