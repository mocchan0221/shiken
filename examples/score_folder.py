"""フォルダの画像を、質問の文だけでスコアリングする (回答 0 件)。

    python examples/score_folder.py photos/ "全然美味しくなさそうな料理の写真" "最高に美味しそうな料理の写真"
"""

import sys
from pathlib import Path

import shiken
from shiken import Score
from shiken.encoders import EmbeddingGemma2

folder, low, high = sys.argv[1], sys.argv[2], sys.argv[3]
paths = sorted(str(p) for p in Path(folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

enc = EmbeddingGemma2(modalities=("image",))
items = enc.embed_files(paths, show_progress=True)
answers = shiken.ask(items, {"q": Score([(1, low), (5, high)])}, encoder=enc)

for path, a in sorted(zip(paths, answers), key=lambda x: -x[1]["q"].score):
    print(f"{a['q'].score:.2f}  {Path(path).name}")
