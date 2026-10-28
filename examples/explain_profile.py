"""価値尺度ファイルの価値尺度を言葉にする (データセットが無くても、答えた項目から計算する)。

    python examples/explain_profile.py my_taste.json food
"""

import sys

import shiken
from shiken.encoders import EmbeddingGemma2

profile, domain = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "food"
enc = EmbeddingGemma2(modalities=())  # 語を埋め込むだけなので、画像や音の部分は読み込まない

for part in ("correction", "all"):
    out = shiken.explain(profile, encoder=enc, words=shiken.words.large(domain), part=part, top=8)
    print("素の価値尺度からのずれ" if part == "correction" else "価値尺度全体")
    print("  高く見る:", "・".join(w for w, _ in out["up"]))
    print("  低く見る:", "・".join(w for w, _ in out["down"]))
