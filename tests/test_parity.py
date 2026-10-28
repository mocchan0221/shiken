"""Python 版とブラウザ版 (js/shiken.js) が同じ答えを出すか。Node.js が無ければ飛ばす。"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import shiken
from shiken import Concepts, Score

from conftest import FakeEncoder

NODE = shutil.which("node")
DRIVER = Path(__file__).with_name("parity.mjs")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js がありません")


def run_js(tmp_path, pack, answers, pairs):
    f = tmp_path / "case.json"
    f.write_text(json.dumps({"pack": pack, "answers": answers, "pairs": pairs}), encoding="utf-8")
    out = subprocess.run([NODE, str(DRIVER), str(f)], capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def answered_taste(S, ids, enc, truth, n, concepts):
    taste = shiken.Taste(Score([(1, "bad"), (5, "good")]), S, ids, enc, concepts=concepts)
    for _ in range(n):
        a, b = taste.next_pair()
        taste.answer(a, b, prefer=a if truth[a] > truth[b] else (None if truth[a] == truth[b] else b))
    # ブラウザ版の fromProfile と同じく、全回答で加減点の大きさを選び直した状態で比べる
    return shiken.Taste.load(taste.to_profile(), items=S, ids=ids, encoder=enc)


@pytest.mark.parametrize("case,n", [("direction", 6), ("direction", 35), ("item_specific", 35)])
def test_python_and_js_agree(tmp_path, case, n):
    rng = np.random.default_rng(1)
    concepts = Concepts(("alpha", "beta", "gamma", "delta"), "thing")
    if case == "direction":
        S = rng.normal(size=(40, 32))
        truth_vec = S @ rng.normal(size=32)
    else:  # 同じ項目が何度も出て、個別の加減点が選ばれる
        S = rng.normal(size=(10, 32))
        truth_vec = rng.normal(size=10) * 3
    ids = [f"x{i}" for i in range(len(S))]
    truth = dict(zip(ids, truth_vec))
    enc = FakeEncoder()
    taste = answered_taste(S, ids, enc, truth, n, concepts)
    if case == "item_specific":
        assert taste.offset_scale > 0
    if n < 10:
        assert taste.offset_scale == 0

    pack = taste.export_pack(decimals=None)
    pairs = [[ids[i], ids[j]] for i, j in rng.integers(0, len(ids), (25, 2)) if i != j]
    js = run_js(tmp_path, pack, taste.answers, pairs)

    assert js["itemVar"] == pytest.approx(taste.offset_scale)
    py_probs = [taste.predict(a, b) for a, b in pairs]
    assert np.abs(np.array(js["probs"]) - py_probs).max() < 1e-6
    assert np.abs(np.array(js["scores"]) - taste.scores()[0]).max() < 1e-6
    assert js["baseWeight"] == pytest.approx(taste.base_weight, abs=1e-6)
    assert [w for w, _ in js["explain"]["up"]] == [w for w, _ in taste.explain(3)["up"]]
    assert [w for w, _ in js["explainAll"]["up"]] == [w for w, _ in taste.explain(3, part="all")["up"]]
