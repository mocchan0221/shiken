import json

import numpy as np
import pytest

import shiken
from shiken import Choice, Concepts, Noul, Score

from conftest import FakeEncoder

QUESTION = Score([(1, "bad"), (5, "good")])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def answer_n(taste, truth: dict, n: int):
    for _ in range(n):
        a, b = taste.next_pair()
        taste.answer(a, b, prefer=a if truth[a] > truth[b] else b)


def make(world, **kw):
    return shiken.Taste(QUESTION, world["S"], world["ids"], world["enc"], **kw)


def test_zero_answers_follow_question(world):
    taste = make(world)
    order = [world["ids"][i] for i in np.argsort(-(world["S"] @ world["axis"]))]
    assert taste.rank() == order


def test_answers_move_toward_hidden_preference(world):
    S, ids = world["S"], world["ids"]
    truth = dict(zip(ids, S @ world["hidden"]))
    taste = make(world)
    before = spearman(taste.scores()[0], S @ world["hidden"])
    answer_n(taste, truth, 40)
    after = spearman(taste.scores()[0], S @ world["hidden"])
    assert after > before + 0.2
    assert taste.hit_rate(10) is not None
    assert taste.n_answers == 40


def test_ties_and_bad_prefer(world):
    taste = make(world)
    a, b = taste.next_pair()
    p = taste.answer(a, b, prefer=None)
    assert 0 < p < 1 and taste.answers[-1]["y"] == 0.5
    with pytest.raises(ValueError):
        taste.answer(a, b, prefer="someone else")


def test_offsets_pick_up_item_specific_likes(rng):
    # 向きでは表せない、項目ごとの好き嫌いだけの本人 (小さなデータセットで同じ項目が何度も出る)
    S = rng.normal(size=(12, 32))
    ids = [f"song{i}" for i in range(12)]
    like = dict(zip(ids, rng.normal(size=12) * 3))
    taste = shiken.Taste(Score([(1, "bad"), (5, "good")]), S, ids, FakeEncoder())
    answer_n(taste, like, 60)
    assert taste.offset_scale > 0
    assert spearman(taste.scores()[0], list(like.values())) > 0.8


def test_weight_vector_matches_scores(world):
    taste = make(world, concepts=Concepts(("alpha", "beta", "gamma"), "thing"))
    truth = dict(zip(world["ids"], world["S"] @ world["hidden"]))
    answer_n(taste, truth, 15)
    w, c = taste.weight_vector()
    mean, _ = taste.scores(world["S"])  # 名前を渡さないので加減点なし
    assert np.allclose(world["S"] @ w + c, mean, atol=1e-8)


def test_save_and_load(world, tmp_path):
    truth = dict(zip(world["ids"], world["S"] @ world["hidden"]))
    taste = make(world, concepts=Concepts(("alpha", "beta"), "thing"))
    answer_n(taste, truth, 20)
    path = tmp_path / "me.json"
    taste.save(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["format"] == "shiken-profile" and len(d["answers"]) == 20
    assert all(k in d["items"] for r in d["answers"] for k in (r["a"], r["b"]))

    back = shiken.Taste.load(path, items=world["S"], ids=world["ids"], encoder=world["enc"])
    assert back.n_answers == 20 and back.skipped == 0
    assert back.rank()[:10] == taste.rank()[:10]
    assert [r["predicted"] for r in back.answers] == [r["predicted"] for r in taste.answers]

    # データセットを渡さなければ、答えた項目だけで作り直す
    only = shiken.Taste.load(path, encoder=world["enc"])
    assert set(only.ids) == set(d["items"])
    assert only.explain(top=2)["up"] or only.explain(top=2)["down"]


def test_load_on_other_dataset_uses_stored_embeddings(world, tmp_path):
    truth = dict(zip(world["ids"], world["S"] @ world["hidden"]))
    taste = make(world)
    answer_n(taste, truth, 20)
    taste.save(tmp_path / "me.json")
    other = np.random.default_rng(5).normal(size=(30, world["S"].shape[1]))
    moved = shiken.Taste.load(tmp_path / "me.json", items=other, encoder=world["enc"])
    assert moved.n_answers == 20 and moved.skipped == 0
    # 別のデータセットでも、補正した価値尺度は隠れた好みに沿う
    assert spearman(moved.scores()[0], other @ world["hidden"]) > spearman(other @ world["axis"], other @ world["hidden"])


def test_load_with_other_encoder_skips_stored(world, tmp_path):
    truth = dict(zip(world["ids"], world["S"] @ world["hidden"]))
    taste = make(world)
    answer_n(taste, truth, 10)
    taste.save(tmp_path / "me.json")
    enc2 = FakeEncoder(name="another-model", known=world["enc"].known)
    # 同じデータセットなら、回答は名前でたどれる
    same_items = shiken.Taste.load(tmp_path / "me.json", items=world["S"], ids=world["ids"], encoder=enc2)
    assert same_items.n_answers == 10
    # 別のデータセットでは、保存した埋め込みベクトルを使えないので回答は使えない
    other = shiken.Taste.load(tmp_path / "me.json", items=world["S"][:5], encoder=enc2)
    assert other.n_answers == 0 and other.skipped == 10
    with pytest.raises(ValueError):
        shiken.Taste.load(tmp_path / "me.json", encoder=enc2)


def test_explain_finds_planted_word(rng):
    dim = 32
    words = Concepts(("spicy", "sweet", "salty", "sour"), "food")
    spicy = rng.normal(size=dim)
    enc = FakeEncoder(known={"spicyのfood": spicy})
    S = rng.normal(size=(100, dim))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    ids = [str(i) for i in range(100)]
    truth = dict(zip(ids, S @ spicy))
    taste = shiken.Taste(Score([(1, "bad"), (5, "good")]), S, ids, enc, concepts=words)
    answer_n(taste, truth, 40)
    out = taste.explain(top=2)
    assert out["up"][0][0] == "spicy"
    assert taste.explain(top=2, part="all")["up"]


def test_choice_is_rejected(world):
    with pytest.raises(ValueError):
        shiken.Taste(Choice({"a": "good", "b": "bad"}), world["S"], world["ids"], world["enc"])


def test_noul_uses_concept_noun_as_neutral(world):
    taste = shiken.Taste(Noul("good"), world["S"], world["ids"], world["enc"],
                         concepts=Concepts(("x",), "thing"))
    assert taste.question.neutral == "thing"


def test_vector_question_round_trip(world, tmp_path):
    q = Noul(world["axis"], contrast=-world["axis"])
    taste = shiken.Taste(q, world["S"], world["ids"], world["enc"])
    truth = dict(zip(world["ids"], world["S"] @ world["hidden"]))
    answer_n(taste, truth, 5)
    taste.save(tmp_path / "v.json")
    back = shiken.Taste.load(tmp_path / "v.json", items=world["S"], ids=world["ids"], encoder=world["enc"])
    assert np.allclose(back.w0, taste.w0, atol=1e-2)
