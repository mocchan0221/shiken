import numpy as np
import pytest

import shiken
from shiken import Choice, File, Noul, Score


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def test_score_follows_question_direction(world):
    S, enc, axis = world["S"], world["enc"], world["axis"]
    ans = shiken.ask(S, {"q": Score([(1, "bad"), (5, "good")])}, encoder=enc)
    scores = np.array([a["q"].score for a in ans])
    assert np.all((scores >= 1) & (scores <= 5))
    for a in ans:
        assert sum(a["q"].probabilities.values()) == pytest.approx(1.0)
    # 質問の向きとの近さと同じ順に並ぶ
    assert spearman(scores, S @ axis) > 0.999


def test_noul_and_choice(world):
    S, enc = world["S"], world["enc"]
    ans = shiken.ask(S, {
        "n": Noul("good", contrast="bad"),
        "neutral": Noul("good", neutral="something"),
        "alone": Noul("good"),
        "c": Choice({"g": "good", "b": "bad", "x": "other"}),
    }, encoder=enc)
    for a in ans:
        assert 0 < a["n"].probability < 1
        assert 0 < a["neutral"].probability < 1
        assert a["c"].choice in ("g", "b", "x")
        assert 0 <= a["c"].confidence <= 1


def test_file_and_vector_descriptions(world):
    S, enc, axis = world["S"], world["enc"], world["axis"]
    ans = shiken.ask(S, {"v": Noul(axis), "f": Noul(File("example.jpg"))}, encoder=enc)
    p = np.array([a["v"].probability for a in ans])
    assert spearman(p, S @ axis) > 0.999


def test_mixed_kinds_warn(world):
    with pytest.warns(UserWarning):
        shiken.ask(world["S"], {"m": Score([(1, "bad"), (5, File("great.jpg"))])}, encoder=world["enc"])


def test_text_needs_encoder(world):
    with pytest.raises(ValueError):
        shiken.ask(world["S"], {"q": Noul("good")})
