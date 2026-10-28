import hashlib

import numpy as np
import pytest

DIM = 32


def _hash_vector(key: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


class FakeEncoder:
    """文やファイル名から決まったベクトルを返す、テスト用のエンコーダ。

    known に {文: ベクトル} を入れておくと、その文はそのベクトルになる (質問の向きを決めておける)。
    """

    def __init__(self, name: str = "fake-encoder", known: dict | None = None):
        self.name = name
        self.known = dict(known or {})
        self.calls = 0

    def _one(self, key: str) -> np.ndarray:
        v = self.known.get(key)
        return _hash_vector(self.name + key) if v is None else v / np.linalg.norm(v)

    def embed_texts(self, texts):
        self.calls += 1
        return np.stack([self._one(t) for t in texts])

    def embed_files(self, paths):
        self.calls += 1
        return np.stack([self._one("file:" + p) for p in paths])

    def info(self):
        return {"model": self.name}


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def world(rng):
    """データセット 80 件と、質問の向き (good − bad) と、本人の隠れた好みの向き。"""
    S = rng.normal(size=(80, DIM))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    axis = rng.normal(size=DIM)
    axis /= np.linalg.norm(axis)
    hidden = 0.3 * axis + rng.normal(size=DIM)
    hidden /= np.linalg.norm(hidden)
    enc = FakeEncoder(known={"good": axis, "bad": -axis})
    ids = [f"item{i:02d}" for i in range(len(S))]
    return {"S": S, "ids": ids, "axis": axis, "hidden": hidden, "enc": enc}
