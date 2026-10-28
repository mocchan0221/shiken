"""説明 (文・ファイル・ベクトル) を埋め込みベクトルにする処理と、int8 での保存形式。

エンコーダは次の形を持つオブジェクト:
    name: str                                    価値尺度ファイルに記録する名前。違えば別の空間とみなす
    embed_texts(texts) -> (n, d) の配列          質問の説明や概念の語 (検索の問い合わせ側)
    embed_files(paths) -> (n, d) の配列          画像・音声・動画。説明にファイルを使うときだけ要る
    info() -> dict                               (任意) 価値尺度ファイルに書く、埋め込みの作り方の詳細
どれも長さ 1 に正規化したベクトルを返すこと。
"""

from __future__ import annotations

import base64

import numpy as np

from shiken.questions import File


def normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return X / np.linalg.norm(X)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def encoder_name(encoder) -> str | None:
    return getattr(encoder, "name", None) if encoder is not None else None


def encoder_info(encoder) -> dict:
    if encoder is None:
        return {}
    info = encoder.info() if hasattr(encoder, "info") else {}
    return {"name": encoder_name(encoder), **info}


class Resolver:
    """説明を埋め込みベクトルにする。同じ文・ファイルは 1 回だけ埋め込む。"""

    def __init__(self, encoder=None):
        self.encoder = encoder
        self._cache: dict = {}

    def _need_encoder(self, what: str):
        if self.encoder is None:
            raise ValueError(f"{what}を埋め込むにはエンコーダ (encoder=...) が要ります")

    def prefetch(self, descriptions: list) -> None:
        texts, files = set(), set()

        def walk(d):
            if isinstance(d, tuple):
                for x in d:
                    walk(x)
            elif isinstance(d, str):
                texts.add(d)
            elif isinstance(d, File):
                files.add(d.path)

        for d in descriptions:
            walk(d)
        texts = sorted(t for t in texts if ("t", t) not in self._cache)
        files = sorted(f for f in files if ("f", f) not in self._cache)
        if texts:
            self._need_encoder("文")
            for t, e in zip(texts, self.encoder.embed_texts(texts)):
                self._cache[("t", t)] = normalize(e)
        if files:
            self._need_encoder("ファイル")
            if not hasattr(self.encoder, "embed_files"):
                raise ValueError("このエンコーダはファイルを埋め込めません (embed_files がありません)")
            for f, e in zip(files, self.encoder.embed_files(files)):
                self._cache[("f", f)] = normalize(e)

    def vector(self, d) -> np.ndarray:
        if isinstance(d, np.ndarray):
            return normalize(d)
        if isinstance(d, tuple):
            return normalize(np.mean([self.vector(x) for x in d], axis=0))
        self.prefetch([d])
        key = ("t", d) if isinstance(d, str) else ("f", d.path)
        return self._cache[key]

    def matrix(self, descriptions: list) -> np.ndarray:
        self.prefetch(descriptions)
        return np.stack([self.vector(d) for d in descriptions])


def kind_of(d) -> str:
    """説明の種類。1 つの質問の中でそろっているかを確かめるのに使う。"""
    if isinstance(d, str):
        return "text"
    if isinstance(d, File):
        return "file"
    if isinstance(d, tuple):
        kinds = {kind_of(x) for x in d}
        return kinds.pop() if len(kinds) == 1 else "mixed"
    return "vector"


# ---- int8 での保存 -------------------------------------------------------------


def pack_vector(e: np.ndarray) -> dict:
    """埋め込みベクトルを int8 + 倍率にして base64 で返す (768 次元で 1,024 文字)。"""
    e = np.asarray(e, dtype=np.float64)
    scale = float(np.abs(e).max() / 127) or 1.0
    q = np.clip(np.round(e / scale), -127, 127).astype(np.int8)
    return {"emb": base64.b64encode(q.tobytes()).decode("ascii"), "scale": scale}


def unpack_vector(d: dict) -> np.ndarray:
    q = np.frombuffer(base64.b64decode(d["emb"]), dtype=np.int8).astype(np.float64)
    return normalize(q * d["scale"])
