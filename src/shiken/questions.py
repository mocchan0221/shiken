"""質問の型 (Score / Choice / Noul) と、型の決まった答え (Answer)。

選択肢の説明は文のほか、ファイル (File) や埋め込みベクトル (numpy 配列) でもよい。
タプルで渡すと、各要素の埋め込みベクトルを平均する (言い回しの言い換えや、例の画像を数枚)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np


@dataclass(frozen=True)
class File:
    """説明として使うファイル (画像・音声・動画)。エンコーダの embed_files で埋め込む。"""

    path: str

    def __post_init__(self):
        object.__setattr__(self, "path", str(self.path).replace("\\", "/"))


Description = Union[str, File, np.ndarray, tuple]


class _Stored:
    """価値尺度ファイルから読んだ質問で、説明がベクトルだった所の目印 (ベクトルは question_embeddings にある)。"""

    def __repr__(self):
        return "STORED"


STORED = _Stored()


@dataclass
class Score:
    """順序のある段階のどこに当たるか。levels: [(値, 説明), ...] を低い順に。

    答えは各段階の確率と、段階の値の期待値。
    """

    levels: list
    instructions: str = ""

    @property
    def labels(self) -> list[str]:
        return [str(v) for v, _ in self.levels]

    @property
    def descriptions(self) -> list:
        return [d for _, d in self.levels]

    @property
    def values(self) -> list[float]:
        return [float(v) for v, _ in self.levels]


@dataclass
class Choice:
    """選択肢から 1 つを選ぶ。options: {ラベル: 説明}。"""

    options: dict
    instructions: str = ""

    @property
    def labels(self) -> list[str]:
        return list(self.options)

    @property
    def descriptions(self) -> list:
        return list(self.options.values())


@dataclass
class Noul:
    """当てはまるかどうか。

    contrast (当てはまらない状態の説明) を渡すと、2 つの説明の 2 択として解く。
    渡さない場合、neutral (分野の名詞だけの中立の文。例:「人の顔写真」) があればその差で、
    無ければ比較用の集合での平均との差で測る。
    """

    statement: Description
    contrast: Description | None = None
    neutral: str | None = None
    instructions: str = ""

    @property
    def descriptions(self) -> list:
        return [self.statement] + ([self.contrast] if self.contrast is not None else [])


Question = Union[Score, Choice, Noul]


# ---- 保存用 ------------------------------------------------------------------------


def _desc_to_json(d):
    if isinstance(d, str):
        return d
    if isinstance(d, File):
        return {"file": d.path}
    if isinstance(d, tuple):
        return {"mean": [_desc_to_json(x) for x in d]}
    if isinstance(d, np.ndarray):
        return {"vector": True}  # ベクトルそのものは価値尺度ファイルの question_embeddings に入る
    raise TypeError(f"説明に使えない型です: {type(d)}")


def _desc_from_json(d):
    if isinstance(d, str):
        return d
    if "file" in d:
        return File(d["file"])
    if "mean" in d:
        return tuple(_desc_from_json(x) for x in d["mean"])
    if d.get("vector"):
        return STORED  # 埋め込みベクトルは question_embeddings から戻す
    raise ValueError(f"説明の形式が読めません: {d}")


def question_to_dict(q: Question) -> dict:
    if isinstance(q, Score):
        return {"type": "score", "levels": [[v, _desc_to_json(d)] for v, d in q.levels],
                "instructions": q.instructions}
    if isinstance(q, Noul):
        return {"type": "noul", "statement": _desc_to_json(q.statement),
                "contrast": None if q.contrast is None else _desc_to_json(q.contrast),
                "neutral": q.neutral, "instructions": q.instructions}
    if isinstance(q, Choice):
        return {"type": "choice", "options": {k: _desc_to_json(v) for k, v in q.options.items()},
                "instructions": q.instructions}
    raise TypeError(f"質問の型ではありません: {type(q)}")


def question_from_dict(d: dict) -> Question:
    if d["type"] == "score":
        return Score([(v, _desc_from_json(t)) for v, t in d["levels"]], d.get("instructions", ""))
    if d["type"] == "noul":
        c = d.get("contrast")
        return Noul(_desc_from_json(d["statement"]), None if c is None else _desc_from_json(c),
                    d.get("neutral"), d.get("instructions", ""))
    if d["type"] == "choice":
        return Choice({k: _desc_from_json(v) for k, v in d["options"].items()}, d.get("instructions", ""))
    raise ValueError(f"質問の型が読めません: {d.get('type')}")


@dataclass
class Answer:
    """型の決まった答え。"""

    kind: str
    probabilities: dict[str, float] = field(default_factory=dict)
    choice: str | None = None
    score: float | None = None
    probability: float | None = None  # Noul: 当てはまる確率
    confidence: float = 0.0  # 1 − 正規化したエントロピー
    margin: float = 0.0  # 1 位と 2 位の確率の差

    def to_dict(self) -> dict:
        d = {"type": self.kind, "confidence": round(self.confidence, 4), "margin": round(self.margin, 4)}
        if self.probabilities:
            d["probabilities"] = {k: round(v, 4) for k, v in self.probabilities.items()}
        if self.choice is not None:
            d["choice"] = self.choice
        if self.score is not None:
            d["score"] = round(self.score, 4)
        if self.probability is not None:
            d["probability"] = round(self.probability, 4)
        return d
