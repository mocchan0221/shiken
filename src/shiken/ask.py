"""素の価値尺度で答える (回答 0 件)。

    z_k = (s · o_k − μ_k) / τ
      s   : 項目の埋め込みベクトル
      o_k : 選択肢 k の説明の埋め込みベクトル
      μ_k : 比較用の集合での s · o_k の平均 (どの項目にも似やすい説明の偏りを除く)
      τ   : 温度
を softmax で確率にする。対比の説明 (contrast) つきの Noul では μ_k を引かない
(比較用の集合の平均が 50% に寄せられ、当てはまる項目が少ない質問で確率がずれるため。
Food-101 で ECE 0.03 → 0.4)。
"""

from __future__ import annotations

import warnings

import numpy as np

from shiken.embedding import Resolver, kind_of, normalize
from shiken.questions import Answer, Choice, Noul, Question, Score


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def norm_entropy(p: np.ndarray) -> np.ndarray:
    h = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=-1)
    return h / np.log(p.shape[-1])


def check_kinds(q: Question) -> None:
    kinds = {kind_of(d) for d in q.descriptions}
    if len(kinds) > 1 or "mixed" in kinds:
        warnings.warn("1 つの質問の中で、説明の種類 (文・ファイル・ベクトル) が混ざっています。"
                      "埋め込みモデルの空間では種類ごとに固まる傾向があり、種類の違いそのものでスコアが決まることがあります",
                      stacklevel=3)


def uses_neutral(q: Noul) -> bool:
    """中立の文との差で測るか (説明が文のときだけ。画像や音と文の差には種類の違いが混ざるため)。"""
    return q.contrast is None and bool(q.neutral) and kind_of(q.statement) == "text"


def probabilities(S: np.ndarray, q: Question, resolver: Resolver, reference: np.ndarray, tau: float) -> np.ndarray:
    """(項目数, 選択肢の数) の確率。Noul は (項目数, 2) で、列 0 が「当てはまる」。"""
    if isinstance(q, Noul) and q.contrast is None:
        descs = [q.statement, q.neutral] if uses_neutral(q) else [q.statement]
        O = resolver.matrix(descs)
        sims = S @ O.T - (reference @ O.T).mean(axis=0)
        z = (sims[:, 0] - sims[:, 1] if len(descs) == 2 else sims[:, 0]) / tau
        p = 1.0 / (1.0 + np.exp(-z))
        return np.stack([p, 1 - p], axis=1)
    O = resolver.matrix(q.descriptions)
    sims = S @ O.T
    if not isinstance(q, Noul):
        sims = sims - (reference @ O.T).mean(axis=0)
    return softmax(sims / tau)


def ask(items, questions: dict[str, Question], encoder=None, reference=None, tau: float = 0.03,
        resolver: Resolver | None = None) -> list[dict[str, Answer]]:
    """データセットの各項目に、質問ごとの型の決まった答えを返す。

    items: (項目数, 次元) の埋め込みベクトル。reference: 比較用の集合 (省くと items)。
    tau: 温度。正解つきのデータで決めない限り、確率の値は目安で、意味があるのは並び順。
    """
    S = normalize(items)
    R = S if reference is None else normalize(reference)
    resolver = resolver or Resolver(encoder)
    out: list[dict[str, Answer]] = [{} for _ in range(len(S))]
    for name, q in questions.items():
        check_kinds(q)
        P = probabilities(S, q, resolver, R, tau)
        conf = 1.0 - norm_entropy(P)
        top2 = np.sort(P, axis=1)[:, ::-1][:, :2]
        margin = top2[:, 0] - top2[:, 1]
        for i, p in enumerate(P):
            a = Answer(kind=type(q).__name__.lower(), confidence=float(conf[i]), margin=float(margin[i]))
            if isinstance(q, Noul):
                a.probability = float(p[0])
            else:
                a.probabilities = dict(zip(q.labels, map(float, p)))
                if isinstance(q, Choice):
                    a.choice = q.labels[int(p.argmax())]
                elif isinstance(q, Score):
                    a.score = float(p @ np.array(q.values))
            out[i][name] = a
    return out
