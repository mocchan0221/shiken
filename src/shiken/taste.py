"""2 択の回答でパーソナライズする窓口 (Taste) と、価値尺度ファイル・パックの読み書き。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shiken.ask import check_kinds, uses_neutral
from shiken.concepts import Concepts
from shiken.embedding import Resolver, encoder_info, encoder_name, normalize, pack_vector, unpack_vector
from shiken.personalize import Features, PersonalAxis, Prior, default_features
from shiken.questions import STORED, Choice, Noul, Question, Score, question_from_dict, question_to_dict

PROFILE_FORMAT = "shiken-profile"
PACK_FORMAT = "shiken-pack"
LEGACY_PROFILE_FORMATS = ("mekiki-profile",)  # 改名 (2026-10-29) より前に保存した価値尺度ファイル
ITEM_VAR_GRID = (0.0, 0.3, 1.0, 2.0, 4.0)
MIN_GAIN = 1.0  # 個別の加減点なしより、周辺尤度が対数でこれだけ高いときだけ加減点を使う
RESELECT_EVERY = 10  # 加減点の大きさを選び直す間隔 (毎回だと補正の時間が約 5 倍になる)


def _neutral(q: Noul, concepts: Concepts | None) -> Noul:
    """中立の文が無い Noul に、概念の語の名詞 (「料理の写真」など) を中立の文として補う。"""
    if q.neutral is None and concepts is not None:
        return Noul(q.statement, q.contrast, concepts.noun, q.instructions)
    return q


def axis_vector(q: Question, vectors: dict) -> np.ndarray:
    """質問を埋め込み空間の 1 本の向き (素の価値尺度 w0) にする。vectors: 説明 → ベクトル。"""
    if isinstance(q, Score):
        O = np.stack(vectors["descriptions"])
        v = np.array(q.values)
        w = ((v - v.mean())[:, None] * O).sum(0)
    elif q.contrast is not None:
        w = vectors["descriptions"][0] - vectors["descriptions"][1]
    elif vectors.get("neutral") is not None:
        w = vectors["descriptions"][0] - vectors["neutral"]
    else:
        w = vectors["descriptions"][0]
    return normalize(w)


class Taste:
    """1 つの質問について、本人の 2 択の回答で価値尺度をパーソナライズする。

    question: 段階 (Score) かはい／いいえ (Noul) の質問
    items, ids: データセットの埋め込みベクトル (項目数, 次元) と、項目の名前
    encoder: 質問の説明と概念の語を埋め込むエンコーダ
    concepts: 概念の語 (shiken.concepts.FOOD など)。省くと埋め込みベクトルそのものだけでずれを表す
    reference: 比較用の集合。省くと items
    """

    def __init__(self, question: Question, items, ids=None, encoder=None, concepts: Concepts | None = None,
                 reference=None, prior: Prior | None = None, seed: int = 0, _vectors: dict | None = None):
        if isinstance(question, Choice):
            raise ValueError("選択 (Choice) の質問は、今はパーソナライズできません。段階 (Score) かはい／いいえ (Noul) を使ってください")
        if isinstance(question, Noul):
            question = _neutral(question, concepts)
        check_kinds(question)
        self.question = question
        self.encoder = encoder
        self.concepts = concepts
        self.S = normalize(items)
        if self.S.ndim != 2 or len(self.S) < 2:
            raise ValueError("items は 2 件以上の (項目数, 次元) の配列にしてください")
        self.ids = [str(i) for i in (ids if ids is not None else range(len(self.S)))]
        if len(self.ids) != len(self.S) or len(set(self.ids)) != len(self.ids):
            raise ValueError("ids は items と同じ数の、重複のない名前にしてください")
        self.index = {k: i for i, k in enumerate(self.ids)}
        self.R = self.S if reference is None else normalize(reference)
        self.resolver = Resolver(encoder)
        self.vectors = _vectors or self._question_vectors()
        self.w0 = axis_vector(question, self.vectors)
        self.concept_vectors = self.resolver.matrix(concepts.texts()) if concepts is not None else None
        if self.concept_vectors is not None:
            features = default_features(self.R, self.concept_vectors)
        else:
            features = Features(self.R, "full")
        self.prior = prior or Prior()
        self.axis = PersonalAxis(self.w0, self.R, features, self.prior)
        self.answers: list[dict] = []
        self.extra: dict[str, np.ndarray] = {}  # データセットにない、回答に出てくる項目 (価値尺度ファイルから)
        self._asked: set = set()
        self.skipped = 0  # 価値尺度ファイルから読めなかった回答の数
        self.rng = np.random.default_rng(seed)

    def _question_vectors(self) -> dict:
        q = self.question
        out = {"descriptions": [self.resolver.vector(d) for d in q.descriptions]}
        if isinstance(q, Noul) and uses_neutral(q):
            out["neutral"] = self.resolver.vector(q.neutral)
        return out

    # ---- 項目 ---------------------------------------------------------------------

    def _vec(self, k: str) -> np.ndarray:
        if k in self.index:
            return self.S[self.index[k]]
        if k in self.extra:
            return self.extra[k]
        raise KeyError(f"項目「{k}」はデータセットにも価値尺度ファイルにもありません")

    # ---- 出題と回答 -----------------------------------------------------------------

    def next_pair(self, n_candidates: int = 3000) -> tuple[str, str]:
        """次に聞く組 (項目の名前の組)。答えを聞くと価値尺度について一番多くを学べる組 (BALD)。"""
        n = len(self.S)
        asked = self._asked if len(self._asked) < n * (n - 1) // 2 else None  # 全部聞いたら重複を許す
        i, j = self.axis.next_pair(self.S, self.rng, n_candidates=n_candidates, asked=asked, ids=self.ids)
        return self.ids[i], self.ids[j]

    def predict(self, a: str, b: str) -> float:
        """答える前の予想: a が選ばれる確率。"""
        return float(self.axis.p_pair(self._vec(a)[None], self._vec(b)[None], [a], [b])[0])

    def answer(self, a: str, b: str, prefer: str | None) -> float:
        """回答を記録して補正し直す。prefer: 選ばれた方 (a か b)、None なら「同じくらい」。答える前の予想を返す。"""
        if prefer not in (a, b, None):
            raise ValueError("prefer には a か b か None (同じくらい) を渡してください")
        y = 0.5 if prefer is None else (1.0 if prefer == a else 0.0)
        p = self.predict(a, b)
        self._add(a, b, y, p)
        if len(self.answers) % RESELECT_EVERY == 0:
            self.axis.select_item_var(ITEM_VAR_GRID, MIN_GAIN)
        else:
            self.axis.fit()
        return p

    def _add(self, a: str, b: str, y: float, predicted: float | None) -> None:
        va, vb = self._vec(a), self._vec(b)  # 無い項目なら KeyError (回答は記録しない)
        self.answers.append({"a": a, "b": b, "y": y, "predicted": predicted})
        self.axis.add_pair(va, vb, y, a, b)
        if a in self.index and b in self.index:
            self._asked.add(frozenset((self.index[a], self.index[b])))

    def hit_rate(self, last: int = 10) -> float | None:
        """直近 last 問で、答える前の予想が当たった割合 (「同じくらい」は除く)。"""
        rs = [r for r in self.answers[-last:] if r["y"] != 0.5 and r["predicted"] is not None]
        if not rs:
            return None
        return float(np.mean([(r["predicted"] > 0.5) == (r["y"] == 1) for r in rs]))

    @property
    def n_answers(self) -> int:
        return len(self.answers)

    @property
    def base_weight(self) -> float:
        """素の価値尺度をどれだけ残しているか (a。回答 0 件で 1)。"""
        return float(self.axis.theta[0])

    @property
    def offset_scale(self) -> float:
        """個別の加減点の大きさ (事前分布の分散 s²)。0 なら加減点を使っていない。"""
        return float(self.axis.item_var)

    # ---- スコアリング -----------------------------------------------------------------

    def scores(self, items=None) -> tuple[np.ndarray, np.ndarray]:
        """スコアの平均と分散。items を省くとデータセットの各項目 (個別の加減点を含む)。"""
        if items is None:
            return self.axis.utility(self.S, self.ids)
        return self.axis.utility(normalize(np.atleast_2d(items)))

    def rank(self) -> list[str]:
        """項目の名前を、パーソナライズしたスコアの高い順に。"""
        mean, _ = self.scores()
        return [self.ids[i] for i in np.argsort(-mean, kind="stable")]

    def weight_vector(self) -> tuple[np.ndarray, float]:
        """価値尺度を 1 本のベクトルに畳む: スコア = w · s + c (個別の加減点は含まない)。"""
        return self.axis.weight_vector()

    # ---- 言葉にする -------------------------------------------------------------------

    def explain(self, top: int = 6, words: Concepts | None = None, part: str = "correction") -> dict:
        """価値尺度を語で表す。比較用の集合の各項目について、スコアと各語らしさの相関をとる。

        words: 照らし合わせる語 (省くと concepts)。part: "correction" は素の価値尺度からのずれだけ、"all" は全体。
        返り値: {"up": [(語, 相関), ...], "down": [...]}
        """
        words = words or self.concepts
        if words is None:
            raise ValueError("言葉にするには、concepts か words に語のリストを渡してください")
        if part not in ("correction", "all"):
            raise ValueError('part は "correction" か "all" にしてください')
        R = self.R
        u = self.axis.utility(R)[0]
        if part == "correction":
            u0 = (R @ self.axis.w0 - self.axis.u_mu) / self.axis.u_sd
            u = u - self.base_weight * u0
        if u.std() < 1e-9:
            return {"up": [], "down": []}
        Z = concept_scores(R, self.resolver.matrix(words.texts()))
        r = (Z * ((u - u.mean()) / u.std())[:, None]).mean(0)
        order = np.argsort(r)
        return {"up": [(words.words[k], float(r[k])) for k in order[::-1][:top] if r[k] > 0],
                "down": [(words.words[k], float(r[k])) for k in order[:top] if r[k] < 0]}

    # ---- 価値尺度ファイル -----------------------------------------------------------------

    def to_profile(self) -> dict:
        """価値尺度ファイルの中身 (JSON にできる dict)。"""
        used = dict.fromkeys(k for r in self.answers for k in (r["a"], r["b"]))
        qv = {"descriptions": [pack_vector(v) for v in self.vectors["descriptions"]]}
        if self.vectors.get("neutral") is not None:
            qv["neutral"] = pack_vector(self.vectors["neutral"])
        return {
            "format": PROFILE_FORMAT,
            "version": 1,
            "question": question_to_dict(self.question),
            "question_embeddings": qv,
            "concepts": self.concepts.to_dict() if self.concepts is not None else None,
            "encoder": encoder_info(self.encoder),
            "answers": [dict(r) for r in self.answers],
            "items": {k: pack_vector(self._vec(k)) for k in used},
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_profile(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, profile, items=None, ids=None, encoder=None, reference=None, prior: Prior | None = None,
             seed: int = 0) -> "Taste":
        """価値尺度ファイルを読み込み、回答から補正し直す。

        items を省くと、価値尺度ファイルに入っている答えた項目をデータセット (と比較用の集合) にする。
        埋め込みモデルの名前が違う場合は、保存した埋め込みベクトルを使わず、
        データセットにある項目への回答だけで補正する (使えなかった回答の数は skipped)。
        """
        d = profile if isinstance(profile, dict) else json.loads(Path(profile).read_text(encoding="utf-8"))
        if d.get("format") not in (PROFILE_FORMAT, *LEGACY_PROFILE_FORMATS):
            raise ValueError("価値尺度ファイルではありません (format が shiken-profile ではない)")
        saved = d.get("encoder", {}).get("name")
        same = saved is not None and saved == encoder_name(encoder)
        stored = {k: unpack_vector(v) for k, v in d.get("items", {}).items()} if same else {}
        if items is None:
            if not stored:
                raise ValueError("データセットを渡さずに読み込むには、価値尺度ファイルと同じ埋め込みモデルのエンコーダが要ります")
            ids, items = list(stored), np.stack(list(stored.values()))
        q = question_from_dict(d["question"])
        concepts = Concepts.from_dict(d["concepts"]) if d.get("concepts") else None
        vectors = None
        qv = d.get("question_embeddings")
        if same and qv:
            vectors = {"descriptions": [unpack_vector(v) for v in qv["descriptions"]]}
            if qv.get("neutral"):
                vectors["neutral"] = unpack_vector(qv["neutral"])
            # ベクトルで渡した説明 (保存時は None) を、保存したベクトルで埋めておく
            q = _fill_vectors(q, vectors["descriptions"])
        taste = cls(q, items, ids, encoder, concepts, reference, prior, seed, _vectors=vectors)
        for r in d.get("answers", []):
            for k in (r["a"], r["b"]):
                if k not in taste.index and k in stored:
                    taste.extra[k] = stored[k]
            try:
                taste._add(r["a"], r["b"], float(r["y"]), r.get("predicted"))
            except KeyError:
                taste.skipped += 1
        if taste.answers:
            taste.axis.select_item_var(ITEM_VAR_GRID, MIN_GAIN)
        return taste

    # ---- ブラウザ版のパック -----------------------------------------------------------

    def export_pack(self, path=None, include_embeddings: bool = False, decimals: int | None = 6,
                    meta: dict | None = None) -> dict:
        """ブラウザ版 (js/shiken.js) に渡す値を書き出す。

        u0: 各項目の素の価値尺度でのスコア (標準化済み)、G: 事前分布の下での項目どうしのスコアの共分散、
        Z: 各項目の概念らしさ (補正の中身の表示用)。ブラウザ版はこれだけで同じ事後分布を求める。
        """
        X = self.axis._x(self.S, False)
        u0 = X[:, 0]
        G = (X / self.axis.Lambda) @ X.T

        def r(a):
            a = np.asarray(a, dtype=np.float64)
            return (np.round(a, decimals) if decimals is not None else a).tolist()

        pack = {
            "format": PACK_FORMAT,
            "version": 1,
            "ids": self.ids,
            "question": question_to_dict(self.question),
            "encoder": encoder_info(self.encoder),
            "prior": {"a0": self.prior.a0, "sigmaA": self.prior.sigma_a},
            "itemVarGrid": list(ITEM_VAR_GRID),
            "minGain": MIN_GAIN,
            "reselectEvery": RESELECT_EVERY,
            "u0": r(u0),
            "G": r(G),
            "meta": meta or {},
        }
        if self.concepts is not None:
            pack["words"] = list(self.concepts.words)
            pack["Z"] = r(concept_scores(self.S, self.concept_vectors))
        if include_embeddings:
            # 価値尺度ファイルを Python 版と同じ形式で書き出すのに使う
            pack["profile"] = {k: v for k, v in self.to_profile().items() if k not in ("answers", "items")}
            pack["embeddings"] = {k: pack_vector(self.S[i]) for i, k in enumerate(self.ids)}
        if path is not None:
            Path(path).write_text(json.dumps(pack, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return pack


def concept_scores(S: np.ndarray, C: np.ndarray) -> np.ndarray:
    """各項目の各語らしさ。どの語の文にも分野の名詞 (「料理の写真」など) が共通するので、
    項目ごとに全語の平均を引いて相対的な強さにし、語ごとに標準化する。"""
    Z = S @ C.T
    Z = Z - Z.mean(1, keepdims=True)
    sd = Z.std(0)
    return (Z - Z.mean(0)) / np.where(sd > 0, sd, 1.0)


def _fill_vectors(q: Question, vecs: list) -> Question:
    """説明がベクトルだった所 (STORED) を、価値尺度ファイルに保存したベクトルで埋める。"""

    def fill(d, v):
        stored = d is STORED or (isinstance(d, tuple) and any(x is STORED for x in d))
        return v if stored else d

    if isinstance(q, Score):
        return Score([(v, fill(d, vecs[i])) for i, (v, d) in enumerate(q.levels)], q.instructions)
    if isinstance(q, Noul):
        c = None if q.contrast is None else fill(q.contrast, vecs[1])
        return Noul(fill(q.statement, vecs[0]), c, q.neutral, q.instructions)
    return q


def explain(profile, encoder, top: int = 6, words: Concepts | None = None, part: str = "correction",
            items=None, ids=None) -> dict:
    """価値尺度ファイルの価値尺度を語で表す。データセットを渡さなければ、答えた項目を比較用の集合にする。"""
    return Taste.load(profile, items=items, ids=ids, encoder=encoder).explain(top=top, words=words, part=part)
