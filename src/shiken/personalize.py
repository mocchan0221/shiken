"""パーソナライズの本体: 素の価値尺度に、本人の 2 択の回答からずれを足す。

    u(s)        = a · u0(s) + Δwᵀ φ(s)       本人のスコア (相対値)
    P(A ≻ B)    = σ(u(A) − u(B))             2 択の回答 (Bradley–Terry)

    u0 : 素の価値尺度 s · w0 を比較用の集合で標準化したもの
    φ  : 項目の特徴。比較用の集合で中心化し、平均ノルムが 1 になるようスケール
    事前分布: a ~ N(a0, σa²), Δw ~ N(0, λ⁻¹ I)

回答 0 件なら a = a0, Δw = 0 で素の並びと同じ。φ のスケールを揃えているので
λ は「個人差が logit で ±1/√λ 程度」という意味になる。
最頻値をニュートン法で求め、事後共分散はラプラス近似 (ヘッセ行列の逆) で出す。

個別の加減点: 回答に出てきた項目 k ごとに ε_k ~ N(0, s²) を足す
    u(s_k) = a · u0(s_k) + Δwᵀ φ(s_k) + ε_k
価値尺度の向きでは表せない「その項目だけの好き嫌い」を拾う。s² は周辺尤度で選ぶ (select_item_var)。
加減点は別のデータセットには移らないので、項目の名前 (ids) を渡さずにスコアリングすれば価値尺度の部分だけになる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_erfc = np.frompyfunc(math.erfc, 1, 1)


def ndtr(x: np.ndarray) -> np.ndarray:
    """標準正規分布の累積分布関数。"""
    return _erfc(-np.asarray(x, dtype=np.float64) / math.sqrt(2)).astype(np.float64) / 2


# ---- 特徴 φ ---------------------------------------------------------------------


class Features:
    """kind:
    "full"    : 768 次元そのまま
    "mrl"     : 先頭 dim 次元
    "pca"     : 比較用の集合の PCA で dim 次元に白色化
    "concept" : 概念の説明文の埋め込み (concepts, (C, 768)) への射影。各概念を比較用の集合で標準化
                → 補正 Δw が「言葉で表せる概念の組み合わせ」に制限される
    """

    def __init__(self, reference: np.ndarray, kind: str = "full", dim: int | None = None,
                 concepts: np.ndarray | None = None):
        self.kind, self.dim = kind, dim
        R = self._pre(reference)
        self.mu = R.mean(0)
        X = R - self.mu
        self.proj = None
        if kind == "pca":
            _, sv, Vt = np.linalg.svd(X, full_matrices=False)
            self.proj = (Vt[:dim] / (sv[:dim, None] / np.sqrt(len(X)))).T  # 各成分を分散 1 に
        elif kind == "concept":
            P = concepts.T
            self.proj = P / (X @ P).std(0)
        Z = X if self.proj is None else X @ self.proj
        self.scale = float(np.sqrt((Z**2).sum(1).mean()))

    def _pre(self, S: np.ndarray) -> np.ndarray:
        if self.kind == "mrl":
            S = S[:, : self.dim]
            return S / np.linalg.norm(S, axis=1, keepdims=True)
        return S

    def __call__(self, S: np.ndarray) -> np.ndarray:
        Z = self._pre(S) - self.mu
        if self.proj is not None:
            Z = Z @ self.proj
        return Z / self.scale

    @property
    def size(self) -> int:
        return len(self.mu) if self.proj is None else self.proj.shape[1]


class ConcatFeatures:
    """複数の特徴を重み付きで連結する (例: 概念 1.0 + full 0.3 → 主に概念で、残りを full で拾う)。"""

    kind = "concat"

    def __init__(self, parts: list[tuple[Features, float]]):
        self.parts = parts

    def __call__(self, S: np.ndarray) -> np.ndarray:
        return np.hstack([f(S) * w for f, w in self.parts])

    @property
    def size(self) -> int:
        return sum(f.size for f, _ in self.parts)


def default_features(reference: np.ndarray, concept_embeddings: np.ndarray) -> ConcatFeatures:
    """シミュレーションで最良だった構成: 概念 (重み 1.0) + 768 次元 (重み 0.3)、λ = 1 と組で使う。"""
    full = Features(reference, "full")
    concept = Features(reference, "concept", concepts=concept_embeddings)
    return ConcatFeatures([(concept, 1.0), (full, 0.3)])


# ---- 本体 -----------------------------------------------------------------------


@dataclass
class Prior:
    a0: float = 1.0  # 素の価値尺度の軸をどれだけ信じるか (標準化した u0 1 単位あたりの logit)
    sigma_a: float = 1.0
    sigma_b: float = 2.0
    lam: float = 1.0


def _sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))  # 1 / (1 + e^-z) をオーバーフローなしで


def _probit_mean(mu, var):
    """σ(z), z ~ N(mu, var) の期待値 (プロビット近似)。"""
    return _sigmoid(mu / np.sqrt(1.0 + np.pi * var / 8.0))


class PersonalAxis:
    def __init__(self, w0: np.ndarray, reference: np.ndarray, features: Features, prior: Prior | None = None,
                 item_var: float = 0.0):
        self.w0 = w0
        u = reference @ w0
        self.u_mu, self.u_sd = float(u.mean()), float(u.std())
        self.features = features
        # 特徴から u0 の成分を除く (直交化)。除かないと「a を大きくして Δw で打ち消す」解が
        # 選ばれ、Δw (= 概念ごとの補正) が本人の好みのずれを表さなくなる
        u0 = (u - self.u_mu) / self.u_sd
        self.beta = features(reference).T @ u0 / (u0 @ u0)
        self.prior = prior or Prior()
        p = self.prior
        k = features.size
        # θ = [a, b, Δw]
        self.m = np.concatenate([[p.a0, 0.0], np.zeros(k)])
        self.Lambda = np.concatenate([[1 / p.sigma_a**2, 1 / p.sigma_b**2], np.full(k, p.lam)])
        self.theta = self.m.copy()
        self.cov = np.diag(1 / self.Lambda)
        self.rows: list[np.ndarray] = []
        self.ys: list[float] = []
        # 個別の加減点: 回答ごとの (項目 a の名前, 項目 b の名前)。名前の無い回答は (None, None)
        self.pair_ids: list[tuple] = []
        self.item_var = float(item_var)
        self.items: dict = {}  # 項目の名前 → 個別の加減点の列番号
        self.eps = np.zeros(0)  # 個別の加減点の MAP
        self.cov_full = self.cov  # [θ, ε] の事後共分散
        self._log_evidence = float("nan")

    # 設計行列の行 (状態ごと)
    def _x(self, S: np.ndarray, with_bias: bool) -> np.ndarray:
        u0 = (S @ self.w0 - self.u_mu) / self.u_sd
        b = np.full(len(S), 1.0 if with_bias else 0.0)
        return np.column_stack([u0, b, self.features(S) - u0[:, None] * self.beta])

    # ---- 回答の追加 ----
    def add_pair(self, sa: np.ndarray, sb: np.ndarray, y: float, ida=None, idb=None) -> None:
        """y = 1: A の方が当てはまる / 0: B / 0.5: 同じくらい。ida / idb: 項目の名前 (個別の加減点に使う)。"""
        self.rows.append(self._x(sa[None], False)[0] - self._x(sb[None], False)[0])
        self.ys.append(float(y))
        self.pair_ids.append((ida, idb))

    def add_label(self, s: np.ndarray, y: float) -> None:
        """y = 1: はい / 0: いいえ。"""
        self.rows.append(self._x(s[None], True)[0])
        self.ys.append(float(y))
        self.pair_ids.append((None, None))

    @property
    def n_answers(self) -> int:
        return len(self.ys)

    # ---- 学習 ----
    def _objective(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        z = X @ theta
        nll = (np.logaddexp(0, z) - y * z).sum()
        d = theta - self.m
        return float(nll + 0.5 * (self.Lambda * d * d).sum())

    def _design(self):
        """[θ, ε] の設計行列・事前の平均・事前の精度。個別の加減点の列は、回答に出てきた項目の数だけ。"""
        X = np.array(self.rows) if self.rows else np.zeros((0, len(self.m)))
        self.items = {}
        # 回答を外から差し替えた (交差検証など) ときは pair_ids の長さが合わないので、個別の加減点は使わない
        if self.item_var > 0 and len(self.pair_ids) == len(self.rows):
            for pair in self.pair_ids:
                for k in pair:
                    if k is not None and k not in self.items:
                        self.items[k] = len(self.items)
        if not self.items:
            return X, self.m, self.Lambda
        E = np.zeros((len(self.rows), len(self.items)))
        for r, (ka, kb) in enumerate(self.pair_ids):
            if ka is not None and kb is not None:
                E[r, self.items[ka]] += 1.0
                E[r, self.items[kb]] -= 1.0
        m = np.concatenate([self.m, np.zeros(len(self.items))])
        lam = np.concatenate([self.Lambda, np.full(len(self.items), 1 / self.item_var)])
        return np.hstack([X, E]), m, lam

    def fit(self, max_iter: int = 50, tol: float = 1e-7) -> None:
        """ニュートン法 + 直線探索 (目的関数が下がるまで歩幅を半分に。素のニュートン法は発散しうる)。"""
        X, m, lam = self._design()
        n0 = len(self.m)
        theta = np.concatenate([self.theta[:n0], np.zeros(X.shape[1] - n0)])
        y = np.array(self.ys)

        def objective(t):
            z = X @ t
            d = t - m
            return float((np.logaddexp(0, z) - y * z).sum() + 0.5 * (lam * d * d).sum())

        f = objective(theta)
        for _ in range(max_iter):
            p = _sigmoid(X @ theta)
            g = X.T @ (p - y) + lam * (theta - m)
            H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(lam)
            step = np.linalg.solve(H, g)
            t = 1.0
            while t > 1e-6:
                cand = theta - t * step
                f_new = objective(cand)
                if f_new <= f:
                    break
                t /= 2
            theta, f = cand, f_new
            if np.abs(t * step).max() < tol:
                break
        p = _sigmoid(X @ theta)
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(lam)
        cov = np.linalg.inv(H)
        self.theta, self.eps = theta[:n0], theta[n0:]
        self.cov, self.cov_full = cov[:n0, :n0], cov
        # 周辺尤度 (ラプラス近似): log p(y|θ̂) − ½ (θ̂−m)ᵀ Λ (θ̂−m) − ½ log|H| + ½ log|Λ|
        z = X @ theta
        d = theta - m
        self._log_evidence = float(-(np.logaddexp(0, z) - y * z).sum() - 0.5 * (lam * d * d).sum()
                                   - 0.5 * np.linalg.slogdet(H)[1] + 0.5 * np.log(lam).sum())

    def log_evidence(self) -> float:
        """直前の fit の周辺尤度 (ラプラス近似)。個別の加減点の大きさ s² を選ぶのに使う。"""
        return self._log_evidence

    def select_item_var(self, grid=(0.0, 0.3, 1.0, 2.0, 4.0), min_gain: float = 1.0) -> float:
        """個別の加減点の大きさ s² を周辺尤度で選び、その値で補正し直す。選んだ s² を返す。

        個別の加減点なし (s² = 0) より周辺尤度が min_gain 以上 (対数のベイズ因子) 高いときだけ個別の加減点を使う。
        でたらめな回答でも、回答が少ないうちは周辺尤度が偶然 +1 程度上がることがあるため
        (生成した料理 104 枚・でたらめな 50 問で 10 回中 3 回、最大 +1.2。150 問では最大 +0.06)。
        """
        best, best_e = 0.0, -np.inf
        base_e = None
        for v in grid:
            self.item_var = v
            self.fit()
            if v == 0.0:
                base_e = self._log_evidence
            elif base_e is not None and self._log_evidence < base_e + min_gain:
                continue
            if self._log_evidence > best_e:
                best, best_e = v, self._log_evidence
        self.item_var = best
        self.fit()
        return best

    # ---- 予測 ----
    def _quad(self, X: np.ndarray) -> np.ndarray:
        """各行 x について xᵀ Σ x (事後分散)。"""
        return ((X @ self.cov) * X).sum(1)

    def _mean_var(self, X: np.ndarray, ids_a=None, ids_b=None) -> tuple[np.ndarray, np.ndarray]:
        """価値尺度の部分 X (行ごと) に、名前のある項目の個別の加減点を足した平均と分散。
        ids_b を渡すと「a − b」の差。まだ回答に出てきていない項目は、個別の加減点の事前分散 s² だけ足す。"""
        if (ids_a is None and ids_b is None) or not self.item_var > 0:
            return X @ self.theta, self._quad(X)
        E = np.zeros((len(X), len(self.items)))
        extra = np.zeros(len(X))
        for ids, sign in ((ids_a, 1.0), (ids_b, -1.0)):
            if ids is None:
                continue
            for r, k in enumerate(ids):
                if k in self.items:
                    E[r, self.items[k]] += sign
                elif k is not None:
                    extra[r] += self.item_var
        F = np.hstack([X, E])
        return X @ self.theta + E @ self.eps, ((F @ self.cov_full) * F).sum(1) + extra

    def utility(self, S: np.ndarray, ids=None) -> tuple[np.ndarray, np.ndarray]:
        """本人のスコア u(s) の平均と分散。ids (項目の名前) を渡すと個別の加減点も足す。"""
        return self._mean_var(self._x(S, False), ids)

    def p_yes(self, S: np.ndarray) -> np.ndarray:
        X = self._x(S, True)
        return _probit_mean(X @ self.theta, self._quad(X))

    def p_pair(self, SA: np.ndarray, SB: np.ndarray, ids_a=None, ids_b=None) -> np.ndarray:
        mean, var = self._mean_var(self._x(SA, False) - self._x(SB, False), ids_a, ids_b)
        return _probit_mean(mean, var)

    def info_gain(self, SA: np.ndarray, SB: np.ndarray, ids_a=None, ids_b=None) -> np.ndarray:
        """BALD (Houlsby+ 2011): その組を聞いたときに θ について得られる情報量 [bit]。"""
        mean, var = self._mean_var(self._x(SA, False) - self._x(SB, False), ids_a, ids_b)
        mu = mean * np.sqrt(np.pi / 8)  # logistic → probit
        var = var * np.pi / 8
        C2 = np.pi * np.log(2) / 2
        p = ndtr(mu / np.sqrt(1 + var))
        h = -(p * np.log2(np.clip(p, 1e-12, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1)))
        return h - np.sqrt(C2 / (var + C2)) * np.exp(-(mu**2) / (2 * (var + C2)))

    def next_pair(self, S: np.ndarray, rng: np.random.Generator, n_candidates: int = 2000, asked=None, ids=None):
        """S の中から次に聞く組 (i, j) を選ぶ。asked は聞いた組の集合 (frozenset)。ids: S の項目の名前。"""
        n = len(S)
        i = rng.integers(0, n, n_candidates)
        j = rng.integers(0, n, n_candidates)
        ok = i != j
        if asked:
            ok &= np.array([frozenset((a, b)) not in asked for a, b in zip(i, j)])
        i, j = i[ok], j[ok]
        if ids is not None:
            gain = self.info_gain(S[i], S[j], [ids[a] for a in i], [ids[b] for b in j])
        else:
            gain = self.info_gain(S[i], S[j])
        k = int(np.argmax(gain))
        return int(i[k]), int(j[k])

    # ---- 1 本のベクトルに畳む ----
    def weight_vector(self) -> tuple[np.ndarray, float]:
        """u(s) = wᵀ s + c の 1 本のベクトルに畳む (検索や配布用)。

        特徴が s のアフィン変換 (full / pca / concept とその連結) のときに正しい。mrl は不可。
        """
        dim = len(self.w0)
        c = float(self.utility(np.zeros((1, dim)))[0][0])
        w = self.utility(np.eye(dim))[0] - c
        return w, c
