/* Shiken のブラウザ版: 2 択の出題・パーソナライズ・スコアリングをブラウザの中だけで行う。
 *
 * Python 版の Taste.export_pack() が書き出すパックを読む。パックの中身は
 *   u0[s]   : 項目 s の素の価値尺度でのスコア (比較用の集合で標準化済み)
 *   G[s][t] : 事前分布の下での項目どうしのスコアの共分散
 *   Z[s][k] : 項目 s の語 k らしさ (補正の中身の表示用)
 * 2 択の回答 (a が b より当てはまる: y=1 / 逆: 0 / 同じくらい: 0.5) から、項目ごとのスコアの
 * 事後分布を、回答数 N の次元で求める (ガウス過程分類のラプラス近似。Rasmussen & Williams,
 * Algorithm 3.1 に事前の平均を足したもの)。Python 版 (PersonalAxis) と同じ事後分布になる。
 * 個別の加減点は、共分散に s²·I を足すことに当たる。s² は周辺尤度で選ぶ (Python 版と同じ規則)。
 */

const sigmoid = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
const log1pexp = (z) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));

function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) L[i][i] = Math.sqrt(Math.max(s, 1e-12));
      else L[i][j] = s / L[j][j];
    }
  }
  return L;
}

function solveLower(L, b) {
  const n = b.length;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

function solveUpperT(L, b) {
  // Lᵀ x = b
  const n = b.length;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/** 標準正規分布の累積分布関数 (Abramowitz & Stegun 26.2.17。誤差 7.5e-8 以下) */
function ndtr(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp((-x * x) / 2);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x > 0 ? 1 - p : p;
}

function pearson(x, y) {
  const n = x.length;
  let mx = 0, my = 0;
  for (let k = 0; k < n; k++) { mx += x[k]; my += y[k]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let k = 0; k < n; k++) {
    const dx = x[k] - mx, dy = y[k] - my;
    sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
}

/**
 * 回答から事後分布を求める。answers: [{i, j, y}] (i, j は項目の番号)。itemVar: 個別の加減点の大きさ s²。
 * 返り値: mean (加減点を含むスコア), axisMean (価値尺度の部分だけ), C (事後共分散), aPost, logEvidence
 */
export function fit(u0, G, answers, { a0 = 1, sigmaA = 1, itemVar = 0, iters = 50 } = {}) {
  const M = u0.length;
  const n = answers.length;
  const g = (s, t) => G[s][t] + (s === t ? itemVar : 0);
  if (n === 0) {
    const mean = Float64Array.from(u0, (v) => a0 * v);
    return { mean, axisMean: mean, C: G.map((r, s) => Float64Array.from(r, (v, t) => g(s, t))), aPost: a0, logEvidence: 0 };
  }
  const I = answers.map((a) => a.i);
  const J = answers.map((a) => a.j);
  const y = Float64Array.from(answers, (a) => a.y);
  // Kx[p][s] = 回答 p の差 (I_p − J_p) と項目 s の共分散
  const Kx = I.map((ip, p) => Float64Array.from({ length: M }, (_, s) => g(ip, s) - g(J[p], s)));
  const K = Kx.map((row) => Float64Array.from(I, (ip, q) => row[ip] - row[J[q]]));
  const z0 = Float64Array.from(I, (ip, p) => a0 * (u0[ip] - u0[J[p]]));
  const Kdot = (v) => K.map((row) => row.reduce((acc, kv, q) => acc + kv * v[q], 0));

  const psi = (a) => {
    const Ka = Kdot(a);
    let s = 0;
    for (let p = 0; p < n; p++) {
      const f = z0[p] + Ka[p];
      s += -0.5 * a[p] * Ka[p] + y[p] * f - log1pexp(f);
    }
    return s;
  };

  const factor = (a) => {
    const Ka = Kdot(a);
    const f = Float64Array.from(Ka, (v, p) => z0[p] + v);
    const pr = Float64Array.from(f, sigmoid);
    const W = Float64Array.from(pr, (v) => v * (1 - v));
    const sW = Float64Array.from(W, Math.sqrt);
    const B = K.map((row, p) => Float64Array.from(row, (kv, q) => (p === q ? 1 : 0) + sW[p] * kv * sW[q]));
    return { f, pr, W, sW, L: cholesky(B) };
  };

  let a = new Float64Array(n);
  for (let it = 0; it < iters; it++) {
    const { f, pr, W, sW, L } = factor(a);
    const b = Float64Array.from(W, (w, p) => w * (f[p] - z0[p]) + (y[p] - pr[p]));
    const Kb = Kdot(b);
    const c = solveUpperT(L, solveLower(L, Float64Array.from(Kb, (v, p) => sW[p] * v)));
    const aNew = Float64Array.from(b, (v, p) => v - sW[p] * c[p]);
    // 目的関数が下がるなら歩幅を半分に (素のニュートン法は発散しうる)
    const base = psi(a);
    let t = 1;
    const cand = (tt) => Float64Array.from(a, (v, p) => v + tt * (aNew[p] - v));
    while (psi(cand(t)) < base - 1e-12 && t > 1e-4) t /= 2;
    const next = cand(t);
    let maxStep = 0;
    for (let p = 0; p < n; p++) maxStep = Math.max(maxStep, Math.abs(next[p] - a[p]));
    a = next;
    if (maxStep < 1e-8) break;
  }
  const { sW, L } = factor(a);
  let logDet = 0;
  for (let p = 0; p < n; p++) logDet += Math.log(L[p][p]);
  const logEvidence = psi(a) - logDet;

  const mean = Float64Array.from(u0, (v) => a0 * v);
  const axisMean = Float64Array.from(u0, (v) => a0 * v);
  for (let p = 0; p < n; p++) {
    for (let s = 0; s < M; s++) {
      mean[s] += Kx[p][s] * a[p];
      axisMean[s] += (G[I[p]][s] - G[J[p]][s]) * a[p];
    }
  }
  // V = L⁻¹ (sW ∘ Kx) (N × M)、C = G + s²I − VᵀV
  const V = Array.from({ length: n }, () => new Float64Array(M));
  for (let s = 0; s < M; s++) {
    const col = solveLower(L, Float64Array.from(Kx, (row, p) => sW[p] * row[s]));
    for (let p = 0; p < n; p++) V[p][s] = col[p];
  }
  const C = G.map((r, s) => Float64Array.from(r, (v, t) => g(s, t)));
  for (let s = 0; s < M; s++) {
    for (let t = s; t < M; t++) {
      let d = 0;
      for (let p = 0; p < n; p++) d += V[p][s] * V[p][t];
      C[s][t] -= d;
      if (t !== s) C[t][s] -= d;
    }
  }
  let aPost = a0;
  for (let p = 0; p < n; p++) aPost += sigmaA * sigmaA * a[p] * (u0[I[p]] - u0[J[p]]);
  return { mean, axisMean, C, aPost, logEvidence };
}

/** s が t より当てはまる確率 (事後分布で平均したもの。プロビット近似) */
export function pairProb(st, s, t) {
  const m = st.mean[s] - st.mean[t];
  const v = Math.max(st.C[s][s] + st.C[t][t] - 2 * st.C[s][t], 0);
  return sigmoid(m / Math.sqrt(1 + (Math.PI * v) / 8));
}

/** 組 (s, t) を聞いたときに価値尺度について得られる情報量 (BALD, Houlsby et al. 2011) */
function infoGain(st, s, t) {
  const C2 = (Math.PI * Math.log(2)) / 2;
  const mu = (st.mean[s] - st.mean[t]) * Math.sqrt(Math.PI / 8);
  const v = Math.max(st.C[s][s] + st.C[t][t] - 2 * st.C[s][t], 0) * (Math.PI / 8);
  const p = Math.min(Math.max(ndtr(mu / Math.sqrt(1 + v)), 1e-12), 1 - 1e-12);
  const h = -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
  return h - Math.sqrt(C2 / (v + C2)) * Math.exp(-(mu * mu) / (2 * (v + C2)));
}

export class Taste {
  /** pack: Python 版の Taste.export_pack() が書き出したもの (JSON を読んだオブジェクト) */
  constructor(pack) {
    if (!["shiken-pack", "mekiki-pack"].includes(pack.format)) throw new Error("パックではありません (format が shiken-pack ではない)");
    this.pack = pack;
    this.ids = pack.ids;
    this.index = new Map(this.ids.map((k, s) => [k, s]));
    this.u0 = Float64Array.from(pack.u0);
    this.G = pack.G.map((r) => Float64Array.from(r));
    this.Z = pack.Z || null;
    this.words = pack.words || null;
    this.prior = { a0: pack.prior?.a0 ?? 1, sigmaA: pack.prior?.sigmaA ?? 1 };
    this.grid = pack.itemVarGrid || [0, 0.3, 1, 2, 4];
    this.minGain = pack.minGain ?? 1;
    this.reselectEvery = pack.reselectEvery ?? 10;
    this.itemVar = 0;
    this.answers = []; // {a, b, y, predicted}
    this.skipped = 0;
    this.asked = new Set();
    this.state = this._fit();
  }

  _rows() {
    return this.answers.map((r) => ({ i: this.index.get(r.a), j: this.index.get(r.b), y: r.y }));
  }

  _fit(itemVar = this.itemVar) {
    return fit(this.u0, this.G, this._rows(), { ...this.prior, itemVar });
  }

  /** 個別の加減点の大きさを周辺尤度で選び直す (加減点なしより minGain 以上よいときだけ使う) */
  selectItemVar() {
    let best = 0, bestE = -Infinity, baseE = null, bestState = null;
    for (const v of this.grid) {
      const st = this._fit(v);
      if (v === 0) baseE = st.logEvidence;
      else if (baseE !== null && st.logEvidence < baseE + this.minGain) continue;
      if (st.logEvidence > bestE) { best = v; bestE = st.logEvidence; bestState = st; }
    }
    this.itemVar = best;
    this.state = bestState;
    return best;
  }

  _key(s, t) {
    return s < t ? `${s}-${t}` : `${t}-${s}`;
  }

  /** 次に聞く組 [a, b] (項目の名前)。答えを聞くと一番多くを学べる組 */
  nextPair() {
    const M = this.ids.length;
    const allAsked = this.asked.size >= (M * (M - 1)) / 2;
    let best = null, bestGain = -Infinity;
    for (let s = 0; s < M; s++) {
      for (let t = s + 1; t < M; t++) {
        if (!allAsked && this.asked.has(this._key(s, t))) continue;
        const gain = infoGain(this.state, s, t);
        if (gain > bestGain) { bestGain = gain; best = [s, t]; }
      }
    }
    return [this.ids[best[0]], this.ids[best[1]]];
  }

  /** 答える前の予想: a が選ばれる確率 */
  predict(a, b) {
    return pairProb(this.state, this._idx(a), this._idx(b));
  }

  _idx(k) {
    const s = this.index.get(k);
    if (s === undefined) throw new Error(`項目「${k}」はパックにありません`);
    return s;
  }

  /** 回答を記録して補正し直す。prefer: 選ばれた方 (a か b)、null なら「同じくらい」。答える前の予想を返す */
  answer(a, b, { prefer } = {}) {
    if (prefer !== a && prefer !== b && prefer !== null) {
      throw new Error("prefer には a か b か null (同じくらい) を渡してください");
    }
    const p = this.predict(a, b);
    this.answers.push({ a, b, y: prefer === null ? 0.5 : prefer === a ? 1 : 0, predicted: p });
    this.asked.add(this._key(this._idx(a), this._idx(b)));
    if (this.answers.length % this.reselectEvery === 0) this.selectItemVar();
    else this.state = this._fit();
    return p;
  }

  /** 項目ごとのスコア (個別の加減点を含む) */
  scores() {
    return Array.from(this.state.mean);
  }

  /** 項目の名前を、パーソナライズしたスコアの高い順に */
  rank() {
    const m = this.state.mean;
    return this.ids.map((k, s) => s).sort((p, q) => m[q] - m[p] || p - q).map((s) => this.ids[s]);
  }

  /** 直近 last 問で、答える前の予想が当たった割合 (「同じくらい」は除く) */
  hitRate(last = 10) {
    const rs = this.answers.slice(-last).filter((r) => r.y !== 0.5 && r.predicted != null);
    if (!rs.length) return null;
    return rs.filter((r) => (r.predicted > 0.5) === (r.y === 1)).length / rs.length;
  }

  get baseWeight() {
    return this.state.aPost;
  }

  /** 価値尺度を語で表す。part: "correction" は素の価値尺度からのずれだけ、"all" は全体 */
  explain(top = 6, { part = "correction" } = {}) {
    if (!this.Z || !this.words) throw new Error("このパックには語が入っていません");
    const st = this.state;
    const u = Float64Array.from(st.axisMean, (m, s) => (part === "correction" ? m - st.aPost * this.u0[s] : m));
    const r = this.words.map((_, k) => pearson(u, this.Z.map((row) => row[k])));
    const order = r.map((v, k) => k).sort((p, q) => r[q] - r[p]);
    return {
      up: order.slice(0, top).filter((k) => r[k] > 0).map((k) => [this.words[k], r[k]]),
      down: order.slice(-top).reverse().filter((k) => r[k] < 0).map((k) => [this.words[k], r[k]]),
    };
  }

  /** 価値尺度ファイル (Python 版と同じ形式)。パックに埋め込みベクトルが入っていないと items は空になる */
  toProfile() {
    const base = this.pack.profile || { format: "shiken-profile", version: 1, question: this.pack.question,
      encoder: this.pack.encoder };
    const items = {};
    const emb = this.pack.embeddings || {};
    for (const r of this.answers) for (const k of [r.a, r.b]) if (emb[k]) items[k] = emb[k];
    return { ...base, answers: this.answers.map((r) => ({ ...r })), items };
  }

  /** 価値尺度ファイルの回答をパックの上で補正し直す (パックにない項目への回答は使わない) */
  static fromProfile(pack, profile) {
    const taste = new Taste(pack);
    for (const r of profile.answers || []) {
      if (taste.index.has(r.a) && taste.index.has(r.b)) {
        taste.answers.push({ a: r.a, b: r.b, y: Number(r.y), predicted: r.predicted ?? null });
        taste.asked.add(taste._key(taste.index.get(r.a), taste.index.get(r.b)));
      } else taste.skipped += 1;
    }
    if (taste.answers.length) taste.selectItemVar();
    return taste;
  }
}
