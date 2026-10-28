import { test } from "node:test";
import assert from "node:assert/strict";
import { Taste } from "./shiken.js";

// 小さなパック: 項目 6 個、素の価値尺度は番号順、共分散は単位行列 + 素の価値尺度の向き
function tinyPack() {
  const ids = ["a", "b", "c", "d", "e", "f"];
  const u0 = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5];
  const G = u0.map((x, s) => u0.map((y, t) => x * y + (s === t ? 1 : 0)));
  const Z = u0.map((x) => [x, -x, 0.1 * x * x]);
  return {
    format: "shiken-pack", version: 1, ids, u0, G, Z, words: ["上がる", "下がる", "端"],
    prior: { a0: 1, sigmaA: 1 }, itemVarGrid: [0, 0.3, 1, 2, 4], minGain: 1, reselectEvery: 10,
    question: { type: "score", levels: [[1, "低い"], [5, "高い"]] }, encoder: { name: "test" },
  };
}

test("回答 0 件では素の価値尺度の順に並ぶ", () => {
  const taste = new Taste(tinyPack());
  assert.deepEqual(taste.rank(), ["f", "e", "d", "c", "b", "a"]);
  assert.equal(taste.hitRate(), null);
});

test("逆向きに答え続けると並びが入れ替わる", () => {
  const taste = new Taste(tinyPack());
  const order = ["a", "b", "c", "d", "e", "f"]; // 本人は番号の小さい方が好き
  for (let k = 0; k < 15; k++) {
    const [x, y] = taste.nextPair();
    assert.notEqual(x, y);
    taste.answer(x, y, { prefer: order.indexOf(x) < order.indexOf(y) ? x : y });
  }
  assert.equal(taste.rank()[0], "a");
  assert.ok(taste.baseWeight < 1);
  assert.ok(taste.hitRate(5) >= 0.6);
});

test("prefer の誤りと、同じくらい", () => {
  const taste = new Taste(tinyPack());
  assert.throws(() => taste.answer("a", "b", { prefer: "z" }));
  const p = taste.answer("a", "b", { prefer: null });
  assert.ok(p > 0 && p < 1);
  assert.equal(taste.answers[0].y, 0.5);
});

test("価値尺度ファイルの書き出しと読み込み", () => {
  const taste = new Taste(tinyPack());
  taste.answer("a", "f", { prefer: "a" });
  taste.answer("b", "e", { prefer: "b" });
  const profile = taste.toProfile();
  assert.equal(profile.format, "shiken-profile");
  assert.equal(profile.answers.length, 2);
  const back = Taste.fromProfile(tinyPack(), { answers: [...profile.answers, { a: "a", b: "zz", y: 1 }] });
  assert.equal(back.answers.length, 2);
  assert.equal(back.skipped, 1);
  assert.deepEqual(back.rank(), taste.rank());
});

test("補正の中身を語で表す", () => {
  const taste = new Taste(tinyPack());
  for (const [x, y] of [["a", "f"], ["b", "e"], ["a", "d"], ["c", "f"]]) taste.answer(x, y, { prefer: x });
  const out = taste.explain(2);
  assert.equal(out.up[0][0], "下がる");
});
