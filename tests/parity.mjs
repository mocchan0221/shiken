// Python 版のテスト (test_parity.py) から呼ばれる: パックと回答を読み、ブラウザ版で計算した結果を出す
import { readFileSync } from "node:fs";
import { Taste } from "../js/shiken.js";

const { pack, answers, pairs } = JSON.parse(readFileSync(process.argv[2], "utf-8"));
const taste = Taste.fromProfile(pack, { answers });
const out = {
  itemVar: taste.itemVar,
  probs: pairs.map(([a, b]) => taste.predict(a, b)),
  scores: taste.scores(),
  baseWeight: taste.baseWeight,
  explain: taste.explain(3),
  explainAll: taste.explain(3, { part: "all" }),
};
process.stdout.write(JSON.stringify(out));
