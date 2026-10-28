# JavaScript で使う

日本語 | [English](en/javascript.md)

ブラウザ版（`js/shiken.js`、依存の無い ES モジュール 1 つ）は、2 択の出題・パーソナライズ・スコアリングをブラウザの中だけで行います。回答はどこにも送られないので、好みのデータをサーバーに集めたくないサービスにも組み込めます。

## 下準備: パックを作る（Python）

ブラウザでは埋め込みモデルを動かしません。代わりに、データセットと質問が決まった時点で必要な値を Python で計算し、パック（JSON ファイル）にしておきます。

```python
taste = shiken.Taste(question=..., items=items, ids=ids, encoder=enc, concepts=shiken.concepts.FOOD)
taste.export_pack("pack.json", include_embeddings=True, meta={"prompt": "どっちが美味しそう？"})
```

パックには次のものが入ります。

| 中身 | 大きさ |
| :-- | :-- |
| 項目の名前 | 項目数 |
| 各項目の、素の価値尺度でのスコア | 項目数 |
| 事前分布の下での、項目どうしのスコアの共分散 | 項目数 × 項目数 |
| 各項目の概念らしさ（補正の中身の表示に使う） | 項目数 × 概念の数 |
| 質問の定義、埋め込みモデルの名前、`meta`（画面で使う値を自由に入れられる） | 小さい |
| 各項目の埋め込みベクトル（`include_embeddings=True` のとき。価値尺度ファイルを Python 版と同じ形式で書き出すのに要る） | 項目数 × 約 1 KB |

共分散が項目数の 2 乗の大きさなので、データセットは数百件までを想定しています。104 件のパックは埋め込みベクトル込みで約 330 KB でした。それより大きいデータセットでは、出題に使う数百件を選んでパックにし、全体のスコアリングは Python 側で `weight_vector` を使って行う形になります。

質問の説明はパックを作るときに埋め込むので、ブラウザ版では利用者が質問をその場で作ることはできません。

## 使い方

```js
import { Taste } from "./shiken.js";

const pack = await (await fetch("pack.json")).json();
const taste = new Taste(pack);

const [a, b] = taste.nextPair();     // 次に聞く組（項目の名前）
taste.predict(a, b);                 // 答える前の予想: a が選ばれる確率
taste.answer(a, b, { prefer: a });   // prefer: null なら「同じくらい」。答える前の予想を返す
taste.rank();                        // パーソナライズした並び
taste.scores();                      // 項目ごとのスコア（パックの ids の順）
taste.hitRate(10);                   // 直近 10 問の予想の的中率
taste.baseWeight;                    // 素の価値尺度をどれだけ残しているか（回答 0 件で 1）
taste.explain(6);                    // { up: [["炒め物", 0.41], ...], down: [...] }
taste.explain(6, { part: "all" });   // 素の価値尺度からのずれではなく、価値尺度全体
```

次に聞く組は、まだ聞いていない組の中から選びます（全部聞き終えたら、聞いた組も選びます）。補正は回答数 × 回答数の大きさの計算で、104 件のデータセットで 1 問あたり約 4 ミリ秒でした。個別の加減点の大きさは、Python 版と同じく 10 問ごとに周辺尤度で選び直します。

## 保存

```js
const profile = taste.toProfile();             // 価値尺度ファイル（JSON のオブジェクト）
const restored = Taste.fromProfile(pack, profile);
restored.skipped;                              // パックにない項目への回答で、使えなかった数
```

`toProfile` は Python 版と同じ形式の価値尺度ファイルを返します（[価値尺度ファイルの形式](profile-format.md)）。Python 版の `Taste.load` でそのまま読めます。ブラウザの保存領域に置くか、ファイルとしてダウンロードさせるか、サーバーに送るかは、組み込む側が決めます。価値尺度ファイルからは本人の好みが分かるので、扱いは個人のデータに準じます。

## Python 版との一致

ブラウザ版と Python 版は、同じ回答から同じ事後分布を求めます。リポジトリのテスト（`tests/test_parity.py`）で、同じパックと回答から両方の予想の確率・スコア・個別の加減点の大きさ・補正の中身を計算して比べています。確率の差は 1e-13 程度でした（[開発](development.md)）。
