# 先行類似事例

Shiken の部品の 1 つ 1 つには先行例があります。凍結した埋め込みの上で軽いベクトルを当てはめて個人化する方法は、論文にも個人の実装にも多数あります。文の埋め込みを出発点に 2 択で動かす方法も、2 択と「迷う組から聞く」で個人の係数を更新する方法も、すでにあります。

調べた範囲で見つからなかったのは、次をまとめて組み込める部品にしたものです。

1. 利用者が自分で質問を書き、回答 0 件でも型の決まった答えを確率つきで返す
2. 本人の 2 択で補正し、答える前の予想の的中率で、本人に合ってきたかを見せる
3. 結果を価値尺度ファイルとして、別のデータセットや別のアプリに持ち運ぶ
4. データセットをまたいで使える価値尺度と、そのデータセットだけの個別の加減点を分けて持つ
5. 画像と音声で同じように動く

## 一番近い研究と実装

| 名前 | 何をしているか | Shiken との違い |
| :-- | :-- | :-- |
| [Gallego 2023](https://arxiv.org/abs/2308.07929)「Fast Adaptation with Bradley-Terry Preference Models」 | CLIP の文の埋め込みを出発点に、2 択から Bradley–Terry モデルで埋め込みを動かす | 出発点の文は固定。回答は集団の投票で、個人ではない。画像だけ |
| [HumorDNA](https://github.com/ragsaniarumal/Personalized-Meme-Recommendation-from-Pairwise-Feedback) | 個人の係数を 2 択で更新し、集団の事前に寄せる。迷う組から聞く | 出発点は公開データの注釈で、利用者の質問ではない。持ち運びと音声は無い |
| [小島ら 2025](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001044)（PLOS Digital Health） | 「美味しそうな料理」などを CLIP で聞き、個人ごとに 896 枚の 8 段階の評価から特性ベクトルを求める | 1 人に数百枚の点数をつけてもらう。質問から始めて 2 択で直す形ではない |
| [Model Sketching](https://arxiv.org/abs/2303.02884)（Lam ら CHI 2023） | 自然文で概念を書き、ゼロショットで評価して組み合わせる | 個人の 2 択による補正は無い |
| [CLIP-IQA](https://ojs.aaai.org/index.php/AAAI/article/view/25353)（Wang ら AAAI 2023） | 対になる文（good photo / bad photo）で画質をゼロショットで評価する | Shiken のはい／いいえの型（対の文）とほぼ同じ方法 |

## 個人の開発者による実装

同じ流れ（埋め込む → 本人に聞く → 軽く当てはめる → 次に何を聞くかを選ぶ）を、プロダクトごとに作っている例が多くあります。Shiken はこの部分を部品にしたものです。

| 名前 | 対象 | 作っていた部分 |
| :-- | :-- | :-- |
| [fineshyt](https://github.com/qweliant/fineshyt) | 写真の選別 | CLIP → 星の評価 → リッジ回帰。作風を文で書いて視覚言語モデルに判定させる方式は、本人の感覚とずれるとして捨てていた |
| [trove](https://github.com/howwohmm/trove) | 画像の収集 | ブラウザ内の CLIP → スワイプ → 好みの重心 |
| [Taste Engine](https://huggingface.co/spaces/Creatiste/taste-engine) | 服 | FashionSigLIP → 好き嫌い → 複数の重心 |
| [Visual Taste Approximator](https://github.com/SelfishGene/visual_taste_approximator) | 画像の 2 分類 | CLIP など → ラベル付けの画面 → 迷う画像から聞く |

## 型の決まった答えを返す判断モデル

画像や音声に、選択・段階・はい／いいえの型の答えを確率で返す判断モデルは、Jev の系統を中心に多数公開されています（一覧は [awesome-jev](https://github.com/yibie/awesome-jev)）。どれも全員に同じ基準で答えます。Shiken の答えの型はこれに合わせていますが、Shiken が加えているのは本人の回答で価値尺度を補正する部分です。

## 報告されている限界

- CLIP の類似度は、「重そう」「フォーマル」のような見た目から推しにくい印象語では、人の印象評価と相関しなかった（柴田 2026、[知能と情報](https://www.jstage.jst.go.jp/article/jsoft/38/1/38_569/_article/-char/ja)）。
- 1 本のベクトルでは好みを表しにくい人がいる（小島ら 2025）。Shiken の個別の加減点は、これに対する 1 つの手当てです。
- プロのデザイナーどうしでも、画面のデザインの 2 択はあまり一致しない（Krippendorff の α = 0.25。[DesignPref](https://arxiv.org/abs/2511.20513)）。個人ごとに合わせたモデルは、少ない例数でも集団に合わせたモデルより個人を当てた。

## 文献

- Bradley, R. A. & Terry, M. E. (1952). Rank analysis of incomplete block designs.
- Houlsby, N. et al. (2011). Bayesian Active Learning for Classification and Preference Learning.（BALD）
