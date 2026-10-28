# 開発

日本語 | [English](en/development.md)

## リポジトリの構成

| パス | 中身 |
| :-- | :-- |
| `src/shiken/` | Python のパッケージ。スコアリングと補正は numpy だけに依存する |
| `src/shiken/encoders/` | 付属のエンコーダ（EmbeddingGemma 2）。`shiken[eg2]` を入れたときだけ使える |
| `js/` | ブラウザ版（依存の無い ES モジュール 1 つ） |
| `tests/` | Python のテストと、Python 版とブラウザ版の一致を確かめるテスト |
| `examples/` | GPU なしでも動く短い作例（EmbeddingGemma 2 は CPU で動く。データを取ってくる作例はネットワークが要る） |
| `demo/` | デモ（書き出し・Cloudflare Pages に上げるスクリプト・ページ）。書き出したデータは git に入れない |
| `docs/` | この文書 |
| `.github/workflows/` | GitHub Actions（テスト） |

## Python 版とブラウザ版を一致させる

2 つの版は計算のしかたが違います。Python 版は特徴の空間（概念の語の数 + 埋め込みの次元、約 870 次元）で補正し、ブラウザ版は回答の空間（回答数の次元）で補正します。数式の上では同じ事後分布になりますが、実装の食い違いはテストでしか見つけられません。

そこで、次の手順のテストを置きます。

1. Python で、小さなデータセット（乱数の埋め込み）と回答の列からパックを作り、各組の予想の確率と並びを JSON に書き出す。
2. Node.js で同じパックと回答をブラウザ版に読ませ、予想の確率と並びを計算する。
3. 両者の差が目安 1e-6 以下であることを確かめる。

個別の加減点を使う場合と使わない場合の両方を確かめます。実験用のコードでは、2 つの版の確率の差は 6e-8 以下でした。

## テストを回す

```bash
pip install -e ".[dev]"
pytest                     # Python のテストと、Python 版とブラウザ版の一致 (Node.js があれば)
node --test js/            # ブラウザ版のテスト
```

テストは、文やファイル名から決まったベクトルを返す偽のエンコーダ（`tests/conftest.py`）で回すので、GPU も埋め込みモデルも使いません。EmbeddingGemma 2 を実際に読み込むテストは、モデルの取得が要るので既定では飛ばします（`pip install -e ".[eg2,dev]"` のあと `pytest -m eg2`）。GitHub Actions（`.github/workflows/test.yml`）では、Python 3.10 と 3.12 で GPU を使わないテストを回します。

## 対応する環境

- Python 3.10 以上（本体は numpy だけに依存する）
- ブラウザ版は、ES モジュールが動く今のブラウザ（Chrome、Edge、Firefox、Safari）と Node.js 20 以上
