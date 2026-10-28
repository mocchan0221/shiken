# Development

[日本語](../development.md) | English

## Repository layout

| Path | Contents |
| :-- | :-- |
| `src/shiken/` | The Python package. Scoring and correction depend only on numpy |
| `src/shiken/encoders/` | The bundled encoder (EmbeddingGemma 2). Available only when `shiken[eg2]` is installed |
| `js/` | The browser version (a single ES module with no dependencies) |
| `tests/` | Python tests, and tests that check the Python and browser versions agree |
| `examples/` | Short examples that run without a GPU (EmbeddingGemma 2 runs on a CPU; examples that fetch data need network access) |
| `demo/` | The demo (export, scripts for uploading to Cloudflare Pages, and the page). Exported data is not committed to git |
| `docs/` | These documents |
| `.github/workflows/` | GitHub Actions (tests) |

## Keeping the Python and browser versions in agreement

The two versions compute differently. The Python version performs the correction in feature space (number of concept words + embedding dimensions, about 870 dimensions), while the browser version performs it in answer space (dimension = number of answers). Mathematically they give the same posterior distribution, but implementation discrepancies can only be caught by tests.

So there is a test with the following steps.

1. In Python, build a pack from a small dataset (random embeddings) and a sequence of answers, and write the predicted probability for each pair and the ranking to JSON.
2. In Node.js, load the same pack and answers into the browser version and compute the predicted probabilities and the ranking.
3. Check that the difference between the two is at most about 1e-6.

Both cases, with and without per-item adjustments, are checked. In the experimental code, the probabilities from the two versions differed by at most 6e-8.

## Running the tests

```bash
pip install -e ".[dev]"
pytest                     # Python tests, plus Python/browser agreement (if Node.js is available)
node --test js/            # browser version tests
```

The tests run with a fake encoder (`tests/conftest.py`) that returns vectors determined by the text or file name, so they use neither a GPU nor an embedding model. Tests that actually load EmbeddingGemma 2 need to download the model, so they are skipped by default (run `pytest -m eg2` after `pip install -e ".[eg2,dev]"`). GitHub Actions (`.github/workflows/test.yml`) runs the non-GPU tests on Python 3.10 and 3.12.

## Supported environments

- Python 3.10 or later (the core depends only on numpy)
- The browser version: current browsers that support ES modules (Chrome, Edge, Firefox, Safari) and Node.js 20 or later
