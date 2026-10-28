"""EmbeddingGemma 2 を実際に読み込むテスト (pytest -m eg2)。モデルの取得 (約 1.4 GB) が要る。"""

import numpy as np
import pytest

pytestmark = pytest.mark.eg2


def test_texts_are_normalized_and_close_for_similar_meaning():
    from shiken.encoders import EmbeddingGemma2

    enc = EmbeddingGemma2(modalities=(), cache_dir=None, device="cpu")
    E = enc.embed_texts(["カレーライス", "ハヤシライス", "株価"])
    assert E.shape == (3, 768)
    assert np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-4)
    assert E[0] @ E[1] > E[0] @ E[2]
    assert enc.name == "google/embeddinggemma-2"
