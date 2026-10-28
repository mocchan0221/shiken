"""付属のエンコーダ。使うときにだけ読み込む (PyTorch などは shiken[eg2] を入れたときだけ要る)。"""


def __getattr__(name):
    if name == "EmbeddingGemma2":
        from shiken.encoders.eg2 import EmbeddingGemma2

        return EmbeddingGemma2
    raise AttributeError(name)


__all__ = ["EmbeddingGemma2"]
