"""EmbeddingGemma 2 のエンコーダ (pip install "shiken[eg2]")。結果はディスクに保存して再利用する。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL_ID = "google/embeddinggemma-2"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}

# EG2 の文書側のプロンプト。画像や音はプレースホルダつきの文書として埋め込む
DOC_PREFIX = "title: none | text: "

# 音声は 1 秒 25 トークンで、EG2 の文脈 8,192 トークンには約 327 秒まで入る。前処理の初期設定は
# 30 秒で後ろを切り捨てるので、切り捨てを止めて曲をまるごと入れる。上限を超える曲は前を切る
AUDIO_SR = 16000
AUDIO_MAX_SECONDS = 320


def modality(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return "text"


class EmbeddingGemma2:
    """EmbeddingGemma 2 の薄いラッパー。

    modalities: 読み込む入力の種類。("image",) なら音声の部分を読み込まない (メモリが少なくて済む)
    cache_dir: ファイルの埋め込みベクトルの保存先。None なら保存しない
    device: "cuda" / "cpu"。省くと GPU があれば GPU
    """

    def __init__(self, modalities: tuple[str, ...] = ("image", "audio", "video"),
                 cache_dir: str | Path | None = "~/.cache/shiken/embeddings", device: str | None = None,
                 doc_prompt: bool = True, model_id: str = MODEL_ID):
        import torch
        from sentence_transformers import SentenceTransformer

        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()  # iPhone の HEIC を開けるようにする
        except ImportError:
            pass

        self.model_id = model_id
        self.doc_prompt = doc_prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # float16 は NaN になるので、GPU では bfloat16、CPU では float32
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        config_kwargs = {}
        if "image" not in modalities and "video" not in modalities:
            config_kwargs["vision_config"] = None
        if "audio" not in modalities:
            config_kwargs["audio_config"] = None
        kwargs = dict(device=self.device, config_kwargs=config_kwargs, model_kwargs={"torch_dtype": dtype})
        try:
            # 取得済みならネットワークに問い合わせない (起動が速く、オフラインでも動く)
            self.model = SentenceTransformer(model_id, local_files_only=True, **kwargs)
        except OSError:
            self.model = SentenceTransformer(model_id, **kwargs)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """価値尺度ファイルに記録する名前。文書のプロンプトを使わない設定は別の空間として扱う。"""
        return self.model_id if self.doc_prompt else f"{self.model_id};no-doc-prompt"

    def info(self) -> dict:
        revision = None
        try:
            from huggingface_hub import try_to_load_from_cache

            f = try_to_load_from_cache(self.model_id, "config.json")
            if isinstance(f, str):
                revision = Path(f).parent.name  # snapshots/<revision>/config.json
        except Exception:
            pass
        return {"model": self.model_id, "revision": revision,
                "options": {"doc_prompt": self.doc_prompt, "audio_max_seconds": AUDIO_MAX_SECONDS}}

    # ---- 文 ----------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """質問の説明や概念の語 (検索の問い合わせ側のプロンプトで埋め込む)。"""
        return self.model.encode(list(texts), prompt_name="SearchQuery", convert_to_numpy=True,
                                 normalize_embeddings=True).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """データセットの項目としての文 (文書側のプロンプトで埋め込む)。"""
        inputs = [self._doc(t) for t in texts]
        return self.model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    # ---- ファイル ------------------------------------------------------------------

    def embed_files(self, paths: list, batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
        """画像・音声・動画・テキストファイルを埋め込む。"""
        paths = [str(p) for p in paths]
        out: list[np.ndarray | None] = [None] * len(paths)
        todo = []
        for i, p in enumerate(paths):
            key = self._cache_key(p)
            if key and (self.cache_dir / f"{key}.npy").exists():
                out[i] = np.load(self.cache_dir / f"{key}.npy")
            else:
                todo.append(i)
        # 音声は長さがまちまちで (最大 8,000 トークン)、まとめるとパディングでメモリを食うので 1 件ずつ
        if any(modality(paths[i]) == "audio" for i in todo):
            batch_size = 1
        steps = range(0, len(todo), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm

                steps = tqdm(steps, desc="embed", unit="batch")
            except ImportError:
                pass
        for start in steps:
            idx = todo[start: start + batch_size]
            inputs = [self._input(paths[i]) for i in idx]
            embs = self.model.encode(inputs, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
                                     processing_kwargs={"audio": {"truncation": False}}).astype(np.float32)
            for i, e in zip(idx, embs):
                out[i] = e
                key = self._cache_key(paths[i])
                if key:
                    np.save(self.cache_dir / f"{key}.npy", e)
        return np.stack(out)

    def _input(self, path: str):
        from PIL import Image

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        kind = modality(p)
        if kind == "image":
            with Image.open(p) as im:
                return self._media("image", im.convert("RGB"))
        if kind == "text":
            return self._doc(p.read_text(encoding="utf-8"))
        if kind == "audio":
            return self._media("audio", {"array": load_audio_tail(p), "sampling_rate": AUDIO_SR})
        return self._media(kind, str(p))

    def _media(self, kind: str, value) -> dict:
        if self.doc_prompt:
            return {"text": f"{DOC_PREFIX}<|{kind}|>", kind: value}
        return {kind: value}

    def _doc(self, text: str) -> str:
        return f"{DOC_PREFIX}{text}" if self.doc_prompt else text

    def _cache_key(self, path: str) -> str | None:
        p = Path(path)
        if not self.cache_dir or not p.is_file():
            return None
        p = p.resolve()
        st = p.stat()
        raw = f"{p}|{st.st_size}|{st.st_mtime_ns}|{self.name}"
        if modality(p) == "audio":
            raw += f"|audio_tail={AUDIO_MAX_SECONDS}"
        return hashlib.sha1(raw.encode()).hexdigest()


def load_audio_tail(path: str | Path, max_seconds: float = AUDIO_MAX_SECONDS) -> np.ndarray:
    """16kHz モノラルで読み込み、長すぎれば最後の max_seconds 秒だけ残す。"""
    from transformers.audio_utils import load_audio

    x = load_audio(str(path), sampling_rate=AUDIO_SR)
    n = int(max_seconds * AUDIO_SR)
    return x[-n:] if len(x) > n else x
