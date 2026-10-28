# Embedding models

[日本語](../embedding-models.md) | English

The core of Shiken works with embedding vectors from any embedding model. The only requirement is that the model can embed the dataset items (images or audio) and the question descriptions into the same space.

## Default model: EmbeddingGemma 2

The bundled encoder (`pip install "shiken[eg2]"`) uses Google's EmbeddingGemma 2 (`google/embeddinggemma-2`, 740M parameters). It embeds images, audio, video, and text into the same 768-dimensional space. All numbers in this document were measured with this model.

| Item | Details |
| :-- | :-- |
| Size | About 1.4 GB (downloaded from Hugging Face on first use) |
| Compute | Uses the GPU if there is one, otherwise the CPU. Results are saved to disk and reused |
| Question descriptions (text) | Embedded with the `SearchQuery` prompt |
| Dataset items | Embedded with the document prompt (`title: none \| text: `) |
| Audio | The whole song is passed in. For songs longer than 320 seconds, the last 320 seconds are used |
| Numeric type | bfloat16 on GPU, float32 on CPU (float16 is not used because it breaks the computation) |
| License | Gemma Terms of Use |

If you build EmbeddingGemma 2 into a commercial product, check the Gemma Terms of Use. They apply separately from the license of Shiken's code.

## Using other models

Depending on the domain, a model that is strong in that domain (FashionSigLIP for clothing, CLAP for music, and so on) may be a better fit. You can swap the model by writing a class that satisfies the encoder interface (`name`, `embed_texts`, and `embed_files` if needed) (see [Using from Python](python.md#preparing-embedding-vectors)).

```python
class MyCLIP:
    name = "openai/clip-vit-large-patch14@<version>"
    def embed_texts(self, texts):
        ...  # return a (len(texts), dim) array normalized to unit length
```

`name` is recorded in the value-scale file and is used on load to decide whether the saved embedding vectors can be reused. If you change the model version or the preprocessing, change the name as well.

## Results with other models

Shiken was built mainly with EmbeddingGemma 2, but we confirmed that it also works with five other embedding models (2026-10-28, RTX 3070 Ti 8 GB). In every case it was enough to write an encoder class; the core of Shiken ran unchanged. This is a record for people who want to extend Shiken, not a recommendation.

| Model | License | Input | Dim. | Label check with zero answers | After answering (images and text) | After answering (music) | Per image | VRAM |
| :-- | :-- | :-- | --: | --: | :-- | :-- | --: | --: |
| EmbeddingGemma 2 (default) | Gemma Terms of Use | text, image, audio, video | 768 | +0.55 | 0.54 → 0.64 | 0.54 → 0.78 | 0.08 s | 2–6 GB |
| SigLIP 2 (`google/siglip2-so400m-patch16-384`) | Apache-2.0 | text, image | 1152 | +0.32 | 0.51 → 0.65 | — | 0.04 s | 2.5 GB |
| Qwen3-VL-Embedding-2B | Apache-2.0 | text, image, video | 2048 | +0.52 | 0.54 → 0.61 | — | 0.29 s | 7.1 GB |
| Ovis-VL-Embedding-2B | Apache-2.0 | text, image, video | 2048 | +0.58 | 0.56 → 0.64 | — | 0.29 s | 5.4 GB |
| Ovis-Omni-Embedding-3B (8-bit quantized) | Apache-2.0 | text, image, audio, video | 2048 | +0.55 | 0.54 → 0.64 | 0.53 → 0.72 | 1.7 s | 7.4 GB |
| Omni-Embed-Nemotron-3B (8-bit quantized) | Non-commercial (research and development only) | text, image, audio, video | 2048 | +0.52 | 0.57 → 0.64 | 0.53 → 0.76 | 1.5 s | 8.2 GB |

- **Label check with zero answers** is the mean rank correlation between the base value scale's ordering and the labels assigned at generation time, over 21 questions on the six demo image sets and the novel synopses (the same idea as [Base value scale vs. ground-truth labels](evaluation.md#base-value-scale-vs-ground-truth-labels)). The questions were left in Japanese
- **After answering** re-embeds the answers in the author's value-scale files (8 for images and text, 4 for music, 30–147 questions each) with each model, computes pairwise-question accuracy with 10-fold cross-validation as "base value scale only → after correction", and averages it. The pairs asked are the ones EmbeddingGemma 2 picked
- **Per image** and **VRAM** are rough figures on that GPU. The two 3B models did not fit in 8 GB and spilled into shared memory at peak, so they appear slow

Findings:

- The correction mechanism does not depend on the model. With every model, answering raised accuracy by 0.07–0.15 on average
- SigLIP 2 struggles with Japanese questions: its ordering with zero answers was weak, and it could barely order long texts such as synopses (text is truncated at 64 tokens). On the image-only value-scale files, however, it was the most accurate after correction
- The 2B-class and 3B-class models were about on par with EmbeddingGemma 2, both with zero answers and after answering. For music, EmbeddingGemma 2 was the best
- In speed and VRAM, EmbeddingGemma 2 and SigLIP 2 are light; the 2B-class models were 3–4 times slower and the 3B-class models about 20 times slower

Per-question values and the detailed settings for each model are recorded in the development repository (not public).

### Tips for writing an encoder

- **Separate the query side from the document side.** Shiken embeds question descriptions and concept words as queries with `embed_texts`, and embeds dataset items (image, audio, and text files) as documents with `embed_files`. If the model card specifies a query instruction or prefix (`query: ` / `passage: ` and so on), follow it. If text items are passed as `.txt` files, embed them as documents inside `embed_files`
- **Normalize to unit length before returning.** Follow the model card for pooling as well (last token, mean, etc.)
- **Put the settings in `name`.** Quantization, instructions, and whether a prompt is used all change the space, so use distinct names, e.g. `"ATH-MaaS/Ovis-Omni-Embedding-3B;8bit"`
- **Set a cap on image size.** The Qwen-family preprocessing has a large default cap, and feeding smartphone photos as-is overflowed VRAM and became extremely slow. Capping at around one megapixel keeps it stable. For audio too, using only the last two minutes or so of long songs fits on an 8 GB GPU
- **Save file embeddings to disk.** Embedding takes time with large models, so caching keyed on path, modification time, and `name`, as the bundled EmbeddingGemma 2 encoder does, makes the second run onward fast
- **Running 3B-class models on an 8 GB GPU.** It worked to quantize only the language-model part to 8 bits with bitsandbytes while leaving the image and audio parts in bfloat16 (`llm_int8_skip_modules=["visual", "audio_tower", "lm_head"]`)

## Shrinking the embedding vectors

EmbeddingGemma 2's embedding vectors are designed so that you can use just the leading dimensions (Matryoshka representation). Results with fewer dimensions or with int8 are in [Results so far](evaluation.md#shrinking-the-embedding-vectors). Converting to int8 while keeping 768 dimensions did not change the results, and reducing the dimensions did not reduce the number of answers needed for personalization.
