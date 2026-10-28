# Value-scale file format

[日本語](../profile-format.md) | English

A value-scale file is a JSON file that bundles, for one person and one question, the question definition and the pairwise answers. Your value scale can be saved in this single lightweight file and shared with others. The file holds the answers themselves rather than the result of the correction (a 768-dimensional value-scale vector), so the loading side can always rebuild the value-scale vector from the answers. As long as the same embedding model is used, it works in other apps and with other datasets, and can be exchanged between people.

## Why store answers instead of the value-scale vector

Storing only the value-scale vector (768 numbers) would keep the file at about 1 KB. Answers are stored anyway, for the following four reasons.

1. **You can continue answering where you left off.** A single vector does not carry "how certain it is" (the spread of the posterior distribution). Shiken uses this spread both when picking the next pair to ask and for the predicted probability before an answer, so if it receives only the vector it cannot keep correcting correctly from there. With the answers, it can rebuild everything, including the spread.
2. **Per-item adjustments are not lost.** Per-item ups and downs such as "I like this one item regardless of the overall tendency" cannot be represented by a single direction. With the answers, the adjustments can be rebuilt when used with the same dataset.
3. **It can adapt to the loading side's dataset.** The correction is performed after standardizing over the reference set, so the same answers produce a slightly different value-scale vector on a different dataset. Having the answers lets it be rebuilt for the loading side's dataset.
4. **Old files keep working when the computation improves.** When the correction method is improved, the value scale can be rebuilt from the answers with the new method.

Even across embedding models, the answers can sometimes be used to rebuild it. The saved embedding vectors cannot be used with a different model, but the record of which item was compared with which and which one was chosen remains, so if you have the original images or texts at hand you can re-embed them with the new model and correct again.

The trade-off is that the file grows to tens to hundreds of KB, and the answered items become guessable to some extent (see "Handling as personal data" below). When you only need the value-scale vector, such as for ranking with vector search, you can export it with [`weight_vector()`](python.md#collapsing-into-a-single-vector).

## Example

```jsonc
{
  "format": "shiken-profile",
  "version": 1,
  "question": {                          // question definition
    "type": "score",
    "levels": [[1, "a photo of a dish that looks not at all tasty"], [5, "a photo of a dish that looks incredibly tasty"]]
  },
  "question_embeddings": {              // embedding vectors of the question descriptions (in description order; same format as items)
    "descriptions": [{"emb": "…", "scale": 0.0012}, {"emb": "…", "scale": 0.0011}]
  },
  "concepts": {                          // concept words used for the correction (so it can be recomputed with the same words)
    "name": "food",
    "noun": "料理の写真",                // "photo of a dish"; the bundled concept words are in Japanese
    "words": ["揚げ物", "焼き料理", "辛い", "..."]   // fried food, grilled dishes, spicy, ...
  },
  "encoder": {                           // how the embedding vectors were made; saved embedding vectors are used only when name matches
    "name": "google/embeddinggemma-2",
    "model": "google/embeddinggemma-2",
    "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
    "options": {"doc_prompt": true, "audio_max_seconds": 320}
  },
  "answers": [                           // answer history; the order is meaningful
    {"a": "dish-0012", "b": "dish-0087",
     "y": 1,                             // 1: a was chosen / 0: b / 0.5: about the same
     "predicted": 0.62}                  // prediction before answering (probability that a is chosen)
  ],
  "items": {                             // items that appear in the answers; keys are the a / b of answers
    "dish-0012": {
      "emb": "…",                        // embedding vector quantized to int8 (base64)
      "scale": 0.00123                   // multiplier to convert int8 back to real values
    }
  }
}
```

## Rules for each part

- **Item names (a / b in answers, keys of items)** are strings chosen by the embedding application. File paths or product IDs are both fine. Use the same name for the same item. Per-item adjustments are tied to these names.
- **Embedding vectors** are stored as int8 at their original dimensions. When used, `emb × scale` is renormalized to length 1. Quantizing to int8 did not change the accuracy of scoring with the base value scale or of personalization compared with float ([Results so far](evaluation.md#shrinking-the-embedding-vectors)). One item at 768 dimensions is 772 bytes (about 1 KB in base64), and a file with 100 questions and about 200 items is around 200 KB.
- **question_embeddings** holds the embedded question descriptions. When a description is an image or audio, the loading side may not have the original file, so the vectors are included in the file. If the embedding model is the same, these vectors are used even when the descriptions are text. When a Yes–No question uses a neutral sentence, its embedding vector is also included as `neutral`.
- If the **encoder** `name` differs from the `name` of the loading side's encoder, the saved embedding vectors are not used, because the spaces are different and mixing them produces meaningless values. In that case, only the answers about items in the loading side's dataset are used for the correction.
- **predicted** is the prediction before answering. Lined up, these show whether the predictions became more accurate as answering went on (whether the value scale came to fit the person).
- **The size of the per-item adjustments** is not stored in the file; it is re-determined from the answers each time the file is loaded.

## Handling as personal data

A value-scale file reveals the person's preferences. Also, since it contains the embedding vectors of the answered items, "what they answered about" can be guessed to some extent even without the original files (searching for images with nearby embedding vectors will turn up similar images).

Shiken never sends value-scale files anywhere. Where they are stored and how they are shared is up to the embedding application. If you build a sharing feature, tell your users about this.

## Open questions

- **Whether to carry the reference set along.** Score standardization and feature centering are done over the reference set (by default, the dataset itself). With a different dataset, the correction result from the same answers changes slightly. We have not decided whether to include a summary of the reference set (such as its mean) in the file.
- **Whether to keep answer timestamps.** Currently only the order is kept. Timestamps would make it possible to see how preferences change over time, but they would make the file weigh more heavily as personal data.
