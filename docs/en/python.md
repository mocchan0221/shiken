# Using from Python

[日本語](../python.md) | English

## Installation

```bash
pip install shiken            # scoring and personalization (numpy only)
pip install "shiken[eg2]"     # also compute embedding vectors with EmbeddingGemma 2 (installs PyTorch)
```

For the first release we plan not to publish to PyPI, and to distribute via `pip install git+https://github.com/<owner>/shiken` instead.

## Preparing embedding vectors

The core of Shiken does not compute embedding vectors. You pass it the dataset's embedding vectors (a numpy array of shape `(number of items, dimensions)`, with each row normalized to length 1) and a way to embed the question descriptions.

The way to embed descriptions is passed as an object (an encoder) with the following shape.

```python
class Encoder(Protocol):
    name: str                                         # name recorded in the value-scale file; a different name means a different space
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...
    def embed_files(self, paths: list[str]) -> np.ndarray: ...   # only needed when images or audio are used as descriptions
    def info(self) -> dict: ...                                  # optional; details such as the version, written to the value-scale file
```

Every method returns vectors normalized to length 1.

When you use EmbeddingGemma 2, the bundled encoder also computes the dataset's embedding vectors.

```python
from shiken.encoders import EmbeddingGemma2

enc = EmbeddingGemma2()                      # uses a GPU if available; results are cached in ~/.cache/shiken and reused
items = enc.embed_files(["a.jpg", "b.jpg"])  # images, audio, video
docs = enc.embed_documents(["product description", ...])  # embed text as dataset items
```

If you only handle images, `EmbeddingGemma2(modalities=("image",))` avoids loading the audio part.

To use another model, write a class with the shape above yourself ([Embedding models](embedding-models.md)). The dataset and the question descriptions must be embedded in the same space of the same model.

## Answering with the base value scale

If you just want typed answers without collecting any answers from the user, use `shiken.ask`.

```python
from shiken import Score, Noul

answers = shiken.ask(items, {
    "spiciness": Score([(1, "a photo of a mild dish with no heat at all"), (5, "a photo of a fiery dish loaded with chili peppers and chili oil")]),
    "soup": Noul("a photo of a soup or brothy dish", contrast="a photo of a dish with no broth"),
}, encoder=enc)

answers[0]["spiciness"].score         # expected value over the levels (e.g. 2.3)
answers[0]["soup"].probability        # probability that it applies
```

Besides the probabilities, each answer comes with a confidence (1 − normalized entropy) and the gap between the probabilities of the first and second choices. How to write questions is covered in [Writing questions](questions.md).

## Personalizing with pairwise questions

```python
taste = shiken.Taste(
    question=Score([(1, "a photo of a dish that looks not at all tasty"), (5, "a photo of a dish that looks incredibly tasty")]),
    items=items,                    # the dataset's embedding vectors
    ids=paths,                      # item names (used for per-item adjustments and the value-scale file)
    encoder=enc,
    concepts=shiken.concepts.FOOD,  # concept words; if omitted, the shift is expressed with the raw embedding vectors only
)
```

| Argument | Meaning |
| :-- | :-- |
| `question` | A Score or Yes–No (Noul) question |
| `items`, `ids` | The dataset's embedding vectors and the item names |
| `encoder` | The encoder used to embed the question descriptions and the concept words |
| `concepts` | A list of concept words. Either a bundled one (food, paintings, app screens, photos, faces, music) or your own made with `shiken.Concepts(words=[...], noun="服の写真")` (noun: "photo of clothing") |
| `reference` | The reference set. If omitted, `items` is used |

### Asking and answering

```python
a, b = taste.next_pair()        # the next pair to ask (item names)
p = taste.predict(a, b)         # prediction before answering: probability that a is chosen
taste.answer(a, b, prefer=a)    # a was chosen; prefer=None means "about the same"
taste.hit_rate(last=10)         # hit rate of the predictions over the last 10 questions
```

`answer` records the answer, recomputes the correction, and returns the prediction made before the answer. The size of the per-item adjustments is re-selected every 10 questions (selecting it every time would make the correction take about 5 times as long). With a dataset of 165 items, the correction plus picking the next pair took 0.3–0.4 seconds per question on a CPU.

### Scoring

```python
taste.rank()                    # item names, ordered by personalized score, highest first
mean, var = taste.scores()      # scores of each item in the dataset, and their uncertainty
mean, var = taste.scores(other_items)   # items not in the dataset (per-item adjustments are not used)
```

Scores are relative and only meaningful for comparisons within the dataset. Even for a Score question, the personalized scores are not mapped back to the level values (1–5).

### Putting the value scale into words

```python
taste.explain(top=6)
# {"up": [("炒め物", 0.41), ("中華料理", 0.38), ("辛い", 0.35)], "down": [("洋菓子", -0.44), ...]}
# (stir-fries, Chinese food, spicy / Western-style sweets — the bundled concept words are in Japanese)
```

For each item in the reference set, this returns the correlation between how much the correction moved it and how strongly it matches each word ([How it works](concepts.md#explaining-the-correction)). From the list of words you can see whether your taste leans toward 「辛い」 (spicy) or toward 「ヘルシー」 (healthy).

| Argument | Meaning |
| :-- | :-- |
| `words` | The words to compare against. If omitted, the words of `concepts` (98 words for food). You can also pass `shiken.words.large("food")` (a broader dictionary that adds color, texture, light, composition, mood and more; about 400 words for food) or your own words made with `shiken.Concepts(words=[...], noun="料理の写真")` (noun: "photo of a dish") |
| `part` | `"correction"` (default) covers only the shift from the base value scale; `"all"` covers the whole value scale |

You can also put a value scale into words from the value-scale file alone. Without a dataset, the embedding vectors of the answered items stored in the value-scale file are used in place of the reference set (about 200 items for 100 questions).

```python
enc = EmbeddingGemma2(modalities=())           # only words are embedded, so the image and audio parts are not loaded
shiken.explain("my_taste.json", encoder=enc, words=shiken.words.large("food"))
```

With the author's "looks tasty" value-scale file (101 questions), the side rated higher listed 「具だくさんの・唐辛子の多い・麻婆豆腐・辛い」 (chunky / full of ingredients, lots of chili, mapo tofu, spicy), and the side rated lower listed 「滑らかな・アイスクリーム・淡い色の」 (smooth, ice cream, pale-colored).

### Saving and loading

```python
taste.save("my_taste.json")
taste = shiken.Taste.load("my_taste.json", items=items, ids=paths, encoder=enc)
```

On loading, the correction is recomputed from the saved answers (the size of the per-item adjustments is also re-selected at this point). If you omit the dataset (`items`), the answered items stored in the value-scale file become the dataset. The loading side's dataset may differ from the saving side's. Because the value-scale file contains the embedding vectors of the answered items, answers about items not in the dataset can still be used for the correction. If the embedding model's name differs, the saved embedding vectors are not used, and the correction uses only the answers about items that are in the loading side's dataset. The number of answers that could not be used is stored in `taste.skipped`. The format is described in [Value-scale file format](profile-format.md).

There are two examples that apply a value-scale file answered and saved in the demo to a different local dataset. Both fetch the data they use at run time.

- `examples/aozora_classics.py`: ranks classics from Aozora Bunko (synopses from Wikipedia) using the taste you taught with the demo's novel synopses
- `examples/met_paintings.py`: ranks works in the Metropolitan Museum of Art's collection (public domain, images CC0) using the taste you taught with the demo's paintings

For text items, results are more stable if you embed them in the same form as when you answered. The demo's synopses are "title on line 1, synopsis from line 2 on". Testing with the author's value-scale file (50 questions), embedding only the synopsis without the title gave a rank correlation of 0.94 compared with including the title, and 7 of the top 10 items overlapped.

### Collapsing into a single vector

```python
w, c = taste.weight_vector()    # score = w · s + c
```

This turns the value scale into a single vector with the same dimensions as the embedding vectors (the value-scale vector). Use it when you want to rank in a vector-search database. Per-item adjustments are not included.

## Planned features

- **Starting from someone else's value-scale file.** Start from a chef's value-scale file and correct it with your own answers.
- **Personalization for Choice questions.**
- **Yes–No answers** (instead of a pairwise question, answering "applies" / "does not apply" for a single item).

`export_pack`, which builds a pack for the browser version, is described in [Using from JavaScript](javascript.md).
