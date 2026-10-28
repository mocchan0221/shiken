[日本語](README.md) | English

# Shiken (四見)

![Each time you answer a pairwise question, the ranking of dishes reshuffles toward your own idea of "looks tasty"](docs/images/readme-image.gif)

Try the in-browser demo here:

https://shiken.mocchan.net/

Intro video (on X):

https://x.com/mocchan_dev/status/2115394802732974511

## What is this?
A **personalizable & shareable**, Jev-like **multimodal scoring system** built on **embedding models**.


1. **Multimodal scoring** Jev-like, it evaluates how strongly a given input matches a given criterion. The focus is on getting the relative order right rather than on the correctness of absolute values.
2. **Personalized by your own answers** The out-of-the-box evaluation follows the directions the embedding model already has, but by correcting it you can build a value scale that is yours alone.
3. **Shareable** The value scale you build by answering questions becomes a single file. As long as the model is the same, it works across apps and can be shared with other people.

All scoring needs is the vectors you get by passing the targets through an embedding model. Once the embedding vectors are computed, scoring and personalization run on numpy alone, with no GPU and no LLM API.

## What can it do?
- Evaluate images with language

![Each time you answer "Which looks tastier?", the ranking of 179 dish photos reshuffles](docs/images/readme-image.gif)

- Evaluate music with an image

![Using a single photo as the question to rank 110 songs (with zero answers)](docs/images/readme-music.gif)

- Evaluate language with language

![120 novel synopses, ranked after a correction from 50 pairwise answers](docs/images/readme-text.gif)

- Put your own values into words

![Describing a corrected value scale by the words it rates high and the words it rates low](docs/images/readme-explain.gif)

## Minimal example (Python)

```python
import shiken
from shiken import Score
from shiken.encoders import EmbeddingGemma2   # pip install "shiken[eg2]"

enc = EmbeddingGemma2()
photos = ["dish01.jpg", "dish02.jpg", ...]
taste = shiken.Taste(
    question=Score([(1, "a photo of a dish that looks not at all tasty"),
                    (5, "a photo of a dish that looks absolutely delicious")]),
    items=enc.embed_files(photos), ids=photos, encoder=enc,
)

taste.rank()                   # with zero answers: the ranking produced by the bare embedding model
a, b = taste.next_pair()       # the next pair to ask about
taste.answer(a, b, prefer=a)   # "a looks tastier"
taste.rank()                   # the ranking tailored to you
taste.save("my_taste.json")    # value-scale file (a file that stores your value scale)
```

You can also use embedding models other than the default EmbeddingGemma 2 (it has been confirmed to work with SigLIP 2, Qwen3-VL-Embedding, Ovis-Embedding, and Omni-Embed-Nemotron). See [Using from Python](docs/en/python.md) and [Embedding models](docs/en/embedding-models.md#results-with-other-models) for details.

## Using it in the browser (JavaScript)

If you prepare in Python the list of vectors obtained by running the dataset through the embedding model, asking pairwise questions and personalization can happen entirely inside the browser. There is no need to send answers to a server. See [Using from JavaScript](docs/en/javascript.md) for details.

## Possible applications

![Automatic selection of generated images, evaluating the look of UI designs, personal search and recommendations, carrying your taste over to other data](docs/images/applications.png)

- **Automatic selection of generated images**: From a large batch of candidates produced by image generation, keep only the top ones, as if you had picked them by eye. It can also serve as the "judge" in a loop that generates while varying the prompt.
- **Evaluating the look of UI designs**: Score screenshots of screens with a team's or users' value scale for "easy to read". You can compare design proposals or check that a change has not made things look worse.
- **Personal search and recommendations**: Re-rank search or recommendation results with your own value scale. If you export the value-scale vector (`weight_vector()`), you can use it directly for re-ranking in a vector search database.
- **Carrying your taste over to other data**: Apply a value-scale file made in the demo to a different dataset. As examples, there is [`examples/aozora_classics.py`](examples/aozora_classics.py), which ranks classic works from Aozora Bunko with a file made on the demo's "novel synopses", and [`examples/met_paintings.py`](examples/met_paintings.py), which ranks works from the Metropolitan Museum of Art's collection (public domain) with a file made on the demo's "paintings".

## Documentation

| Document | Contents |
| :-- | :-- |
| [How it works](docs/en/concepts.md) | How this system works |
| [Writing questions](docs/en/questions.md) | How to write questions for scoring |
| [Using from Python](docs/en/python.md) | The Python API |
| [Using from JavaScript](docs/en/javascript.md) | How to use it in the browser |
| [Value-scale file format](docs/en/profile-format.md) | What the saved file contains, and how to treat it as personal data |
| [Embedding models](docs/en/embedding-models.md) | The default model, swapping in other models, licenses |
| [Results so far](docs/en/evaluation.md) | Per subject: how well it does with zero answers, and how closely it matches after how many answers |
| [Development](docs/en/development.md) | Repository layout, tests that keep the Python and JavaScript versions in agreement |
| [Demo](demo/README.en.md) | About the demo |
| [Related work](docs/en/related-work.md) | Existing similar tools, research, etc. |

## About the name

Shiken (四見) comes from the Buddhist (Yogācāra) phrase "一水四見" (*issui shiken*, "one water, four views"). The same water appears to heavenly beings as ground adorned with jewels, to humans as water, to fish as a dwelling, and to hungry ghosts as pus and blood. It means that the same thing looks different depending on who is looking. The starting point of this system is that even for the same dish photos, the "looks tasty" ranking differs from person to person.

It is also a pun on "私見" (*shiken*), meaning "a personal view" — a way of seeing that is yours alone.

## License

Apache-2.0. The code's license and the terms of the default embedding model (EmbeddingGemma 2 is under the Gemma Terms of Use) apply separately.
