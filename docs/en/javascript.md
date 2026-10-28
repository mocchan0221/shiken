# Using from JavaScript

[日本語](../javascript.md) | English

The browser version (`js/shiken.js`, a single ES module with no dependencies) does pairwise asking, personalization and scoring entirely inside the browser. Answers are never sent anywhere, so it can be embedded in services that do not want to collect preference data on a server.

## Preparation: building a pack (Python)

The browser does not run the embedding model. Instead, once the dataset and the question are fixed, the values it needs are computed in Python and stored in a pack (a JSON file).

```python
taste = shiken.Taste(question=..., items=items, ids=ids, encoder=enc, concepts=shiken.concepts.FOOD)
taste.export_pack("pack.json", include_embeddings=True, meta={"prompt": "Which looks tastier?"})
```

A pack contains the following.

| Contents | Size |
| :-- | :-- |
| Item names | number of items |
| Each item's score on the base value scale | number of items |
| Covariance of scores between items under the prior distribution | number of items × number of items |
| Each item's concept scores (used for explaining the correction) | number of items × number of concepts |
| The question definition, the embedding model's name, and `meta` (free-form values for your UI) | small |
| Each item's embedding vector (when `include_embeddings=True`; needed to write value-scale files in the same format as the Python version) | number of items × about 1 KB |

Since the covariance grows with the square of the number of items, datasets of up to a few hundred items are expected. A 104-item pack, including embedding vectors, was about 330 KB. For larger datasets, pick a few hundred items to use for asking and put those in the pack, then score the whole dataset on the Python side with `weight_vector`.

Question descriptions are embedded when the pack is built, so in the browser version users cannot create questions on the spot.

## Usage

```js
import { Taste } from "./shiken.js";

const pack = await (await fetch("pack.json")).json();
const taste = new Taste(pack);

const [a, b] = taste.nextPair();     // the next pair to ask (item names)
taste.predict(a, b);                 // prediction before answering: probability that a is chosen
taste.answer(a, b, { prefer: a });   // prefer: null means "about the same"; returns the prediction made before the answer
taste.rank();                        // personalized ranking
taste.scores();                      // score per item (in the order of the pack's ids)
taste.hitRate(10);                   // hit rate of the predictions over the last 10 questions
taste.baseWeight;                    // how much of the base value scale is retained (1 with zero answers)
taste.explain(6);                    // { up: [["炒め物", 0.41], ...], down: [...] }  (bundled concept words are Japanese; 炒め物 = stir-fries)
taste.explain(6, { part: "all" });   // the whole value scale, rather than the shift from the base value scale
```

The next pair is picked from pairs not yet asked (once all of them have been asked, already-asked pairs are picked too). The correction is a computation of size number of answers × number of answers, and took about 4 milliseconds per question with a 104-item dataset. As in the Python version, the size of the per-item adjustments is re-selected by marginal likelihood every 10 questions.

## Saving

```js
const profile = taste.toProfile();             // value-scale file (a JSON object)
const restored = Taste.fromProfile(pack, profile);
restored.skipped;                              // number of answers about items not in the pack, which could not be used
```

`toProfile` returns a value-scale file in the same format as the Python version ([Value-scale file format](profile-format.md)). The Python version's `Taste.load` can read it as is. Whether to keep it in browser storage, let the user download it as a file, or send it to a server is up to the embedding application. A value-scale file reveals the person's preferences, so handle it as personal data.

## Matching the Python version

The browser version and the Python version compute the same posterior distribution from the same answers. The repository's test (`tests/test_parity.py`) computes the prediction probabilities, scores, size of the per-item adjustments and contents of the correction from the same pack and answers with both, and compares them. The probabilities differed by about 1e-13 ([Development](development.md)).
