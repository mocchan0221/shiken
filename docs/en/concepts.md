[日本語](../concepts.md) | English

# How it works

## What is an embedding model?

An embedding model converts inputs such as text or images into a fixed-length sequence of numbers (an embedding vector). The conversion is trained so that inputs with similar meanings end up as nearby vectors. The texts "curry rice" and "hayashi rice" become nearby vectors, while "curry rice" and "stock prices" become distant ones.

Some models can embed not only text but also images and audio into the same space (multimodal embedding models). With such models, the vector of a photo of curry and the vector of the text "curry rice" are also close. In other words, inputs of different kinds can be compared directly by how close their vectors are.

![Text, images, and sounds pass through the embedding model and become points in the same space. The text "カレーライス" (curry rice) is close to the photo of curry (inner product 0.79) and far from "株価" (stock prices) (0.68)](../images/embedding.gif)

EmbeddingGemma 2 (EG2), which Shiken uses by default, is a multimodal embedding model published by Google. It can embed text, images, audio, and video into the same 768-dimensional space. At 740 million parameters (about 1.4 GB) it is small, and it runs on consumer GPUs, or even on a CPU if you are willing to wait.

## Scoring with an embedding model

Taking the inner product of two vectors (the sum of the element-wise products) gives a single number for how close the two are. A common use of embedding models is search. Make a vector for the text "sunset at the seaside", rank your photos by their inner product with it in descending order, and you can search photos by text.

![Top 8 of the demo's 120 scenery photos when searching with the text "海辺の夕焼け" (sunset at the seaside) (computed with EG2)](../images/search.png)

EG2 itself is described in its [model card on Hugging Face](https://huggingface.co/google/embeddinggemma-2).

Shiken uses this mechanism for scoring. Turn the two texts "a photo of a dish that looks absolutely delicious" and "a photo of a dish that looks not at all tasty" into vectors and take their difference, and you get a single direction in the space, "looks untasty → looks tasty" — something like a value scale. The inner product between a dish photo's vector and this "looks untasty → looks tasty" vector is that photo's "looks-tasty" score.

The material for a value scale does not have to be text. Anything that can be embedded into the same space — images or sounds — can be used as a question. For example, using a photo as the question to evaluate music is possible in principle (though it is not guaranteed to work well).

![The difference between the vectors of the two texts "looks not at all tasty" and "looks absolutely delicious" becomes the base value scale, and the dish photos line up along it in score order](../images/value-scale.gif)

However, closeness between embedding vectors reflects the tendencies of the data the model was trained on (a large amount of text and images collected by Google). A value scale built from text represents the average "looks tasty" in that data, and it is the same for everyone.

One way to tailor it to an individual is to retrain (fine-tune) the embedding model itself. But that requires a large number of answers per person and GPU training, and the model may become over-optimized for a particular value scale and lose the judgment it originally had (so-called catastrophic forgetting). On top of that, what you end up with is a model of several GB per person, so using it in other apps or exchanging it with others is not practical.

## How Shiken works

Shiken leaves the embedding model untouched and corrects only the value scale on top of it (768 numbers and a few coefficients) using the person's answers. The flow is as follows.

1. **Write a question.** When you write a question in text (or with images or sounds), you get the model's base value scale. At this point it can already score with zero answers.
2. **Answer pairwise questions.** Shiken picks two items from the dataset and shows them, and the person picks the one that better fits the question ("about the same" is also an option). It uses pairwise questions instead of asking for scores because people are said to give more consistent answers when comparing two things than when assigning absolute scores.
3. **Correct the value scale.** After each answer, the value scale is corrected to fit the person's answers. The next pair to show is also chosen as the pair whose answer would teach it the most.
4. **Check whether it fits.** Before the person answers, Shiken predicts which one will be picked. As these predictions start to come true, the value scale is fitting the person.
5. **Save it as a file.** The question and the answer history are saved as a value-scale file (JSON). As long as the same embedding model is used, you can score with the same value scale in other apps and on other datasets. You can also exchange it with other people.

Since a value scale is a single direction in the space, it cannot express item-specific likes and dislikes such as "regardless of the trend, I just like this one dish". So Shiken also keeps separate per-item adjustments for the items that appeared in the answers. The value scale can be carried over to other datasets, but per-item adjustments are used only within the dataset that contains those items.

After correction, it can show in words how the value scale was corrected, e.g. "shifted toward spicy dishes" or "lowered Western-style desserts".

## The actual personalization computation

From here on, we look at the computation itself.

### The base value scale and answers with zero answers

A question is written as a description (text, image, or sound) for each option. Let o_k be the embedding of option k's description and s the embedding of an item. The plausibility of option k is

```
z_k = (s · o_k − μ_k) / τ
```

which is turned into probabilities with softmax. μ_k is the mean of s · o_k over the reference set (by default, the dataset itself), subtracted to remove the bias of descriptions that tend to resemble every item. τ is the temperature. A Score question returns the probability of each level and the expected value of the level values; a Choice question returns the probability of each option; and a Yes–No question (Noul) returns the probability that it applies. Unless the temperature is fitted on labeled data, the probability values themselves are only a rough guide; what is meaningful is the ranking.

The base value scale w0 used for personalization is, for a Score question, the sum of the description embeddings weighted by the level values (minus their mean), and for a Yes–No question, the difference between the "applies" description and the "does not apply" description.

### The shape of a person's score

A person's score is expressed as the sum of the score under the base value scale and the deviation from it.

```
u(s) = a · u0(s) + Δwᵀ φ(s)
```

- u0(s) is s · w0 standardized over the reference set (mean 0, standard deviation 1).
- a is how much of the base value scale to keep. With zero answers, a = 1, giving the same ranking as the base. If the person's answers are unrelated to the base value scale, a approaches 0.
- φ(s) is the item's features used to express the deviation. It concatenates the closeness to the concept words (for dishes, 98 words such as "deep-fried", "spicy", "beautifully plated") and the embedding vector itself (scaled by 0.3). Because the deviation Δw leans toward combinations of concepts that can be expressed in words, the correction works even with few answers. The component in the same direction as u0 is removed from φ. Without this, the solution "increase a and cancel it out with Δw" would be chosen, and Δw would no longer represent the person's deviation.

### Handling answers and the correction

Pairwise answers are handled with the Bradley–Terry model.

```
P(a is picked) = σ(u(a) − u(b))      σ is the logistic function
```

Answers are recorded as 1 (a), 0 (b), or 0.5 (about the same). a and Δw get priors of "a around 1, Δw around 0", so while there are few answers the result stays close to the base value scale, and as answers increase it leans toward the person's answers. After each answer, the mode of the posterior distribution is found with Newton's method, and the spread around it with the Laplace approximation. Predicted probabilities are computed including this spread, so for pairs it is not yet confident about they come out close to 0.5.

### The next pair to ask

The next pair to ask is the one whose answer would yield the most information about the value scale (BALD, Houlsby et al. 2011). Roughly, it is a pair on which the candidate value scales disagree in their predictions. Pairs where it is confident which one will be picked, and pairs whose answer would not narrow down the candidates, are not chosen.

### Per-item adjustments

For each item k that appeared in the answers, an adjustment ε_k is added.

```
u(s_k) = a · u0(s_k) + Δwᵀ φ(s_k) + ε_k
```

The size of ε_k (the prior variance s²) is determined automatically from the answers. The correction is redone for each candidate (0, 0.3, 1, 2, 4), and the candidates are compared by how well they explain the answers (marginal likelihood). Adjustments are used only when they beat no adjustments (s² = 0) by 1.0 or more in log. This is because, even with random answers, the marginal likelihood can go up by about 1 by chance while there are few answers. On small datasets where the same items are asked about repeatedly (e.g. a few dozen songs), adjustments tend to be chosen; on datasets with many items, they are not.

### Explaining the correction

For each item in the reference set, the correlation between how much the correction moved it (u − a · u0) and how strongly it exhibits each concept is computed, and the words with the largest correlations are listed. The components of Δw are not read directly because similar concepts compete for weight, making the magnitudes of individual values unreliable. The words to compare against are not limited to the concept words used for the correction. Since it only takes correlations, a larger dictionary or your own word list can also be used. When there is no dataset, the answered items stored in the value-scale file stand in for the reference set.

### Rebuilding from a value-scale file

A value-scale file stores not the result of the correction but the question and the answers (including the embedding vectors of the answered items), and the correction is redone every time it is loaded. This is so that it can be rebuilt from the answers even if the computation differs between the side that saved it and the side that loads it. If the embedding model is different, the space is a different one, so in that case the saved embedding vectors are not used.

### Computation in the browser

The browser version computes the same posterior distribution. If u0 for each item in the dataset and the covariance between item scores under the prior are precomputed in Python, the correction becomes a computation whose size is "number of answers × number of answers" (the same form as the Laplace approximation for Gaussian process classification). Per-item adjustments correspond to adding s² to this covariance. At around 100 answers, one correction takes a few milliseconds.
