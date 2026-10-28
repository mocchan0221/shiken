# Results so far

[日本語](../evaluation.md) | English

These are the results of the author's experiments with EmbeddingGemma 2. For each subject, we summarize, in order: how accurate the base value scale was, how far accuracy rose after how many answers, whether the results are reproducible, and whether they carry over to another dataset. Parts not yet measured are marked "not measured".

The questions were asked in Japanese; they are shown below in the original Japanese with an English gloss.

The public datasets used for evaluation (Food-101, WikiArt, RICO, Unsplash, DEAM) include some that cannot be redistributed, so they are not included in this repository.

## Summary by subject

Values are the fraction of the person's own pairwise-question answers that were predicted correctly (accuracy). "About the same" answers are excluded.

| Subject | Question | Who answered | With zero answers | After answering | Reproducibility | Carrying over to another dataset |
| :-- | :-- | :-- | --: | :-- | :-- | :-- |
| Food photos (165 taken by the author) | 美味しそう ("looks tasty") | Author | 0.47 | 10 questions 0.52 → 25 questions 0.61 → 50 questions 0.66 → 75 questions 0.70 | Not measured | Not measured |
| Paintings (WikiArt) | メリハリがあって視線が一点に集まる ("has strong contrast that draws the eye to one point") | Author | 0.57 | About 100 questions 0.66 | Not measured | Not measured |
| App screens (RICO) | Same as above | Author | 0.54 | About 100 questions 0.62 | Not measured | Not measured |
| Photos (Unsplash) | Same as above | Author | 0.47 | About 100 questions 0.55 | Not measured | Not measured |
| Faces (240 generated) | Same as above | Author | 0.61 | About 100 questions 0.64 | Not measured | Not measured |
| Food photos (104 generated) | 美味しそう ("looks tasty") | 3 of the author's family members | Not tallied | 127–200 questions 0.56–0.60, 0.63–0.73 with per-item adjustments | Not measured | Not measured |
| Music (40 game soundtrack songs) | 思わず聴き入ってしまう、印象に残る ("draws you in, memorable") | Author | Not tallied | 114 questions 0.73, 0.84 with per-item adjustments | Not measured | Not measured |

How these were measured:

- **With zero answers** is the fraction of the person's answers predicted correctly by the base value scale. It is around 50% because Shiken picked the "most uncertain pair" when asking, so it cannot be compared directly with numbers from prior work that measures on random pairs.
- **n questions** means correcting with n answers chosen at random and predicting the remaining answers (mean of 20 runs).
- **About 100 questions** and the family and music values use 10-fold cross-validation (correct with 90% of the answers, predict the remaining 10%).

We also record the hit rate of the prediction made before each answer (whether the value scale at the time of asking predicted that answer). From the first half to the second half of the answers, it rose from 0.63 to 0.72 for food and from 0.62 to 0.70 for paintings. For photos it was 0.49 in both halves, so personalization did not work.

All three family members made almost no use of the base value scale (a = 0.02–0.14). In other words, their "looks tasty" was determined more by likes and dislikes for individual dishes than by visual tendencies, and adding per-item adjustments raised accuracy. Music followed the same pattern.

### Carrying over across domains

With the same question (the contrast question), we applied the author's value scale built in one domain to the author's answers in another domain. Accuracy varied from 0.36 to 0.68 depending on the combination, so we cannot say it predicts consistently across domains. Carrying over to a different dataset in the same domain (e.g. using a value scale built on the author's food photos for the generated food photos) has not been measured yet.

## Results with simulated users

We had simulated users with hidden preferences answer questions, and measured the rank correlation with the hidden preference on test images. Unlike real people, their answers have no noise and the ground truth is known, which makes it easy to compare how close the scale gets after how many questions.

| | 0 questions | 5 questions | 20 questions | 80 questions | 1,000 questions |
| :-- | --: | --: | --: | --: | --: |
| Price (generated food, question 高そう "looks expensive") | 0.70 | 0.72 | 0.76 | 0.78 | 0.82 |
| Preference (Food-101, question 美味しそう "looks tasty") | 0.49 | 0.56 | 0.66 | 0.70 | 0.78 |

- Even with 1,000 answers, the rank correlation plateaued around 0.8. Because a value scale is a single direction in the space, preferences that cannot be expressed along that direction cannot be expressed no matter how many answers you add.
- Learning from the answers alone, without starting from the base value scale, reached a rank correlation of only 0.2–0.35 at 5 questions.
- Asking uncertain pairs gave higher results at 5–10 questions than asking random pairs (preference, at 10 questions: 0.49 → 0.62).

## Base value scale vs. ground-truth labels

On data with ground-truth labels, we measured how accurate scoring with zero answers is. Except for food, each value is the better of two ways of building the description (difference from a sentence with the opposite meaning / difference from a neutral sentence).

| Domain | Evaluation (metric) | Value |
| :-- | :-- | --: |
| Food (Food-101) | 101-class classification (top-1 / top-5) | 0.83 / 0.97 |
| | Sweet / soup (AUROC) | 0.996 / 0.98 |
| | Spicy (AUROC, depends on the description) | 0.60–0.92 |
| Paintings (WikiArt) | Realistic vs. abstract (AUROC) | 0.99 |
| Faces (240 generated) | Glasses / smiling (AUROC) | 0.99 / 0.99 |
| Photos (Unsplash) | Sea / flowers (AUROC) | 0.90 / 0.93 |
| App screens (RICO) | Contains a map (AUROC) | 0.97 |
| | Information density vs. number of elements (rank correlation) | 0.21 |
| Music (DEAM) | Brightness vs. valence (rank correlation) | 0.52 |
| Generated food (104 images) | "Looks expensive" score vs. price range (rank correlation) | 0.7 |

Probability values can be calibrated by setting the temperature on labeled data (on Food-101 classification, setting the temperature on half of the data and measuring on the rest gives an ECE of about 0.02).

### Demo datasets

For the nine demo sets (generated images and music, and synopses of fictional novels), we used the labels assigned at generation time as a guide and measured the rank correlation with the ordering produced with zero answers. The labels are not the answers to the questions themselves but indicators of related properties, so they serve to check whether the direction is right (positive where the correlation should be positive). All 25 questions came out in the expected direction.

| Data | Question | Label compared | Count | Rank correlation |
| :-- | :-- | :-- | --: | --: |
| Food | 美味しそう ("looks tasty") | Not a very ordinary meal (plain=False) | 179 | +0.48 |
| Food | 辛そう ("looks spicy") | Spiciness 0–3 | 179 | +0.50 |
| Food | ヘルシー ("healthy") | Healthiness 1–5 | 179 | +0.63 |
| Food | 高そう ("looks expensive") | Price range, 4 levels | 179 | +0.70 |
| Rooms | 住んでみたい ("would like to live here") | Luxury 1–3 | 119 | +0.26 |
| Rooms | 落ち着く ("calming") | Tidiness 1–3 | 119 | +0.51 |
| Rooms | 片付いている ("tidy") | Tidiness 1–3 | 119 | +0.67 |
| Landscapes | 寒そう ("looks cold") | Coldness 1–3 | 120 | +0.77 |
| Landscapes | にぎやか ("lively") | Liveliness 1–3 | 120 | +0.67 |
| Animals | かわいい ("cute") | Danger 1–3 (should correlate negatively) | 100 | -0.31 |
| Animals | 怖い ("scary") | Danger 1–3 | 100 | +0.47 |
| Animals | 飼ってみたい ("would like to keep as a pet") | Is a pet (group=ペット, "pet") | 100 | +0.51 |
| Pictures | かわいい ("cute") | Cute style 0/1 | 120 | +0.40 |
| Pictures | 不気味 ("creepy") | Dark style 0/1 | 120 | +0.59 |
| Pictures | 写実的 ("realistic") | Abstraction 1–3 (should correlate negatively) | 120 | -0.59 |
| Clothes | フォーマル ("formal") | Neatness 1–3 | 100 | +0.67 |
| Clothes | 派手 ("flashy") | Flashiness 1–3 | 100 | +0.53 |
| World music | ノれる ("makes you groove") | BPM | 110 | +0.54 |
| World music | 落ち着く ("calming") | BPM (should correlate negatively) | 110 | -0.50 |
| Japanese songs | ノれる ("makes you groove") | BPM | 100 | +0.45 |
| Japanese songs | 懐かしい ("nostalgic") | In the Shōwa / traditional top-level category | 100 | +0.26 |
| Novel synopses | 怖そう ("sounds scary") | Scariness 1–3 | 120 | +0.57 |
| Novel synopses | 泣けそう ("sounds like a tearjerker") | Poignancy 1–3 | 120 | +0.50 |
| Novel synopses | 笑えそう ("sounds funny") | Funniness 1–3 | 120 | +0.61 |
| Novel synopses | 壮大 ("epic") | Scale of the story 1–3 | 120 | +0.57 |

Results for the same 24 questions with other embedding models are in [Embedding models](embedding-models.md#results-with-other-models).

## Do per-item adjustments react to random answers?

We checked whether per-item adjustments get wrongly selected with random answers (104 generated food images, 10 runs each). With 50 questions they were selected in 3 of 10 runs (the largest marginal-likelihood gain was +1.2), and with 150 questions in 1 run (largest +0.06). In neither case did accuracy improve. The gains for the family (+1.9 to +5.6) and for music (+4.2) are clearly larger than this level.

## Shrinking the embedding vectors

A comparison of truncating the embedding vectors to their leading dimensions, or converting them to int8 or 1 bit. "Base value scale" is the mean difference over 34 metrics with ground-truth labels; "After answering" is the mean cross-validated accuracy over 8 of the author's value-scale files.

| Setting | Bytes per item | Base value scale (diff. from 768 dims) | After answering | Accuracy at 10 / 20 / 40 / 80 questions |
| :-- | --: | --: | --: | :-- |
| 768-dim float | 3,072 | Baseline | 0.668 | 0.555 / 0.591 / 0.617 / 0.650 |
| 768-dim int8 | 772 | 0.000 | 0.665 | 0.555 / 0.590 / 0.616 / 0.652 |
| 512-dim float | 2,048 | −0.003 | 0.667 | 0.557 / 0.584 / 0.612 / 0.657 |
| 256-dim int8 | 260 | −0.030 | 0.663 | 0.557 / 0.582 / 0.617 / 0.654 |
| 128-dim float | 512 | −0.073 | 0.657 | 0.557 / 0.583 / 0.614 / 0.646 |
| 768-dim 1-bit | 96 | −0.036 | 0.648 | 0.546 / 0.575 / 0.600 / 0.619 |

Converting to int8 while keeping 768 dimensions does not change the results. At 256 dimensions the base value scale drops slightly, with larger drops for music (−0.08) and app screens (−0.04). At 128 dimensions and at 1 bit the drop is noticeable.

Reducing the dimensions did not reduce the number of answers needed for personalization. The improvement from 10 to 80 questions is almost the same at 768 and at 128 dimensions. This is presumably because accuracy with few answers is determined mainly by the base value scale and the prior distribution.

## Speed

- Python version: with a 165-item dataset, the correction after each answer plus picking the next pair takes 0.3–0.4 seconds on CPU (up to 0.7 seconds).
- Browser version: the correction after each answer takes a few milliseconds.
- Computing the embedding vectors is separate; with EmbeddingGemma 2 it is fast if you have a GPU, and the results are saved and reused.

## Not yet measured

- **Reproducibility.** The agreement rate when the same person re-answers the same pairs on a later day (noise on the human side), and the variance of the correction when the question order changes (noise on Shiken's side).
- **Carrying over to another dataset in the same domain.** Applying a value-scale file built on one dataset to answers the same person gave on a different dataset.
- **The rest of the other embedding models.** Results with five models are in [Embedding models](embedding-models.md#results-with-other-models). Music-specific models (CLAP, etc.), comparison with English-translated questions, and comparison with simulated users have not been done yet.
- **Questions written with images or audio** (only ordering songs with a photo has been tried; see [Writing questions](questions.md#results-ranking-songs-with-a-photo)).
