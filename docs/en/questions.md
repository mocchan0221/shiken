[日本語](../questions.md) | English

# Writing questions

A question is made of a type and a description for each option. Descriptions can be text, images, or sounds. Shiken scores by the closeness between the description's embedding vector and the item's embedding vector, so how well it does with zero answers depends heavily on how the descriptions are written. For a question detecting "spicy" (辛い) in dishes, the AUROC ranged from 0.60 to 0.92 depending on how the description text was worded (descriptions written in Japanese).

## The three types

### Score

Asks where an item falls on a set of ordered levels. The answer is the probability of each level and the expected value of the level values.

```python
Score([(1, "a photo of a mild, gently flavored dish with no heat"),
       (5, "a photo of a fiery dish loaded with chili peppers and chili oil")])
```

The levels can be just the two ends, or you can fill in between to make five. This type and the Yes–No type are the ones that can be personalized with pairwise answers.

### Choice

Picks one of the options. The answer is the probability of each option.

```python
Choice({"Japanese": "a photo of Japanese food", "Chinese": "a photo of Chinese food", "Italian": "a photo of Italian food"})
```

The Choice type currently cannot be personalized with pairwise answers (it answers with the base value scale only).

### Yes–No (Noul)

Asks whether something applies. The answer is the probability that it applies.

```python
Noul("a photo of a soup or brothy dish", contrast="a photo of a dish with no liquid")
Noul("a photo of the face of a person wearing glasses", neutral="a photo of a person's face")
```

If you write `contrast` (a description of the state where it does not apply), it is solved as a two-choice question between the two descriptions. If you do not, it is measured by the difference from a neutral sentence consisting only of the domain noun passed as `neutral` (such as "a photo of a person's face"). If there is no `neutral` either, it is measured by the difference from the mean over the reference set (if concept words are passed when personalizing, their noun is used for the neutral sentence). For "present or absent" questions such as glasses or maps, leaving the negation out and comparing against a neutral sentence worked better than writing a negative sentence such as "a person not wearing glasses".

## Describing with text

- **Describe what matching items look like, not a single word.** "唐辛子やラー油がたっぷり入った激辛料理の写真" ("a photo of a fiery dish loaded with chili peppers and chili oil") works better than "辛い" ("spicy"). Embedding models are good at measuring closeness to text that describes the item itself.
- **Be careful with color words.** Writing a color such as "真っ赤な" ("bright red") pushes up other things of that color (such as red velvet cake, which has nothing to do with spiciness).
- **You can write several phrasings and average them.** If you pass a tuple of texts instead of a single description, the embedding vectors are averaged. Putting Japanese and English side by side never beat the better of the two languages, but it avoided landing on the worse one.
- **Questions that distinguish proper names, such as dish names, work better in English.** In 101-class classification on Food-101, top-1 was 0.83 with English dish names and 0.67–0.70 with Japanese ones (names of foreign dishes written in katakana put Japanese at a disadvantage). For questions about impressions or qualities, there was no consistent difference between Japanese and English.

## Describing with images or sounds

> Only the use of ranking songs with a photo has been tested, on the demo data (see [Results: ranking songs with a photo](#results-ranking-songs-with-a-photo) below). The other combinations describe what the mechanism makes possible; how well they work has not been verified yet.

You can also pass files or embedding vectors as descriptions.

```python
from shiken import File

# Rank songs that suit a single photo
Noul(File("sunset.jpg"))

# Turn "this kind of dish" into levels using example photos
Score([(1, (File("plain1.jpg"), File("plain2.jpg"), File("plain3.jpg"))),
       (5, (File("great1.jpg"), File("great2.jpg"), File("great3.jpg")))])
```

There are two caveats known from how the mechanism works.

- **Within one question, keep the kinds of descriptions the same.** In the space of a multimodal embedding model, images tend to cluster with images and text with text (the modality gap, known from CLIP and others). If level 1 is text and level 5 is an image, "image-ness" itself gets mixed into the direction of their difference, and items end up ranked by image-ness rather than by what the question asks. On the other hand, when there is a single description and the items are of a single kind, as in ranking songs with one photo, this bias is added equally to every item, so the ranking does not change. For the same reason, when a Yes–No question's description is an image or sound, it is measured by the difference from the mean over the reference set rather than from a neutral sentence.
- **Several example images are better than one.** When evaluating images with images, whether colors and composition are similar has a strong effect. Passing several examples of the same level and averaging them should make it easier to keep the qualities the examples share.

### Results: ranking songs with a photo

Using a single scenery photo from the demo as the question (`Noul(File(photo))`, a single description with no neutral sentence), the demo's 110 "world music" songs were ranked with zero answers. This was checked by rank correlation with the songs' BPM and with the rankings from the text questions 「落ち着く」 ("relaxing") and 「ノれる」 ("groovy") (EmbeddingGemma 2; the text questions were in Japanese).

| Photo used as the question | Correlation with 「落ち着く」 ("relaxing") | Correlation with BPM | Mean BPM of top 10 songs (overall: 108) |
| :-- | --: | --: | --: |
| Lakeside cabin on a misty morning | +0.67 | −0.31 | 90 |
| Glacial lake on a calm morning | +0.79 | −0.33 | 86 |
| Big-city intersection on a rainy night | −0.32 | +0.33 | 131 |
| Asian night market on a summer night | −0.21 | +0.32 | 114 |
| Beach on a sunny southern island | +0.40 | −0.04 | 101 |
| Lava volcano at dusk | +0.43 | −0.09 | 103 |

With quiet photos, slow and calm songs came to the top; with photos of the city at night or a night market, fast and intense songs did. On the other hand, some songs tended to come out on top for every photo (ambient, and dubstep that starts on piano), and for photos with an in-between mood such as the beach or the volcano, the ranking was not clear-cut.

Using two photos with different moods as the matching side and the non-matching side (`Noul(File("lakeside.jpg"), contrast=File("intersection.jpg"))`) gave a correlation of +0.77 with "relaxing" and −0.52 with BPM, stronger than with a single photo. If you choose songs with a photo, we recommend adding a contrasting photo.

## What the base value scale gets wrong

Since the base value scale is a single direction in the space, it struggles with evaluations like the following.

- **Evaluations that require counting.** The rank correlation between "amount of information" on app screens and the actual number of elements was 0.07–0.24.
- **Impressions that are hard to infer from appearance or sound.** The rank correlation between the "brightness" of music and human-rated valence was 0.52. It has also been reported that with CLIP, impression words such as "looks heavy" or "formal" did not correlate with human ratings (Shibata 2026).
- **The probability values themselves.** Unless the temperature is fitted on labeled data, what is meaningful is the ranking.

Even for such questions, personalization with pairwise answers is possible. However, the value scale is still a single direction, so there is a limit to how closely it can be fitted.
