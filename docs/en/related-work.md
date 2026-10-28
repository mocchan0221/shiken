# Related work

[日本語](../related-work.md) | English

Each of Shiken's components has precedents. Personalizing by fitting a lightweight vector on top of frozen embeddings appears in many papers and many individual implementations. Starting from a text embedding and moving it with pairwise questions exists, as does updating per-person coefficients with pairwise questions and "asking the most uncertain pair".

What we did not find, within what we surveyed, is a component that bundles all of the following into something you can embed in your own app:

1. Users write their own questions, and even with zero answers it returns typed answers with probabilities
2. It corrects with the person's own pairwise answers, and shows whether it is fitting the person through the hit rate of predictions made before each answer
3. The result can be carried to another dataset or another app as a value-scale file
4. It keeps a value scale that works across datasets separate from per-item adjustments specific to one dataset
5. It works the same way for images and audio

## Closest research and implementations

| Name | What it does | Difference from Shiken |
| :-- | :-- | :-- |
| [Gallego 2023](https://arxiv.org/abs/2308.07929) "Fast Adaptation with Bradley-Terry Preference Models" | Starts from a CLIP text embedding and moves the embedding with a Bradley–Terry model from pairwise comparisons | The starting text is fixed. Answers are crowd votes, not an individual's. Images only |
| [HumorDNA](https://github.com/ragsaniarumal/Personalized-Meme-Recommendation-from-Pairwise-Feedback) | Updates per-person coefficients with pairwise questions, shrunk toward a population prior. Asks the most uncertain pair | The starting point is annotations from a public dataset, not the user's question. No portability and no audio |
| [小島 (Kojima) et al. 2025](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001044) (PLOS Digital Health) | Queries CLIP with prompts such as "a tasty-looking dish" and derives a per-person characteristic vector from 8-level ratings of 896 images | Each person rates several hundred images. Not a start-from-a-question, correct-with-pairwise-questions design |
| [Model Sketching](https://arxiv.org/abs/2303.02884) (Lam et al., CHI 2023) | Describes concepts in natural language, evaluates them zero-shot, and combines them | No correction from an individual's pairwise answers |
| [CLIP-IQA](https://ojs.aaai.org/index.php/AAAI/article/view/25353) (Wang et al., AAAI 2023) | Evaluates image quality zero-shot with paired prompts (good photo / bad photo) | Essentially the same method as Shiken's Yes–No type (paired sentences) |

## Implementations by individual developers

Many developers build the same pipeline (embed → ask the person → fit something lightweight → choose what to ask next) separately for each product. Shiken turns this part into a component.

| Name | Target | What they built |
| :-- | :-- | :-- |
| [fineshyt](https://github.com/qweliant/fineshyt) | Photo culling | CLIP → star ratings → ridge regression. An approach that described the style in text and had a vision-language model judge it was dropped because it diverged from the person's own sense |
| [trove](https://github.com/howwohmm/trove) | Image collecting | In-browser CLIP → swipes → preference centroid |
| [Taste Engine](https://huggingface.co/spaces/Creatiste/taste-engine) | Clothing | FashionSigLIP → likes/dislikes → multiple centroids |
| [Visual Taste Approximator](https://github.com/SelfishGene/visual_taste_approximator) | Binary image classification | CLIP etc. → labeling UI → asks about the most uncertain images |

## Judgment models that return typed answers

Many judgment models that return Choice, Score, or Yes–No answers with probabilities for images and audio have been published, mostly in the Jev lineage (see [awesome-jev](https://github.com/yibie/awesome-jev) for a list). All of them answer everyone by the same standard. Shiken's answer types follow these, but what Shiken adds is the part that corrects the value scale from the person's own answers.

## Reported limitations

- CLIP similarity did not correlate with human impression ratings for impression words that are hard to infer from appearance, such as "heavy-looking" or "formal" (柴田 (Shibata) 2026, [知能と情報](https://www.jstage.jst.go.jp/article/jsoft/38/1/38_569/_article/-char/ja)).
- Some people's preferences are hard to represent with a single vector (小島 (Kojima) et al. 2025). Shiken's per-item adjustments are one remedy for this.
- Even among professional designers, pairwise judgments of screen designs do not agree much (Krippendorff's α = 0.25; [DesignPref](https://arxiv.org/abs/2511.20513)). Models adapted to each individual predicted individuals better than population-level models, even with few examples.

## References

- Bradley, R. A. & Terry, M. E. (1952). Rank analysis of incomplete block designs.
- Houlsby, N. et al. (2011). Bayesian Active Learning for Classification and Preference Learning. (BALD)
