# ToxSense — Multi-Label Toxicity Classifier

Fine-tuned DistilBERT for simultaneous detection of 6 toxicity categories in online comments, with word-level explainability via LIME and a weighted severity score from 0 to 100.

&nbsp;

## Links

| Resource | Link |
|---|---|
| Live Demo | [huggingface.co/spaces/ayman005/toxsense](https://huggingface.co/spaces/ayman005/toxsense) |
| Model Weights | [huggingface.co/ayman005/toxsense-model](https://huggingface.co/ayman005/toxsense-model) |
| Training Notebook | [View on GitHub](https://github.com/aymankhan555/toxsense/blob/main/toxsense_notebook.ipynb) |
| Portfolio | [aymankhan555.github.io](https://aymankhan555.github.io) |

&nbsp;

## Problem Statement

Most toxicity classifiers treat the problem as binary — toxic or not toxic. In practice, harmful content comes in many forms that require different moderation responses. A single binary label also gives moderators no insight into why a comment was flagged.

ToxSense solves three problems. First, multi-label classification — it detects 6 overlapping toxicity types simultaneously rather than just one. Second, class imbalance — rare categories like threat (0.30%) and identity hate (0.88%) make up less than 1% of the data and naive models ignore them entirely. Third, explainability — moderators need to know which specific words triggered a flag, not just a probability score.

&nbsp;

## Features

6-label toxicity detection covering toxic, severe toxic, obscene, identity hate, insult, and threat. A severity score from 0 to 100 converts the 6 label probabilities into a single urgency score for moderation prioritization. LIME explainability highlights which words drove each prediction with a visual bar chart. Real-time inference is served through an interactive Streamlit app. Weighted BCE loss with per-label pos_weight ensures rare classes are not ignored during training.

&nbsp;

## Model Architecture

The model uses `keras_nlp.models.DistilBertClassifier` loaded from the `distilbert-base-uncased` preset. This provides the full DistilBERT backbone with a built-in classification head.

```
Raw comment text
      |
DistilBertTokenizerFast  (max_len=128, padding, truncation)
      |
DistilBertClassifier backbone  (distilbert-base-uncased, pretrained)
      |
Built-in classification head  (hidden_dim=64, dropout=0.3)
      |
6 output units  (no activation — raw scores output)
      |
Sigmoid applied inside loss during training
Sigmoid applied manually at inference to get probabilities
```

**Why DistilBERT?**
DistilBERT is 40% smaller than BERT and 60% faster at inference while retaining 97% of BERT's language understanding performance. This makes it practical for deployment on free-tier compute without meaningful loss in prediction quality.

**Why keras_nlp instead of TFAutoModel from HuggingFace?**
`keras_nlp` is natively compatible with TensorFlow 2.19 and Keras 3.x. The HuggingFace `TFAutoModel` and `TFDistilBertModel` classes have known import errors on newer Keras versions due to the Keras 2 to Keras 3 migration. Using `keras_nlp` avoids all of these version conflicts entirely.

**Why raw scores with no activation on the final layer?**
The loss function `tf.nn.weighted_cross_entropy_with_logits` applies sigmoid internally, which is numerically more stable than applying sigmoid in the model and then computing log-loss separately. This also allows `pos_weight` to be applied correctly inside the loss.

&nbsp;

## Dataset

Jigsaw Toxic Comment Classification Challenge — [kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)

| Property | Value |
|---|---|
| Source | Wikipedia talk page comments |
| Total size | 159,571 labeled comments |
| Label type | 6 binary labels, multi-label (overlapping) |
| Language | English |
| Train split | 127,656 comments (80%) |
| Validation split | 15,957 comments (10%) |
| Test split | 15,958 comments (10%) |

**Label distribution and pos_weight:**

| Label | Positive Count | Positive Rate | pos_weight |
|---|---|---|---|
| toxic | 15,294 | 9.58% | 9.4 |
| severe_toxic | 1,595 | 1.00% | 98.4 |
| obscene | 8,449 | 5.29% | 17.9 |
| identity_hate | 1,405 | 0.88% | 113.4 |
| insult | 7,877 | 4.94% | 19.3 |
| threat | 478 | 0.30% | 334.1 |

The `threat` label has a pos_weight of 334.1, meaning the model is penalized 334 times more heavily for missing a threat than for a false alarm. Without this correction the model would learn to predict clean for everything and still achieve high accuracy.

&nbsp;

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.19 / Keras 3.13 |
| Pretrained Model | DistilBERT (distilbert-base-uncased) via keras_nlp 0.26 |
| Tokenizer | HuggingFace DistilBertTokenizerFast |
| Explainability | LIME (lime.lime_text.LimeTextExplainer) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn (ROC-AUC, F1, Confusion Matrix) |
| Compute | Kaggle GPU (Tesla T4) |
| Deployment | HuggingFace Spaces (Docker + Streamlit) |
| Model Hosting | HuggingFace Hub |

&nbsp;

## Key Design Decisions

**pos_weight for class imbalance**

pos_weight is calculated per label as negative count divided by positive count. For the threat label with a 0.30% positive rate this produces a weight of 334.1, forcing the model to pay heavy attention to the rare positive examples it encounters during training.

**Weighted binary cross-entropy loss**

`tf.nn.weighted_cross_entropy_with_logits` treats each of the 6 labels as an independent binary classification problem, which is mathematically correct for multi-label classification where a comment can belong to multiple categories simultaneously.

**Threshold tuning on validation set**

After training, thresholds from 0.20 to 0.80 in steps of 0.05 are swept on the validation set and the threshold maximizing macro F1 is selected. The tuned threshold (0.80) is then applied once, unchanged, to the held-out test set to avoid data leakage.

**Severity score**

Six label probabilities are combined into a single 0 to 100 urgency score using a weighted sum. Labels with higher potential for direct harm receive higher weights.

| Label | Weight |
|---|---|
| toxic | 0.30 |
| severe_toxic | 0.50 |
| obscene | 0.20 |
| threat | 0.50 |
| insult | 0.20 |
| identity_hate | 0.40 |

&nbsp;

## Results

**Overall metrics on held-out test set:**

| Metric | Value |
|---|---|
| Macro AUC | 0.9838 |
| Macro F1 | 0.4592 |
| Best Threshold | 0.80 |

**Per-label breakdown:**

| Label | AUC | F1 |
|---|---|---|
| toxic | 0.9831 | 0.7832 |
| severe_toxic | 0.9901 | 0.2899 |
| obscene | 0.9881 | 0.7096 |
| identity_hate | 0.9790 | 0.2124 |
| insult | 0.9873 | 0.6638 |
| threat | 0.9751 | 0.0964 |

**Why is Macro AUC high but Macro F1 lower?**

AUC measures how well the model ranks toxic comments above non-toxic ones, independently of any threshold. All 6 labels score above 0.975 AUC, showing the model has strong discriminative ability. The lower macro F1 reflects the difficulty of converting those rankings into precise binary predictions for very rare labels like threat (0.30% positive rate) and identity hate (0.88%). These labels have too few positive training examples for precise threshold calibration regardless of the loss weighting strategy.

**Severity score on example inputs:**

| Comment | Severity | Result |
|---|---|---|
| "I love this community!" | 1.8 | Safe |
| "You are such an idiot, I hate you." | 97.0 | Flagged |
| "I will find you and hurt you." | 83.8 | Flagged |

&nbsp;

## LIME Explainability

LIME creates hundreds of perturbed versions of the input comment by randomly masking words, runs each through the model, and identifies which words most influenced the prediction by observing how the toxic probability changes when each word is removed.

Red bars show words pushing the prediction toward toxic. Green bars show words pushing the prediction away from toxic.

For "You are a worthless piece of trash" (Score: 0.992), the words `trash`, `piece`, and `worthless` are the strongest toxic contributors. The model correctly identifies the core harmful phrase rather than flagging based on surrounding neutral words.

For "Thanks for the great explanation!" (Score: 0.019), the words `explanation` and `Thanks` are the strongest non-toxic signals. The model correctly identifies positive, constructive language.

&nbsp;

## Run Locally

```bash
git clone https://github.com/aymankhan555/toxsense.git
cd toxsense
pip install -r requirements.txt
streamlit run app.py
```

The model (266MB) is downloaded automatically from HuggingFace Hub on first run and cached locally. No manual download needed.

&nbsp;

## Project Structure

```
toxsense/
├── app.py                    # Streamlit demo app
├── requirements.txt          # Dependencies
├── .gitignore
├── LICENSE                   # MIT
├── README.md
└── toxsense_notebook.ipynb   # Full training notebook
```

Model weights and tokenizer are hosted on HuggingFace Hub and are not stored in this repository.

&nbsp;

## Limitations and Ethics

The dataset is sourced from Wikipedia talk page comments in English only. The model reflects the toxicity patterns of editorial disputes on Wikipedia and may not generalize perfectly to other platforms or other languages.

The threat and identity hate labels show low F1 scores (0.0964 and 0.2124) due to severe class imbalance. These labels have strong AUC scores (good ranking ability) but low F1 (imprecise threshold calibration) because there are too few positive examples in the training data.

Predictions are probabilistic and should not be used as the sole criterion in production content moderation. Human review of flagged content is strongly recommended.

Like all models trained on internet data, this model may reflect demographic biases present in the Wikipedia editor community.

&nbsp;

## License

MIT License — see [LICENSE](LICENSE) for details.

&nbsp;

## Author

**Md Ayman Khan**

GitHub: [@aymankhan555](https://github.com/aymankhan555)

Kaggle: [@aymankhan555](https://www.kaggle.com/aymankhan555)

LinkedIn: [ayman-khan-6a4485345](https://www.linkedin.com/in/ayman-khan-6a4485345)

Portfolio: [aymankhan555.github.io](https://aymankhan555.github.io)
