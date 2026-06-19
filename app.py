"""
ToxSense — Multi-Label Toxicity Classifier
Streamlit App

Loads the fine-tuned DistilBERT model and tokenizer from Hugging Face Hub,
provides real-time toxicity analysis with severity scoring and LIME explainability.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import keras
import keras_nlp
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast
from huggingface_hub import hf_hub_download, snapshot_download
from lime.lime_text import LimeTextExplainer


HF_REPO_ID = "ayman005/toxsense-model"
MAX_LEN    = 128

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

LABEL_META = {
    'toxic':         {'emoji': '⚠️',  'desc': 'Generally harmful or rude content'},
    'severe_toxic':  {'emoji': '🚨',  'desc': 'Extremely offensive or aggressive'},
    'obscene':       {'emoji': '🤬',  'desc': 'Sexually explicit or vulgar language'},
    'threat':        {'emoji': '🔪',  'desc': 'Direct threats of violence or harm'},
    'insult':        {'emoji': '😤',  'desc': 'Personal attacks or derogatory language'},
    'identity_hate': {'emoji': '🏳️', 'desc': 'Hate speech targeting identity groups'},
}

SEVERITY_WEIGHTS = np.array([0.30, 0.50, 0.20, 0.50, 0.20, 0.40], dtype='float32')
BEST_THRESHOLD   = 0.80


# ── Custom loss (needed to load the model) 

def weighted_bce(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    pos_weight = tf.ones(len(LABELS), dtype=tf.float32)  # dummy — not used at inference
    loss = tf.nn.weighted_cross_entropy_with_logits(
        labels=y_true, logits=y_pred, pos_weight=pos_weight
    )
    return tf.reduce_mean(loss)


# ── Load model & tokenizer (cached so it only downloads once) 

@st.cache_resource(show_spinner="Loading model from Hugging Face Hub...")
def load_model_and_tokenizer():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="toxsense_model.keras")
    model = keras.models.load_model(
        model_path, custom_objects={'weighted_bce': weighted_bce}
    )
    tokenizer_dir = snapshot_download(repo_id=HF_REPO_ID)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_dir)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# ── Inference functions 

def severity_score(probs):
    return np.clip(
        (probs * SEVERITY_WEIGHTS).sum(axis=1) / SEVERITY_WEIGHTS.sum() * 100, 0, 100
    )


def predict_text(texts):
    enc = tokenizer(
        list(texts), max_length=MAX_LEN,
        padding='max_length', truncation=True, return_tensors='np'
    )
    logits = model(
        {'token_ids': enc['input_ids'], 'padding_mask': enc['attention_mask']},
        training=False
    )
    probs = tf.sigmoid(tf.cast(logits, tf.float32)).numpy()
    return probs, severity_score(probs)


def predict_fn_lime(texts):
    probs, _ = predict_text(texts)
    toxic_prob = probs[:, 0]
    non_toxic_prob = 1.0 - toxic_prob
    return np.column_stack([non_toxic_prob, toxic_prob])


explainer = LimeTextExplainer(class_names=['non-toxic', 'toxic'])


def lime_plot(text):
    exp = explainer.explain_instance(
        text, predict_fn_lime, num_features=8, num_samples=300
    )
    word_list = exp.as_list(label=1)
    words   = [x[0] for x in word_list]
    weights = [x[1] for x in word_list]
    words.reverse()
    weights.reverse()

    colors = ['#ff4d4d' if w > 0 else '#2ecc71' for w in weights]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    ax.barh(words, weights, color=colors, height=0.6)
    ax.set_title("Word Influence on Toxicity Score", color='white', fontsize=12, pad=10)
    ax.tick_params(colors='white', labelsize=10)
    ax.axvline(0, color='white', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('#555555')

    plt.tight_layout()
    return fig


# ── Streamlit UI 

st.set_page_config(page_title="ToxSense", page_icon="🔍", layout="centered")

st.title("🔍 ToxSense")
st.caption("Multi-Label Toxicity Classifier · DistilBERT + LIME Explainability")

st.markdown("---")

# ── Initialize session state for the single shared text box
if 'comment_text' not in st.session_state:
    st.session_state['comment_text'] = ""

# ── Examples — placed ABOVE the input, buttons write into the same key 
st.subheader("Try an example")
examples = [
    "I love this community!",
    "You are such an idiot, I hate you.",
    "I will find you and hurt you.",
]
cols = st.columns(len(examples))
for i, (col, ex) in enumerate(zip(cols, examples)):
    if col.button(ex[:25] + "...", key=f"example_btn_{i}"):
        st.session_state['comment_text'] = ex
        st.rerun()

st.markdown("---")

# ── Single input box, bound to session_state via key 
text_input = st.text_area(
    "Enter a comment to analyze",
    placeholder="Type or paste a comment here...",
    height=100,
    key="comment_text"
)

analyze = st.button("Analyze", type="primary", key="analyze_button")

if analyze and text_input.strip():
    probs, scores = predict_text([text_input])
    probs = probs[0]
    score = scores[0]

    flagged = [LABELS[i] for i, p in enumerate(probs) if p >= BEST_THRESHOLD]

    if not flagged:
        st.success(f"✅ Clean — Severity Score: {score:.1f}/100")
    else:
        st.error(f"🚩 Flagged: {', '.join(flagged)} — Severity Score: {score:.1f}/100")

    st.subheader("Label Breakdown")
    for i, label in enumerate(LABELS):
        meta = LABEL_META[label]
        pct  = probs[i] * 100
        st.write(f"{meta['emoji']} **{label.replace('_', ' ').title()}** — {pct:.1f}%")
        st.progress(min(int(pct), 100))

    st.subheader("Why this prediction?")
    with st.spinner("Generating explanation..."):
        fig = lime_plot(text_input)
        st.pyplot(fig)

    st.markdown("---")
    st.caption(
        "⚠️ This model is trained on the Jigsaw Toxic Comment dataset (Wikipedia "
        "talk page comments) and may reflect biases present in that data. "
        "Predictions are probabilistic and should not be the sole basis for "
        "moderation decisions."
    )

elif analyze:
    st.warning("Please enter some text to analyze.")