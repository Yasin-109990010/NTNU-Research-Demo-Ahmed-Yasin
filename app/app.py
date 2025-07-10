import os
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# Load LaBSE model for multilingual sentence embeddings
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/LaBSE')

model = load_model()

# Load dataset captions
DATASET_DIR = Path(__file__).resolve().parent.parent / 'dataset_samples'
caption_files = sorted(DATASET_DIR.glob('*.txt'))

captions = []
for path in caption_files:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        captions.append({'file': path.stem, 'text': text})

caption_texts = [c['text'] for c in captions]
caption_embeddings = model.encode(caption_texts, normalize_embeddings=True)

st.title('Multilingual Image–Text Retrieval Demo')
query = st.text_input('Enter a description to search for:')

if query:
    query_emb = model.encode(query, normalize_embeddings=True)
    # Compute cosine similarity via dot product because embeddings are normalized
    scores = np.dot(caption_embeddings, query_emb)
    best_idx = int(np.argmax(scores))
    best = captions[best_idx]
    st.subheader('Best Match')
    # Placeholder image display
    st.write(f"Image placeholder: {best['file']}.jpg")
    st.write(f"Caption: {best['text']}")
