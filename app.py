import streamlit as st
from pathlib import Path
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json

# Load model (LaBSE or similar multilingual model)
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased")

model = load_model()

# Path setup
DATA_DIR = Path(__file__).parent / "dataset_samples"
EMBEDDING_FILE = DATA_DIR / "embeddings.json"

# Collect data files
@st.cache_data
def load_dataset():
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    data = []
    for txt_file in txt_files:
        stem = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = txt_file.with_suffix(ext)
            if candidate.exists():
                image_path = str(candidate)
                break
        if image_path:
            data.append({"id": stem, "text": text, "image_path": image_path})
    return data

data = load_dataset()

# Compute or load embeddings
@st.cache_data
def compute_or_load_embeddings():
    if EMBEDDING_FILE.exists():
        with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    embeddings = {}
    for item in data:
        emb = model.encode(item["text"], convert_to_tensor=True).tolist()
        embeddings[item["id"]] = emb
    with open(EMBEDDING_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    return embeddings

embeddings = compute_or_load_embeddings()

# UI: Query input
st.title("🔍 Multilingual Image–Text Retrieval Prototype")
st.write("App loaded successfully ✅")
query = st.text_input("Enter your query (Yoruba, Mandarin, English, etc.):")

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Score and rank
    scores = []
    for item in data:
        item_embedding = torch.tensor(embeddings[item["id"]])
        sim = util.cos_sim(query_embedding, item_embedding).item()
        scores.append((sim, item))

    scores.sort(reverse=True, key=lambda x: x[0])
    
    # Display top results
    st.markdown("### Top Matches:")
    for sim, item in scores[:3]:
        st.image(item["image_path"], caption=f"Score: {sim:.2f} | {item['text']}", use_container_width=True)
        feedback = st.radio(f"Was this relevant? ({item['id']})", ["Yes", "No"], horizontal=True, key=item['id'])
        # (You could log feedback to file/db here)
        st.markdown("---")
else:
    st.info("👋 Start by entering a query in Yoruba, Mandarin, or English to retrieve relevant cultural images.")
