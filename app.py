import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent / "dataset_samples"

st.title("Multilingual Image–Text Retrieval Viewer")

for txt_file in sorted(DATA_DIR.glob("*.txt")):
    st.subheader(txt_file.stem)
    st.write(txt_file.read_text())
    image_path = txt_file.with_suffix(".jpg")
    if image_path.exists():
        st.image(str(image_path))
    else:
        st.warning(f"Image not found: {image_path.name}")