import streamlit as st
from sentence_transformers import SentenceTransformer

# Define the loader with caching
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Defining the embedding function
def emb_text(text, model):
    return model.encode(text)