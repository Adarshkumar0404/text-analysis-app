import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Google Car Manual Search", layout="centered")

st.title("Google Car Manual - Semantic Search")
st.write("Ask a question and find the most relevant document.")

DOCUMENT1 = {
    "title": "Operating the Climate Control System",
    "content": "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console. Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. Auto, Cool, Heat, Defrost."
}

DOCUMENT2 = {
    "title": "Touchscreen",
    "content": "Your Googlecar has a large touchscreen display that provides access to navigation, entertainment, and climate control. Touch icons like Navigation or Music to use them."
}

DOCUMENT3 = {
    "title": "Shifting Gears",
    "content": "Your Googlecar has an automatic transmission. To shift gears, move the shift lever. Park locks wheels, Reverse backs up, Neutral stops power, Drive moves forward, Low is for snow or slippery roads."
}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
documents_df = pd.DataFrame(documents)

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def embed_documents(texts):
    return model.encode(texts)

documents_df["embedding"] = list(embed_documents(documents_df["content"].tolist()))

def cosine_similarity(a,b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = st.text_input("Enter your question", placeholder = "How to shift gears in the Google car?")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        qv = model.encode(query)

        similarities = [
            cosine_similarity(qv, emb) for emb in documents_df["embedding"]
        ]

        documents_df["similarity"] = similarities
        best_idx = np.argmax(similarities)

        st.subheader("Most Relevant Document")
        st.markdown(f"**Title:** {documents_df.iloc[best_idx]['title']}")
        st.write(documents_df.iloc[best_idx]["content"])

        st.subheader("Similarity Scores")

        score_df = documents_df[["title", "similarity"]].sort_values(
            by="similarity",
            ascending=False
        )

        st.dataframe(score_df)
