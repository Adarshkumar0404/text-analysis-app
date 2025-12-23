import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the function from other file
from emb_txt import load_model, emb_text
from similarity import compute_similarity

st.set_page_config(page_title="Text Analysis", layout="wide")
st.title("Text Analysis App")

model = load_model() # Load model once here

tab1, tab2 = st.tabs(["Generating Embedding", "Cosine Similarity"])

# Tab 1 : Text embedding
with tab1:
    st.header("Generate Vector Representation")
    st.write("Enter a text to see its numerical representation")
    text_input = st.text_area("Input Text:", height=150, key="vector_input")

    if st.button("Generate Vector"):
        if text_input:
            # Split input into a list of lines
            lines = [line.strip() for line in text_input.split('\n') if line.strip()]

            if lines:
                # Get embedding for all lines at once
                embeddings = emb_text(lines, model)
                st.success(f"Generated {len(lines)} vectors!")
                # Create a DataFrame 
                df_vectors = pd.DataFrame(embeddings, index=lines)
                st.write("### Vector Data Table")
                st.write("Each row matches a line of your text. Each column is a dimension of the vector.")
                # Display Dataframe
                st.dataframe(df_vectors)
            else:
                st.warning("Please enter valid text(not just blank lines).")
        else:
            st.warning("Please type something first.")
 
# Tab2 : Cosine Similarity
with tab2:
    st.header("Cosine Similarity")
    st.write("Enter sentences below (one per line) to compare similarity.")

    user_input = st.text_area("Input Sentences:", height=200, key="sim_input")

    if st.button("Calculate Similarity"):
        if user_input:
            # Clean the input
            sentence_list = [s.strip() for s in user_input.split("\n") if s.strip()]

            if len(sentence_list) < 2:
                st.warning("Please enter at least 2 different sentences to compare.")
            else:
                # Calculate and Show answer
                df_matrix = compute_similarity(sentence_list, model)
                st.success("Calculation Complete!")

                st.write("Values range from **0.0** (Different) to **1.0** (Exact Match).")
                # Display the Colourful Table
                st.subheader("1.Detailed Data Table")
                st.dataframe(df_matrix.style.background_gradient(cmap="Greens", axis=None))

                st.subheader("2.Visual Heatmap")
                st.write("Darker colors means the sentences are more similar.")
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_matrix, annot=True, cmap="Greens", ax=ax, vmin=0, vmax=1)
                # Display the plot
                st.pyplot(fig)

        else:
            st.warning("Please enter some sentences first.")