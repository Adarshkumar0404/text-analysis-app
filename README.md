# Text Analysis & Similarity App

A Python web application built with **Streamlit** that allows users to generate vector embeddings for text and calculate cosine similarity between multiple sentences. It includes visual heatmaps and data download options.

## Features

* **Vector Generation (Tab 1):**
    * Convert single or multiple lines of text into numerical vector representations.
    * View vector dimensions in a structured table.
* **Cosine Similarity (Tab 2):**
    * Compare multiple sentences to see how similar they are (0.0 to 1.0).
    * **Visual Heatmap:** A color-coded grid (Green) showing similarity scores.
    * **Data Table:** Detailed raw numbers for precision.

## Technologies Used

* **Streamlit:** For the web interface.
* **Sentence-Transformers:** For generating text embeddings (Model: `all-MiniLM-L6-v2`).
* **Pandas:** For data handling and table display.
* **Seaborn & Matplotlib:** For generating the similarity heatmap.
