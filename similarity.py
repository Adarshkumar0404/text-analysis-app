from sentence_transformers import util
import pandas as pd

def compute_similarity(texts, model):
    # Encode all texts at once
    embeddings = model.encode(texts)
    
    # Calculate Cosine Similarity
    matrix = util.cos_sim(embeddings, embeddings)

    # Convert to a Pandas DataFrame for easy reading
    df_matrix = pd.DataFrame(matrix.numpy(), index=texts, columns=texts)

    return df_matrix