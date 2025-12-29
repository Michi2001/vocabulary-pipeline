"""Test clustering with Sentence Transformer embeddings."""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


# MODEL_NAME = "BAAI/bge-m3"
# MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
MODEL_NAME = "sentence-transformers/LaBSE"
CSV_PATH = "experiments/cluster_test_sentence_transformer.csv"

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)


if __name__ == "__main__":
    # Read CSV (no header)
    df = pd.read_csv(CSV_PATH, header=None, names=["spanish", "german", "definition"])
    print(f"Found {len(df)} word meanings\n")

    # Compute embeddings
    print("Computing embeddings...")
    word_meanings = []
    embeddings = []

    for _, row in df.iterrows():
        spanish = row["spanish"].strip()
        german = row["german"].strip()
        definition = row["definition"].strip()

        # Combine spanish and german for multilingual embedding
        combined_text = f"{definition}"
        # Get embedding using SentenceTransformer
        embedding = model.encode(combined_text, convert_to_numpy=True)

        word_meanings.append(f"{spanish}, {german}")
        embeddings.append(embedding)

        print(f"  {spanish}, {german}")

    embeddings = np.array(embeddings)

    # Cluster
    n_clusters = 9
    print(
        f"\nClustering {len(word_meanings)} word meanings into {n_clusters} groups..."
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=44, n_init=9)
    clusters = kmeans.fit_predict(embeddings)

    # Display results
    print("\n" + "=" * 80)
    print("CLUSTERING RESULTS (SentenceTransformer Embeddings)")
    print("=" * 80)

    for cluster_id in range(n_clusters):
        indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        print(f"\nCLUSTER {cluster_id} ({len(indices)} word meanings):")
        for idx in indices:
            print(f"  {word_meanings[idx]}")
