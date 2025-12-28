"""Test clustering of meaning vectors."""

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re


MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
CSV_PATH = "experiments/cluster_test_sentences.csv"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()


def extract_marked_word(sentence: str):
    """Extract word in square brackets and return word and sentence without brackets."""
    match = re.search(r"\[([^\]]+)\]", sentence)
    if not match:
        raise ValueError(f"No marked word found in: {sentence}")
    marked_word = match.group(1)
    clean_sentence = sentence.replace("[", "").replace("]", "")
    return marked_word, clean_sentence


def get_word_embedding(sentence: str, marked_word: str):
    """Compute average token embedding for the marked word in context."""
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state

    word_tokens = tokenizer.tokenize(marked_word.lower())
    sentence_tokens = tokenizer.tokenize(sentence.lower())

    word_positions = []
    word_len = len(word_tokens)
    for i in range(len(sentence_tokens) - word_len + 1):
        if sentence_tokens[i : i + word_len] == word_tokens:
            word_positions.extend(range(i + 1, i + 1 + word_len))
            break

    if not word_positions:
        raise ValueError(f"Could not find word '{marked_word}' in tokenized sentence")

    word_embeddings = hidden_states[0, word_positions, :]
    avg_embedding = torch.mean(word_embeddings, dim=0)

    return avg_embedding.numpy()


if __name__ == "__main__":
    # Read CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} sentences\n")

    # Compute embeddings
    meanings = []
    embeddings = []

    print("Computing embeddings...")
    for idx, row in df.iterrows():
        sentence = row["sentence"]
        marked_word, clean_sentence = extract_marked_word(sentence)
        embedding = get_word_embedding(clean_sentence, marked_word)

        meanings.append(
            {
                "index": idx,
                "sentence": sentence,
                "marked_word": marked_word,
                "clean_sentence": clean_sentence,
            }
        )
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    # Cluster into 10 groups
    n_clusters = 10
    print(f"\nClustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # Display clusters
    print("\n" + "=" * 80)
    print("CLUSTERING RESULTS")
    print("=" * 80)

    for cluster_id in range(n_clusters):
        indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        print(f"\nCLUSTER {cluster_id} ({len(indices)} sentences):")
        for idx in indices:
            print(f"  [{idx}] {meanings[idx]['sentence']}")
