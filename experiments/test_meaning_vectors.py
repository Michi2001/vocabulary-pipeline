"""
Test script for computing meaning vectors using BERT embeddings.

Reads Spanish example sentences with marked words in square brackets,
computes contextualized embeddings for each marked word, and finds
the most and least similar meanings based on cosine similarity.
"""

import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import random


# Load Spanish BERT model
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
CSV_PATH = "experiments/example_sentences.csv"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()


def extract_marked_word(sentence: str):
    """Extract word in square brackets and return word and sentence without brackets."""
    match = re.search(r"\[([^\]]+)\]", sentence)
    if not match:
        raise ValueError(f"No marked word found in: {sentence}")

    marked_word = match.group(1)
    # Remove brackets but keep the word
    clean_sentence = sentence.replace("[", "").replace("]", "")

    return marked_word, clean_sentence


def get_word_embedding(sentence: str, marked_word: str):
    """
    Compute average token embedding for the marked word in context.

    Returns the mean of all token embeddings that belong to the marked word.
    """
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)

    # Get BERT outputs
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # Shape: (1, seq_len, hidden_dim)

    # Tokenize the marked word separately to find its tokens
    word_tokens = tokenizer.tokenize(marked_word.lower())

    # Find positions of the marked word in the tokenized sentence
    sentence_tokens = tokenizer.tokenize(sentence.lower())

    # Find all occurrences of the word token sequence
    word_positions = []
    word_len = len(word_tokens)
    for i in range(len(sentence_tokens) - word_len + 1):
        if sentence_tokens[i : i + word_len] == word_tokens:
            # +1 because of [CLS] token at position 0
            word_positions.extend(range(i + 1, i + 1 + word_len))
            break

    if not word_positions:
        raise ValueError(f"Could not find word '{marked_word}' in tokenized sentence")

    # Extract embeddings for the marked word tokens and average them
    word_embeddings = hidden_states[
        0, word_positions, :
    ]  # Shape: (num_tokens, hidden_dim)
    avg_embedding = torch.mean(word_embeddings, dim=0)  # Shape: (hidden_dim,)

    return avg_embedding.numpy()


if __name__ == "__main__":
    # Read CSV file
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} sentences\n")

    # Extract marked words and compute embeddings
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

    print(f"\nSuccessfully processed {len(meanings)} sentences")

    embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings)

    # For each meaning, find top 5 most similar and least similar
    print("\n" + "=" * 80)
    print("SIMILARITY ANALYSIS")
    print("=" * 80)

    # Select random sample of meanings to display (for brevity)
    sample_indices = random.sample(range(len(meanings)), k=min(10, len(meanings)))

    for i in sample_indices:
        meaning = meanings[i]
        print(f"\n[{i}] {meaning['sentence']}")
        print(f"    Marked word: '{meaning['marked_word']}'")

        # Get similarities for this meaning (excluding itself)
        similarities = similarity_matrix[i].copy()

        # Get top 10 most similar
        most_similar_indices = np.argsort(similarities)[-10:][::-1]

        print("\n  TOP 10 MOST SIMILAR:")
        for rank, idx in enumerate(most_similar_indices, 1):
            sim_score = similarities[idx]
            print(
                f"    {rank}. [{idx}] (sim: {sim_score:.4f}) {meanings[idx]['sentence']}"
            )

        # Get top 5 least similar
        least_similar_indices = np.argsort(similarities)[:5]

        print("\n  TOP 5 LEAST SIMILAR:")
        for rank, idx in enumerate(least_similar_indices, 1):
            sim_score = similarities[idx]
            print(
                f"    {rank}. [{idx}] (sim: {sim_score:.4f}) {meanings[idx]['sentence']}"
            )

        print("\n" + "-" * 80)
