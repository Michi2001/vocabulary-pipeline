"""
Train a Word2Vec model on Spanish corpus data.

Uses the same corpus files as the main frequency pipeline.
"""

from pathlib import Path
from gensim.models import Word2Vec
import spacy

# === CONFIGURATION ===
CORPUS_PATH = "data/Leipzig_Wortschatz/spa_news_2023_1M/spa_news_2023_1M-sentences.txt"
MODEL_OUTPUT_PATH = "models/spanish_word2vec.model"

# Word2Vec hyperparameters
VECTOR_SIZE = 100  # Embedding dimension
WINDOW_SIZE = 5  # Context window
MIN_COUNT = 5  # Minimum word frequency
WORKERS = 4  # Parallel workers
EPOCHS = 5  # Training epochs
MAX_SENTENCES = None  # Limit sentences (None = all)


def read_sentences(corpus_path: str, max_sentences: int | None = None):
    """Read and tokenize sentences from Leipzig corpus format."""
    print(f"Loading spaCy model for tokenization...")
    nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])

    print(f"Reading corpus from {corpus_path}")
    sentences = []
    count = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                text = parts[1]
                # Tokenize and lemmatize
                doc = nlp(text)
                tokens = [
                    token.lemma_.lower()
                    for token in doc
                    if not token.is_punct and not token.is_space
                ]
                if tokens:
                    sentences.append(tokens)
                    count += 1

                    if count % 50000 == 0:
                        print(f"  Processed {count} sentences...")

                    if max_sentences and count >= max_sentences:
                        break

    print(f"Total sentences: {len(sentences)}")
    return sentences


def train_model(sentences: list[list[str]]) -> Word2Vec:
    """Train Word2Vec model on tokenized sentences."""
    print(f"\nTraining Word2Vec model...")
    print(f"  vector_size={VECTOR_SIZE}")
    print(f"  window={WINDOW_SIZE}")
    print(f"  min_count={MIN_COUNT}")
    print(f"  epochs={EPOCHS}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
        sg=1,  # Skip-gram (better for small datasets)
    )

    print(f"Vocabulary size: {len(model.wv)}")
    return model


def main():
    # Read and tokenize sentences
    sentences = read_sentences(CORPUS_PATH, MAX_SENTENCES)

    # Train model
    model = train_model(sentences)

    # Save model
    Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")

    # Quick test
    print("\nSample predictions for 'creer':")
    if "creer" in model.wv:
        predictions = model.predict_output_word(["creer"], topn=10)
        for word, prob in predictions:
            print(f"  {word}: {prob:.4f}")
    else:
        print("  'creer' not in vocabulary")


if __name__ == "__main__":
    main()
