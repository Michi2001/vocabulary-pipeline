"""
Word Sense Disambiguation using BERT contextual embeddings.

Uses cosine similarity between contextualized BERT embeddings to match
a target word in a sentence with its correct meaning from example sentences.
"""

import torch
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Global variables: Example sentences for different word meanings
# Target word must be enclosed in square brackets [word]

# Example 1: banco (bank/bench/shoal)
EXAMPLES = [
    {
        "target": "Trajo dinero al [banco].",
        "meanings": [
            "Me senté en el [banco] del parque.",  # bench
            "Deposité mi dinero en el [banco].",  # bank (financial)
            "El pescador se acercó al [banco] de peces.",  # shoal
        ],
    },
    # Example 2: llama (flame/call/llama animal)
    {
        "target": "Vi una [llama] en el zoológico.",
        "meanings": [
            "La [llama] del fuego era muy alta.",  # flame
            "Hice una [llama] telefónica urgente.",  # phone call
            "La [llama] es un animal andino.",  # llama animal
        ],
    },
    # Example 3: vela (candle/sail/vigil)
    {
        "target": "El barco desplegó la [vela] blanca.",
        "meanings": [
            "Encendí una [vela] aromática.",  # candle
            "El velero tiene una [vela] grande.",  # sail
            "Hicimos una [vela] por el difunto.",  # vigil
        ],
    },
    # Example 4: cabo (end/corporal/cape)
    {
        "target": "El [cabo] dirigió a los soldados.",
        "meanings": [
            "Al [cabo] de tres días regresó.",  # at the end
            "El [cabo] tiene dos años de servicio.",  # military corporal
            "Visitamos el [cabo] de Hornos.",  # geographic cape
        ],
    },
    # Example 5: sierra (saw/mountain range)
    {
        "target": "Subimos a la [sierra] de Guadarrama.",
        "meanings": [
            "Corté la madera con una [sierra].",  # saw tool
            "La [sierra] estaba cubierta de nieve.",  # mountain range
        ],
    },
]

# Use first example as default
TARGET_SENTENCE = EXAMPLES[0]["target"]
MEANING_SENTENCES = EXAMPLES[0]["meanings"]


def load_bert_model(model_name="dccuchile/bert-base-spanish-wwm-uncased"):
    """Load Spanish BERT/RoBERTa model and tokenizer.

    Supports both BERT and RoBERTa models via AutoModel.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def extract_word_and_sentence(sentence):
    """Extract target word in brackets and clean sentence."""
    match = re.search(r"\[([^\]]+)\]", sentence)
    if not match:
        raise ValueError(f"No word in brackets found in: {sentence}")

    target_word = match.group(1)
    clean_sentence = sentence.replace("[", "").replace("]", "")

    return target_word, clean_sentence


def get_word_embedding(sentence, target_word, tokenizer, model):
    """Get BERT embedding for target word in sentence context.

    Uses average of last 4 layers for better semantic representation.
    """
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Get BERT embeddings from all hidden layers
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states: tuple of (num_layers + 1) tensors
        # Each tensor: [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states

        # Use last 4 layers (indices -4, -3, -2, -1)
        last_four_layers = torch.stack(
            hidden_states[-4:]
        )  # [4, batch_size, seq_len, hidden_dim]

        # Average across the 4 layers
        embeddings = last_four_layers.mean(dim=0)[0]  # [seq_len, hidden_dim]

    # Find target word token positions
    target_word_lower = target_word.lower()
    target_positions = []

    for i, token in enumerate(tokens):
        # Handle BERT subword tokens (##)
        clean_token = token.replace("##", "").lower()
        if clean_token in target_word_lower or target_word_lower in clean_token:
            target_positions.append(i)

    if not target_positions:
        raise ValueError(f"Target word '{target_word}' not found in tokens: {tokens}")

    # Average embeddings of all subword tokens
    word_embedding = embeddings[target_positions].mean(dim=0)

    return word_embedding.numpy()


def disambiguate(target_sentence, meaning_sentences, tokenizer, model):
    """Find best matching meaning sentence using cosine similarity."""
    # Extract target word and clean sentence
    target_word, clean_target = extract_word_and_sentence(target_sentence)

    # Get embedding for target word in context
    target_embedding = get_word_embedding(clean_target, target_word, tokenizer, model)

    # Get embeddings for each meaning
    meaning_embeddings = []
    meaning_words = []
    clean_meanings = []

    for meaning_sent in meaning_sentences:
        meaning_word, clean_meaning = extract_word_and_sentence(meaning_sent)
        meaning_words.append(meaning_word)
        clean_meanings.append(clean_meaning)

        meaning_emb = get_word_embedding(clean_meaning, meaning_word, tokenizer, model)
        meaning_embeddings.append(meaning_emb)

    # Calculate cosine similarities
    similarities = []
    for meaning_emb in meaning_embeddings:
        sim = cosine_similarity(
            target_embedding.reshape(1, -1), meaning_emb.reshape(1, -1)
        )[0][0]
        similarities.append(sim)

    # Find highest similarity (most similar meaning)
    best_idx = np.argmax(similarities)

    return {
        "target_sentence": target_sentence,
        "target_word": target_word,
        "best_match_idx": best_idx,
        "best_match_sentence": meaning_sentences[best_idx],
        "similarities": similarities,
        "meaning_sentences": meaning_sentences,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BERT Word Sense Disambiguation")
    print("=" * 60)

    # Load model once
    print("\nLade BERT Modell...")
    # Alternative Modelle:
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"  # BETO uncased
    # "dccuchile/bert-base-spanish-wwm-cased" - BETO cased (empfohlen)
    # model_name = "bert-base-multilingual-cased"  # - mBERT
    # "xlm-roberta-base" - XLM-RoBERTa
    # "bertin-project/bertin-roberta-base-spanish" - BERTIN
    # model_name = "dccuchile/bert-base-spanish-wwm-cased"
    tokenizer, model = load_bert_model(model_name=model_name)

    # Run all examples
    for example_idx, example in enumerate(EXAMPLES, 1):
        print(f"\n{'='*60}")
        print(f"BEISPIEL {example_idx}")
        print("=" * 60)

        target_sent = example["target"]
        meaning_sents = example["meanings"]

        print(f"\nZiel-Satz: {target_sent}")
        print("\nBedeutungs-Sätze:")
        for i, sent in enumerate(meaning_sents, 1):
            print(f"  {i}. {sent}")

        result = disambiguate(target_sent, meaning_sents, tokenizer, model)

        # Display results
        print("\nERGEBNIS:")
        print(f"Ziel-Wort: '{result['target_word']}'")
        print(f"Beste Übereinstimmung: Meaning {result['best_match_idx'] + 1}")
        print(f"→ {result['best_match_sentence']}")

        print("\nCosine Similarities:")
        for i, (sent, sim) in enumerate(
            zip(result["meaning_sentences"], result["similarities"]), 1
        ):
            marker = " ← BESTE" if i - 1 == result["best_match_idx"] else ""
            print(f"  {i}. {sim:.4f} - {sent}{marker}")
