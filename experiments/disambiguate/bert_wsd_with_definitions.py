"""
Word Sense Disambiguation using BERT with definitions instead of example sentences.

Uses contextualized BERT embeddings of the target word compared against
embeddings of the same word in definition contexts.
"""

import torch
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Examples with definitions instead of example sentences
EXAMPLES = [
    {
        "target": "Trajo dinero al [banco].",
        "meanings": [
            "Asiento público para varias personas: [banco]",  # bench
            "Institución financiera donde se guarda dinero: [banco]",  # bank
            "Conjunto de peces nadando juntos: [banco]",  # shoal
        ],
    },
    {
        "target": "Vi una [llama] en el zoológico.",
        "meanings": [
            "Masa gaseosa en combustión que desprende luz: [llama]",  # flame
            "Acción de comunicarse por teléfono: [llama]",  # call
            "Animal rumiante de los Andes: [llama]",  # llama animal
        ],
    },
    {
        "target": "El barco desplegó la [vela] blanca.",
        "meanings": [
            "Cilindro de cera con mecha que da luz: [vela]",  # candle
            "Pieza de tela que impulsa una embarcación: [vela]",  # sail
            "Acto de permanecer despierto cuidando un difunto: [vela]",  # vigil
        ],
    },
    {
        "target": "El [cabo] dirigió a los soldados.",
        "meanings": [
            "Extremo o final de algo: [cabo]",  # end
            "Militar de rango inferior al sargento: [cabo]",  # corporal
            "Punta de tierra que penetra en el mar: [cabo]",  # cape
        ],
    },
    {
        "target": "Subimos a la [sierra] de Guadarrama.",
        "meanings": [
            "Herramienta para cortar madera con dientes: [sierra]",  # saw
            "Cordillera de montañas con forma de dientes: [sierra]",  # mountain range
        ],
    },
]


def load_bert_model(model_name="dccuchile/bert-base-spanish-wwm-uncased"):
    """Load Spanish BERT/RoBERTa model and tokenizer."""
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
        hidden_states = outputs.hidden_states

        # Use last 4 layers (indices -4, -3, -2, -1)
        last_four_layers = torch.stack(hidden_states[-4:])

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


def disambiguate(target_sentence, meaning_definitions, tokenizer, model):
    """Find best matching meaning using definition-based embeddings."""
    # Extract target word and clean sentence
    target_word, clean_target = extract_word_and_sentence(target_sentence)

    # Get embedding for target word in context
    target_embedding = get_word_embedding(clean_target, target_word, tokenizer, model)

    # Get embeddings for each meaning definition
    meaning_embeddings = []
    meaning_words = []
    clean_definitions = []

    for definition in meaning_definitions:
        def_word, clean_def = extract_word_and_sentence(definition)
        meaning_words.append(def_word)
        clean_definitions.append(clean_def)

        meaning_emb = get_word_embedding(clean_def, def_word, tokenizer, model)
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
        "best_match_definition": meaning_definitions[best_idx],
        "similarities": similarities,
        "meaning_definitions": meaning_definitions,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BERT WSD with Definitions")
    print("=" * 60)

    # Load model once
    print("\nLade BERT Modell...")
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer, model = load_bert_model(model_name=model_name)
    print(f"Modell: {model_name}")

    # Run all examples
    for example_idx, example in enumerate(EXAMPLES, 1):
        print(f"\n{'='*60}")
        print(f"BEISPIEL {example_idx}")
        print("=" * 60)

        target_sent = example["target"]
        meaning_defs = example["meanings"]

        print(f"\nZiel-Satz: {target_sent}")
        print("\nDefinitionen:")
        for i, def_sent in enumerate(meaning_defs, 1):
            # Show clean version without brackets
            clean_def = def_sent.replace("[", "").replace("]", "")
            print(f"  {i}. {clean_def}")

        result = disambiguate(target_sent, meaning_defs, tokenizer, model)

        # Display results
        print("\nERGEBNIS:")
        print(f"Ziel-Wort: '{result['target_word']}'")
        print(f"Beste Übereinstimmung: Meaning {result['best_match_idx'] + 1}")
        clean_best = result["best_match_definition"].replace("[", "").replace("]", "")
        print(f"→ {clean_best}")

        print("\nCosine Similarities:")
        for i, (def_sent, sim) in enumerate(
            zip(result["meaning_definitions"], result["similarities"]), 1
        ):
            marker = " ← BESTE" if i - 1 == result["best_match_idx"] else ""
            clean_def = def_sent.replace("[", "").replace("]", "")
            print(f"  {i}. {sim:.4f} - {clean_def}{marker}")
