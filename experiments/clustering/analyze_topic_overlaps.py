"""Analyze topic overlaps between words for clustering evaluation."""

import json
import random

# === CONFIGURATION ===
TRANSLATIONS_FILE = "output/spanish_lemma_frequencies/20251229_185353/translations.json"
NUM_SAMPLES = 5  # Number of words to sample
TOP_N = 10  # Show top N overlaps per word
SPECIFIC_WORDS = (
    None  # Set to list like ["vender|VERB", "coche|NOUN"] or None for random
)
SEED = 42  # Random seed for reproducibility (None for random)
MIN_OVERLAP = 1  # Minimum overlap count to show
# =====================


def load_translations(filepath: str) -> list[dict]:
    """Load translations.json file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_word_topics(
    translations: list[dict],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """
    Extract all topics for each word (lemma|pos).

    Combines topics from all meanings into one set per word.
    Returns tuple of (word_topics, word_labels) where labels contain primary translation.
    """
    word_topics = {}
    word_labels = {}

    for entry in translations:
        lemma = entry["lemma"]
        pos = entry["pos"]
        key = f"{lemma}|{pos}"

        # Collect all topics from all meanings
        all_topics = set()
        for meaning in entry.get("meanings", []):
            topics = meaning.get("topics", [])
            all_topics.update(topics)

        if all_topics:
            word_topics[key] = all_topics

            # Get primary translation (first translation of first meaning)
            primary_translation = lemma  # fallback
            if entry.get("meanings") and entry["meanings"][0].get("translations"):
                primary_translation = entry["meanings"][0]["translations"][0]["text"]

            word_labels[key] = primary_translation

    return word_topics, word_labels


def find_overlaps(
    word_topics: dict[str, set[str]],
    word_labels: dict[str, str],
    sample_words: list[str],
) -> dict:
    """
    Find topic overlaps between sampled words and all other words.

    Returns dict mapping each sampled word to its overlaps with other words.
    """
    results = {}

    for word in sample_words:
        if word not in word_topics:
            continue

        word_topic_set = word_topics[word]
        overlaps = []

        for other_word, other_topics in word_topics.items():
            if other_word == word:
                continue

            # Find intersection
            common_topics = word_topic_set & other_topics

            if common_topics:
                overlaps.append(
                    {
                        "word": other_word,
                        "label": word_labels.get(other_word, other_word),
                        "common_topics": sorted(common_topics),
                        "overlap_count": len(common_topics),
                    }
                )

        # Sort by overlap count descending
        overlaps.sort(key=lambda x: x["overlap_count"], reverse=True)

        results[word] = {
            "label": word_labels.get(word, word),
            "topics": sorted(word_topic_set),
            "topic_count": len(word_topic_set),
            "overlaps": overlaps,
        }

    return results


def print_results(results: dict, top_n: int = 10):
    """Print overlap results in a readable format."""
    for word, data in results.items():
        print(f"\n{'='*60}")
        print(f"📝 {data['label']} ({word})")
        print(f"   Topics ({data['topic_count']}): {', '.join(data['topics'])}")
        print(f"\n   🔗 Top {top_n} Überschneidungen:")
        print(f"   {'-'*50}")

        for overlap in data["overlaps"][:top_n]:
            label = overlap["label"]
            other_word = overlap["word"]
            count = overlap["overlap_count"]
            common = ", ".join(overlap["common_topics"])
            print(f"   • {label} - {other_word} ({count} topics)")
            print(f"     → {common}")

        if len(data["overlaps"]) > top_n:
            print(
                f"\n   ... und {len(data['overlaps']) - top_n} weitere Wörter mit Überschneidungen"
            )


def main():
    # Set seed if provided
    if SEED is not None:
        random.seed(SEED)

    # Load data
    print(f"Loading translations from {TRANSLATIONS_FILE}...")
    translations = load_translations(TRANSLATIONS_FILE)
    print(f"Loaded {len(translations)} entries")

    # Extract topics
    word_topics, word_labels = extract_word_topics(translations)
    print(f"Extracted topics for {len(word_topics)} words")

    # Determine which words to analyze
    if SPECIFIC_WORDS:
        sample_words = SPECIFIC_WORDS
    else:
        available_words = list(word_topics.keys())
        sample_words = random.sample(
            available_words, min(NUM_SAMPLES, len(available_words))
        )

    print(f"\nAnalyzing {len(sample_words)} words...")

    # Find overlaps
    results = find_overlaps(word_topics, word_labels, sample_words)

    # Filter by minimum overlap
    if MIN_OVERLAP > 1:
        for word in results:
            results[word]["overlaps"] = [
                o
                for o in results[word]["overlaps"]
                if o["overlap_count"] >= MIN_OVERLAP
            ]

    # Print results
    print_results(results, top_n=TOP_N)

    # Summary statistics
    print(f"\n{'='*60}")
    print("📊 Zusammenfassung:")
    for word, data in results.items():
        avg_overlap = (
            sum(o["overlap_count"] for o in data["overlaps"]) / len(data["overlaps"])
            if data["overlaps"]
            else 0
        )
        print(
            f"   {word}: {len(data['overlaps'])} Wörter mit Überschneidung, ∅ {avg_overlap:.1f} Topics"
        )


if __name__ == "__main__":
    main()
