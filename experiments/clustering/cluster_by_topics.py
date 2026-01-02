"""Graph-based clustering of words using topic overlaps."""

import json
import random
import networkx as nx
from collections import defaultdict

# === CONFIGURATION ===
TRANSLATIONS_FILE = "output/spanish_lemma_frequencies/20251229_185353/translations.json"
MIN_OVERLAP = 3  # Minimum shared topics to create an edge
MIN_CLUSTER_SIZE = 3  # Minimum words in a cluster to display
MAX_CLUSTER_SIZE = 5  # Maximum words in a cluster (splits larger clusters)
TOP_CLUSTERS = 20  # Number of top clusters to show
RESOLUTION = 1.0  # Louvain resolution (higher = more smaller clusters)

# Filtering options (set to None to disable)
SAMPLE_SIZE = 50  # Random sample size (e.g., 100) or None for all
SLICE_RANGE = None  # Tuple like (0, 500) for entries 0-499, or None for all
POS_FILTER = None  # List like ["VERB", "NOUN"] or None for all POS
# =====================


def load_translations(filepath: str) -> list[dict]:
    """Load translations.json file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_translations(
    translations: list[dict],
    slice_range: tuple[int, int] | None,
    sample_size: int | None,
    pos_filter: list[str] | None,
) -> list[dict]:
    """Filter translations by slice, sample, or POS."""
    result = translations

    # Apply slice first (before sampling)
    if slice_range is not None:
        start, end = slice_range
        result = result[start:end]

    # Filter by POS
    if pos_filter is not None:
        result = [t for t in result if t["pos"] in pos_filter]

    # Random sample
    if sample_size is not None and sample_size < len(result):
        result = random.sample(result, sample_size)

    return result


def extract_word_data(translations: list[dict]) -> tuple[dict, dict]:
    """Extract topics and labels for each word."""
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

            # Get primary translation
            primary_translation = lemma
            if entry.get("meanings") and entry["meanings"][0].get("translations"):
                primary_translation = entry["meanings"][0]["translations"][0]["text"]
            word_labels[key] = primary_translation

    return word_topics, word_labels


def build_graph(word_topics: dict, min_overlap: int) -> nx.Graph:
    """Build a graph where edges connect words with shared topics."""
    G = nx.Graph()

    # Add all words as nodes
    for word in word_topics:
        G.add_node(word)

    # Add edges for overlapping topics
    words = list(word_topics.keys())
    for i, word1 in enumerate(words):
        for word2 in words[i + 1 :]:
            common_topics = word_topics[word1] & word_topics[word2]
            if len(common_topics) >= min_overlap:
                G.add_edge(
                    word1, word2, weight=len(common_topics), topics=common_topics
                )

    return G


def find_clusters(
    G: nx.Graph, resolution: float, max_size: int | None
) -> dict[int, list[str]]:
    """Find clusters using Louvain community detection, splitting large clusters."""
    # Only cluster connected components
    if G.number_of_edges() == 0:
        return {}

    communities = nx.community.louvain_communities(
        G, weight="weight", resolution=resolution
    )

    clusters = {}
    cluster_id = 0

    for community in communities:
        community_list = list(community)

        # Split large clusters if max_size is set
        if max_size is not None and len(community_list) > max_size:
            split_clusters = split_cluster(G, community_list, max_size)
            for sub_cluster in split_clusters:
                clusters[cluster_id] = sub_cluster
                cluster_id += 1
        else:
            clusters[cluster_id] = community_list
            cluster_id += 1

    return clusters


def split_cluster(G: nx.Graph, words: list[str], max_size: int) -> list[list[str]]:
    """Split a large cluster into smaller ones by finding sub-communities."""
    if len(words) <= max_size:
        return [words]

    # Create subgraph for this cluster
    subgraph = G.subgraph(words).copy()

    if subgraph.number_of_edges() == 0:
        # No edges - just split arbitrarily
        return [words[i : i + max_size] for i in range(0, len(words), max_size)]

    # Try to find sub-communities with higher resolution
    sub_communities = nx.community.louvain_communities(
        subgraph, weight="weight", resolution=2.0
    )

    result = []
    for sub_comm in sub_communities:
        sub_list = list(sub_comm)
        if len(sub_list) > max_size:
            # Still too large - split recursively or arbitrarily
            result.extend(
                [sub_list[i : i + max_size] for i in range(0, len(sub_list), max_size)]
            )
        else:
            result.append(sub_list)

    return result


def get_cluster_topics(cluster_words: list[str], word_topics: dict) -> dict[str, int]:
    """Get common topics in a cluster with their frequency."""
    topic_counts = defaultdict(int)

    for word in cluster_words:
        for topic in word_topics.get(word, []):
            topic_counts[topic] += 1

    return dict(topic_counts)


def print_clusters(
    clusters: dict,
    word_topics: dict,
    word_labels: dict,
    min_size: int,
    top_n: int,
):
    """Print clusters with their words and common topics."""
    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    # Filter by minimum size
    filtered_clusters = [
        (cid, words) for cid, words in sorted_clusters if len(words) >= min_size
    ]

    print(f"\n{'='*70}")
    print(f"📊 Gefundene Cluster: {len(filtered_clusters)} (mit >= {min_size} Wörtern)")
    print(f"{'='*70}")

    for i, (cluster_id, words) in enumerate(filtered_clusters[:top_n]):
        # Get topic frequencies for this cluster
        topic_counts = get_cluster_topics(words, word_topics)

        # Find topics that appear in most words of the cluster
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        top_topics = [t for t, c in sorted_topics[:8] if c >= len(words) * 0.3]

        print(f"\n🔷 Cluster {i+1} ({len(words)} Wörter)")
        print(f"   Haupt-Topics: {', '.join(top_topics)}")
        print(f"   {'-'*60}")

        # Print words with their labels
        for word in sorted(words, key=lambda w: word_labels.get(w, w)):
            label = word_labels.get(word, word)
            print(f"   • {label} ({word})")

    if len(filtered_clusters) > top_n:
        print(f"\n   ... und {len(filtered_clusters) - top_n} weitere Cluster")


def print_graph_stats(G: nx.Graph):
    """Print statistics about the graph."""
    print(f"\n📈 Graph-Statistiken:")
    print(f"   Knoten (Wörter): {G.number_of_nodes()}")
    print(f"   Kanten (Verbindungen): {G.number_of_edges()}")

    if G.number_of_edges() > 0:
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        print(
            f"   Durchschnittliche Overlap-Stärke: {sum(weights)/len(weights):.1f} Topics"
        )
        print(f"   Max Overlap: {max(weights)} Topics")

    # Connected components
    components = list(nx.connected_components(G))
    print(f"   Zusammenhängende Komponenten: {len(components)}")

    isolated = sum(1 for c in components if len(c) == 1)
    print(f"   Isolierte Wörter: {isolated}")


def main():
    # Load data
    print(f"Loading translations from {TRANSLATIONS_FILE}...")
    translations = load_translations(TRANSLATIONS_FILE)
    print(f"Loaded {len(translations)} entries")

    # Filter translations
    filtered = filter_translations(translations, SLICE_RANGE, SAMPLE_SIZE, POS_FILTER)
    print(f"Filtered to {len(filtered)} entries")

    # Extract word data
    word_topics, word_labels = extract_word_data(filtered)
    print(f"Extracted topics for {len(word_topics)} words")

    # Build graph
    print(f"\nBuilding graph with min_overlap={MIN_OVERLAP}...")
    G = build_graph(word_topics, MIN_OVERLAP)
    print_graph_stats(G)

    # Find clusters
    print(
        f"\nFinding clusters (resolution={RESOLUTION}, max_size={MAX_CLUSTER_SIZE})..."
    )
    clusters = find_clusters(G, RESOLUTION, MAX_CLUSTER_SIZE)
    print(f"Found {len(clusters)} clusters")

    # Print results
    print_clusters(clusters, word_topics, word_labels, MIN_CLUSTER_SIZE, TOP_CLUSTERS)

    # Summary
    total_clustered = sum(
        len(words) for words in clusters.values() if len(words) >= MIN_CLUSTER_SIZE
    )
    print(f"\n{'='*70}")
    print(f"📊 Zusammenfassung:")
    print(f"   {total_clustered} Wörter in Clustern mit >= {MIN_CLUSTER_SIZE} Wörtern")
    print(
        f"   {len(word_topics) - total_clustered} Wörter nicht oder in kleinen Clustern"
    )


if __name__ == "__main__":
    main()
