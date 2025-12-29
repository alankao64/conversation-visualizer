"""
Similarity detection module.
Identifies when topics return by calculating semantic similarity between segments.
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class SimilarityDetector:
    """Detects topic returns using semantic similarity."""

    def __init__(self, threshold: float = 0.75, min_distance: int = 3):
        """
        Initialize similarity detector.

        Args:
            threshold: Similarity threshold for detecting topic returns (0-1)
            min_distance: Minimum segment distance to consider (avoid adjacent segments)
        """
        self.threshold = threshold
        self.min_distance = min_distance

    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise cosine similarity between all embeddings.

        Args:
            embeddings: Array of segment embeddings (n_segments x embedding_dim)

        Returns:
            Similarity matrix (n_segments x n_segments)
        """
        print("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

    def detect_topic_returns(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Detect when topics return by finding high-similarity segments.

        Args:
            embeddings: Segment embeddings
            chunks: List of chunk dictionaries
            verbose: Whether to show progress

        Returns:
            Updated chunks with similarity information
        """
        n_segments = len(embeddings)
        similarity_matrix = self.calculate_similarity_matrix(embeddings)

        # For each segment, find previous segments with high similarity
        for i in tqdm(range(n_segments), disable=not verbose, desc="Detecting topic returns"):
            similar_to = []
            similarity_scores = []

            # Only look at previous segments (j < i)
            # Skip segments too close (within min_distance)
            for j in range(max(0, i - n_segments), i - self.min_distance):
                sim_score = similarity_matrix[i, j]

                if sim_score >= self.threshold:
                    similar_to.append(j)
                    similarity_scores.append(float(sim_score))

            # Sort by similarity score (highest first)
            if similar_to:
                sorted_pairs = sorted(
                    zip(similar_to, similarity_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                similar_to = [pair[0] for pair in sorted_pairs]
                similarity_scores = [pair[1] for pair in sorted_pairs]

            chunks[i]['similar_to'] = similar_to
            chunks[i]['similarity_scores'] = similarity_scores
            chunks[i]['has_returns'] = len(similar_to) > 0

        # Calculate statistics
        total_returns = sum(1 for chunk in chunks if chunk['has_returns'])
        if verbose:
            print(f"\nTopic return statistics:")
            print(f"  Segments with returns: {total_returns}/{n_segments} ({total_returns/n_segments*100:.1f}%)")
            print(f"  Threshold used: {self.threshold}")

        return chunks

    def get_return_pairs(self, chunks: List[Dict]) -> List[Tuple[int, int, float]]:
        """
        Get all topic return pairs (from_segment, to_segment, similarity).

        Args:
            chunks: List of chunks with similarity information

        Returns:
            List of (from_id, to_id, similarity_score) tuples
        """
        pairs = []

        for chunk in chunks:
            to_id = chunk['segment_id']
            if chunk['has_returns']:
                for from_id, score in zip(chunk['similar_to'], chunk['similarity_scores']):
                    pairs.append((from_id, to_id, score))

        return pairs

    def group_returns_by_topic(
        self,
        chunks: List[Dict],
        return_pairs: List[Tuple[int, int, float]]
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        """
        Group topic returns by topic ID.

        Args:
            chunks: List of chunks
            return_pairs: List of return pairs

        Returns:
            Dictionary mapping topic_id to list of return pairs
        """
        topic_returns = {}

        for from_id, to_id, score in return_pairs:
            # Get topic of the target segment
            topic_id = chunks[to_id]['topic_id']

            if topic_id not in topic_returns:
                topic_returns[topic_id] = []

            topic_returns[topic_id].append((from_id, to_id, score))

        return topic_returns

    def calculate_return_strength(self, chunks: List[Dict]) -> List[Dict]:
        """
        Calculate a 'return strength' metric for each segment.

        Args:
            chunks: List of chunks

        Returns:
            Updated chunks with return_strength field
        """
        for chunk in chunks:
            if chunk['has_returns']:
                # Average of top 3 similarity scores (or all if fewer)
                top_scores = chunk['similarity_scores'][:3]
                chunk['return_strength'] = np.mean(top_scores) if top_scores else 0.0
            else:
                chunk['return_strength'] = 0.0

        return chunks


if __name__ == "__main__":
    # Example usage with dummy embeddings
    np.random.seed(42)

    # Create 10 dummy embeddings (5D for simplicity)
    # Segments 0, 5 similar (topic A)
    # Segments 2, 7 similar (topic B)
    embeddings = np.random.randn(10, 5)
    embeddings[5] = embeddings[0] + np.random.randn(5) * 0.1  # Make similar
    embeddings[7] = embeddings[2] + np.random.randn(5) * 0.1  # Make similar

    # Create dummy chunks
    chunks = [{'segment_id': i, 'text': f"Segment {i}"} for i in range(10)]

    # Detect returns
    detector = SimilarityDetector(threshold=0.7, min_distance=2)
    chunks = detector.detect_topic_returns(embeddings, chunks)

    # Show results
    print("\nSegments with topic returns:")
    for chunk in chunks:
        if chunk['has_returns']:
            print(f"  Segment {chunk['segment_id']}: similar to {chunk['similar_to']} "
                  f"(scores: {[f'{s:.2f}' for s in chunk['similarity_scores']]})")
