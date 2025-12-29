"""
Topic modeling module using BERTopic.
Identifies topics in podcast transcripts using transformer-based embeddings.
"""

from typing import List, Dict, Tuple
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


class TopicModeler:
    """Handles topic detection using BERTopic."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        nr_topics: any = "auto",
        min_topic_size: int = 5,
        calculate_probabilities: bool = True,
        verbose: bool = True
    ):
        """
        Initialize topic modeler.

        Args:
            embedding_model: Name of sentence-transformers model
            nr_topics: Number of topics ('auto' or integer)
            min_topic_size: Minimum size for a topic cluster
            calculate_probabilities: Whether to calculate topic probabilities
            verbose: Whether to print progress
        """
        self.embedding_model_name = embedding_model
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.calculate_probabilities = calculate_probabilities
        self.verbose = verbose

        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model)
        self.topic_model = None
        self.embeddings = None

    def fit_transform(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Fit BERTopic model and assign topics to documents.

        Args:
            documents: List of text documents (chunks)

        Returns:
            Tuple of (topics, probabilities)
        """
        if self.verbose:
            print("Generating embeddings...")

        # Generate embeddings
        self.embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=self.verbose
        )

        if self.verbose:
            print(f"Embeddings shape: {self.embeddings.shape}")
            print("Fitting BERTopic model...")

        # Use CountVectorizer with improved settings for better topic labels
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2
        )

        # Initialize and fit BERTopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            nr_topics=self.nr_topics,
            min_topic_size=self.min_topic_size,
            calculate_probabilities=self.calculate_probabilities,
            verbose=self.verbose,
            vectorizer_model=vectorizer_model
        )

        topics, probabilities = self.topic_model.fit_transform(
            documents,
            embeddings=self.embeddings
        )

        if self.verbose:
            unique_topics = set(topics)
            print(f"Found {len(unique_topics)} topics (including outliers)")
            print(f"Topic range: {min(topics)} to {max(topics)}")

        return topics, probabilities

    def get_topic_info(self) -> Dict:
        """
        Get information about discovered topics.

        Returns:
            Dictionary with topic information
        """
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet")

        topic_info = self.topic_model.get_topic_info()
        return topic_info

    def get_topic_label(self, topic_id: int, max_words: int = 5) -> str:
        """
        Get a human-readable label for a topic.

        Args:
            topic_id: Topic ID
            max_words: Maximum number of words in label

        Returns:
            Topic label string
        """
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet")

        # Handle outlier topic
        if topic_id == -1:
            return "Miscellaneous"

        # Get top words for topic
        topic_words = self.topic_model.get_topic(topic_id)
        if not topic_words:
            return f"Topic {topic_id}"

        # Create label from top words
        words = [word for word, _ in topic_words[:max_words]]
        label = ", ".join(words).title()

        return label

    def assign_topics_to_chunks(
        self,
        chunks: List[Dict],
        topics: List[int],
        probabilities: np.ndarray
    ) -> List[Dict]:
        """
        Assign topic information to chunks.

        Args:
            chunks: List of chunk dictionaries
            topics: Topic assignments
            probabilities: Topic probabilities

        Returns:
            Updated chunks with topic information
        """
        if len(chunks) != len(topics):
            raise ValueError("Number of chunks and topics must match")

        for i, chunk in enumerate(chunks):
            topic_id = topics[i]
            chunk['topic_id'] = int(topic_id)
            chunk['topic_label'] = self.get_topic_label(topic_id)

            # Add probability if available
            if probabilities is not None and len(probabilities) > 0:
                if len(probabilities.shape) == 2:
                    # Get probability for assigned topic
                    chunk['topic_probability'] = float(probabilities[i][topic_id + 1])
                else:
                    chunk['topic_probability'] = None
            else:
                chunk['topic_probability'] = None

        return chunks

    def get_embeddings(self) -> np.ndarray:
        """
        Get the embeddings generated during fitting.

        Returns:
            Numpy array of embeddings
        """
        if self.embeddings is None:
            raise ValueError("Embeddings have not been generated yet")

        return self.embeddings

    def reduce_topics(self, n_topics: int) -> None:
        """
        Reduce the number of topics after initial fitting.

        Args:
            n_topics: Target number of topics
        """
        if self.topic_model is None:
            raise ValueError("Model has not been fitted yet")

        if self.verbose:
            print(f"Reducing to {n_topics} topics...")

        self.topic_model.reduce_topics(docs=None, nr_topics=n_topics)

        if self.verbose:
            print(f"Topics reduced successfully")


if __name__ == "__main__":
    # Example usage
    sample_docs = [
        "The economy is growing and unemployment is down",
        "Stock markets reached new highs today",
        "AI and machine learning are transforming industries",
        "Neural networks can now generate realistic images",
        "Climate change is affecting weather patterns globally",
        "Renewable energy adoption is increasing worldwide"
    ]

    modeler = TopicModeler(min_topic_size=2)
    topics, probs = modeler.fit_transform(sample_docs)

    print("\nTopic assignments:")
    for i, (doc, topic) in enumerate(zip(sample_docs, topics)):
        label = modeler.get_topic_label(topic)
        print(f"{i}: [{label}] {doc}")
