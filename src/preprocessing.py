"""
Preprocessing module for podcast transcript data.
Handles loading, cleaning, and chunking of transcript text.
"""

import re
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize


class TranscriptPreprocessor:
    """Handles transcript preprocessing and chunking."""

    def __init__(self, chunk_size: int = 7):
        """
        Initialize preprocessor.

        Args:
            chunk_size: Number of sentences per chunk (default: 7)
        """
        self.chunk_size = chunk_size
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

    def load_transcript(self, file_path: str) -> str:
        """
        Load transcript from file.

        Args:
            file_path: Path to transcript text file

        Returns:
            Raw transcript text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def clean_text(self, text: str) -> str:
        """
        Clean transcript text by removing annotations and extra whitespace.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned text
        """
        # Remove common transcript annotations
        text = re.sub(r'\[LAUGHTER\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[MUSIC\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[APPLAUSE\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[CROSSTALK\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)  # Remove any other bracketed annotations

        # Remove timestamps (various formats)
        text = re.sub(r'\d{1,2}:\d{2}:\d{2}', '', text)
        text = re.sub(r'\d{1,2}:\d{2}', '', text)

        # Remove speaker labels (common formats like "Speaker 1:", "JOE:", etc.)
        text = re.sub(r'^[A-Z\s]+:\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Speaker \d+:\s*', '', text, flags=re.MULTILINE)

        # Clean up whitespace
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        text = re.sub(r' +', ' ', text)    # Multiple spaces to single
        text = text.strip()

        return text

    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into chunks of sentences.

        Args:
            text: Cleaned transcript text

        Returns:
            List of dictionaries containing chunk information
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)

        chunks = []
        chunk_id = 0

        # Group sentences into chunks
        for i in range(0, len(sentences), self.chunk_size):
            chunk_sentences = sentences[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_sentences)

            # Skip very short chunks
            if len(chunk_text.split()) < 10:
                continue

            chunks.append({
                'segment_id': chunk_id,
                'text': chunk_text,
                'sentence_start': i,
                'sentence_end': min(i + self.chunk_size, len(sentences)),
                'num_sentences': len(chunk_sentences)
            })
            chunk_id += 1

        return chunks

    def process(self, file_path: str) -> List[Dict[str, any]]:
        """
        Complete preprocessing pipeline: load, clean, and chunk.

        Args:
            file_path: Path to transcript file

        Returns:
            List of processed chunks
        """
        print(f"Loading transcript from {file_path}...")
        raw_text = self.load_transcript(file_path)
        print(f"Loaded {len(raw_text):,} characters")

        print("Cleaning text...")
        cleaned_text = self.clean_text(raw_text)
        print(f"Cleaned text: {len(cleaned_text):,} characters")

        print(f"Chunking text (chunk_size={self.chunk_size} sentences)...")
        chunks = self.chunk_text(cleaned_text)
        print(f"Created {len(chunks)} chunks")

        return chunks


if __name__ == "__main__":
    # Example usage
    preprocessor = TranscriptPreprocessor(chunk_size=7)

    # Test with a sample text
    sample_text = """
    [00:00:15] Speaker 1: Welcome to the show. Today we have an amazing guest.
    [MUSIC]
    Speaker 2: Thanks for having me. It's great to be here.
    Speaker 1: So let's talk about your new book. What inspired you to write it?
    Speaker 2: Well, I've been thinking about this topic for years. [LAUGHTER]
    It all started when I was working in Silicon Valley.
    """

    cleaned = preprocessor.clean_text(sample_text)
    print("Cleaned sample:")
    print(cleaned)
