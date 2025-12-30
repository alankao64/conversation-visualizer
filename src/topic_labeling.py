"""
Topic labeling module using LLM for semantic topic names.
Replaces keyword-based labels with human-readable semantic labels.
"""

from typing import List, Dict, Optional
import anthropic
from collections import defaultdict


class TopicLabeler:
    """Generates semantic topic labels using Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-20241022",
        max_chunks_per_topic: int = 5,
        label_style: str = "short",
        verbose: bool = True
    ):
        """
        Initialize topic labeler.

        Args:
            api_key: Anthropic API key
            model: Claude model to use (haiku is cheap and fast for labeling)
            max_chunks_per_topic: Number of representative chunks to send per topic
            label_style: Style of labels ('short', 'medium', 'detailed')
            verbose: Whether to print progress
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_chunks_per_topic = max_chunks_per_topic
        self.label_style = label_style
        self.verbose = verbose

    def _get_representative_chunks(
        self,
        topic_id: int,
        chunks: List[Dict]
    ) -> List[str]:
        """
        Get representative chunks for a topic.

        Args:
            topic_id: Topic ID
            chunks: List of all chunks

        Returns:
            List of representative chunk texts
        """
        # Get all chunks for this topic
        topic_chunks = [c for c in chunks if c['topic_id'] == topic_id]

        if not topic_chunks:
            return []

        # Sort by topic probability (if available) to get most representative
        topic_chunks_sorted = sorted(
            topic_chunks,
            key=lambda x: x.get('topic_probability', 0) or 0,
            reverse=True
        )

        # Take top N chunks
        representative = topic_chunks_sorted[:self.max_chunks_per_topic]

        return [c['text'] for c in representative]

    def _create_labeling_prompt(
        self,
        chunk_texts: List[str],
        current_label: str
    ) -> str:
        """
        Create prompt for Claude to generate topic label.

        Args:
            chunk_texts: Representative chunk texts
            current_label: Current keyword-based label

        Returns:
            Prompt string
        """
        # Build chunk examples
        chunks_str = "\n\n".join([
            f"Chunk {i+1}: {text[:300]}..."
            for i, text in enumerate(chunk_texts)
        ])

        # Style-specific instructions
        if self.label_style == "short":
            style_instruction = "Generate a concise topic label (2-5 words) that captures the main theme."
        elif self.label_style == "medium":
            style_instruction = "Generate a descriptive topic label (3-8 words) that explains what's being discussed."
        else:  # detailed
            style_instruction = "Generate a detailed topic label (5-12 words) that fully describes the conversation topic."

        prompt = f"""You are analyzing segments from a podcast conversation. Below are representative chunks from a single topic cluster.

Current keyword-based label: "{current_label}"

Representative chunks from this topic:
{chunks_str}

{style_instruction}

Examples of good labels:
- "The Assassination Attempt"
- "2020 Election Fraud Claims"
- "UFC and Combat Sports Discussion"
- "California Water Management Issues"
- "Transition from The Apprentice to Politics"

Return ONLY the topic label, nothing else. No explanations, no quotes, just the label."""

        return prompt

    def generate_label_for_topic(
        self,
        topic_id: int,
        chunks: List[Dict],
        current_label: str
    ) -> str:
        """
        Generate semantic label for a single topic using Claude.

        Args:
            topic_id: Topic ID
            chunks: List of all chunks
            current_label: Current keyword-based label

        Returns:
            Generated semantic label
        """
        # Handle outlier topic
        if topic_id == -1:
            return "Miscellaneous"

        # Get representative chunks
        representative_chunks = self._get_representative_chunks(topic_id, chunks)

        if not representative_chunks:
            return current_label  # Fallback to keyword label

        # Create prompt
        prompt = self._create_labeling_prompt(representative_chunks, current_label)

        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=50,  # Labels are short
                temperature=0.3,  # Lower temperature for more consistent labels
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract label from response
            label = response.content[0].text.strip()

            # Clean up label (remove quotes if present)
            label = label.strip('"').strip("'").strip()

            if self.verbose:
                print(f"  Topic {topic_id}: '{current_label}' → '{label}'")

            return label

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Failed to generate label for topic {topic_id}: {e}")
            return current_label  # Fallback to keyword label

    def improve_topic_labels(
        self,
        chunks: List[Dict],
        topic_model = None
    ) -> List[Dict]:
        """
        Improve all topic labels in chunks using Claude API.

        Args:
            chunks: List of chunks with topic assignments
            topic_model: Optional topic model (not used, kept for compatibility)

        Returns:
            Updated chunks with improved labels
        """
        if self.verbose:
            print("\nGenerating semantic topic labels with Claude...")

        # Get unique topics
        unique_topics = sorted(set(c['topic_id'] for c in chunks))

        if self.verbose:
            print(f"Processing {len(unique_topics)} topics...")

        # Generate new labels for each topic
        label_mapping = {}
        for topic_id in unique_topics:
            # Get current label from first chunk with this topic
            current_label = next(
                c['topic_label'] for c in chunks if c['topic_id'] == topic_id
            )

            # Generate new semantic label
            new_label = self.generate_label_for_topic(
                topic_id,
                chunks,
                current_label
            )

            label_mapping[topic_id] = new_label

        # Update all chunks with new labels
        for chunk in chunks:
            topic_id = chunk['topic_id']
            chunk['topic_label'] = label_mapping[topic_id]
            # Store original keyword label for reference
            chunk['original_keyword_label'] = chunk.get('topic_label')

        if self.verbose:
            print(f"✓ Generated {len(label_mapping)} semantic topic labels")

        return chunks

    def get_topic_summary(
        self,
        topic_id: int,
        chunks: List[Dict]
    ) -> Optional[str]:
        """
        Generate a longer summary/description for a topic (future feature).

        Args:
            topic_id: Topic ID
            chunks: List of all chunks

        Returns:
            Topic summary (1-2 sentences)
        """
        # TODO: Future enhancement - generate longer descriptions for tooltips
        # This could be used in interactive visualizations
        pass


if __name__ == "__main__":
    # Example usage
    import os

    # Sample chunks (simulating BERTopic output)
    sample_chunks = [
        {
            'segment_id': 0,
            'text': 'I was at the rally and suddenly I felt this sharp pain in my ear. It was the most surreal moment of my life.',
            'topic_id': 0,
            'topic_label': 'Beautiful, Bed, Surreal, White House, Lincoln',
            'topic_probability': 0.92
        },
        {
            'segment_id': 1,
            'text': 'The bullet just grazed my ear. I touched it and saw blood. The Secret Service immediately rushed me off stage.',
            'topic_id': 0,
            'topic_label': 'Beautiful, Bed, Surreal, White House, Lincoln',
            'topic_probability': 0.88
        },
        {
            'segment_id': 2,
            'text': 'We had massive issues with mail-in ballots. Millions of ballots went out and we don\'t know who filled them out.',
            'topic_id': 1,
            'topic_label': 'Ballots, Want, Mail, Need, Thing',
            'topic_probability': 0.85
        },
    ]

    # Initialize labeler (requires API key)
    api_key = os.getenv('ANTHROPIC_API_KEY', 'your-api-key-here')

    if api_key and api_key != 'your-api-key-here':
        labeler = TopicLabeler(api_key=api_key, verbose=True)

        # Generate improved labels
        improved_chunks = labeler.improve_topic_labels(sample_chunks)

        print("\n--- Results ---")
        for chunk in improved_chunks:
            print(f"Topic {chunk['topic_id']}: {chunk['topic_label']}")
            print(f"  Text: {chunk['text'][:100]}...")
            print()
    else:
        print("Set ANTHROPIC_API_KEY environment variable to test")
