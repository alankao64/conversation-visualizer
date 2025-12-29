#!/usr/bin/env python3
"""
Main pipeline for Podcast Conversation Flow Visualizer.

Orchestrates the complete pipeline:
1. Load and preprocess transcript
2. Detect topics using BERTopic
3. Calculate semantic similarity for topic returns
4. Generate visualizations
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

from src.preprocessing import TranscriptPreprocessor
from src.topic_modeling import TopicModeler
from src.similarity import SimilarityDetector
from src.visualize import ConversationVisualizer


def main(
    transcript_path: str,
    output_dir: str = "output",
    chunk_size: int = 7,
    nr_topics: any = "auto",
    min_topic_size: int = 5,
    similarity_threshold: float = 0.75,
    show_returns: bool = True,
    save_intermediate: bool = True
):
    """
    Run the complete conversation visualization pipeline.

    Args:
        transcript_path: Path to transcript text file
        output_dir: Directory for output files
        chunk_size: Number of sentences per chunk
        nr_topics: Number of topics ('auto' or int)
        min_topic_size: Minimum topic size for BERTopic
        similarity_threshold: Threshold for detecting topic returns
        show_returns: Whether to visualize topic returns
        save_intermediate: Whether to save intermediate JSON data
    """
    print("=" * 80)
    print("PODCAST CONVERSATION FLOW VISUALIZER")
    print("=" * 80)
    print(f"Transcript: {transcript_path}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size} sentences")
    print(f"Topics: {nr_topics}")
    print(f"Min topic size: {min_topic_size}")
    print(f"Similarity threshold: {similarity_threshold}")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================================
    # STEP 1: Preprocessing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: PREPROCESSING")
    print("=" * 80)

    preprocessor = TranscriptPreprocessor(chunk_size=chunk_size)
    chunks = preprocessor.process(transcript_path)

    print(f"\nPreprocessing complete:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average words per chunk: {sum(len(c['text'].split()) for c in chunks) / len(chunks):.1f}")
    print()

    # =========================================================================
    # STEP 2: Topic Modeling
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: TOPIC MODELING")
    print("=" * 80)

    # Extract texts for modeling
    texts = [chunk['text'] for chunk in chunks]

    modeler = TopicModeler(
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=True
    )

    topics, probabilities = modeler.fit_transform(texts)

    # Get topic information
    topic_info = modeler.get_topic_info()
    print("\nTopic Information:")
    print(topic_info)

    # Assign topics to chunks
    chunks = modeler.assign_topics_to_chunks(chunks, topics, probabilities)

    print(f"\nTopic modeling complete:")
    print(f"  Unique topics found: {len(set(topics))}")
    print(f"  Topics: {sorted(set(topics))}")
    print()

    # =========================================================================
    # STEP 3: Similarity Detection
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: SIMILARITY DETECTION")
    print("=" * 80)

    # Get embeddings from topic model
    embeddings = modeler.get_embeddings()

    detector = SimilarityDetector(
        threshold=similarity_threshold,
        min_distance=3
    )

    chunks = detector.detect_topic_returns(embeddings, chunks, verbose=True)

    # Calculate return strength
    chunks = detector.calculate_return_strength(chunks)

    # Get return pairs
    return_pairs = detector.get_return_pairs(chunks)

    print(f"\nSimilarity detection complete:")
    print(f"  Total return connections: {len(return_pairs)}")
    print()

    # =========================================================================
    # STEP 4: Visualization
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: VISUALIZATION")
    print("=" * 80)

    visualizer = ConversationVisualizer(figsize=(24, 14))

    # Generate summary statistics
    stats = visualizer.generate_summary_stats(chunks)
    print("\nConversation Statistics:")
    for key, value in stats.items():
        if key != 'topic_distribution':
            print(f"  {key}: {value}")

    # Create visualizations
    timeline_path = os.path.join(output_dir, f"timeline_{timestamp}.png")
    river_path = os.path.join(output_dir, f"river_diagram_{timestamp}.png")
    plotly_path = os.path.join(output_dir, f"interactive_{timestamp}.html")

    print("\nGenerating visualizations...")

    # Matplotlib timeline
    visualizer.create_timeline_matplotlib(
        chunks,
        timeline_path,
        show_returns=show_returns
    )

    # River diagram
    visualizer.create_river_diagram(chunks, river_path)

    # Plotly interactive (if available)
    try:
        visualizer.create_timeline_plotly(chunks, plotly_path, show_returns=show_returns)
    except Exception as e:
        print(f"Warning: Could not create Plotly visualization: {e}")

    # =========================================================================
    # STEP 5: Save Data
    # =========================================================================
    if save_intermediate:
        print("\n" + "=" * 80)
        print("STEP 5: SAVING INTERMEDIATE DATA")
        print("=" * 80)

        data_path = os.path.join(output_dir, f"conversation_data_{timestamp}.json")

        # Prepare data for JSON (convert numpy types)
        json_chunks = []
        for chunk in chunks:
            json_chunk = chunk.copy()
            # Convert any numpy types to Python types
            for key, value in json_chunk.items():
                if hasattr(value, 'item'):  # numpy scalar
                    json_chunk[key] = value.item()
                elif isinstance(value, list) and value and hasattr(value[0], 'item'):
                    json_chunk[key] = [v.item() for v in value]
            json_chunks.append(json_chunk)

        output_data = {
            'metadata': {
                'transcript_path': transcript_path,
                'timestamp': timestamp,
                'chunk_size': chunk_size,
                'nr_topics': nr_topics,
                'min_topic_size': min_topic_size,
                'similarity_threshold': similarity_threshold
            },
            'statistics': stats,
            'chunks': json_chunks,
            'topic_info': topic_info.to_dict()
        }

        with open(data_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved conversation data to: {data_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Timeline: {timeline_path}")
    print(f"  River diagram: {river_path}")
    if os.path.exists(plotly_path):
        print(f"  Interactive: {plotly_path}")
    if save_intermediate:
        print(f"  Data: {data_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize podcast conversation flow and topic dynamics"
    )

    parser.add_argument(
        "transcript",
        type=str,
        help="Path to transcript text file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=7,
        help="Number of sentences per chunk (default: 7)"
    )

    parser.add_argument(
        "--nr-topics",
        type=str,
        default="auto",
        help="Number of topics or 'auto' (default: auto)"
    )

    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=5,
        help="Minimum topic size (default: 5)"
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for topic returns (default: 0.75)"
    )

    parser.add_argument(
        "--no-returns",
        action="store_true",
        help="Don't show topic return connections"
    )

    parser.add_argument(
        "--no-save-data",
        action="store_true",
        help="Don't save intermediate JSON data"
    )

    args = parser.parse_args()

    # Convert nr_topics to int if not 'auto'
    nr_topics = args.nr_topics
    if nr_topics != "auto":
        nr_topics = int(nr_topics)

    main(
        transcript_path=args.transcript,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        nr_topics=nr_topics,
        min_topic_size=args.min_topic_size,
        similarity_threshold=args.similarity_threshold,
        show_returns=not args.no_returns,
        save_intermediate=not args.no_save_data
    )
