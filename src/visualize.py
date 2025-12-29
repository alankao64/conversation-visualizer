"""
Visualization module for podcast conversation flow.
Creates timeline visualizations showing topic flow over time.
"""

from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict


class ConversationVisualizer:
    """Creates visualizations of conversation topic flow."""

    def __init__(self, figsize: Tuple[int, int] = (20, 12)):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size for matplotlib (width, height)
        """
        self.figsize = figsize
        self.color_map = {}

    def _rgb_string_to_hex(self, rgb_string: str) -> str:
        """
        Convert Plotly RGB string format to hex color for matplotlib.

        Args:
            rgb_string: Color in format 'rgb(r,g,b)' or hex '#RRGGBB'

        Returns:
            Hex color string '#RRGGBB'
        """
        # If already hex, return as-is
        if rgb_string.startswith('#'):
            return rgb_string

        # Parse 'rgb(r,g,b)' format
        if rgb_string.startswith('rgb('):
            rgb_values = rgb_string[4:-1].split(',')
            r, g, b = [int(v.strip()) for v in rgb_values]
            return f'#{r:02x}{g:02x}{b:02x}'

        # Return as-is if unrecognized format
        return rgb_string

    def _get_topic_colors(self, chunks: List[Dict]) -> Dict[int, str]:
        """
        Generate a color map for topics.

        Args:
            chunks: List of chunks with topic assignments

        Returns:
            Dictionary mapping topic_id to color hex code
        """
        # Get unique topics
        unique_topics = sorted(set(chunk['topic_id'] for chunk in chunks))

        # Use a qualitative color palette
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Set2

        # Assign colors (outlier topic -1 gets gray)
        color_map = {}
        color_idx = 0

        for topic_id in unique_topics:
            if topic_id == -1:
                color_map[topic_id] = '#CCCCCC'  # Gray for outliers
            else:
                # Convert Plotly RGB string to hex for matplotlib compatibility
                plotly_color = colors[color_idx % len(colors)]
                color_map[topic_id] = self._rgb_string_to_hex(plotly_color)
                color_idx += 1

        return color_map

    def create_timeline_matplotlib(
        self,
        chunks: List[Dict],
        output_path: str,
        show_returns: bool = True,
        return_threshold: float = 0.8
    ):
        """
        Create timeline visualization using matplotlib.

        Args:
            chunks: List of chunks with topic and similarity information
            output_path: Path to save output image
            show_returns: Whether to show topic return connections
            return_threshold: Minimum similarity to show return connection
        """
        print("Creating matplotlib timeline visualization...")

        # Get color map
        color_map = self._get_topic_colors(chunks)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        n_segments = len(chunks)

        # Plot each segment as a colored bar
        for i, chunk in enumerate(chunks):
            topic_id = chunk['topic_id']
            color = color_map[topic_id]

            # Draw bar for this segment
            ax.barh(
                y=topic_id,
                width=1,
                left=i,
                height=0.8,
                color=color,
                edgecolor='white',
                linewidth=0.5
            )

        # Draw topic return connections if requested
        if show_returns:
            for chunk in chunks:
                if chunk.get('has_returns', False):
                    to_idx = chunk['segment_id']
                    to_topic = chunk['topic_id']

                    for from_idx, sim_score in zip(chunk['similar_to'], chunk['similarity_scores']):
                        if sim_score >= return_threshold:
                            # Draw arc connecting segments
                            from_topic = chunks[from_idx]['topic_id']

                            # Draw a curved line
                            x = [from_idx + 0.5, to_idx + 0.5]
                            y = [from_topic, to_topic]

                            # Alpha based on similarity strength
                            alpha = min(0.6, (sim_score - return_threshold) / (1 - return_threshold))

                            ax.plot(
                                x, y,
                                color='black',
                                alpha=alpha,
                                linewidth=1.5,
                                linestyle='--'
                            )

        # Get unique topics and their labels
        topic_labels = {}
        for chunk in chunks:
            topic_id = chunk['topic_id']
            if topic_id not in topic_labels:
                topic_labels[topic_id] = chunk['topic_label']

        # Set y-axis labels
        sorted_topics = sorted(topic_labels.keys())
        ax.set_yticks(sorted_topics)
        ax.set_yticklabels([topic_labels[tid] for tid in sorted_topics])

        # Labels and title
        ax.set_xlabel('Segment Index (Time Flow →)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Topics', fontsize=14, fontweight='bold')
        ax.set_title(
            'Podcast Conversation Flow - Topic Timeline',
            fontsize=18,
            fontweight='bold',
            pad=20
        )

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)

        # Set x-axis limits
        ax.set_xlim(-0.5, n_segments + 0.5)

        # Tight layout
        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved matplotlib visualization to: {output_path}")

        plt.close()

    def create_timeline_plotly(
        self,
        chunks: List[Dict],
        output_path: str,
        show_returns: bool = True
    ):
        """
        Create interactive timeline visualization using Plotly.

        Args:
            chunks: List of chunks with topic and similarity information
            output_path: Path to save output HTML/PNG
            show_returns: Whether to show topic return connections
        """
        print("Creating Plotly timeline visualization...")

        # Get color map
        color_map = self._get_topic_colors(chunks)

        fig = go.Figure()

        # Group chunks by topic to create continuous bands
        topic_segments = defaultdict(list)
        for chunk in chunks:
            topic_id = chunk['topic_id']
            segment_id = chunk['segment_id']
            topic_segments[topic_id].append(segment_id)

        # Create timeline bars for each topic
        for topic_id, segment_ids in topic_segments.items():
            # Get topic label
            topic_label = chunks[segment_ids[0]]['topic_label']
            color = color_map[topic_id]

            # Create segments
            for seg_id in segment_ids:
                chunk = chunks[seg_id]

                # Hover text
                hover_text = (
                    f"<b>{topic_label}</b><br>"
                    f"Segment: {seg_id}<br>"
                    f"Text: {chunk['text'][:100]}..."
                )

                fig.add_trace(go.Bar(
                    x=[1],
                    y=[topic_label],
                    orientation='h',
                    name=topic_label,
                    marker=dict(color=color),
                    showlegend=False,
                    hovertext=hover_text,
                    hoverinfo='text',
                    base=seg_id,
                    width=0.8
                ))

        # Add topic return connections
        if show_returns:
            for chunk in chunks:
                if chunk.get('has_returns', False):
                    to_idx = chunk['segment_id']
                    to_topic = chunk['topic_label']

                    for from_idx, sim_score in zip(chunk['similar_to'], chunk['similarity_scores']):
                        from_chunk = chunks[from_idx]
                        from_topic = from_chunk['topic_label']

                        # Add line
                        fig.add_trace(go.Scatter(
                            x=[from_idx + 0.5, to_idx + 0.5],
                            y=[from_topic, to_topic],
                            mode='lines',
                            line=dict(
                                color='rgba(0,0,0,0.3)',
                                width=2,
                                dash='dash'
                            ),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Podcast Conversation Flow - Topic Timeline',
                'font': {'size': 24, 'family': 'Arial, bold'}
            },
            xaxis_title='Segment Index (Time Flow →)',
            yaxis_title='Topics',
            barmode='stack',
            height=800,
            width=1600,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Save as HTML (interactive)
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"Saved interactive Plotly visualization to: {html_path}")

        # Save as static PNG
        try:
            fig.write_image(output_path, width=1600, height=800)
            print(f"Saved static Plotly visualization to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save static image: {e}")
            print("Install kaleido for static image export: pip install kaleido")

    def create_river_diagram(
        self,
        chunks: List[Dict],
        output_path: str
    ):
        """
        Create a river/stream diagram showing topic flow.

        Args:
            chunks: List of chunks with topic information
            output_path: Path to save output image
        """
        print("Creating river diagram...")

        # Get color map
        color_map = self._get_topic_colors(chunks)

        # Count topic occurrences per segment position
        n_segments = len(chunks)
        unique_topics = sorted(set(chunk['topic_id'] for chunk in chunks))

        # Create data matrix: segments x topics
        topic_matrix = np.zeros((n_segments, len(unique_topics)))

        topic_to_idx = {topic_id: idx for idx, topic_id in enumerate(unique_topics)}

        for chunk in chunks:
            seg_id = chunk['segment_id']
            topic_id = chunk['topic_id']
            topic_idx = topic_to_idx[topic_id]
            topic_matrix[seg_id, topic_idx] = 1

        # Create stacked area chart
        fig, ax = plt.subplots(figsize=self.figsize)

        # X coordinates
        x = np.arange(n_segments)

        # Cumulative y values for stacking
        y_cumulative = np.zeros(n_segments)

        # Plot each topic as a stream
        for topic_idx, topic_id in enumerate(unique_topics):
            y_values = topic_matrix[:, topic_idx]

            # Get label
            topic_label = next(
                chunk['topic_label'] for chunk in chunks if chunk['topic_id'] == topic_id
            )

            # Plot filled area
            ax.fill_between(
                x,
                y_cumulative,
                y_cumulative + y_values,
                label=topic_label,
                color=color_map[topic_id],
                alpha=0.8,
                edgecolor='white',
                linewidth=1
            )

            y_cumulative += y_values

        # Labels and title
        ax.set_xlabel('Segment Index (Time Flow →)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Topic Presence', fontsize=14, fontweight='bold')
        ax.set_title(
            'Podcast Conversation Flow - River Diagram',
            fontsize=18,
            fontweight='bold',
            pad=20
        )

        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)

        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved river diagram to: {output_path}")

        plt.close()

    def generate_summary_stats(self, chunks: List[Dict]) -> Dict:
        """
        Generate summary statistics about the conversation.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with summary statistics
        """
        # Topic distribution
        topic_counts = defaultdict(int)
        for chunk in chunks:
            topic_counts[chunk['topic_id']] += 1

        # Return statistics
        total_returns = sum(1 for chunk in chunks if chunk.get('has_returns', False))

        # Topic transitions
        transitions = 0
        for i in range(1, len(chunks)):
            if chunks[i]['topic_id'] != chunks[i-1]['topic_id']:
                transitions += 1

        stats = {
            'total_segments': len(chunks),
            'unique_topics': len(topic_counts),
            'segments_with_returns': total_returns,
            'return_percentage': (total_returns / len(chunks) * 100) if chunks else 0,
            'topic_transitions': transitions,
            'topic_distribution': dict(topic_counts)
        }

        return stats


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)

    # Create dummy chunks
    chunks = []
    topics = [0, 0, 1, 1, 2, 0, 1, 2, 2, 0]  # Some topic returns

    for i, topic_id in enumerate(topics):
        chunks.append({
            'segment_id': i,
            'text': f'Sample text for segment {i}',
            'topic_id': topic_id,
            'topic_label': f'Topic {topic_id}',
            'similar_to': [],
            'similarity_scores': [],
            'has_returns': False
        })

    # Add some returns
    chunks[5]['similar_to'] = [0, 1]
    chunks[5]['similarity_scores'] = [0.85, 0.82]
    chunks[5]['has_returns'] = True

    chunks[9]['similar_to'] = [0, 5]
    chunks[9]['similarity_scores'] = [0.88, 0.81]
    chunks[9]['has_returns'] = True

    # Create visualizer
    viz = ConversationVisualizer()

    # Generate visualizations
    viz.create_timeline_matplotlib(chunks, '/tmp/test_timeline.png')
    viz.create_river_diagram(chunks, '/tmp/test_river.png')

    # Print stats
    stats = viz.generate_summary_stats(chunks)
    print("\nSummary statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
