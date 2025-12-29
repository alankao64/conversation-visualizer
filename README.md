# 🎙️ Podcast Conversation Flow Visualizer

A powerful tool that visualizes how topics flow through long-form podcast conversations. Transform hours of dialogue into clear, visual topic timelines.

## 🎯 What It Does

Takes a podcast transcript and creates visual "topic maps" showing:
- How topics emerge and evolve over time
- When conversations switch between subjects
- Where topics return after tangents
- The overall conversational journey

**Perfect for**: Navigating 3-hour Joe Rogan episodes, finding specific topics in long interviews, understanding conversation structure, or analyzing discussion patterns.

## 🖼️ Example Output

The tool generates three types of visualizations:

1. **Timeline View**: Horizontal bars showing topics over time with return connections
2. **River Diagram**: Flowing stream chart of topic presence
3. **Interactive HTML**: Clickable timeline with hover details

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/conversation-visualizer.git
cd conversation-visualizer

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Basic Usage

```bash
# Run with default settings
python main.py data/your_transcript.txt

# Custom parameters
python main.py data/your_transcript.txt \
    --chunk-size 10 \
    --nr-topics 15 \
    --similarity-threshold 0.8 \
    --output-dir output
```

## 📊 How It Works

### Pipeline Overview

```
Transcript → Preprocessing → Topic Modeling → Similarity Detection → Visualization
```

1. **Preprocessing**: Cleans transcript, removes annotations, chunks into segments
2. **Topic Modeling**: Uses BERTopic (transformer-based) to identify topics
3. **Similarity Detection**: Calculates semantic similarity to find topic returns
4. **Visualization**: Generates timeline and river diagrams

### Technical Details

- **Topic Detection**: BERTopic with `all-MiniLM-L6-v2` embeddings
- **Similarity**: Cosine similarity between segment embeddings
- **Chunking**: Configurable sentence grouping (default: 7 sentences)
- **Visualization**: Matplotlib (static) + Plotly (interactive)

## 📁 Project Structure

```
conversation-visualizer/
├── data/                           # Input transcripts
│   └── jre_2219_trump.txt         # Example transcript
├── src/
│   ├── preprocessing.py            # Text cleaning and chunking
│   ├── topic_modeling.py           # BERTopic wrapper
│   ├── similarity.py               # Topic return detection
│   └── visualize.py                # Visualization generation
├── output/                         # Generated visualizations
│   ├── timeline_YYYYMMDD_HHMMSS.png
│   ├── river_diagram_YYYYMMDD_HHMMSS.png
│   ├── interactive_YYYYMMDD_HHMMSS.html
│   └── conversation_data_YYYYMMDD_HHMMSS.json
├── main.py                         # Main pipeline script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `transcript` | (required) | Path to transcript text file |
| `--output-dir` | `output` | Directory for output files |
| `--chunk-size` | `7` | Sentences per segment |
| `--nr-topics` | `auto` | Number of topics (or 'auto') |
| `--min-topic-size` | `5` | Minimum cluster size for topics |
| `--similarity-threshold` | `0.75` | Threshold for topic returns (0-1) |
| `--no-returns` | `False` | Hide topic return connections |
| `--no-save-data` | `False` | Don't save intermediate JSON |

### Tuning Parameters

**For shorter podcasts (<1 hour):**
```bash
python main.py transcript.txt --chunk-size 5 --min-topic-size 3
```

**For very long podcasts (3+ hours):**
```bash
python main.py transcript.txt --chunk-size 10 --min-topic-size 8
```

**For more granular topics:**
```bash
python main.py transcript.txt --nr-topics 20 --min-topic-size 3
```

**For stricter topic returns:**
```bash
python main.py transcript.txt --similarity-threshold 0.85
```

## 📝 Input Format

### Transcript Requirements

The tool accepts plain text transcripts. It automatically handles:
- Speaker labels (`Speaker 1:`, `JOE:`, etc.)
- Timestamps (`00:15:30`, `1:45:20`)
- Annotations (`[LAUGHTER]`, `[MUSIC]`, `[APPLAUSE]`)

### Supported Formats

```
# Simple format
This is the conversation text.
It can span multiple lines.
No special formatting needed.

# With speaker labels
Speaker 1: Welcome to the show.
Speaker 2: Thanks for having me.

# With timestamps
[00:01:30] Let's talk about AI.
[00:02:15] That's a fascinating topic.
```

## 🎨 Customization

### Modifying Visualizations

Edit `src/visualize.py` to customize:
- Color schemes
- Figure sizes
- Font styles
- Layout options

### Adjusting Topic Labels

BERTopic auto-generates labels. To improve them:

1. Increase `ngram_range` in `topic_modeling.py`
2. Use custom topic labels post-processing
3. Integrate with GPT for better descriptions (future feature)

## 🧪 Example Workflow

```python
# In Python script or notebook
from src.preprocessing import TranscriptPreprocessor
from src.topic_modeling import TopicModeler
from src.similarity import SimilarityDetector
from src.visualize import ConversationVisualizer

# Load and process
preprocessor = TranscriptPreprocessor(chunk_size=7)
chunks = preprocessor.process("data/transcript.txt")

# Detect topics
texts = [c['text'] for c in chunks]
modeler = TopicModeler()
topics, probs = modeler.fit_transform(texts)
chunks = modeler.assign_topics_to_chunks(chunks, topics, probs)

# Find returns
embeddings = modeler.get_embeddings()
detector = SimilarityDetector(threshold=0.75)
chunks = detector.detect_topic_returns(embeddings, chunks)

# Visualize
viz = ConversationVisualizer()
viz.create_timeline_matplotlib(chunks, "output/timeline.png")
```

## 📦 Dependencies

Core libraries:
- `bertopic` - Topic modeling
- `sentence-transformers` - Embeddings
- `scikit-learn` - ML utilities
- `matplotlib` - Static visualizations
- `plotly` - Interactive visualizations
- `nltk` - Text processing

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### Common Issues

**"No module named 'bertopic'"**
```bash
pip install bertopic
```

**"NLTK punkt not found"**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**"Cannot save plotly PNG"**
```bash
pip install kaleido
```

**Too many small topics**
- Increase `--min-topic-size`
- Reduce `--nr-topics`
- Increase `--chunk-size`

**Topics not coherent**
- Adjust `--chunk-size` (try 5-10)
- Increase `--min-topic-size`
- Try manual topic reduction

## 🔮 Future Enhancements

- [ ] Speaker differentiation (dual-thread visualization)
- [ ] Sentiment overlay
- [ ] GPT-powered topic label improvement
- [ ] Time-based scaling (duration on topic)
- [ ] Audio timestamp integration
- [ ] Real-time visualization updates
- [ ] Web interface
- [ ] Batch processing multiple episodes

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [BERTopic](https://github.com/MaartenGr/BERTopic) by Maarten Grootendorst
- [Sentence Transformers](https://www.sbert.net/) by UKP Lab
- [Plotly](https://plotly.com/) for interactive visualizations

## 📧 Contact

Questions? Issues? Suggestions?
- Open an issue on GitHub
- Submit a pull request
- Contact: [your-email@example.com]

---

**Made with ❤️ for podcast enthusiasts and conversation analysts**
