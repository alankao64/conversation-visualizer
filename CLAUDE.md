# 🤖 Guide for AI Assistants

Welcome! You're working on the **Podcast Conversation Flow Visualizer** - a tool that uses AI to visualize how topics flow through long-form podcast conversations.

## 🎯 Project Overview

**What it does:** Takes a 3-hour podcast transcript → Identifies topics using BERTopic → Detects when topics return → Creates visual timeline showing conversational flow

**Tech stack:** Python, BERTopic, Sentence Transformers, UMAP, HDBSCAN, Matplotlib, Plotly

**Status:** ✅ Production ready (v1.0 - full pipeline implemented)

---

## 📂 Codebase Structure

```
src/
├── preprocessing.py      # Load, clean, chunk transcripts
├── topic_modeling.py     # BERTopic wrapper for topic discovery
├── similarity.py         # Semantic similarity & topic return detection
└── visualize.py          # Timeline, river, and interactive visualizations

main.py                   # CLI orchestration script
scripts/                  # Utility scripts (transcript download, etc.)
data/                     # Input transcripts
output/                   # Generated visualizations
```

### Key Files to Know

- **main.py** - Entry point, orchestrates entire pipeline
- **src/preprocessing.py** - Handles transcript cleaning (removes [LAUGHTER], timestamps, speaker labels)
- **src/topic_modeling.py** - BERTopic integration, uses `all-MiniLM-L6-v2` embeddings
- **src/similarity.py** - Calculates cosine similarity to find topic returns
- **src/visualize.py** - Creates 3 types of visualizations (timeline, river, interactive)
- **MEMORY.md** - 📝 **Project memory - CHECK THIS FIRST!**

---

## 🚀 Common Tasks

### Running the Pipeline

```bash
# Basic usage
python main.py data/transcript.txt

# With custom parameters
python main.py data/transcript.txt \
  --chunk-size 7 \
  --nr-topics 15 \
  --similarity-threshold 0.75 \
  --output-dir output
```

### Testing Changes

```bash
# Quick test with sample data
python main.py data/sample_transcript.txt --chunk-size 5 --min-topic-size 2

# Test individual modules
python src/preprocessing.py
python src/topic_modeling.py
python src/similarity.py
python src/visualize.py
```

**⚡ Fast Testing for Bug Fixes (Skip Full Pipeline)**

When fixing bugs in isolated code sections, avoid reinstalling dependencies each session:

```bash
# For logic/syntax fixes: Just review the code
# - Read the file and verify logic
# - Check similar patterns in codebase
# - No need to run if confident in fix

# For import/syntax errors: Use Python's parser
python -m py_compile src/topic_modeling.py  # Validates syntax only

# For testing individual functions: Import in Python REPL
python -c "from src.topic_modeling import TopicModeler; print('OK')"
```

**When to skip full testing:**
- ✅ Index/offset fixes (like probabilities[i][topic_id] → probabilities[i][topic_id])
- ✅ Type conversions (int(), float(), str())
- ✅ Conditional logic fixes (if/else corrections)
- ✅ String formatting changes
- ✅ Variable renaming

**When you MUST test:**
- ❌ Algorithm changes
- ❌ New dependencies
- ❌ External API/model calls
- ❌ File I/O changes
- ❌ User-facing behavior changes

**Pro tip:** User can test locally after PR merge. Focus on correct logic in code review.

### Adding Dependencies

```bash
# Add to requirements.txt
echo "new-package>=1.0.0" >> requirements.txt

# Install
pip install -r requirements.txt
```

---

## 🎨 Code Conventions

- **Python 3.9+** required
- **Type hints** for function signatures
- **Docstrings** for all classes and public methods
- **Clear variable names** (prefer `chunk_size` over `cs`)
- **Modular design** - each file has single responsibility
- **Progress indicators** using `tqdm` or print statements
- **Error handling** with informative messages

---

## 🐛 Known Gotchas

1. **Hugging Face downloads** - First run downloads ~500MB of models (requires internet)
2. **Small datasets** - Need at least 15-20 chunks for meaningful topic modeling
3. **Memory usage** - Large transcripts (10k+ chunks) can use significant RAM
4. **Topic -1** - BERTopic creates outlier topic (-1) for segments that don't fit clusters
5. **Similarity threshold** - Default 0.75 works well; too low = noise, too high = misses returns

---

## 📝 CRITICAL: Update MEMORY.md

**🚨 Before ending your session, update MEMORY.md if you:**

- ✅ Completed a feature or significant work
- ✅ Made architectural decisions
- ✅ Solved a tricky bug
- ✅ Created or merged a PR
- ✅ Learned something important about the codebase
- ✅ Changed dependencies or configuration

**How to update:**

1. **Always update** "🎯 Current State" section
2. Add entry to "📅 Session Log" (use template at bottom of MEMORY.md)
3. Document decisions in "💬 Key Discussions & Intent"
4. Add bugs/solutions to "🐛 Known Issues & Fixes"
5. Archive old sessions if getting long (keep last ~10)

**Quick check:** If you're about to commit/PR, MEMORY.md should reflect your changes!

---

## 🔄 Git Workflow

```bash
# Check current branch
git status

# Develop on feature branch (usually starts with claude/)
git checkout -b claude/feature-name-SESSION_ID

# Commit with descriptive message
git add -A
git commit -m "Feature: Add X functionality

- Implemented Y
- Fixed Z
- Updated docs"

# Push to remote
git push -u origin claude/feature-name-SESSION_ID
```

**Branch naming:** `claude/<description>-<session-id>` (session ID critical for permissions)

---

## 🧪 Quality Checklist

Before completing work:

- [ ] Code works as expected
- [ ] No syntax errors or warnings
- [ ] Dependencies added to requirements.txt
- [ ] Docstrings added/updated
- [ ] README updated if user-facing changes
- [ ] **MEMORY.md updated** ⚠️
- [ ] Changes committed with clear message
- [ ] Pushed to correct branch

---

## 💡 Development Tips

### When Adding Features

1. Check MEMORY.md for relevant past discussions
2. Look at existing code patterns (preprocessing.py is a good example)
3. Add progress indicators for long operations
4. Test with sample_transcript.txt first
5. Update docstrings and type hints
6. Document decisions in MEMORY.md

### When Debugging

1. Check "🐛 Known Issues" in MEMORY.md first
2. Enable verbose mode in modules (most have `verbose` parameter)
3. Check intermediate outputs (JSON files in output/)
4. Common issues: network access, small datasets, memory limits

### When Reviewing Code

1. Read MEMORY.md for context on why things were built certain ways
2. Check for consistent style with existing code
3. Verify type hints and docstrings present
4. Test with both small and large inputs

---

## 🗺️ Future Roadmap

See "🗺️ Roadmap Notes" in MEMORY.md for discussed features:

- Speaker differentiation (dual-thread view)
- Sentiment analysis overlay
- GPT-powered topic labeling
- Web interface
- Real-time processing
- Audio timestamp integration

---

## 📚 Useful References

- **BERTopic docs**: https://maartengr.github.io/BERTopic/
- **Sentence Transformers**: https://www.sbert.net/
- **UMAP**: https://umap-learn.readthedocs.io/
- **HDBSCAN**: https://hdbscan.readthedocs.io/

---

## 🆘 Getting Help

1. **Check MEMORY.md** - Past decisions and solutions documented there
2. **Check README.md** - User-facing documentation
3. **Check PROJECT_STATUS.md** - Implementation details and status
4. **Check docstrings** - Most modules have detailed documentation
5. **Check example code** - Each module has `if __name__ == "__main__"` examples

---

## 🤝 Working with This Codebase

**Philosophy:** This project values clarity over cleverness. Code should be readable, well-documented, and maintainable. When in doubt, choose the simpler approach.

**Style:** Follow existing patterns. Look at how preprocessing.py structures classes and methods - that's the template.

**Documentation:** If you spent time figuring something out, document it in MEMORY.md so the next AI (or human) doesn't have to.

**Testing:** Always test with sample_transcript.txt before committing. It's fast and catches most issues.

---

## 🎯 Your Mission

Help improve this tool while maintaining code quality and documentation. When you make changes, think about the next person (AI or human) who will work on this code. Leave it better than you found it.

**Remember:** Update MEMORY.md before you commit! 📝

---

**Happy coding! 🚀**
