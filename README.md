# 🔄 Self-Refinement Loop Project

A system that iteratively improves AI-generated answers using a self-refinement loop and a separate ML critic.

## 📁 Project Structure

```
refine-loop/
├── config/                    # Configuration files
│   ├── API_KEY.txt           # Gemini API key
│   └── requirements.txt      # Python dependencies
├── data/                      # Input data
│   ├── prompts.csv           # Training prompts
│   └── test_prompts.csv      # Test prompts
├── models/                    # Trained models
│   ├── improved_svm_critic_model.pkl
│   ├── improved_svm_critic_vectorizer.pkl
│   └── improved_svm_text_extractor.pkl
├── logs/                      # Output logs and results
├── scripts/                   # Main execution scripts
│   ├── run_analysis.py       # Analysis launcher
│   ├── run_demo.py           # Demo launcher
│   └── test_improved_critic.py # Critic testing
├── src/                       # Source code
│   ├── core/                  # Core functionality
│   │   ├── critics/          # Critic implementations
│   │   ├── refinement/       # Refinement logic
│   │   └── training/         # Model training
│   ├── analysis/             # Analysis tools
│   ├── utils/                # Utilities
│   └── ui/                   # User interface
└── docs/                     # Documentation
    ├── DETAILED_README.md    # Detailed documentation
    └── IMPROVEMENTS_SUMMARY.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd refine-loop

# Install dependencies
pip install -r config/requirements.txt

# Create a .env with your key (preferred)
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

### 2. Run Analysis

```bash
# Run analysis tools
python scripts/run_analysis.py

# Or run individual analyses
python src/analysis/simple_analysis.py
```

### 3. Launch Interactive Demo

```bash
# Start the Streamlit demo
python scripts/run_demo.py

# Or run directly with streamlit
streamlit run ui/demo_app.py
```

### 4. Reproduce Results

```bash
# End-to-end reproduction (generate data → train critic → eval → print metrics)
bash scripts/reproduce_results.sh
```

## 📊 Current Performance

- **Success Rate**: 60.7% (answers deemed sufficient by critic)
- **Average BLEU**: 0.64
- **Average ROUGE-L**: 0.813
- **Critic**: Improved SVM with TF‑IDF + handcrafted features; threshold 0.67

## 🔧 Key Components

### Core Functionality (`src/core/`)
- **Critics** (`critics/`): ML-powered critic implementations
- **Refinement** (`refinement/`): Self-refinement logic and batch processing
- **Training** (`training/`): Model training utilities

### Analysis Tools (`src/analysis/`)
- Performance evaluation and visualization
- Model improvement analysis
- Results analysis and metrics

### User Interface (`ui/`)
- Interactive Streamlit demo

## 📚 Documentation

For more depth, see:
- `docs/DETAILED_README.md` – Comprehensive project documentation
- `docs/IMPROVEMENTS_SUMMARY.md` – Recent improvements and next steps

## 🎯 Features

- Batch processing with checkpointing
- ML-powered critics for answer quality
- Analysis tools and visualizations
- Interactive demo for quick testing

## 📈 Performance Metrics

- Success Rate: 60.7%
- BLEU: 0.64
- ROUGE‑L: 0.813
- Improved Critic Training Accuracy: 98.8%

The project is organized for maintainability and easy navigation.
