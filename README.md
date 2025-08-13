<<<<<<< HEAD
# self-refine-loop
Self-Refine Loop
=======
# 🔄 Self-Refinement Loop Project

A sophisticated system that uses machine learning critics to iteratively improve AI-generated answers through self-reflection and refinement.

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
    ├── README.md             # Detailed documentation
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

# Add your Gemini API key
echo "your-api-key-here" > config/API_KEY.txt
```

### 2. Run Analysis

```bash
# Run analysis tools
python scripts/run_analysis.py

# Or run individual analyses
python src/analysis/simple_analysis.py
python src/analysis/improve_critic.py
```

### 3. Launch Interactive Demo

```bash
# Start the Streamlit demo
python scripts/run_demo.py

# Or run directly with streamlit
streamlit run ui/demo_app.py
```

### 4. Test Critic

```bash
# Test the improved critic
python scripts/test_improved_critic.py
```

## 📊 Current Performance

- **Success Rate**: 60.7% (answers deemed sufficient by critic)
- **Average BLEU Score**: 0.64
- **Average ROUGE-L Score**: 0.813
- **Model**: Improved SVM-based critic with optimized threshold (0.67)

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
- Real-time refinement testing

## 📚 Documentation

For detailed documentation, see the `docs/` directory:
- `docs/README.md` - Comprehensive project documentation
- `docs/IMPROVEMENTS_SUMMARY.md` - Recent improvements and next steps

## 🎯 Features

- **Batch Processing**: Process large datasets with automatic refinement
- **ML-Powered Critics**: Trained models that evaluate answer quality
- **Advanced Feedback**: Detailed analysis with specific improvement suggestions
- **Interactive Demo**: Web interface for real-time testing
- **Comprehensive Analysis**: Tools to understand and visualize results

## 📈 Performance Metrics

- **Accuracy**: 96%
- **Success Rate**: 60.7%
- **False Negatives**: 5 (7.8%)
- **False Positives**: 2 (2%)
- **Improved Critic Training Accuracy**: 98.8%

The project is now organized for better maintainability and easier navigation! 
>>>>>>> 5112d03 (Initial commit: push full self-refine-loop project)
