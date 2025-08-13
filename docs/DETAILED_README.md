# 🔄 Self-Refinement Loop Project

A sophisticated system that uses machine learning critics to iteratively improve AI-generated answers through self-reflection and refinement.

## 🎯 Project Overview

This project implements a self-refinement loop where an AI model generates answers, a critic evaluates their quality, and the system iteratively improves responses based on feedback. The system includes:

- **Batch Processing**: Process large datasets of prompts with automatic refinement
- **ML-Powered Critics**: Trained models that evaluate answer quality
- **Advanced Feedback**: Detailed analysis with specific improvement suggestions
- **Interactive Demo**: Web interface for real-time testing
- **Comprehensive Analysis**: Tools to understand and visualize results

## 📊 Current Performance

- **Success Rate**: 60.7% (answers deemed sufficient by critic)
- **Average BLEU Score**: 0.64
- **Average ROUGE-L Score**: 0.813
- **Model**: SVM-based critic with TF-IDF features

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd refine-loop

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo "your-api-key-here" > API_KEY.txt
```

### 2. Run Analysis

```bash
# Run analysis tools
python run_analysis.py

# Or run individual analyses
python src/analysis/simple_analysis.py
python src/analysis/improve_critic.py
```

### 3. Launch Interactive Demo

```bash
# Start the Streamlit demo
python run_demo.py

# Or run directly with streamlit
streamlit run ui/demo_app.py
```

## 📁 Project Structure

```
refine-loop/
├── data/                    # Input prompts and test data
├── logs/                    # Output results and metrics
├── models/                  # Trained critic models
├── ui/                      # User interface
│   ├── __init__.py         # UI package initialization
│   └── demo_app.py         # Interactive Streamlit demo
├── src/                     # Source code
│   ├── core/               # Core functionality
│   │   ├── __init__.py     # Core package initialization
│   │   ├── batch_self_refine.py     # Batch processing system
│   │   ├── test_with_critic.py      # External critic testing
│   │   ├── advanced_critic.py       # Detailed feedback system
│   │   ├── svm_critic_head.py      # SVM critic implementation
│   │   ├── random_forest_critic_head.py  # Random Forest critic
│   │   └── one_shot.py             # One-shot refinement
│   ├── analysis/           # Analysis and evaluation tools
│   │   ├── __init__.py     # Analysis package initialization
│   │   ├── analyze_results.py      # Results analysis & visualization
│   │   ├── simple_analysis.py      # Quick analysis script
│   │   ├── improve_critic.py       # Performance improvement analysis
│   │   └── evaluate_metrics.py     # BLEU/ROUGE evaluation
│   ├── utils/              # Utility functions
│   │   ├── __init__.py     # Utils package initialization
│   │   ├── config.py               # Configuration and paths
│   │   └── csv_loader.py           # Safe CSV loading utilities
│   └── __init__.py         # Main package initialization
├── run_demo.py             # Demo launcher script
├── run_analysis.py         # Analysis launcher script
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🔧 Key Components

### 1. Batch Self-Refinement (`batch_self_refine.py`)
- Processes large datasets with checkpointing
- Simple feedback loop with Gemini
- Rate limiting and error handling

### 2. External Critic System (`test_with_critic.py`)
- Uses trained ML models to judge answer quality
- SVM-based critic with TF-IDF features
- Automatic refinement based on critic decisions

### 3. Advanced Critic (`advanced_critic.py`)
- Detailed feedback with specific suggestions
- Multi-dimensional quality assessment
- Rule-based and ML-powered evaluation

### 4. Interactive Demo (`demo_app.py`)
- Real-time refinement testing
- Visual progress tracking
- Results analysis dashboard

## 📈 Next Steps & Recommendations

### Immediate Actions (High Impact)

1. **Run Comprehensive Analysis**
   ```bash
   python src/analyze_results.py
   ```
   - Understand current performance patterns
   - Identify weak areas by question type
   - Generate actionable insights

2. **Test Advanced Critic**
   ```bash
   python src/advanced_critic.py
   ```
   - Evaluate detailed feedback quality
   - Compare with existing binary critic
   - Refine feedback generation rules

3. **Launch Interactive Demo**
   ```bash
   streamlit run src/demo_app.py
   ```
   - Test real-time refinement
   - Gather user feedback
   - Identify UI/UX improvements

### Medium-Term Improvements

4. **Enhance Critic Models**
   - Train on larger, more diverse datasets
   - Implement ensemble methods (SVM + Random Forest)
   - Add domain-specific critics for different question types

5. **Improve Refinement Strategies**
   - Implement adaptive iteration limits
   - Add context-aware improvement prompts
   - Experiment with different feedback mechanisms

6. **Expand Evaluation Metrics**
   - Add human evaluation framework
   - Implement factuality checking
   - Add coherence and consistency metrics

### Advanced Features

7. **Multi-Modal Refinement**
   - Support for code generation and refinement
   - Image-based question answering
   - Structured data generation

8. **Personalization**
   - User preference learning
   - Domain-specific adaptation
   - Difficulty level adjustment

9. **Production Deployment**
   - API endpoint for refinement service
   - Batch processing pipeline
   - Monitoring and alerting

## 🧪 Experiment Ideas

### A/B Testing Framework
- Compare different critic models
- Test various refinement strategies
- Evaluate prompt engineering approaches

### Domain-Specific Optimization
- Technical vs. general knowledge questions
- Code generation vs. text explanation
- Factual vs. analytical questions

### Ensemble Methods
- Combine multiple critic models
- Weighted voting systems
- Confidence-based decision making

## 📊 Performance Monitoring

### Key Metrics to Track
- **Success Rate**: Percentage of answers deemed sufficient
- **Iteration Efficiency**: Average iterations needed
- **Quality Improvement**: BLEU/ROUGE score changes
- **User Satisfaction**: Human evaluation scores

### Visualization Dashboard
- Real-time performance tracking
- Question type analysis
- Model comparison charts
- Improvement trends over time

## 🤝 Contributing

### Development Workflow
1. Create feature branch
2. Implement changes with tests
3. Run analysis to validate improvements
4. Update documentation
5. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include type hints where appropriate
- Write unit tests for new features

## 📚 Research Context

This project builds on recent advances in:
- **Self-Refinement**: Models that improve their own outputs
- **Critic Networks**: Learned evaluation functions
- **Iterative Generation**: Multi-step text generation
- **Quality Assessment**: Automated evaluation metrics

## 🔗 Related Work

- [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [RLHF and Human Feedback](https://arxiv.org/abs/2203.02155)

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- Gemini API for text generation
- Scikit-learn for ML models
- Streamlit for web interface
- Evaluate library for metrics

---

**Ready to improve AI answer quality? Start with the analysis and demo! 🚀** 