import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
UI_DIR = PROJECT_ROOT / "ui"

# Source directories
SRC_DIR = PROJECT_ROOT / "src"
CORE_DIR = SRC_DIR / "core"
ANALYSIS_DIR = SRC_DIR / "analysis"
UTILS_DIR = SRC_DIR / "utils"

# File paths
API_KEY_FILE = PROJECT_ROOT / "config" / "API_KEY.txt"
REQUIREMENTS_FILE = PROJECT_ROOT / "config" / "requirements.txt"
README_FILE = PROJECT_ROOT / "docs" / "DETAILED_README.md"

# Model file paths (Improved SVM Critic)
IMPROVED_SVM_MODEL_FILE = MODELS_DIR / "improved_svm_critic_model.pkl"
IMPROVED_SVM_VECTORIZER_FILE = MODELS_DIR / "improved_svm_critic_vectorizer.pkl"
IMPROVED_SVM_TEXT_EXTRACTOR_FILE = MODELS_DIR / "improved_svm_text_extractor.pkl"

# Data file paths
PROMPTS_FILE = DATA_DIR / "prompts.csv"
TEST_PROMPTS_FILE = DATA_DIR / "test_prompts.csv"

# Log file paths
REFINE_OUTPUTS_FILE = LOGS_DIR / "refine_outputs.csv"
FINAL_RESULTS_FILE = LOGS_DIR / "final_refinement_results.csv"
TEST_OUTPUTS_FILE = LOGS_DIR / "test_refine_outputs.csv"
EVALUATED_OUTPUTS_FILE = LOGS_DIR / "evaluated_outputs.csv"
TEST_SUMMARY_FILE = LOGS_DIR / "test_results_summary.csv"

# Gemini API Configuration
GEMINI_MODEL = "gemini-1.5-flash"
MAX_RETRIES = 3
DEFAULT_DELAY = 1.5

# Critic Configuration
DEFAULT_THRESHOLD = 0.5
OPTIMAL_THRESHOLD = 0.67
MAX_ITERATIONS = 5

# Analysis Configuration
CHECKPOINT_INTERVAL = 10
VISUALIZATION_DPI = 300

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        DATA_DIR, LOGS_DIR, MODELS_DIR, UI_DIR,
        SRC_DIR, CORE_DIR, ANALYSIS_DIR, UTILS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Ensured directory exists: {directory}")

def get_api_key():
    """Get the API key from file"""
    if API_KEY_FILE.exists():
        with open(API_KEY_FILE, 'r') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"API key file not found at {API_KEY_FILE}")

def validate_setup():
    """Validate that the project is properly set up"""
    issues = []
    
    # Check for API key
    if not API_KEY_FILE.exists():
        issues.append(f"API key file missing: {API_KEY_FILE}")
    
    # Check for data files
    if not PROMPTS_FILE.exists():
        issues.append(f"Prompts file missing: {PROMPTS_FILE}")
    
    # Check for model files
    model_files = [
        IMPROVED_SVM_MODEL_FILE, IMPROVED_SVM_VECTORIZER_FILE, IMPROVED_SVM_TEXT_EXTRACTOR_FILE
    ]
    for model_file in model_files:
        if not model_file.exists():
            issues.append(f"Model file missing: {model_file}")
    
    if issues:
        print("⚠️  Setup validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Project setup validation passed")
        return True 