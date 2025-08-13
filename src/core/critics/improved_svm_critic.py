"""Improved SVM Critic with optimized decision threshold and class balancing"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ImprovedTextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Enhanced text feature extractor"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = pd.DataFrame()
        
        features['text_length'] = X.str.len()
        features['word_count'] = X.str.split().str.len()
        features['avg_word_length'] = X.str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
        features['unique_word_ratio'] = X.apply(lambda x: len(set(word_tokenize(x.lower()))) / len(word_tokenize(x)) if x else 0)
        features['stop_word_ratio'] = X.apply(lambda x: len([w for w in word_tokenize(x.lower()) if w in self.stop_words]) / len(word_tokenize(x)) if x else 0)
        features['sentence_count'] = X.apply(lambda x: len(nltk.sent_tokenize(x)))
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'].replace(0, 1)
        features['has_code_block'] = X.str.contains(r'```', regex=False).astype(int)
        features['has_bullet_points'] = X.str.contains(r'^\s*[\*\-]', regex=True, na=False).astype(int)
        features['has_numbered_list'] = X.str.contains(r'^\s*\d+\.', regex=True, na=False).astype(int)
        features['has_headers'] = X.str.contains(r'^#+\s', regex=True, na=False).astype(int)
        features['question_marks'] = X.str.count(r'\?')
        features['exclamation_marks'] = X.str.count(r'!')
        features['quotes'] = X.str.count(r'["\']')
        features['parentheses'] = X.str.count(r'[\(\)]')
        features['technical_terms'] = X.str.lower().str.count(r'\b(algorithm|function|method|class|variable|parameter|return|loop|condition)\b')
        
        return features

def preprocess_text(text):
    """Enhanced text preprocessing"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\`\#\-\*\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class ImprovedSVMCritic:
    """Enhanced SVM critic"""
    
    def __init__(self, decision_threshold=0.67, use_class_weights=True):
        self.decision_threshold = decision_threshold
        self.use_class_weights = use_class_weights
        self.model = None
        self.vectorizer = None
        self.text_extractor = None
        self.feature_names = None
        
    def train(self, data_path=None):
        """Train the improved critic model"""
        
        # Get script directory for path resolution
        script_dir = Path(__file__).parent
        
        # Read and prepare data
        if data_path is None:
            data_path = script_dir.parent.parent / "logs" / "refine_outputs.csv"
        
        df = pd.read_csv(data_path)
        
        # Filter valid rows
        df = df.dropna(subset=['original_answer', 'revised_answer', 'needs_improvement'])
        df = df[df['original_answer'].str.strip().astype(bool)]
        df = df[df['revised_answer'].str.strip().astype(bool)]
        df = df[df['needs_improvement'].isin([0, 1])]
        
        print(f"Training with {len(df)} samples")
        print(f"Class distribution: {df['needs_improvement'].value_counts().to_dict()}")
        
        # Preprocess text
        df['original_clean'] = df['original_answer'].apply(preprocess_text)
        df['revised_clean'] = df['revised_answer'].apply(preprocess_text)
        df['feedback_clean'] = df['feedback'].fillna('').apply(preprocess_text)
        
        # Extract features
        text_extractor = ImprovedTextFeatureExtractor()
        original_features = text_extractor.transform(df['original_answer'])
        revised_features = text_extractor.transform(df['revised_answer'])
        feedback_features = text_extractor.transform(df['feedback'])
        
        # Calculate relative features
        relative_features = pd.DataFrame()
        for col in original_features.columns:
            relative_features[f'relative_{col}'] = (revised_features[col] - original_features[col]) / (original_features[col] + 1e-6)
        
        # Combine all features
        X_numeric = pd.concat([
            original_features,
            revised_features,
            feedback_features,
            relative_features
        ], axis=1)
        
        # TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Combine text for TF-IDF
        combined_text = df['original_answer'] + " " + df['revised_answer'] + " " + df['feedback'].fillna('')
        X_tfidf = vectorizer.fit_transform(combined_text)
        
        # Convert to dense array for feature selection
        X_tfidf_dense = X_tfidf.toarray()
        
        # Combine all features
        X_combined = np.hstack([X_numeric.values, X_tfidf_dense])
        
        # Target variable
        y = df['needs_improvement'].values
        
        # Compute class weights if requested
        class_weights = None
        if self.use_class_weights:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y),
                y=y
            )
            class_weights = dict(zip(np.unique(y), class_weights))
            print(f"Class weights: {class_weights}")
        
        # Create and train SVM with optimized parameters
        self.model = LinearSVC(
            C=1.0,
            class_weight=class_weights,
            max_iter=2000,
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_combined, y)
        
        # Store components
        self.vectorizer = vectorizer
        self.text_extractor = text_extractor
        self.feature_names = list(X_numeric.columns) + [f"tfidf_{i}" for i in range(X_tfidf.shape[1])]
        
        # Evaluate performance
        y_pred = self.model.predict(X_combined)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Training accuracy: {accuracy:.3f}")
        print(f"Decision threshold: {self.decision_threshold}")
        
        return self
    
    def predict(self, original_answer, revised_answer, feedback=""):
        """Predict if answer needs improvement with optimized threshold"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess inputs
        original_clean = preprocess_text(original_answer)
        revised_clean = preprocess_text(revised_answer)
        feedback_clean = preprocess_text(feedback)
        
        # Extract features
        original_features = self.text_extractor.transform(pd.Series([original_answer]))
        revised_features = self.text_extractor.transform(pd.Series([revised_answer]))
        feedback_features = self.text_extractor.transform(pd.Series([feedback]))
        
        # Calculate relative features
        relative_features = pd.DataFrame()
        for col in original_features.columns:
            relative_features[f'relative_{col}'] = (revised_features[col] - original_features[col]) / (original_features[col] + 1e-6)
        
        # Combine numeric features
        X_numeric = pd.concat([
            original_features,
            revised_features,
            feedback_features,
            relative_features
        ], axis=1)
        
        # TF-IDF features
        combined_text = original_answer + " " + revised_answer + " " + feedback
        X_tfidf = self.vectorizer.transform([combined_text])
        
        # Combine all features
        X_combined = np.hstack([X_numeric.values, X_tfidf.toarray()])
        
        # Get decision function score
        decision_score = self.model.decision_function(X_combined)[0]
        
        # Apply optimized threshold
        needs_improvement = decision_score < self.decision_threshold
        
        return {
            'needs_improvement': bool(needs_improvement),
            'confidence_score': abs(decision_score),
            'decision_score': decision_score,
            'threshold': self.decision_threshold
        }
    
    def save_model(self, models_dir):
        """Save the trained model and components"""
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(self.model, models_dir / "improved_svm_critic_model.pkl")
        joblib.dump(self.vectorizer, models_dir / "improved_svm_critic_vectorizer.pkl")
        joblib.dump(self.text_extractor, models_dir / "improved_svm_text_extractor.pkl")
        
        print(f"Model saved to {models_dir}")
    
    @classmethod
    def load_model(cls, models_dir):
        """Load a trained model"""
        models_dir = Path(models_dir)
        
        critic = cls()
        critic.model = joblib.load(models_dir / "improved_svm_critic_model.pkl")
        critic.vectorizer = joblib.load(models_dir / "improved_svm_critic_vectorizer.pkl")
        
        # Recreate the text extractor since it can't be pickled properly
        critic.text_extractor = ImprovedTextFeatureExtractor()
        
        return critic

def main():
    """Train and test the improved critic"""
    
    # Initialize improved critic
    critic = ImprovedSVMCritic(decision_threshold=0.67, use_class_weights=True)
    
    # Train the model
    print("🧠 Training Improved SVM Critic...")
    critic.train()
    
    # Save the model
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent.parent / "models"
    critic.save_model(models_dir)
    
    # Test the model
    print("\n🧪 Testing Improved Critic...")
    
    test_cases = [
        {
            'original': "Machine learning is AI.",
            'revised': "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            'feedback': "Good improvement in detail and clarity"
        },
        {
            'original': "Photosynthesis makes food.",
            'revised': "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
            'feedback': "Much better explanation"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = critic.predict(case['original'], case['revised'], case['feedback'])
        print(f"\nTest Case {i}:")
        print(f"Needs improvement: {result['needs_improvement']}")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Decision score: {result['decision_score']:.3f}")

if __name__ == "__main__":
    main() 