from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel, VarianceThreshold

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Get script directory for path resolution
script_dir = Path(__file__).parent

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = pd.DataFrame()
        
        # Basic text features
        features['text_length'] = X.str.len()
        features['word_count'] = X.str.split().str.len()
        features['avg_word_length'] = X.str.split().apply(lambda x: np.mean([len(w) for w in x]) if x else 0)
        
        # Advanced text features
        features['unique_word_ratio'] = X.apply(lambda x: len(set(word_tokenize(x.lower()))) / len(word_tokenize(x)) if x else 0)
        features['stop_word_ratio'] = X.apply(lambda x: len([w for w in word_tokenize(x.lower()) if w in self.stop_words]) / len(word_tokenize(x)) if x else 0)
        features['sentence_count'] = X.apply(lambda x: len(nltk.sent_tokenize(x)))
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'].replace(0, 1)
        
        # Code and formatting features
        features['has_code_block'] = X.str.contains(r'```', regex=False).astype(int)
        features['has_bullet_points'] = X.str.contains(r'^\s*[\*\-]', regex=True, na=False).astype(int)
        features['has_numbered_list'] = X.str.contains(r'^\s*\d+\.', regex=True, na=False).astype(int)
        features['has_headers'] = X.str.contains(r'^#+\s', regex=True, na=False).astype(int)
        
        return features

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters but keep important ones for code
    text = re.sub(r'[^\w\s\`\#\-\*\.]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Read and prepare data
df = pd.read_csv(script_dir.parent / "logs" / "refine_outputs.csv")

# Filter valid rows
df = df.dropna(subset=['original_answer', 'revised_answer', 'needs_improvement'])
df = df[df['original_answer'].str.strip().astype(bool)]
df = df[df['revised_answer'].str.strip().astype(bool)]
df = df[df['needs_improvement'].isin([0, 1])]

# Preprocess text
df['original_clean'] = df['original_answer'].apply(preprocess_text)
df['revised_clean'] = df['revised_answer'].apply(preprocess_text)
df['feedback_clean'] = df['feedback'].fillna('').apply(preprocess_text)

# Extract features
text_extractor = TextFeatureExtractor()
original_features = text_extractor.transform(df['original_answer'])
revised_features = text_extractor.transform(df['revised_answer'])
feedback_features = text_extractor.transform(df['feedback'])

# Calculate relative features
relative_features = pd.DataFrame()
for col in original_features.columns:
    relative_features[f'relative_{col}'] = (revised_features[col] - original_features[col]) / (original_features[col] + 1e-6)

# Combine all features
X_numeric = pd.concat([
    original_features.add_prefix('original_'),
    revised_features.add_prefix('revised_'),
    feedback_features.add_prefix('feedback_'),
    relative_features
], axis=1)

# Create text vectorizer with improved parameters - further reduced feature space
text_vectorizer = TfidfVectorizer(
    max_features=800,   # Further reduced from 1000
    min_df=10,         # Increased from 7 to require more frequent terms
    max_df=0.75,       # Reduced from 0.80 to remove more common terms
    ngram_range=(1, 2), # Keep at (1,2)
    stop_words='english',
    sublinear_tf=True,
    use_idf=True,
    smooth_idf=True
)

# Transform text features
X_text = text_vectorizer.fit_transform(
    df['original_clean'] + ' ' + 
    df['revised_clean'] + ' ' + 
    df['feedback_clean']
)

# Combine features
X_combined = np.hstack([X_text.toarray(), X_numeric])
y = df['needs_improvement']

# Instead of single train-test split, we'll use k-fold CV
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Define model pipeline with further reduced complexity and feature selection
base_model = Pipeline([
    ('scaler', StandardScaler()),
    ('variance_threshold', VarianceThreshold(threshold=0.01)),  # Remove low variance features
    ('feature_selector', SelectFromModel(
        LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42),
        max_features=100  # Limit to top 100 features
    )),
    ('classifier', RandomForestClassifier(
        n_estimators=30,      # Reduced from 50
        max_depth=3,          # Reduced from 6
        min_samples_split=15, # Increased from 12
        min_samples_leaf=6,   # Increased from 5
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
        max_samples=0.6       # Reduced from 0.8 to use less data per tree
    ))
])

# Define parameter grid for GridSearchCV - more conservative
param_grid = {
    'classifier__n_estimators': [20, 30, 40],    # Reduced range
    'classifier__max_depth': [2, 3, 4],          # Reduced range
    'classifier__min_samples_split': [15, 20, 25], # Increased range
    'classifier__min_samples_leaf': [5, 6, 7],   # Increased range
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_samples': [0.5, 0.6, 0.7],  # Reduced range
    'feature_selector__max_features': [80, 100, 120]  # Added feature selection options
}

# Perform grid search with k-fold CV
print("\n🔍 Training model with grid search and k-fold cross-validation...")
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=skf,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

# Train model
grid_search.fit(X_combined, y)

# Get best model
best_model = grid_search.best_estimator_

# Print best parameters
print("\n🔍 Best Parameters:")
print(grid_search.best_params_)

# Perform k-fold cross-validation with the best model
print("\n📊 Performing k-fold cross-validation with best model...")
cv_predictions = cross_val_predict(best_model, X_combined, y, cv=skf)

# Calculate metrics for each fold
fold_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y), 1):
    X_fold_train, X_fold_val = X_combined[train_idx], X_combined[val_idx]
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]
    
    # Train on this fold
    fold_model = best_model.fit(X_fold_train, y_fold_train)
    fold_pred = fold_model.predict(X_fold_val)
    
    # Calculate metrics
    fold_accuracy = accuracy_score(y_fold_val, fold_pred)
    fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(
        y_fold_val, fold_pred, average='weighted'
    )
    
    fold_scores.append({
        'fold': fold,
        'accuracy': fold_accuracy,
        'precision': fold_precision,
        'recall': fold_recall,
        'f1': fold_f1
    })
    
    print(f"\n📊 Fold {fold} Results:")
    print(f"Accuracy: {fold_accuracy:.3f}")
    print(f"Precision: {fold_precision:.3f}")
    print(f"Recall: {fold_recall:.3f}")
    print(f"F1 Score: {fold_f1:.3f}")

# Calculate and print average metrics across folds
avg_scores = {
    metric: np.mean([score[metric] for score in fold_scores])
    for metric in ['accuracy', 'precision', 'recall', 'f1']
}

print("\n📊 Average Performance Across Folds:")
print(f"Average Accuracy: {avg_scores['accuracy']:.3f} (±{np.std([s['accuracy'] for s in fold_scores]):.3f})")
print(f"Average Precision: {avg_scores['precision']:.3f} (±{np.std([s['precision'] for s in fold_scores]):.3f})")
print(f"Average Recall: {avg_scores['recall']:.3f} (±{np.std([s['recall'] for s in fold_scores]):.3f})")
print(f"Average F1 Score: {avg_scores['f1']:.3f} (±{np.std([s['f1'] for s in fold_scores]):.3f})")

# Print overall classification report
print("\n📊 Overall Classification Report:")
print(classification_report(y, cv_predictions))

# Print confusion matrix
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y, cv_predictions))

# Get feature importances
feature_names = (
    text_vectorizer.get_feature_names_out().tolist() + 
    X_numeric.columns.tolist()
)
importances = best_model.named_steps['classifier'].feature_importances_
top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
print("\n🔍 Top 20 Most Important Features:")
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")

# After getting feature importances, add feature selection analysis
selected_features = best_model.named_steps['feature_selector'].get_support()
selected_feature_names = [f for f, selected in zip(feature_names, selected_features) if selected]
print("\n🔍 Number of selected features:", sum(selected_features))
print("\n🔍 Top 10 Selected Features:")
for feature, importance in sorted(zip(selected_feature_names, 
                                    best_model.named_steps['classifier'].feature_importances_),
                                key=lambda x: x[1], 
                                reverse=True)[:10]:
    print(f"{feature}: {importance:.4f}")

# Add feature importance stability analysis across folds
fold_feature_importances = []
selected_features_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y), 1):
    X_fold_train = X_combined[train_idx]
    y_fold_train = y[train_idx]
    
    # Train model on this fold
    fold_model = best_model.fit(X_fold_train, y_fold_train)
    
    # Get selected features for this fold
    selected_features = fold_model.named_steps['feature_selector'].get_support()
    selected_features_per_fold.append(selected_features)
    
    # Get feature importances for selected features only
    fold_importances = fold_model.named_steps['classifier'].feature_importances_
    fold_feature_importances.append(fold_importances)

# Calculate which features were consistently selected across folds
consistent_features = np.all(selected_features_per_fold, axis=0)
consistent_feature_names = [f for f, selected in zip(feature_names, consistent_features) if selected]

print("\n🔍 Feature Selection Stability:")
print(f"Number of features selected in all folds: {sum(consistent_features)}")
print("\nConsistently selected features across all folds:")
for feature in consistent_feature_names:
    print(f"- {feature}")

# Calculate average importance for consistently selected features
avg_importances = np.mean(fold_feature_importances, axis=0)
std_importances = np.std(fold_feature_importances, axis=0)

print("\n🔍 Feature Importance Stability (for consistently selected features):")
for feature, (avg_imp, std_imp) in sorted(zip(consistent_feature_names, 
                                            zip(avg_importances, std_importances)),
                                        key=lambda x: x[1][0], 
                                        reverse=True)[:10]:
    print(f"{feature}:")
    print(f"  Average Importance: {avg_imp:.4f}")
    print(f"  Std Dev: {std_imp:.4f}")

# Save model and vectorizer
joblib.dump(best_model, script_dir.parent / "models" / "critic_model.pkl")
joblib.dump(text_vectorizer, script_dir.parent / "models" / "critic_vectorizer.pkl")
joblib.dump(text_extractor, script_dir.parent / "models" / "text_extractor.pkl")