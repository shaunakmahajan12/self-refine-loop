"""
Robust Model Evaluation Script

Implements comprehensive evaluation including:
- Cross-validation with standard deviation
- Held-out test set evaluation
- Model calibration curves
- Feature importance analysis
- Regularization assessment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from core.improved_svm_critic import ImprovedSVMCritic
from utils.config import LOGS_DIR, MODELS_DIR

def load_and_prepare_data():
    """Load and prepare data for evaluation"""
    print("📊 Loading and preparing data...")
    
    # Load the refinement results
    data_path = LOGS_DIR / "refine_outputs.csv"
    df = pd.read_csv(data_path)
    
    # Filter valid rows
    df = df.dropna(subset=['original_answer', 'revised_answer', 'needs_improvement'])
    df = df[df['original_answer'].str.strip().astype(bool)]
    df = df[df['revised_answer'].str.strip().astype(bool)]
    df = df[df['needs_improvement'].isin([0, 1])]
    
    print(f"✅ Loaded {len(df)} valid samples")
    print(f"📈 Class distribution: {df['needs_improvement'].value_counts().to_dict()}")
    
    return df

def cross_validation_evaluation(df, n_folds=5):
    """Perform cross-validation with detailed metrics"""
    print(f"\n🔄 Cross-Validation Evaluation ({n_folds} folds)")
    print("=" * 50)
    
    # Prepare features (simplified for evaluation)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    
    # Create feature matrix
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['original_answer'] + " " + df['revised_answer'])
    
    # Add basic numeric features
    numeric_features = pd.DataFrame({
        'text_length': df['original_answer'].str.len(),
        'word_count': df['original_answer'].str.split().str.len(),
        'revised_length': df['revised_answer'].str.len(),
        'revised_word_count': df['revised_answer'].str.split().str.len(),
    })
    
    # Combine features
    X = np.hstack([X_text.toarray(), numeric_features.values])
    y = df['needs_improvement'].values
    
    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Metrics storage
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = LinearSVC(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        
        print(f"Fold {fold}: Acc={metrics['accuracy'][-1]:.3f}, "
              f"Prec={metrics['precision'][-1]:.3f}, "
              f"Rec={metrics['recall'][-1]:.3f}, "
              f"F1={metrics['f1'][-1]:.3f}")
    
    # Calculate statistics
    print("\n📊 Cross-Validation Results:")
    print("-" * 30)
    for metric, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.capitalize()}: {mean_val:.3f} ± {std_val:.3f}")
    
    return metrics

def held_out_test_evaluation(df, test_size=0.2):
    """Evaluate on completely held-out test set"""
    print(f"\n🧪 Held-Out Test Set Evaluation ({test_size*100}% test)")
    print("=" * 50)
    
    # Prepare features
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['original_answer'] + " " + df['revised_answer'])
    
    numeric_features = pd.DataFrame({
        'text_length': df['original_answer'].str.len(),
        'word_count': df['original_answer'].str.split().str.len(),
        'revised_length': df['revised_answer'].str.len(),
        'revised_word_count': df['revised_answer'].str.split().str.len(),
    })
    
    X = np.hstack([X_text.toarray(), numeric_features.values])
    y = df['needs_improvement'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"📊 Train set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    print(f"📊 Test class distribution: {np.bincount(y_test)}")
    
    # Train model
    model = LinearSVC(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📈 Test Set Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_scores': y_scores
    }

def model_calibration_analysis(results):
    """Analyze model calibration and confidence"""
    print(f"\n🎯 Model Calibration Analysis")
    print("=" * 40)
    
    y_test = results['y_test']
    y_scores = results['y_scores']
    
    # Convert scores to probabilities (approximate)
    y_probs = 1 / (1 + np.exp(-y_scores))
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_probs, n_bins=10
    )
    
    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Model Calibration Curve")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = LOGS_DIR / "calibration_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Calibration curve saved to {plot_path}")
    
    # Calculate calibration metrics
    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    print(f"📊 Calibration Error: {calibration_error:.3f}")
    
    return {
        'calibration_error': calibration_error,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }

def feature_importance_analysis(df):
    """Analyze feature importance and check for over-reliance"""
    print(f"\n🔍 Feature Importance Analysis")
    print("=" * 40)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Prepare features
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['original_answer'] + " " + df['revised_answer'])
    
    numeric_features = pd.DataFrame({
        'text_length': df['original_answer'].str.len(),
        'word_count': df['original_answer'].str.split().str.len(),
        'revised_length': df['revised_answer'].str.len(),
        'revised_word_count': df['revised_answer'].str.split().str.len(),
        'length_ratio': df['revised_answer'].str.len() / (df['original_answer'].str.len() + 1),
        'word_ratio': df['revised_answer'].str.split().str.len() / (df['original_answer'].str.split().str.len() + 1),
    })
    
    X = np.hstack([X_text.toarray(), numeric_features.values])
    y = df['needs_improvement'].values
    
    # Feature names
    text_feature_names = [f"text_{i}" for i in range(X_text.shape[1])]
    numeric_feature_names = numeric_features.columns.tolist()
    feature_names = text_feature_names + numeric_feature_names
    
    # Train model and get feature importance
    model = LinearSVC(class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    # Get feature importance (coefficients)
    importance = np.abs(model.coef_[0])
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("📊 Top 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Check for over-reliance
    top_5_importance = importance_df.head(5)['importance'].sum()
    total_importance = importance_df['importance'].sum()
    concentration_ratio = top_5_importance / total_importance
    
    print(f"\n🔍 Feature Concentration Analysis:")
    print(f"Top 5 features account for {concentration_ratio:.1%} of total importance")
    
    if concentration_ratio > 0.8:
        print("⚠️  WARNING: High feature concentration - consider regularization")
    elif concentration_ratio > 0.6:
        print("⚠️  CAUTION: Moderate feature concentration")
    else:
        print("✅ Good feature distribution")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    
    # Save plot
    plot_path = LOGS_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Feature importance plot saved to {plot_path}")
    
    return importance_df, concentration_ratio

def regularization_assessment(df):
    """Assess need for regularization"""
    print(f"\n⚖️  Regularization Assessment")
    print("=" * 40)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    
    # Prepare features
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['original_answer'] + " " + df['revised_answer'])
    
    numeric_features = pd.DataFrame({
        'text_length': df['original_answer'].str.len(),
        'word_count': df['original_answer'].str.split().str.len(),
        'revised_length': df['revised_answer'].str.len(),
        'revised_word_count': df['revised_answer'].str.split().str.len(),
    })
    
    X = np.hstack([X_text.toarray(), numeric_features.values])
    y = df['needs_improvement'].values
    
    # Test different regularization strengths
    C_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    cv_scores = []
    
    print("📊 Testing regularization strengths:")
    for C in C_values:
        model = LinearSVC(C=C, class_weight='balanced', random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        mean_score = scores.mean()
        std_score = scores.std()
        cv_scores.append(mean_score)
        print(f"C={C}: F1 = {mean_score:.3f} ± {std_score:.3f}")
    
    # Find optimal C
    optimal_C = C_values[np.argmax(cv_scores)]
    print(f"\n🎯 Optimal C value: {optimal_C}")
    
    # Check for overfitting
    if cv_scores[0] > cv_scores[-1]:
        print("✅ Stronger regularization (lower C) improves performance")
    elif cv_scores[-1] > cv_scores[0]:
        print("⚠️  Weaker regularization (higher C) improves performance")
    else:
        print("✅ Current regularization level appears appropriate")
    
    return {
        'C_values': C_values,
        'cv_scores': cv_scores,
        'optimal_C': optimal_C
    }

def main():
    """Run comprehensive evaluation"""
    print("🔬 ROBUST MODEL EVALUATION")
    print("=" * 50)
    
    # Load data
    df = load_and_prepare_data()
    
    # 1. Cross-validation evaluation
    cv_metrics = cross_validation_evaluation(df)
    
    # 2. Held-out test set evaluation
    test_results = held_out_test_evaluation(df)
    
    # 3. Model calibration analysis
    calibration_results = model_calibration_analysis(test_results)
    
    # 4. Feature importance analysis
    importance_df, concentration_ratio = feature_importance_analysis(df)
    
    # 5. Regularization assessment
    reg_results = regularization_assessment(df)
    
    # Summary
    print(f"\n📋 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"✅ Cross-validation F1: {np.mean(cv_metrics['f1']):.3f} ± {np.std(cv_metrics['f1']):.3f}")
    print(f"✅ Test set F1: {test_results['f1']:.3f}")
    print(f"✅ Calibration error: {calibration_results['calibration_error']:.3f}")
    print(f"✅ Feature concentration: {concentration_ratio:.1%}")
    print(f"✅ Optimal regularization C: {reg_results['optimal_C']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 30)
    
    if test_results['f1'] < np.mean(cv_metrics['f1']) - np.std(cv_metrics['f1']):
        print("⚠️  Test performance lower than CV - consider regularization")
    
    if calibration_results['calibration_error'] > 0.1:
        print("⚠️  High calibration error - consider probability calibration")
    
    if concentration_ratio > 0.8:
        print("⚠️  High feature concentration - consider feature selection")
    
    print("✅ Model evaluation complete!")

if __name__ == "__main__":
    main() 