"""Enhanced Model Evaluation Script"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
import time
import psutil
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from core.improved_svm_critic import ImprovedSVMCritic
from utils.config import LOGS_DIR, MODELS_DIR

def calculate_confidence_intervals(y_true, y_pred, y_scores, n_bootstrap=1000):
    """Calculate confidence intervals using bootstrapping"""
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'brier_score': []
    }
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = resample(range(len(y_true)), n_samples=len(y_true), random_state=i)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_scores_boot = y_scores[indices]
        
        # Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        metrics['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        
        # Brier score (convert scores to probabilities)
        y_probs_boot = 1 / (1 + np.exp(-y_scores_boot))
        metrics['brier_score'].append(brier_score_loss(y_true_boot, y_probs_boot))
    
    # Calculate confidence intervals (95%)
    confidence_intervals = {}
    for metric, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        confidence_intervals[metric] = {
            'mean': mean_val,
            'std': std_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_range': f"{mean_val:.3f} ± {1.96*std_val:.3f}",
            'ci_interval': f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        }
    
    return confidence_intervals

def detailed_calibration_analysis(y_true, y_scores):
    """Comprehensive calibration analysis with multiple metrics"""
    
    # Convert scores to probabilities
    y_probs = 1 / (1 + np.exp(-y_scores))
    
    # 1. Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_probs[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)
    
    # 2. Brier Score
    brier_score = brier_score_loss(y_true, y_probs)
    
    # 3. Reliability diagram
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_probs, n_bins=n_bins
    )
    
    # 4. Isotonic calibration deviation
    from sklearn.isotonic import IsotonicRegression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    y_iso = iso_reg.fit_transform(y_probs, y_true)
    isotonic_deviation = np.mean(np.abs(y_probs - y_iso))
    
    # 5. Plot calibration curve
    plt.figure(figsize=(12, 8))
    
    # Main calibration plot
    plt.subplot(2, 2, 1)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model", linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram of predicted probabilities
    plt.subplot(2, 2, 2)
    plt.hist(y_probs, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Probabilities")
    plt.grid(True, alpha=0.3)
    
    # Reliability diagram
    plt.subplot(2, 2, 3)
    bin_centers = (bin_lowers + bin_uppers) / 2
    plt.bar(bin_centers, fraction_of_positives - mean_predicted_value, 
            width=0.1, alpha=0.7, label="Calibration Error")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Calibration Error")
    plt.title("Reliability Diagram")
    plt.grid(True, alpha=0.3)
    
    # Isotonic calibration
    plt.subplot(2, 2, 4)
    plt.scatter(y_probs, y_true, alpha=0.5, s=20, label="Original")
    plt.scatter(y_probs, y_iso, alpha=0.7, s=20, label="Isotonic")
    plt.plot([0, 1], [0, 1], "k:", alpha=0.5)
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.title("Isotonic Calibration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = LOGS_DIR / "detailed_calibration_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Detailed calibration plots saved to {plot_path}")
    
    # Print calibration metrics
    print(f"\n📊 Calibration Metrics:")
    print(f"Expected Calibration Error (ECE): {ece:.3f}")
    print(f"Brier Score: {brier_score:.3f}")
    print(f"Isotonic Calibration Deviation: {isotonic_deviation:.3f}")
    
    # Interpretation
    print(f"\n💡 Calibration Interpretation:")
    if ece < 0.05:
        print("✅ Excellent calibration (ECE < 0.05)")
    elif ece < 0.1:
        print("✅ Good calibration (ECE < 0.1)")
    elif ece < 0.2:
        print("⚠️  Fair calibration (ECE < 0.2)")
    else:
        print("❌ Poor calibration (ECE >= 0.2)")
    
    if brier_score < 0.1:
        print("✅ Excellent Brier score (< 0.1)")
    elif brier_score < 0.2:
        print("✅ Good Brier score (< 0.2)")
    else:
        print("⚠️  Poor Brier score (>= 0.2)")
    
    return {
        'ece': ece,
        'brier_score': brier_score,
        'isotonic_deviation': isotonic_deviation,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }

def performance_complexity_tradeoff(df):
    """Analyze performance vs complexity tradeoff"""
    print(f"\n⚡ Performance vs Complexity Tradeoff Analysis")
    print("=" * 60)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import time
    import psutil
    
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Test different models
    models = {
        'SVM (C=0.1)': LinearSVC(C=0.1, class_weight='balanced', random_state=42),
        'SVM (C=1.0)': LinearSVC(C=1.0, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔍 Testing {name}...")
        
        # Training time and memory
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        training_memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
        
        # Inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Model size (approximate)
        if hasattr(model, 'coef_'):
            model_size = model.coef_.nbytes / 1024  # KB
        else:
            model_size = 100  # Approximate for tree-based models
        
        # Performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time,
            'inference_time': inference_time,
            'training_memory': training_memory,
            'model_size': model_size
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Training Time: {training_time:.3f}s")
        print(f"  Inference Time: {inference_time:.6f}s")
        print(f"  Training Memory: {training_memory:.1f}MB")
        print(f"  Model Size: {model_size:.1f}KB")
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # F1 Score vs Training Time
    names = list(results.keys())
    f1_scores = [results[name]['f1_score'] for name in names]
    training_times = [results[name]['training_time'] for name in names]
    
    ax1.scatter(training_times, f1_scores, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax1.annotate(name, (training_times[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Training Time (s)')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('Performance vs Training Time')
    ax1.grid(True, alpha=0.3)
    
    # F1 Score vs Model Size
    model_sizes = [results[name]['model_size'] for name in names]
    ax2.scatter(model_sizes, f1_scores, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax2.annotate(name, (model_sizes[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Model Size (KB)')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Performance vs Model Size')
    ax2.grid(True, alpha=0.3)
    
    # Inference Time vs Model Size
    inference_times = [results[name]['inference_time'] for name in names]
    ax3.scatter(model_sizes, inference_times, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax3.annotate(name, (model_sizes[i], inference_times[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('Model Size (KB)')
    ax3.set_ylabel('Inference Time (s)')
    ax3.set_title('Inference Time vs Model Size')
    ax3.grid(True, alpha=0.3)
    
    # Memory Usage vs Training Time
    memory_usage = [results[name]['training_memory'] for name in names]
    ax4.scatter(training_times, memory_usage, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax4.annotate(name, (training_times[i], memory_usage[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax4.set_xlabel('Training Time (s)')
    ax4.set_ylabel('Memory Usage (MB)')
    ax4.set_title('Memory Usage vs Training Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = LOGS_DIR / "performance_complexity_tradeoff.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance vs complexity plots saved to {plot_path}")
    
    # Recommendations
    print(f"\n💡 Deployment Recommendations:")
    
    # Find best model for different criteria
    best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
    fastest_inference = min(results.items(), key=lambda x: x[1]['inference_time'])
    smallest_model = min(results.items(), key=lambda x: x[1]['model_size'])
    
    print(f"🎯 Best Performance: {best_f1[0]} (F1={best_f1[1]['f1_score']:.3f})")
    print(f"⚡ Fastest Inference: {fastest_inference[0]} ({fastest_inference[1]['inference_time']:.6f}s)")
    print(f"📦 Smallest Model: {smallest_model[0]} ({smallest_model[1]['model_size']:.1f}KB)")
    
    return results

def main():
    """Run enhanced evaluation"""
    print("🔬 ENHANCED MODEL EVALUATION")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Held-out test set evaluation
    test_results = held_out_test_evaluation(df)
    
    # 1. Confidence intervals
    print(f"\n📊 Confidence Intervals (95%)")
    print("=" * 40)
    confidence_intervals = calculate_confidence_intervals(
        test_results['y_test'], 
        test_results['y_pred'], 
        test_results['y_scores']
    )
    
    for metric, stats in confidence_intervals.items():
        print(f"{metric.capitalize()}: {stats['ci_range']}")
        print(f"  Confidence Interval: {stats['ci_interval']}")
    
    # 2. Detailed calibration analysis
    calibration_results = detailed_calibration_analysis(
        test_results['y_test'], 
        test_results['y_scores']
    )
    
    # 3. Performance vs complexity tradeoff
    complexity_results = performance_complexity_tradeoff(df)
    
    # Summary
    print(f"\n📋 ENHANCED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"✅ Test F1: {confidence_intervals['f1']['ci_range']}")
    print(f"✅ Calibration ECE: {calibration_results['ece']:.3f}")
    print(f"✅ Brier Score: {calibration_results['brier_score']:.3f}")
    print(f"✅ Best Model: {max(complexity_results.items(), key=lambda x: x[1]['f1_score'])[0]}")
    
    # Save detailed results
    results_summary = {
        'confidence_intervals': confidence_intervals,
        'calibration_results': calibration_results,
        'complexity_results': complexity_results
    }
    
    # Save to file
    import json
    results_path = LOGS_DIR / "enhanced_evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results_summary.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        json_results[key][k] = {k2: float(v2) if isinstance(v2, (np.floating, float)) else v2 
                                              for k2, v2 in v.items()}
                    else:
                        json_results[key][k] = float(v) if isinstance(v, (np.floating, float)) else v
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    print(f"✅ Detailed results saved to {results_path}")
    print("✅ Enhanced evaluation complete!")

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
    model = LinearSVC(C=0.1, class_weight='balanced', random_state=42)
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_scores': y_scores
    }

if __name__ == "__main__":
    main() 