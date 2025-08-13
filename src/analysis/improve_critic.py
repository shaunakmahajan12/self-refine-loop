import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_data():
    """Load the training data and analyze current performance"""
    script_dir = Path(__file__).parent
    
    # Load the training data (you'll need to recreate this from your results)
    # For now, let's create a synthetic dataset based on your metrics
    
    print("🔍 ANALYZING CURRENT CRITIC PERFORMANCE")
    print("=" * 50)
    
    # Based on your confusion matrix: [[97 2], [5 59]]
    # This means: 99 good answers, 64 needing improvement
    # But 5 good answers were incorrectly marked as needing improvement
    # And 2 answers needing improvement were incorrectly marked as good
    
    print("📊 Current Performance Metrics:")
    print("Total samples: 163")
    print("Good answers: 99 (60.7%)")
    print("Needs improvement: 64 (39.3%)")
    print("False positives: 2")
    print("False negatives: 5")
    print("Accuracy: 96%")
    print("Precision: 95.9%")
    print("Recall: 95.7%")
    
    return {
        'total_samples': 163,
        'good_answers': 99,
        'needs_improvement': 64,
        'false_positives': 2,
        'false_negatives': 5
    }

def identify_improvement_areas():
    """Identify specific areas for improvement"""
    print("\n🎯 IMPROVEMENT OPPORTUNITIES")
    print("=" * 50)
    
    print("1. **Class Imbalance Issue**:")
    print("   - 60.7% good answers vs 39.3% needing improvement")
    print("   - This can bias the model toward the majority class")
    print("   - Solution: Use class weights or balanced sampling")
    
    print("\n2. **False Negative Problem**:")
    print("   - 5 good answers were marked as needing improvement")
    print("   - This is more costly than false positives")
    print("   - Solution: Adjust decision threshold or use asymmetric costs")
    
    print("\n3. **Feature Engineering**:")
    print("   - Current features focus on text statistics")
    print("   - Missing semantic and contextual features")
    print("   - Solution: Add domain-specific features")
    
    print("\n4. **Ensemble Methods**:")
    print("   - Single SVM model may be limiting")
    print("   - Solution: Combine multiple models")

def create_improved_critic():
    """Create an improved critic with better performance"""
    print("\n🔧 CREATING IMPROVED CRITIC")
    print("=" * 50)
    
    # 1. Class-balanced SVM
    print("1. **Class-Balanced SVM**:")
    print("   - Using class weights to handle imbalance")
    print("   - Adjusting decision threshold for better recall")
    
    # 2. Ensemble approach
    print("\n2. **Ensemble Model**:")
    print("   - Combining SVM with Random Forest")
    print("   - Voting mechanism for final decision")
    
    # 3. Feature improvements
    print("\n3. **Enhanced Features**:")
    print("   - Adding semantic similarity scores")
    print("   - Question-answer alignment metrics")
    print("   - Domain-specific quality indicators")
    
    return {
        'class_balanced_svm': True,
        'ensemble_model': True,
        'enhanced_features': True
    }

def optimize_decision_threshold():
    """Optimize the decision threshold for better performance"""
    print("\n⚖️ OPTIMIZING DECISION THRESHOLD")
    print("=" * 50)
    
    print("Current threshold: 0.5 (default)")
    print("Recommended threshold: 0.6-0.7")
    print("Reason: Reduce false negatives (good answers marked as needing improvement)")
    
    # Calculate optimal threshold based on cost matrix
    cost_false_negative = 2.0  # More costly to mark good answer as needing improvement
    cost_false_positive = 1.0  # Less costly to mark bad answer as good
    
    optimal_threshold = cost_false_negative / (cost_false_negative + cost_false_positive)
    print(f"Cost-based optimal threshold: {optimal_threshold:.2f}")
    
    return optimal_threshold

def create_ensemble_critic():
    """Create an ensemble critic model"""
    print("\n🎯 ENSEMBLE CRITIC DESIGN")
    print("=" * 50)
    
    ensemble_config = {
        'models': [
            {
                'name': 'SVM',
                'type': 'svm',
                'params': {
                    'C': 1.0,
                    'class_weight': 'balanced',
                    'probability': True
                },
                'weight': 0.4
            },
            {
                'name': 'Random Forest',
                'type': 'rf',
                'params': {
                    'n_estimators': 100,
                    'class_weight': 'balanced',
                    'max_depth': 10
                },
                'weight': 0.3
            },
            {
                'name': 'Advanced Critic',
                'type': 'rule_based',
                'params': {
                    'clarity_weight': 0.3,
                    'completeness_weight': 0.3,
                    'structure_weight': 0.2,
                    'alignment_weight': 0.2
                },
                'weight': 0.3
            }
        ],
        'voting_method': 'weighted_average',
        'threshold': 0.65
    }
    
    print("Ensemble Configuration:")
    for model in ensemble_config['models']:
        print(f"  - {model['name']}: {model['weight']:.1f} weight")
    
    print(f"Voting method: {ensemble_config['voting_method']}")
    print(f"Decision threshold: {ensemble_config['threshold']}")
    
    return ensemble_config

def implement_feature_improvements():
    """Implement enhanced features for better performance"""
    print("\n🔍 ENHANCED FEATURE ENGINEERING")
    print("=" * 50)
    
    new_features = [
        {
            'name': 'semantic_similarity',
            'description': 'Cosine similarity between question and answer',
            'implementation': 'Use sentence transformers or word embeddings'
        },
        {
            'name': 'question_type_alignment',
            'description': 'How well answer type matches question type',
            'implementation': 'Classify question/answer types and check alignment'
        },
        {
            'name': 'factual_consistency',
            'description': 'Check for factual accuracy indicators',
            'implementation': 'Look for specific facts, numbers, citations'
        },
        {
            'name': 'completeness_score',
            'description': 'How completely the question is answered',
            'implementation': 'Keyword matching and coverage analysis'
        },
        {
            'name': 'clarity_indicators',
            'description': 'Language clarity and structure quality',
            'implementation': 'Sentence complexity, jargon detection, structure analysis'
        }
    ]
    
    print("New Features to Add:")
    for feature in new_features:
        print(f"  - {feature['name']}: {feature['description']}")
    
    return new_features

def create_performance_monitoring():
    """Create a performance monitoring system"""
    print("\n📊 PERFORMANCE MONITORING SYSTEM")
    print("=" * 50)
    
    monitoring_config = {
        'metrics_to_track': [
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'false_positive_rate',
            'false_negative_rate'
        ],
        'alert_thresholds': {
            'accuracy': 0.95,
            'false_negative_rate': 0.05,
            'false_positive_rate': 0.02
        },
        'logging': {
            'log_predictions': True,
            'log_confidence_scores': True,
            'log_feature_importance': True
        }
    }
    
    print("Monitoring Configuration:")
    print(f"  - Track {len(monitoring_config['metrics_to_track'])} metrics")
    print(f"  - Alert thresholds: {monitoring_config['alert_thresholds']}")
    print(f"  - Logging: {monitoring_config['logging']}")
    
    return monitoring_config

def generate_implementation_plan():
    """Generate a step-by-step implementation plan"""
    print("\n📋 IMPLEMENTATION PLAN")
    print("=" * 50)
    
    plan = [
        {
            'step': 1,
            'task': 'Implement class-balanced SVM',
            'effort': 'Low',
            'impact': 'High',
            'description': 'Add class weights to existing SVM model'
        },
        {
            'step': 2,
            'task': 'Optimize decision threshold',
            'effort': 'Low',
            'impact': 'Medium',
            'description': 'Adjust threshold to reduce false negatives'
        },
        {
            'step': 3,
            'task': 'Add semantic similarity features',
            'effort': 'Medium',
            'impact': 'High',
            'description': 'Implement question-answer similarity scoring'
        },
        {
            'step': 4,
            'task': 'Create ensemble model',
            'effort': 'Medium',
            'impact': 'High',
            'description': 'Combine SVM with Random Forest and rule-based critic'
        },
        {
            'step': 5,
            'task': 'Implement performance monitoring',
            'effort': 'Low',
            'impact': 'Medium',
            'description': 'Add real-time performance tracking and alerts'
        }
    ]
    
    print("Step-by-Step Implementation:")
    for item in plan:
        print(f"{item['step']}. {item['task']} ({item['effort']} effort, {item['impact']} impact)")
        print(f"   {item['description']}")
    
    return plan

def main():
    """Main function to analyze and improve critic performance"""
    print("🚀 CRITIC PERFORMANCE IMPROVEMENT ANALYSIS")
    print("=" * 60)
    
    # Analyze current performance
    current_metrics = load_and_analyze_data()
    
    # Identify improvement areas
    identify_improvement_areas()
    
    # Create improved critic design
    improved_critic = create_improved_critic()
    
    # Optimize decision threshold
    optimal_threshold = optimize_decision_threshold()
    
    # Design ensemble model
    ensemble_config = create_ensemble_critic()
    
    # Plan feature improvements
    new_features = implement_feature_improvements()
    
    # Create monitoring system
    monitoring_config = create_performance_monitoring()
    
    # Generate implementation plan
    implementation_plan = generate_implementation_plan()
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("\n💡 RECOMMENDATION:")
    print("Your current 96% accuracy is excellent, but there's room for improvement.")
    print("Focus on reducing false negatives (good answers marked as needing improvement)")
    print("and implementing the ensemble approach for more robust performance.")

if __name__ == "__main__":
    main() 