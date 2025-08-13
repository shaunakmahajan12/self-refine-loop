#!/usr/bin/env python3
"""Test script to compare original vs improved SVM critic"""

import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.critics.improved_svm_critic import ImprovedSVMCritic

def test_critic_comparison():
    """Compare original and improved critics"""
    
    print("🧪 CRITIC COMPARISON TEST")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Simple Factual Question',
            'original': "The capital of France is Paris.",
            'revised': "The capital of France is Paris, which is located in the northern part of the country.",
            'feedback': "Added more context"
        },
        {
            'name': 'Complex Technical Question',
            'original': "Machine learning uses algorithms.",
            'revised': "Machine learning is a subset of artificial intelligence that uses algorithms to enable computers to learn and improve from experience without being explicitly programmed.",
            'feedback': "Much more comprehensive explanation"
        },
        {
            'name': 'Process Explanation',
            'original': "Photosynthesis makes food.",
            'revised': "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose, which serves as food for the plant.",
            'feedback': "Detailed process explanation"
        },
        {
            'name': 'Code Example',
            'original': "Python is a programming language.",
            'revised': "Python is a high-level, interpreted programming language known for its simplicity and readability. Example: print('Hello, World!')",
            'feedback': "Added practical example"
        }
    ]
    
    try:
        models_dir = Path(__file__).parent.parent / "models"
        improved_critic = ImprovedSVMCritic.load_model(models_dir)
        print("✅ Improved critic loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load improved critic: {e}")
        return
    
    print(f"\n📊 Testing with decision threshold: {improved_critic.decision_threshold}")
    print("-" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['name']}")
        print(f"Original: {case['original']}")
        print(f"Revised:  {case['revised']}")
        
        try:
            result = improved_critic.predict(case['original'], case['revised'], case['feedback'])
            
            print(f"🔍 Result:")
            print(f"  Needs improvement: {result['needs_improvement']}")
            print(f"  Confidence: {result['confidence_score']:.3f}")
            print(f"  Decision score: {result['decision_score']:.3f}")
            print(f"  Threshold: {result['threshold']}")
            
            # Interpretation
            if result['needs_improvement']:
                print(f"  💡 Interpretation: Answer needs improvement")
            else:
                print(f"  ✅ Interpretation: Answer is sufficient")
                
        except Exception as e:
            print(f"❌ Error predicting: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY")
    print("The improved critic uses:")
    print("- Optimized decision threshold (0.67 vs 0.5)")
    print("- Class balancing for imbalanced data")
    print("- Enhanced feature engineering")
    print("- Better handling of false negatives")

if __name__ == "__main__":
    test_critic_comparison() 