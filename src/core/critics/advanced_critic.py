import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedCritic:
    """
    Advanced critic system that provides detailed feedback and improvement suggestions
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.vectorizer = None
        self.feedback_categories = [
            'clarity', 'completeness', 'accuracy', 'structure', 'depth'
        ]
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def extract_detailed_features(self, prompt, answer, feedback=""):
        """Extract comprehensive features for detailed analysis"""
        features = {}
        
        # Text preprocessing
        clean_answer = self._preprocess_text(answer)
        clean_prompt = self._preprocess_text(prompt)
        clean_feedback = self._preprocess_text(feedback)
        
        # Basic text features
        features['answer_length'] = len(answer)
        features['answer_word_count'] = len(answer.split())
        features['prompt_length'] = len(prompt)
        features['prompt_word_count'] = len(prompt.split())
        
        # Complexity features
        features['avg_sentence_length'] = self._avg_sentence_length(answer)
        features['unique_words_ratio'] = len(set(answer.lower().split())) / max(len(answer.split()), 1)
        features['technical_terms'] = self._count_technical_terms(answer)
        
        # Content quality indicators
        features['has_examples'] = 1 if any(word in answer.lower() for word in ['example', 'instance', 'such as']) else 0
        features['has_steps'] = 1 if any(word in answer.lower() for word in ['step', 'process', 'first', 'then']) else 0
        features['has_definitions'] = 1 if any(word in answer.lower() for word in ['defined', 'definition', 'means']) else 0
        
        # Question-answer alignment
        features['question_type'] = self._classify_question_type(prompt)
        features['answer_type'] = self._classify_answer_type(answer)
        features['alignment_score'] = self._calculate_alignment(prompt, answer)
        
        # Specific quality indicators
        features['clarity_score'] = self._assess_clarity(answer)
        features['completeness_score'] = self._assess_completeness(prompt, answer)
        features['structure_score'] = self._assess_structure(answer)
        
        return features
    
    def _preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\`\#\-\*\.]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _avg_sentence_length(self, text):
        """Calculate average sentence length"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        return sum(len(s.split()) for s in sentences) / len(sentences)
    
    def _count_technical_terms(self, text):
        """Count technical/specialized terms"""
        technical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b\d+(?:\.\d+)?\b',  # Numbers
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        count = 0
        for pattern in technical_patterns:
            count += len(re.findall(pattern, text))
        return count
    
    def _classify_question_type(self, prompt):
        """Classify the type of question"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['what is', 'define', 'explain']):
            return 'definition'
        elif any(word in prompt_lower for word in ['how', 'process', 'mechanism']):
            return 'process'
        elif any(word in prompt_lower for word in ['compare', 'difference', 'vs']):
            return 'comparison'
        elif any(word in prompt_lower for word in ['why', 'cause', 'reason']):
            return 'explanation'
        elif any(word in prompt_lower for word in ['calculate', 'solve', 'equation']):
            return 'calculation'
        else:
            return 'factual'
    
    def _classify_answer_type(self, answer):
        """Classify the type of answer"""
        answer_lower = answer.lower()
        
        if any(word in answer_lower for word in ['step', 'process', 'first', 'then']):
            return 'procedural'
        elif any(word in answer_lower for word in ['example', 'instance', 'such as']):
            return 'exemplary'
        elif any(word in answer_lower for word in ['because', 'due to', 'as a result']):
            return 'explanatory'
        elif len(answer.split()) < 20:
            return 'concise'
        else:
            return 'detailed'
    
    def _calculate_alignment(self, prompt, answer):
        """Calculate how well the answer aligns with the question type"""
        question_type = self._classify_question_type(prompt)
        answer_type = self._classify_answer_type(answer)
        
        alignment_scores = {
            ('definition', 'detailed'): 0.9,
            ('definition', 'concise'): 0.7,
            ('process', 'procedural'): 0.9,
            ('process', 'detailed'): 0.8,
            ('comparison', 'detailed'): 0.9,
            ('comparison', 'exemplary'): 0.8,
            ('explanation', 'explanatory'): 0.9,
            ('explanation', 'detailed'): 0.8,
            ('calculation', 'concise'): 0.9,
            ('factual', 'concise'): 0.8,
            ('factual', 'detailed'): 0.7,
        }
        
        return alignment_scores.get((question_type, answer_type), 0.5)
    
    def _assess_clarity(self, answer):
        """Assess the clarity of the answer"""
        score = 0.5  # Base score
        
        # Positive indicators
        if any(word in answer.lower() for word in ['clearly', 'specifically', 'in other words']):
            score += 0.2
        if self._avg_sentence_length(answer) < 25:
            score += 0.1
        if len(answer.split()) > 50:
            score += 0.1
        
        # Negative indicators
        if any(word in answer.lower() for word in ['complicated', 'complex', 'difficult']):
            score -= 0.1
        if self._avg_sentence_length(answer) > 40:
            score -= 0.2
        
        return min(max(score, 0), 1)
    
    def _assess_completeness(self, prompt, answer):
        """Assess how complete the answer is"""
        score = 0.5  # Base score
        
        # Check if answer addresses key question words
        question_words = set(prompt.lower().split())
        answer_words = set(answer.lower().split())
        
        # Simple overlap metric
        overlap = len(question_words.intersection(answer_words)) / max(len(question_words), 1)
        score += overlap * 0.3
        
        # Length consideration
        if len(answer.split()) > 100:
            score += 0.2
        elif len(answer.split()) > 50:
            score += 0.1
        
        return min(max(score, 0), 1)
    
    def _assess_structure(self, answer):
        """Assess the structure and organization of the answer"""
        score = 0.5  # Base score
        
        # Check for structural elements
        if any(word in answer.lower() for word in ['first', 'second', 'third', 'finally']):
            score += 0.2
        if any(word in answer.lower() for word in ['however', 'although', 'but', 'nevertheless']):
            score += 0.1
        if any(word in answer.lower() for word in ['in conclusion', 'to summarize', 'overall']):
            score += 0.1
        
        # Check for paragraphs
        paragraphs = answer.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.1
        
        return min(max(score, 0), 1)
    
    def generate_detailed_feedback(self, prompt, answer):
        """Generate detailed feedback with specific improvement suggestions"""
        features = self.extract_detailed_features(prompt, answer)
        
        feedback = {
            'overall_score': 0,
            'categories': {},
            'suggestions': [],
            'needs_improvement': True
        }
        
        # Calculate category scores
        feedback['categories']['clarity'] = features['clarity_score']
        feedback['categories']['completeness'] = features['completeness_score']
        feedback['categories']['structure'] = features['structure_score']
        feedback['categories']['alignment'] = features['alignment_score']
        
        # Overall score (weighted average)
        weights = {'clarity': 0.3, 'completeness': 0.3, 'structure': 0.2, 'alignment': 0.2}
        feedback['overall_score'] = sum(
            feedback['categories'][cat] * weights[cat] 
            for cat in weights.keys()
        )
        
        # Generate specific suggestions
        if features['clarity_score'] < 0.6:
            feedback['suggestions'].append(
                "Improve clarity by using shorter sentences and simpler language"
            )
        
        if features['completeness_score'] < 0.6:
            feedback['suggestions'].append(
                "Provide more comprehensive coverage of the topic"
            )
        
        if features['structure_score'] < 0.6:
            feedback['suggestions'].append(
                "Organize the answer with clear sections and logical flow"
            )
        
        if features['alignment_score'] < 0.6:
            feedback['suggestions'].append(
                f"Better align the answer with the question type ({features['question_type']})"
            )
        
        if features['has_examples'] == 0 and features['question_type'] in ['definition', 'process']:
            feedback['suggestions'].append(
                "Include concrete examples to illustrate the concept"
            )
        
        if features['has_steps'] == 0 and features['question_type'] == 'process':
            feedback['suggestions'].append(
                "Break down the process into clear steps"
            )
        
        # Determine if improvement is needed
        feedback['needs_improvement'] = feedback['overall_score'] < 0.7 or len(feedback['suggestions']) > 2
        
        return feedback
    
    def train(self, training_data):
        """Train the critic model on labeled data"""
        # This would be implemented with actual training data
        # For now, we'll use rule-based approach
        print("Training advanced critic model...")
        print("Model ready for detailed feedback generation")
    
    def save_model(self, path):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feedback_categories': self.feedback_categories
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.feedback_categories = model_data['feedback_categories']
        print(f"Model loaded from {path}")

def test_advanced_critic():
    """Test the advanced critic with sample data"""
    critic = AdvancedCritic()
    
    # Sample test cases
    test_cases = [
        {
            'prompt': 'What is machine learning?',
            'answer': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.'
        },
        {
            'prompt': 'How does photosynthesis work?',
            'answer': 'Photosynthesis is the process by which plants convert light energy into chemical energy. First, chlorophyll absorbs sunlight. Then, this energy is used to convert carbon dioxide and water into glucose and oxygen. The process occurs in two main stages: light-dependent reactions and the Calvin cycle.'
        },
        {
            'prompt': 'Compare Python and JavaScript',
            'answer': 'Python and JavaScript are both programming languages but serve different purposes. Python is primarily used for backend development, data science, and automation. JavaScript is mainly used for web development, both frontend and backend. Python has simpler syntax and is more readable, while JavaScript is more flexible and runs in browsers.'
        }
    ]
    
    print("🧪 TESTING ADVANCED CRITIC")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}")
        print(f"Prompt: {case['prompt']}")
        print(f"Answer: {case['answer'][:100]}...")
        
        feedback = critic.generate_detailed_feedback(case['prompt'], case['answer'])
        
        print(f"\n📊 Feedback:")
        print(f"Overall Score: {feedback['overall_score']:.2f}")
        print(f"Needs Improvement: {feedback['needs_improvement']}")
        
        print("Category Scores:")
        for category, score in feedback['categories'].items():
            print(f"  {category.title()}: {score:.2f}")
        
        if feedback['suggestions']:
            print("Suggestions:")
            for suggestion in feedback['suggestions']:
                print(f"  • {suggestion}")
        
        print("-" * 30)

if __name__ == "__main__":
    test_advanced_critic() 