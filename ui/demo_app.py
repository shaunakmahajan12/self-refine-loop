import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import joblib

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.append(str(SRC_DIR))

from core.critics.advanced_critic import AdvancedCritic
from utils.config import LOGS_DIR, MODELS_DIR, get_api_key, GEMINI_MODEL
from utils.csv_loader import load_all_results

PRIMARY_COLOR = "#1f77b4"
BACKGROUND_COLOR = "#f0f2f6"
FEEDBACK_COLOR = "#e8f4fd"

st.set_page_config(
    page_title="Self-Refinement Loop Demo",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: {PRIMARY_COLOR};
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: {BACKGROUND_COLOR};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {PRIMARY_COLOR};
    }}
    .feedback-box {{
        background-color: {FEEDBACK_COLOR};
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid {PRIMARY_COLOR};
    }}
</style>
""", unsafe_allow_html=True)

with st.spinner("Loading data..."):
    time.sleep(1)

@st.cache_resource
def load_models():
    """Load trained models and API configuration"""
    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
    except FileNotFoundError:
        st.error("Google API key not found. Set GOOGLE_API_KEY in your environment or .env.")
        return None, None, None, None, None
    
    # Load improved critic models
    try:
        from core.critics.improved_svm_critic import ImprovedSVMCritic
        improved_critic = ImprovedSVMCritic.load_model(MODELS_DIR)
        critic = improved_critic
        vectorizer = improved_critic.vectorizer
        text_extractor = improved_critic.text_extractor
        st.success("✅ Improved critic loaded successfully")
    except Exception as e:
        st.warning(f"Improved critic not found: {e}. Using rule-based critic.")
        critic = None
        vectorizer = None
        text_extractor = None
    
    # Initialize advanced critic
    advanced_critic = AdvancedCritic()
    
    return model, advanced_critic, critic, vectorizer, text_extractor

@st.cache_data
def load_existing_results():
    """Load existing refinement results for analysis"""
    return load_all_results(LOGS_DIR)

def call_gemini_safe(model, prompt, max_retries=3):
    """Safely call Gemini with retry logic"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                st.error(f"Error calling Gemini: {str(e)}")
                return None
    return None

def extract_features_simple(answer, vectorizer=None, text_extractor=None):
    """Extract features for the simple critic"""
    if vectorizer is None or text_extractor is None:
        # Fallback to simple rule-based approach
        return np.array([[len(answer), len(answer.split()), 1 if len(answer) > 100 else 0]])
    
    # Use the trained models
    import re
    
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s\`\#\-\*\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    clean_text = preprocess_text(answer)
    dummy_feedback = ""
    clean_feedback = preprocess_text(dummy_feedback)
    
    answer_features = text_extractor.transform(pd.Series([answer]))
    feedback_features = text_extractor.transform(pd.Series([dummy_feedback]))
    
    answer_features = answer_features.fillna(0)
    feedback_features = feedback_features.fillna(0)
    
    relative_features = pd.DataFrame()
    for col in answer_features.columns:
        relative_features[f'relative_{col}'] = 0
    
    X_numeric = pd.concat([
        answer_features.add_prefix('original_'),
        answer_features.add_prefix('revised_'),
        feedback_features.add_prefix('feedback_'),
        relative_features
    ], axis=1)
    
    X_numeric = X_numeric.fillna(0)
    text_features = vectorizer.transform([clean_text + ' ' + clean_text + ' ' + clean_feedback])
    combined_features = np.hstack([text_features.toarray(), X_numeric])
    
    if np.isnan(combined_features).any():
        combined_features = np.nan_to_num(combined_features, nan=0.0)
    
    return combined_features

def run_refinement_loop(prompt, model, advanced_critic, simple_critic=None, vectorizer=None, text_extractor=None, max_iterations=5):
    """Run the self-refinement loop"""
    history = []
    
    # Initial answer
    initial_prompt = f"Please provide a clear and accurate answer to the following question.\n\nQuestion: {prompt}"
    current_answer = call_gemini_safe(model, initial_prompt)
    
    if current_answer is None:
        return history
    
    history.append({
        'iteration': 1,
        'answer': current_answer,
        'feedback': 'Initial answer generated',
        'critic_score': None,
        'needs_improvement': True
    })
    
    for iteration in range(2, max_iterations + 1):
        # Get detailed feedback from advanced critic
        detailed_feedback = advanced_critic.generate_detailed_feedback(prompt, current_answer)
        
        # Get simple critic prediction if available
        if simple_critic is not None and hasattr(simple_critic, 'predict'):
            # Check if it's the improved critic
            if hasattr(simple_critic, 'decision_threshold'):
                # Use improved critic prediction
                result = simple_critic.predict("", current_answer, "")
                needs_improvement = result['needs_improvement']
            else:
                # Use old critic prediction
                features = extract_features_simple(current_answer, vectorizer, text_extractor)
                simple_prediction = simple_critic.predict(features)[0]
                needs_improvement = simple_prediction == 1
        else:
            needs_improvement = detailed_feedback['needs_improvement']
        
        # Update history
        history[-1]['critic_score'] = detailed_feedback['overall_score']
        history[-1]['needs_improvement'] = needs_improvement
        history[-1]['feedback'] = f"Advanced Critic Score: {detailed_feedback['overall_score']:.2f}"
        
        if not needs_improvement:
            break
        
        # Generate improvement prompt
        suggestions = detailed_feedback['suggestions']
        improvement_prompt = f"""Question: {prompt}
Previous Answer: {current_answer}

The answer needs improvement. Here are specific suggestions:
{chr(10).join(f"• {suggestion}" for suggestion in suggestions[:3])}

Please provide an improved version of the answer that addresses these suggestions."""
        
        # Get revised answer
        revised_answer = call_gemini_safe(model, improvement_prompt)
        
        if revised_answer is None:
            break
        
        history.append({
            'iteration': iteration,
            'answer': revised_answer,
            'feedback': f"Revision based on: {', '.join(suggestions[:2])}",
            'critic_score': None,
            'needs_improvement': True
        })
        
        current_answer = revised_answer
        time.sleep(1)  # Rate limiting
    
    return history

def create_visualizations(history, detailed_feedback):
    """Create interactive visualizations"""
    
    # 1. Iteration progress
    if len(history) > 1:
        iterations = [h['iteration'] for h in history]
        scores = [h.get('critic_score', 0) for h in history]
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=iterations, y=scores,
            mode='lines+markers',
            name='Critic Score',
            line=dict(color=PRIMARY_COLOR, width=3),
            marker=dict(size=8)
        ))
        fig1.update_layout(
            title='Critic Score Progress',
            xaxis_title='Iteration',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    # 2. Detailed feedback radar chart
    if detailed_feedback and 'categories' in detailed_feedback:
        categories = list(detailed_feedback['categories'].keys())
        scores = list(detailed_feedback['categories'].values())
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Current Answer',
            line_color=PRIMARY_COLOR
        ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title='Answer Quality Breakdown',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

def main():
    """Main demo application"""
    
    # Header
    st.markdown('<div class="main-header">Self-Refinement Loop Demo</div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        model, advanced_critic, simple_critic, vectorizer, text_extractor = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🎛️ Configuration")
    
    # Load existing results
    try:
        existing_results = load_existing_results()
    except Exception as e:
        st.warning(f"Could not load existing results: {str(e)}")
        existing_results = {}
    
    # Demo mode selection
    demo_mode = st.sidebar.selectbox(
        "Demo Mode",
        ["Interactive Refinement", "Results Analysis"]
    )
    
    if demo_mode == "Interactive Refinement":
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        # Initialize session state for prompt
        if 'current_prompt' not in st.session_state:
            st.session_state.current_prompt = ""
        
        with col1:
            user_prompt = st.text_area(
                "Enter your question:",
                value=st.session_state.current_prompt,
                placeholder="e.g., What is machine learning and how does it work?",
                height=100,
                key="prompt_input"
            )
        
        with col2:
            max_iterations = st.slider("Max Iterations", 1, 10, 5)
            st.markdown("### Quick Examples")
            
            example_prompts = [
                "What is the difference between supervised and unsupervised learning?",
                "How does a neural network learn?",
                "Explain the concept of overfitting in machine learning",
                "What are the advantages of using Python for data science?"
            ]
            
            # Create buttons for examples
            for i, example in enumerate(example_prompts):
                # Truncate long examples for button text
                button_text = example[:50] + "..." if len(example) > 50 else example
                if st.button(button_text, key=f"ex_{i}"):
                    st.session_state.current_prompt = example
                    st.rerun()
            
            # Add a clear button
            if st.button("Clear Prompt", key="clear_prompt"):
                st.session_state.current_prompt = ""
                st.rerun()
        
        # Run refinement
        if st.button("🚀 Start Refinement", type="primary") and user_prompt:
            with st.spinner("Running refinement loop..."):
                history = run_refinement_loop(
                    user_prompt, model, advanced_critic, 
                    simple_critic, vectorizer, text_extractor, 
                    max_iterations
                )
            
            if history:
                # Display results
                st.success(f"✅ Refinement completed in {len(history)} iterations!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Iterations", len(history))
                with col2:
                    final_score = history[-1].get('critic_score', 0)
                    st.metric("Final Score", f"{final_score:.2f}")
                with col3:
                    improvement = len(history) > 1
                    st.metric("Improved", "Yes" if improvement else "No")
                
                # Display iterations
                st.subheader("📝 Refinement History")
                
                for i, step in enumerate(history):
                    with st.expander(f"Iteration {step['iteration']}", expanded=i==len(history)-1):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown("**Answer:**")
                            st.write(step['answer'])
                        
                        with col2:
                            if step.get('critic_score') is not None:
                                st.metric("Score", f"{step['critic_score']:.2f}")
                            
                            if step['needs_improvement']:
                                st.warning("Needs improvement")
                            else:
                                st.success("Sufficient")
                        
                        st.markdown(f"**Feedback:** {step['feedback']}")
                
                # Visualizations
                if len(history) > 1:
                    st.subheader("📊 Analysis")
                    
                    # Get detailed feedback for final answer
                    final_feedback = advanced_critic.generate_detailed_feedback(
                        user_prompt, history[-1]['answer']
                    )
                    
                    create_visualizations(history, final_feedback)
                    
                    # Show detailed feedback
                    if final_feedback['suggestions']:
                        st.markdown("### 💡 Improvement Suggestions")
                        for suggestion in final_feedback['suggestions']:
                            st.markdown(f"• {suggestion}")
    
    elif demo_mode == "Results Analysis":
        
        if not existing_results:
            st.warning("No existing results found. Run some refinements first!")
            return
        
        # Load and display existing results
        if 'final_refinement_results' in existing_results:
            df = existing_results['final_refinement_results']
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Responses", len(df))
            
            with col2:
                unique_prompts = df['prompt'].nunique()
                st.metric("Unique Prompts", unique_prompts)
            
            with col3:
                success_rate = (df['feedback'] == 'Critic: Answer is sufficient.').sum() / len(df) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col4:
                avg_iterations = df.groupby('prompt').size().mean()
                st.metric("Avg Iterations", f"{avg_iterations:.1f}")
            
            # Iteration distribution
            st.subheader("📈 Iteration Distribution")
            iteration_counts = df.groupby('prompt').size()
            
            fig = px.histogram(
                x=iteration_counts.values,
                nbins=10,
                title="Distribution of Iterations Needed",
                labels={'x': 'Number of Iterations', 'y': 'Number of Prompts'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample results
            st.subheader("📋 Sample Results")
            st.dataframe(df.head(10), use_container_width=True)
    
    elif demo_mode == "Model Comparison":
        
        st.info("This section would compare different critic models and their performance.")
        
        # Placeholder for model comparison
        st.markdown("""
        ### Planned Features:
        - Compare SVM vs Random Forest critics
        - Analyze performance by question type
        - A/B testing of different refinement strategies
        - Ensemble model performance
        """)

if __name__ == "__main__":
    main()