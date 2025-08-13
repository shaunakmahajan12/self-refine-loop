import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
from collections import Counter

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data():
    """Load and clean all available data files"""
    script_dir = Path(__file__).parent
    logs_dir = script_dir.parent / "logs"
    
    data = {}
    
    # Load different result files
    files_to_load = [
        "final_refinement_results.csv",
        "test_refine_outputs.csv", 
        "evaluated_outputs.csv",
        "test_results_summary.csv"
    ]
    
    for file in files_to_load:
        file_path = logs_dir / file
        if file_path.exists():
            try:
                # Try with default settings first
                data[file.replace('.csv', '')] = pd.read_csv(file_path)
                print(f"✅ Loaded {file}")
            except pd.errors.ParserError:
                # If that fails, try with different quoting options
                try:
                    data[file.replace('.csv', '')] = pd.read_csv(file_path, quoting=1)  # QUOTE_ALL
                    print(f"✅ Loaded {file} (with QUOTE_ALL)")
                except:
                    data[file.replace('.csv', '')] = pd.read_csv(file_path, quoting=3)  # QUOTE_NONE
                    print(f"✅ Loaded {file} (with QUOTE_NONE)")
        else:
            print(f"⚠️  {file} not found")
    
    return data

def analyze_refinement_patterns(data):
    """Analyze how many iterations were needed for different types of questions"""
    if 'final_refinement_results' not in data:
        print("❌ No final refinement results found")
        return
    
    df = data['final_refinement_results']
    
    # Count iterations per prompt
    iteration_counts = df.groupby('prompt').size()
    
    print("\n📊 REFINEMENT PATTERN ANALYSIS")
    print("=" * 50)
    print(f"Total unique prompts: {len(iteration_counts)}")
    print(f"Average iterations needed: {iteration_counts.mean():.2f}")
    print(f"Max iterations needed: {iteration_counts.max()}")
    print(f"Min iterations needed: {iteration_counts.min()}")
    
    # Distribution of iterations
    print(f"\nIteration Distribution:")
    for i in range(1, int(iteration_counts.max()) + 1):
        count = (iteration_counts == i).sum()
        percentage = (count / len(iteration_counts)) * 100
        print(f"  {i} iteration(s): {count} prompts ({percentage:.1f}%)")
    
    # Questions that needed most refinement
    print(f"\n🔍 Questions needing most refinement:")
    most_refined = iteration_counts.nlargest(5)
    for prompt, iterations in most_refined.items():
        print(f"  {iterations} iterations: {prompt[:80]}...")
    
    return iteration_counts

def analyze_answer_quality(data):
    """Analyze answer quality improvements"""
    if 'final_refinement_results' not in data:
        print("❌ No final refinement results found")
        return
    
    df = data['final_refinement_results']
    
    print("\n📈 ANSWER QUALITY ANALYSIS")
    print("=" * 50)
    
    # Analyze feedback patterns
    feedback_counts = df['feedback'].value_counts()
    print(f"Feedback Distribution:")
    for feedback, count in feedback_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {feedback}: {count} ({percentage:.1f}%)")
    
    # Success rate (answers that were sufficient)
    success_count = (df['feedback'] == 'Critic: Answer is sufficient.').sum()
    success_rate = (success_count / len(df)) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}% ({success_count}/{len(df)})")

def analyze_metrics(data):
    """Analyze BLEU and ROUGE metrics if available"""
    if 'evaluated_outputs' not in data:
        print("❌ No evaluated outputs found")
        return
    
    df = data['evaluated_outputs']
    
    print("\n📊 METRICS ANALYSIS")
    print("=" * 50)
    
    if 'BLEU_Original_vs_Revised' in df.columns:
        bleu_scores = df['BLEU_Original_vs_Revised']
        print(f"BLEU Scores:")
        print(f"  Mean: {bleu_scores.mean():.3f}")
        print(f"  Median: {bleu_scores.median():.3f}")
        print(f"  Std: {bleu_scores.std():.3f}")
        print(f"  Min: {bleu_scores.min():.3f}")
        print(f"  Max: {bleu_scores.max():.3f}")
    
    if 'ROUGE-L_Original_vs_Revised' in df.columns:
        rouge_scores = df['ROUGE-L_Original_vs_Revised']
        print(f"\nROUGE-L Scores:")
        print(f"  Mean: {rouge_scores.mean():.3f}")
        print(f"  Median: {rouge_scores.median():.3f}")
        print(f"  Std: {rouge_scores.std():.3f}")
        print(f"  Min: {rouge_scores.min():.3f}")
        print(f"  Max: {rouge_scores.max():.3f}")

def categorize_questions(prompts):
    """Categorize questions by type"""
    categories = {
        'definition': ['what is', 'define', 'explain'],
        'comparison': ['compare', 'difference', 'vs', 'versus'],
        'how': ['how', 'process', 'mechanism'],
        'why': ['why', 'cause', 'reason'],
        'factual': ['what is the', 'who', 'when', 'where'],
        'mathematical': ['calculate', 'solve', 'equation', 'formula'],
        'programming': ['code', 'program', 'algorithm', 'function', 'loop', 'git', 'sql']
    }
    
    question_types = []
    for prompt in prompts:
        prompt_lower = prompt.lower()
        categorized = False
        for category, keywords in categories.items():
            if any(keyword in prompt_lower for keyword in keywords):
                question_types.append(category)
                categorized = True
                break
        if not categorized:
            question_types.append('other')
    
    return question_types

def analyze_by_question_type(data):
    """Analyze performance by question type"""
    if 'final_refinement_results' not in data:
        print("❌ No final refinement results found")
        return
    
    df = data['final_refinement_results']
    
    # Categorize questions
    df['question_type'] = categorize_questions(df['prompt'])
    
    print("\n📋 PERFORMANCE BY QUESTION TYPE")
    print("=" * 50)
    
    # Group by question type
    type_analysis = df.groupby('question_type').agg({
        'feedback': lambda x: (x == 'Critic: Answer is sufficient.').sum() / len(x) * 100,
        'prompt': 'count'
    }).rename(columns={'feedback': 'success_rate', 'prompt': 'count'})
    
    type_analysis = type_analysis.sort_values('success_rate', ascending=False)
    
    for qtype, row in type_analysis.iterrows():
        print(f"{qtype.title()}: {row['success_rate']:.1f}% success ({row['count']} questions)")

def create_visualizations(data):
    """Create comprehensive visualizations"""
    if 'final_refinement_results' not in data:
        print("❌ No final refinement results found")
        return
    
    df = data['final_refinement_results']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Self-Refinement Loop Analysis', fontsize=16, fontweight='bold')
    
    # 1. Iteration distribution
    iteration_counts = df.groupby('prompt').size()
    axes[0, 0].hist(iteration_counts.values, bins=range(1, iteration_counts.max() + 2), 
                   alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Iterations Needed')
    axes[0, 0].set_xlabel('Number of Iterations')
    axes[0, 0].set_ylabel('Number of Prompts')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Success rate by iteration
    success_by_iteration = df.groupby('iteration')['feedback'].apply(
        lambda x: (x == 'Critic: Answer is sufficient.').sum() / len(x) * 100
    )
    axes[0, 1].plot(success_by_iteration.index, success_by_iteration.values, 
                   marker='o', linewidth=2, markersize=8)
    axes[0, 1].set_title('Success Rate by Iteration')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Answer length analysis
    df['answer_length'] = df['answer'].str.len()
    df['revised_length'] = df['revised'].str.len()
    
    axes[1, 0].scatter(df['answer_length'], df['revised_length'], alpha=0.6)
    axes[1, 0].plot([df['answer_length'].min(), df['answer_length'].max()], 
                   [df['answer_length'].min(), df['answer_length'].max()], 
                   'r--', alpha=0.8, label='No change')
    axes[1, 0].set_title('Answer Length: Original vs Revised')
    axes[1, 0].set_xlabel('Original Answer Length')
    axes[1, 0].set_ylabel('Revised Answer Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Question type analysis
    df['question_type'] = categorize_questions(df['prompt'])
    type_success = df.groupby('question_type')['feedback'].apply(
        lambda x: (x == 'Critic: Answer is sufficient.').sum() / len(x) * 100
    ).sort_values(ascending=True)
    
    axes[1, 1].barh(range(len(type_success)), type_success.values)
    axes[1, 1].set_yticks(range(len(type_success)))
    axes[1, 1].set_yticklabels([t.title() for t in type_success.index])
    axes[1, 1].set_title('Success Rate by Question Type')
    axes[1, 1].set_xlabel('Success Rate (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "logs" / "refinement_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    plt.show()

def generate_recommendations(data):
    """Generate actionable recommendations based on analysis"""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    if 'final_refinement_results' in data:
        df = data['final_refinement_results']
        iteration_counts = df.groupby('prompt').size()
        success_rate = (df['feedback'] == 'Critic: Answer is sufficient.').sum() / len(df) * 100
        
        print(f"1. **Current Performance**: {success_rate:.1f}% success rate")
        
        if iteration_counts.mean() > 2:
            print("2. **Refinement Efficiency**: Consider improving initial answer quality to reduce iterations")
        
        # Analyze question types that perform poorly
        df['question_type'] = categorize_questions(df['prompt'])
        type_success = df.groupby('question_type')['feedback'].apply(
            lambda x: (x == 'Critic: Answer is sufficient.').sum() / len(x) * 100
        )
        
        poor_performers = type_success[type_success < 50]
        if not poor_performers.empty:
            print(f"3. **Weak Areas**: Focus on improving {', '.join(poor_performers.index)} questions")
        
        print("4. **Next Steps**:")
        print("   - Implement adaptive refinement strategies")
        print("   - Add domain-specific critic models")
        print("   - Experiment with different feedback mechanisms")
        print("   - Consider ensemble critic approaches")

def main():
    """Main analysis function"""
    print("🔍 SELF-REFINEMENT LOOP ANALYSIS")
    print("=" * 60)
    
    # Load data
    data = load_and_clean_data()
    
    if not data:
        print("❌ No data files found. Please run the refinement scripts first.")
        return
    
    # Run analyses
    analyze_refinement_patterns(data)
    analyze_answer_quality(data)
    analyze_metrics(data)
    analyze_by_question_type(data)
    
    # Create visualizations
    create_visualizations(data)
    
    # Generate recommendations
    generate_recommendations(data)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 