import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv_safely(file_path):
    """Load CSV with proper handling of quoted fields with newlines"""
    try:
        # Try with different engines and quoting options
        df = pd.read_csv(file_path, engine='python', quoting=1)
        return df
    except:
        try:
            df = pd.read_csv(file_path, engine='c', quoting=1)
            return df
        except:
            # Last resort: read manually
            return read_csv_manually(file_path)

def read_csv_manually(file_path):
    """Manual CSV reading for problematic files"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    header = lines[0].strip().split(',')
    
    current_row = []
    in_quoted_field = False
    quoted_content = []
    
    for line in lines[1:]:
        if not in_quoted_field:
            # Check if line starts a quoted field
            if line.strip().startswith('"'):
                in_quoted_field = True
                quoted_content = [line.strip()[1:]]  # Remove opening quote
            else:
                # Regular line, split by comma
                fields = line.strip().split(',')
                if len(fields) == 5:  # Expected number of columns
                    data.append(fields)
        else:
            # We're in a quoted field
            if line.strip().endswith('"'):
                # End of quoted field
                quoted_content.append(line.strip()[:-1])  # Remove closing quote
                current_row.extend(quoted_content)
                in_quoted_field = False
                
                # Get remaining fields
                remaining = line.strip().split(',')[-1]  # Last field after quote
                if remaining and not remaining.endswith('"'):
                    current_row.append(remaining)
                
                if len(current_row) == 5:
                    data.append(current_row)
                current_row = []
                quoted_content = []
            else:
                # Continue quoted field
                quoted_content.append(line.strip())
    
    return pd.DataFrame(data, columns=header)

def analyze_refinement_results():
    """Analyze the refinement results"""
    script_dir = Path(__file__).parent
    logs_dir = script_dir.parent / "logs"
    
    # Load the main results file
    results_file = logs_dir / "final_refinement_results.csv"
    
    if not results_file.exists():
        print("❌ No refinement results found!")
        return
    
    print("🔍 Loading refinement results...")
    df = load_csv_safely(results_file)
    
    print(f"✅ Loaded {len(df)} rows of data")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Basic statistics
    print("\n📈 BASIC STATISTICS")
    print("=" * 50)
    
    # Count iterations per prompt
    iteration_counts = df.groupby('prompt').size()
    print(f"Total unique prompts: {len(iteration_counts)}")
    print(f"Average iterations needed: {iteration_counts.mean():.2f}")
    print(f"Max iterations needed: {iteration_counts.max()}")
    print(f"Min iterations needed: {iteration_counts.min()}")
    
    # Success rate
    success_count = (df['feedback'] == 'Critic: Answer is sufficient.').sum()
    success_rate = (success_count / len(df)) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}% ({success_count}/{len(df)})")
    
    # Iteration distribution
    print(f"\n📊 ITERATION DISTRIBUTION")
    print("=" * 50)
    
    for i in range(1, int(iteration_counts.max()) + 1):
        count = (iteration_counts == i).sum()
        percentage = (count / len(iteration_counts)) * 100
        print(f"{i} iteration(s): {count} prompts ({percentage:.1f}%)")
    
    # Questions that needed most refinement
    print(f"\n🔍 QUESTIONS NEEDING MOST REFINEMENT")
    print("=" * 50)
    
    most_refined = iteration_counts.nlargest(5)
    for prompt, iterations in most_refined.items():
        print(f"{iterations} iterations: {prompt[:80]}...")
    
    # Answer length analysis
    print(f"\n📏 ANSWER LENGTH ANALYSIS")
    print("=" * 50)
    
    df['answer_length'] = df['answer'].str.len()
    df['revised_length'] = df['revised'].str.len()
    
    print(f"Average original answer length: {df['answer_length'].mean():.0f} characters")
    print(f"Average revised answer length: {df['revised_length'].mean():.0f} characters")
    print(f"Length change: {df['revised_length'].mean() - df['answer_length'].mean():.0f} characters")
    
    # Create simple visualizations
    create_simple_plots(df, iteration_counts)
    
    # Generate recommendations
    generate_recommendations(df, iteration_counts, success_rate)

def create_simple_plots(df, iteration_counts):
    """Create simple visualizations"""
    print(f"\n📊 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Self-Refinement Loop Analysis', fontsize=16, fontweight='bold')
    
    # 1. Iteration distribution
    axes[0, 0].hist(iteration_counts.values, bins=range(1, iteration_counts.max() + 2), 
                   alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].set_title('Distribution of Iterations Needed')
    axes[0, 0].set_xlabel('Number of Iterations')
    axes[0, 0].set_ylabel('Number of Prompts')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Success rate by iteration
    success_by_iteration = df.groupby('iteration')['feedback'].apply(
        lambda x: (x == 'Critic: Answer is sufficient.').sum() / len(x) * 100
    )
    axes[0, 1].plot(success_by_iteration.index, success_by_iteration.values, 
                   marker='o', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_title('Success Rate by Iteration')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Answer length comparison
    axes[1, 0].scatter(df['answer_length'], df['revised_length'], alpha=0.6, color='orange')
    max_len = max(df['answer_length'].max(), df['revised_length'].max())
    axes[1, 0].plot([0, max_len], [0, max_len], 'r--', alpha=0.8, label='No change')
    axes[1, 0].set_title('Answer Length: Original vs Revised')
    axes[1, 0].set_xlabel('Original Answer Length')
    axes[1, 0].set_ylabel('Revised Answer Length')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Feedback distribution
    feedback_counts = df['feedback'].value_counts()
    axes[1, 1].pie(feedback_counts.values, labels=feedback_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Feedback Distribution')
    
    plt.tight_layout()
    
    # Save the plot
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "logs" / "simple_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")
    
    plt.show()

def generate_recommendations(df, iteration_counts, success_rate):
    """Generate actionable recommendations"""
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    print(f"1. **Current Performance**: {success_rate:.1f}% success rate")
    
    avg_iterations = iteration_counts.mean()
    print(f"2. **Refinement Efficiency**: Average {avg_iterations:.1f} iterations needed")
    
    if avg_iterations > 2:
        print("   → Consider improving initial answer quality to reduce iterations")
    
    # Analyze question types that perform poorly
    print(f"3. **Improvement Areas**:")
    
    # Simple question type classification
    def classify_question(prompt):
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['what is', 'define', 'explain']):
            return 'definition'
        elif any(word in prompt_lower for word in ['how', 'process', 'mechanism']):
            return 'process'
        elif any(word in prompt_lower for word in ['compare', 'difference', 'vs']):
            return 'comparison'
        elif any(word in prompt_lower for word in ['why', 'cause', 'reason']):
            return 'explanation'
        else:
            return 'factual'
    
    df['question_type'] = df['prompt'].apply(classify_question)
    
    # Find question types that need more iterations
    type_iterations = df.groupby('question_type').agg({
        'prompt': 'count',
        'feedback': lambda x: (x == 'Critic: Answer is sufficient.').sum() / len(x) * 100
    }).rename(columns={'prompt': 'count', 'feedback': 'success_rate'})
    
    print("   Question type performance:")
    for qtype, row in type_iterations.iterrows():
        print(f"   • {qtype.title()}: {row['success_rate']:.1f}% success ({row['count']} questions)")
    
    print(f"\n4. **Next Steps**:")
    print("   - Implement adaptive refinement strategies")
    print("   - Add domain-specific critic models")
    print("   - Experiment with different feedback mechanisms")
    print("   - Consider ensemble critic approaches")

def main():
    """Main analysis function"""
    print("🔍 SIMPLE SELF-REFINEMENT LOOP ANALYSIS")
    print("=" * 60)
    
    analyze_refinement_results()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 