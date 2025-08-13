#!/usr/bin/env python3
"""Self-Refinement Loop Analysis Runner"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run analysis tools"""
    project_root = Path(__file__).parent.parent
    analysis_dir = project_root / "src" / "analysis"
    
    if not analysis_dir.exists():
        print(f"❌ Analysis directory not found at {analysis_dir}")
        sys.exit(1)
    
    print("🔍 Self-Refinement Loop Analysis Tools")
    print("=" * 50)
    print("Available analysis tools:")
    print("1. Simple Analysis (quick overview)")
    print("2. Advanced Critic Test")
    print("3. Performance Improvement Analysis")
    print("4. All analyses")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (0-4): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                print("\n📊 Running Simple Analysis...")
                subprocess.run([sys.executable, str(analysis_dir / "simple_analysis.py")])
            elif choice == "2":
                print("\n🧠 Testing Advanced Critic...")
                critics_dir = project_root / "src" / "core" / "critics"
                subprocess.run([sys.executable, str(critics_dir / "advanced_critic.py")])
            elif choice == "3":
                print("\n🚀 Running Performance Improvement Analysis...")
                subprocess.run([sys.executable, str(analysis_dir / "improve_critic.py")])
            elif choice == "4":
                print("\n📊 Running All Analyses...")
                print("1. Simple Analysis...")
                subprocess.run([sys.executable, str(analysis_dir / "simple_analysis.py")])
                print("\n2. Advanced Critic Test...")
                critics_dir = project_root / "src" / "core" / "critics"
                subprocess.run([sys.executable, str(critics_dir / "advanced_critic.py")])
                print("\n3. Performance Improvement Analysis...")
                subprocess.run([sys.executable, str(analysis_dir / "improve_critic.py")])
                print("\n✅ All analyses complete!")
            else:
                print("❌ Invalid choice. Please select 0-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Analysis stopped by user")
            break
        except Exception as e:
            print(f"❌ Error running analysis: {e}")

if __name__ == "__main__":
    main() 