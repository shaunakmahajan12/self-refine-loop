#!/usr/bin/env python3
"""
Self-Refinement Loop Demo Runner

This script launches the interactive Streamlit demo for the self-refinement loop project.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit demo"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    demo_path = project_root / "ui" / "demo_app.py"
    
    if not demo_path.exists():
        print(f"❌ Demo app not found at {demo_path}")
        print("Please ensure the project structure is correct.")
        sys.exit(1)
    
    print("🚀 Launching Self-Refinement Loop Demo...")
    print(f"📁 Demo path: {demo_path}")
    print("🌐 The demo will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the demo")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 