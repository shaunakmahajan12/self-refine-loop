import pandas as pd
from pathlib import Path

def load_csv_safely(file_path):
    """Load CSV with proper handling of quoted fields with newlines"""
    try:
        # Try with different engines and quoting options
        return pd.read_csv(file_path, engine='python', quoting=1)
    except:
        try:
            return pd.read_csv(file_path, engine='c', quoting=1)
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

def load_all_results(logs_dir):
    """Load all result files from logs directory"""
    results = {}
    
    files_to_load = [
        "final_refinement_results.csv",
        "test_refine_outputs.csv",
        "evaluated_outputs.csv",
        "test_results_summary.csv"
    ]
    
    for file in files_to_load:
        file_path = Path(logs_dir) / file
        if file_path.exists():
            try:
                results[file.replace('.csv', '')] = load_csv_safely(file_path)
                print(f"✅ Loaded {file}")
            except Exception as e:
                print(f"⚠️  Could not load {file}: {str(e)}")
        else:
            print(f"⚠️  {file} not found")
    
    return results 