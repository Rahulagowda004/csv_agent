# server.py
from fastmcp import FastMCP
from pathlib import Path
import os, subprocess, tempfile, json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file in project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))
mcp = FastMCP("CSV-Data-Analysis")

# # ── tools ────────────────────────────────────────────────────────────

# @mcp.tool
# def get_system_prompt() -> str:
#     """Return the system prompt for the CSV data analysis agent."""
#     return """You are a specialized CSV Data Analysis Agent with expertise in data science, statistics, and visualization. Your primary role is to help users analyze CSV data files through comprehensive data profiling, statistical analysis, and visualization.

# ## Core Capabilities:
# - **Data Analysis**: Perform comprehensive data profiling, statistical analysis, and data quality assessment
# - **Visualization**: Create various types of charts and plots using matplotlib, seaborn, and plotly
# - **Data Querying**: Execute complex data filtering, aggregation, and transformation operations
# - **Statistical Insights**: Provide statistical summaries, correlations, and data patterns
# - **Data Quality**: Identify missing values, duplicates, outliers, and data inconsistencies

# ## Available Tools:
# 1. **analyze_csv_data(folder)**: Analyzes a single CSV file in the specified folder and returns comprehensive data profiling including:
#    - File information and data structure
#    - Missing data analysis
#    - Statistical summaries for numeric columns
#    - Categorical data analysis
#    - Sample data preview
#    - Data quality metrics

# 2. **execute_code(script)**: Executes Python code with access to:
#    - pandas, numpy for data manipulation
#    - matplotlib, seaborn, plotly for visualization
#    - scipy, scikit-learn for advanced analytics
#    - All standard Python libraries

# ## Guidelines:
# - Always start by using `analyze_csv_data` to understand the data structure
# - Provide clear explanations of your analysis and findings
# - Suggest appropriate visualizations based on data types and patterns
# - Help users interpret statistical results and data quality issues
# - Offer actionable insights and recommendations
# - Use best practices for data visualization (appropriate chart types, clear labels, etc.)
# - Handle missing data and outliers appropriately
# - Ensure code is well-commented and reproducible

# ## Response Style:
# - Be thorough but concise in explanations
# - Provide context for statistical findings
# - Suggest follow-up analyses when appropriate
# - Explain the significance of data quality issues
# - Offer practical recommendations for data improvement

# Remember: Your goal is to make data analysis accessible and insightful for users of all technical levels."""

@mcp.tool
def analyze_csv_data(user_folder: str) -> dict:
    """Analyze CSV data from the user's folder. Reads the single CSV file and provides comprehensive data profiling."""
    print(f"🔍 DEBUG: analyze_csv_data called for user folder: {user_folder}")
    
    try:
        folder_path = Path(user_folder)
        print(f"🔍 DEBUG: folder_path = {folder_path}")
        print(f"🔍 DEBUG: folder exists: {folder_path.exists()}")
        
        if not folder_path.exists():
            error_msg = f"User folder '{user_folder}' does not exist"
            print(f"❌ ERROR: {error_msg}")
            return {"error": error_msg}
        
        # Find CSV files in the folder
        csv_files = list(folder_path.glob("*.csv"))
        print(f"🔍 DEBUG: Found CSV files: {csv_files}")
        
        if not csv_files:
            error_msg = f"No CSV files found in user folder '{user_folder}'"
            print(f"❌ ERROR: {error_msg}")
            return {"error": error_msg}
        
        if len(csv_files) > 1:
            error_msg = f"Multiple CSV files found in user folder '{user_folder}'. Please ensure only one CSV file is present."
            print(f"❌ ERROR: {error_msg}")
            return {"error": error_msg}
        
        csv_file = csv_files[0]
        print(f"🔍 DEBUG: Using CSV file: {csv_file}")
        
        # Read the CSV file with encoding detection
        try:
            print(f"🔍 DEBUG: Reading CSV file...")
            
            # Try multiple encodings in order of preference
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            df = None
            encoding_used = None
            
            for encoding in encodings_to_try:
                try:
                    print(f"🔍 DEBUG: Trying encoding: {encoding}")
                    df = pd.read_csv(csv_file, encoding=encoding)
                    encoding_used = encoding
                    print(f"🔍 DEBUG: Successfully read CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    print(f"🔍 DEBUG: Failed with encoding: {encoding}")
                    continue
                except Exception as e:
                    print(f"🔍 DEBUG: Other error with encoding {encoding}: {str(e)}")
                    continue
            
            if df is None:
                # Try with error handling
                try:
                    print(f"🔍 DEBUG: Trying with error handling (ignore)")
                    df = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
                    encoding_used = 'utf-8 (with errors ignored)'
                    print(f"🔍 DEBUG: Successfully read CSV with error handling")
                except Exception as e:
                    error_msg = f"Failed to read CSV file '{csv_file.name}' with any encoding: {str(e)}"
                    print(f"❌ ERROR: {error_msg}")
                    return {"error": error_msg}
            
            print(f"🔍 DEBUG: Successfully read CSV. Shape: {df.shape}")
            print(f"🔍 DEBUG: Encoding used: {encoding_used}")
            print(f"🔍 DEBUG: Columns: {list(df.columns)}")
            
        except Exception as e:
            error_msg = f"Failed to read CSV file '{csv_file.name}': {str(e)}"
            print(f"❌ ERROR: {error_msg}")
            return {"error": error_msg}
        
        # Basic file information
        file_info = {
            "filename": csv_file.name,
            "file_size_bytes": csv_file.stat().st_size,
            "file_path": str(csv_file),
            "encoding_used": encoding_used
        }
        
        # Data shape and structure
        data_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
        
        # Missing values analysis
        missing_data = {
            "total_missing": int(df.isnull().sum().sum()),
            "missing_by_column": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            "missing_percentage": {k: float(v) for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()}
        }
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_stats = {}
        if len(numeric_cols) > 0:
            # Convert numpy types to native Python types for JSON serialization
            desc_stats = df[numeric_cols].describe()
            numeric_stats = {}
            for col in desc_stats.columns:
                numeric_stats[col] = {}
                for stat in desc_stats.index:
                    value = desc_stats.loc[stat, col]
                    # Convert numpy types to native Python types
                    if isinstance(value, (np.integer, np.int64, np.int32)):
                        numeric_stats[col][stat] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        numeric_stats[col][stat] = float(value)
                    else:
                        numeric_stats[col][stat] = value
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                "unique_values": int(df[col].nunique()),
                "most_frequent": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                "frequency": {k: int(v) for k, v in df[col].value_counts().head(5).to_dict().items()}
            }
        
        # Sample data
        sample_data = {
            "first_5_rows": df.head().to_dict('records'),
            "last_5_rows": df.tail().to_dict('records')
        }
        
        # Data quality summary
        quality_summary = {
            "has_duplicates": bool(df.duplicated().any()),
            "duplicate_count": int(df.duplicated().sum()),
            "memory_usage_mb": float(round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))
        }
        
        result = {
            "file_info": file_info,
            "data_info": data_info,
            "missing_data": missing_data,
            "numeric_statistics": numeric_stats,
            "categorical_statistics": categorical_stats,
            "sample_data": sample_data,
            "quality_summary": quality_summary,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
        
        print(f"✅ DEBUG: Analysis completed successfully!")
        print(f"🔍 DEBUG: Result keys: {list(result.keys())}")
        return result
        
    except Exception as e:
        error_msg = f"Failed to analyze CSV data: {str(e)}"
        print(f"❌ ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}

@mcp.tool
def manipulate_table(script: str) -> str:
    """Execute Python script for table manipulation operations. Returns JSON with processed data."""
    try:
        # Create temp file for table manipulation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add imports for data analysis
            imports = f"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Note: Folder paths will be defined directly in the user's script

"""
            f.write(imports + script)
            temp_path = f.name
        
        # Execute using virtual environment
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        venv_dir = os.path.join(project_root, "venv")
        venv_python = os.path.join(venv_dir, "bin", "python")
        
        if not os.path.exists(venv_python):
            return "ERROR: Virtual environment not found. Please run 'python3 -m venv venv' in the project root directory."
        
        result = subprocess.run(
            [venv_python, temp_path], 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd=project_root
        )
        
        os.unlink(temp_path)
        
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        if result.returncode != 0:
            output += f"EXIT CODE: {result.returncode}\n"
            
        return output if output else "Table manipulation executed successfully with no output."
        
    except subprocess.TimeoutExpired:
        return "ERROR: Table manipulation timed out after 1 minute."
    except Exception as e:
        return f"ERROR: Failed to execute table manipulation: {str(e)}"

@mcp.tool
def create_visualization(script: str) -> str:
    """Execute Python script for creating visualizations and charts. Returns execution output."""
    try:
        # Create temp file for visualization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add imports for data analysis and visualization
            imports = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
plt.switch_backend('Agg')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Note: Folder paths will be defined directly in the user's script

"""
            f.write(imports + script)
            temp_path = f.name
        
        # Execute using virtual environment
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        venv_dir = os.path.join(project_root, "venv")
        venv_python = os.path.join(venv_dir, "bin", "python")
        
        if not os.path.exists(venv_python):
            return "ERROR: Virtual environment not found. Please run 'python3 -m venv venv' in the project root directory."
        
        result = subprocess.run(
            [venv_python, temp_path], 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd=project_root
        )
        
        os.unlink(temp_path)
        
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        if result.returncode != 0:
            output += f"EXIT CODE: {result.returncode}\n"
            
        return output if output else "Visualization executed successfully with no output."
        
    except subprocess.TimeoutExpired:
        return "ERROR: Visualization timed out after 1 minute."
    except Exception as e:
        return f"ERROR: Failed to execute visualization: {str(e)}"

@mcp.tool
def execute_code(script: str) -> str:
    """Execute Python script for data analysis and visualization with comprehensive data science libraries."""
    try:
        # Create a more robust temp file path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add imports for data analysis and visualization
            imports = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
plt.switch_backend('Agg')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Note: Folder paths will be defined directly in the user's script

"""
            f.write(imports + script)
            temp_path = f.name
        
        # Get the path to the virtual environment's Python interpreter
        # The venv is in the csv_agent project root folder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        venv_dir = os.path.join(project_root, "venv")
        venv_python = os.path.join(venv_dir, "bin", "python")
        
        # Check if virtual environment exists
        if not os.path.exists(venv_python):
            return "ERROR: Virtual environment not found. Please run 'python3 -m venv venv' in the project root directory."
        
        # Execute with better error capture using the virtual environment's Python
        result = subprocess.run(
            [venv_python, temp_path], 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd=project_root  # Set working directory to project root
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        if result.returncode != 0:
            output += f"EXIT CODE: {result.returncode}\n"
            
        return output if output else "Script executed successfully with no output."
        
    except subprocess.TimeoutExpired:
        return "ERROR: Script execution timed out after 5 minutes."
    except Exception as e:
        return f"ERROR: Failed to execute script: {str(e)}"


if __name__ == "__main__":
    mcp.run()
