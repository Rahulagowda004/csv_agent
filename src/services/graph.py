from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import subprocess
from pathlib import Path

from src.constants.prompts import CSV_AGENT_SYSTEM_PROMPT

# Load environment variables first
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# DIRECT TOOL IMPLEMENTATIONS (replacing MCP tools)
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def analyze_csv_data(folder: str) -> dict:
    """Analyze CSV data from a folder. Reads the single CSV file and provides comprehensive data profiling."""
    print(f"🔍 DEBUG: analyze_csv_data called with folder: {folder}")
    
    try:
        folder_path = Path(folder)
        print(f"🔍 DEBUG: folder_path = {folder_path}")
        print(f"🔍 DEBUG: folder exists: {folder_path.exists()}")
        
        if not folder_path.exists():
            error_msg = f"Folder '{folder}' does not exist"
            print(f"❌ ERROR: {error_msg}")
            return {"error": error_msg}
        
        # Find CSV files in the folder
        csv_files = list(folder_path.glob("*.csv"))
        print(f"🔍 DEBUG: Found CSV files: {csv_files}")
        
        if not csv_files:
            error_msg = f"No CSV files found in folder '{folder}'"
            print(f"❌ ERROR: {error_msg}")
            return {"error": error_msg}
        
        if len(csv_files) > 1:
            error_msg = f"Multiple CSV files found in folder '{folder}'. Please ensure only one CSV file is present."
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

@tool
def execute_code(script: str) -> str:
    """Execute Python script for data analysis and visualization with comprehensive data science libraries."""
    try:
        # Create a more robust temp file path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add imports for data analysis and visualization
            imports = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
plt.switch_backend('Agg')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

"""
            f.write(imports + script)
            temp_path = f.name
        
        # Get the current working directory
        current_dir = os.getcwd()
        
        # Get Python executable (try virtual environment first, then system python)
        venv_python_paths = [
            os.path.join(current_dir, ".venv", "Scripts", "python.exe"),  # Windows venv
            os.path.join(current_dir, ".venv", "bin", "python"),         # Unix venv
            os.path.join(current_dir, "venv", "Scripts", "python.exe"),  # Windows venv alt
            os.path.join(current_dir, "venv", "bin", "python"),          # Unix venv alt
            "python"  # System python fallback
        ]
        
        python_exe = "python"  # Default fallback
        for path in venv_python_paths:
            if os.path.exists(path):
                python_exe = path
                break
        
        print(f"🔍 DEBUG: Using Python executable: {python_exe}")
        
        # Execute with better error capture
        result = subprocess.run(
            [python_exe, temp_path], 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd=current_dir  # Set working directory
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

# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Create list of tools
tools = [analyze_csv_data, execute_code]

print(f"Loaded {len(tools)} tools:")
for tool in tools:
    print(f"- {tool.name}")

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o")
# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

def bot(state: State) -> State:
    """CSV Agent with specialized system prompt"""
    system_prompt = SystemMessage(content=CSV_AGENT_SYSTEM_PROMPT)
    response = llm_with_tools.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def to_continue(state: State) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

graph = StateGraph(State)

# Create tool node
tool_node = ToolNode(tools)

# Add nodes
graph.add_node("tools", tool_node)
graph.add_node("agent", bot)

# Set entry point
graph.set_entry_point("agent")

# Add edges
graph.add_conditional_edges("agent",
                            to_continue,
                            {
                                "continue": "tools",
                                "end": END
                            })
graph.add_edge("tools", "agent")

# Compile the graph
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    inputs = {"messages": [("user", """Please analyze the CSV data in the ./data folder. 
                After analyzing the data, create meaningful visualizations (charts, plots) based on the data patterns you find. 
                Save all generated visualizations as image files (PNG format) in the same data directory. 
                Provide a comprehensive analysis report including:
                1. Data overview and structure
                2. Statistical insights
                3. Data quality assessment
                4. Key findings and patterns
                5. Recommendations based on the analysis

                Make sure to save any plots or charts you create to help visualize the data insights.""")]}
    
    print("=" * 80)
    print("🚀 CSV AGENT - SIMPLIFIED VERSION")
    print("=" * 80)
    print_stream(app.stream(inputs, stream_mode="values"))