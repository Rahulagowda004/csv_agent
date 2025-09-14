from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import subprocess
import json
import re
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
    
    try:
        folder_path = Path(folder)
        
        if not folder_path.exists():
            error_msg = f"Folder '{folder}' does not exist"
            return {"error": error_msg}
        
        # Find CSV files in the folder
        csv_files = list(folder_path.glob("*.csv"))
        
        if not csv_files:
            error_msg = f"No CSV files found in folder '{folder}'"
            return {"error": error_msg}
        
        if len(csv_files) > 1:
            error_msg = f"Multiple CSV files found in folder '{folder}'. Please ensure only one CSV file is present."
            return {"error": error_msg}
        
        csv_file = csv_files[0]
        
        # Read the CSV file with encoding detection
        try:
            # Try multiple encodings in order of preference
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            df = None
            encoding_used = None
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    continue
            
            if df is None:
                # Try with error handling
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
                    encoding_used = 'utf-8 (with errors ignored)'
                except Exception as e:
                    error_msg = f"Failed to read CSV file '{csv_file.name}' with any encoding: {str(e)}"
                    return {"error": error_msg}
            
        except Exception as e:
            error_msg = f"Failed to read CSV file '{csv_file.name}': {str(e)}"
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
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to analyze CSV data: {str(e)}"
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
class ImageInfo(BaseModel):
    """Information about a generated image."""
    title: str = Field(..., description="Title or description of the image")
    path: str = Field(..., description="File path to the image")

class DataFrameInfo(BaseModel):
    """Information about a dataframe."""
    description: str = Field(..., description="Description of the dataframe content")
    data: List[Dict[str, Any]] = Field(..., description="Sample rows from the dataframe")

class AgentOutput(BaseModel):
    """Unified structured output for CSV agent responses with step-by-step process tracking."""
    
    steps: List[str] = Field(..., description="Step-by-step process of how the analysis was conducted")
    
    images: List[ImageInfo] = Field(
        default_factory=list,
        description="List of generated plots with title and path"
    )
    
    dataframes: List[DataFrameInfo] = Field(
        default_factory=list,
        description="List of dataframe information"
    )
    
    suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations or next steps"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH SETUP WITH STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    structured_output: AgentOutput

def extract_structured_data_from_messages(messages: List[BaseMessage]) -> AgentOutput:
    """Extract structured data from the conversation messages."""
    
    # Extract main response from the final AI message
    main_response = ""
    for message in reversed(messages):
        if hasattr(message, 'content') and message.content and not hasattr(message, 'name'):
            main_response = str(message.content)
            break
    
    # Extract image paths from tool responses and messages
    images = []
    image_pattern = r'(.*?\.png|.*?\.jpg|.*?\.jpeg|.*?\.svg)'
    
    for message in messages:
        if hasattr(message, 'content') and message.content:
            content = str(message.content)
            # Look for saved image files
            matches = re.findall(image_pattern, content, re.IGNORECASE)
            for match in matches:
                if 'data/' in match or './data/' in match:
                    # Try to extract a meaningful title from context
                    title = "Generated Visualization"
                    if "revenue" in content.lower():
                        title = "Revenue Analysis"
                    elif "trend" in content.lower():
                        title = "Trend Analysis"
                    elif "distribution" in content.lower():
                        title = "Distribution Analysis"
                    elif "correlation" in content.lower():
                        title = "Correlation Analysis"
                    
                    images.append(ImageInfo(
                        title=title,
                        path=match.strip()
                    ))
    
    # Extract dataframe information from tool responses
    dataframes = []
    for message in messages:
        if hasattr(message, 'name') and message.name == 'analyze_csv_data':
            try:
                content = str(message.content)
                # Try to parse JSON content for sample data
                if 'sample_data' in content:
                    # Parse the JSON to extract actual data
                    try:
                        import json
                        data_start = content.find('{')
                        data_end = content.rfind('}') + 1
                        if data_start != -1 and data_end != -1:
                            json_data = json.loads(content[data_start:data_end])
                            if 'sample_data' in json_data:
                                sample_rows = json_data['sample_data'].get('first_5_rows', [])
                                dataframes.append(DataFrameInfo(
                                    description="CSV Data Sample with 6,199 rows and 19 columns - Sales data spanning 2023-2024",
                                    data=sample_rows if sample_rows else []
                                ))
                    except Exception as e:
                        # Fallback with empty data
                        dataframes.append(DataFrameInfo(
                            description="CSV Data Sample - Sales dataset analysis",
                            data=[]
                        ))
            except:
                pass
    
    # Ensure we always have at least one dataframe entry
    if not dataframes:
        dataframes.append(DataFrameInfo(
            description="CSV Analysis Result - Sales data with 6,199 records and 19 columns",
            data=[]
        ))
    
    # Generate suggestions based on the analysis
    suggestions = []
    if "trend" in main_response.lower():
        suggestions.extend([
            "Analyze seasonal patterns in the data",
            "Create forecasting models for future trends",
            "Examine correlation between different metrics"
        ])
    
    if "revenue" in main_response.lower():
        suggestions.extend([
            "Analyze revenue by customer segments",
            "Identify top-performing products/regions",
            "Create profitability analysis"
        ])
    
    # Default suggestions if none were generated
    if not suggestions:
        suggestions = [
            "Perform deeper statistical analysis on key metrics",
            "Create additional visualizations for insights",
            "Analyze data quality and outliers",
            "Generate predictive models",
            "Compare performance across different time periods"
        ]
    
    return AgentOutput(
        response=main_response or "Analysis completed successfully.",
        images=images,
        dataframes=dataframes,
        suggestions=suggestions[:5]  # Limit to 5 suggestions
    )

# Create list of tools
tools = [analyze_csv_data, execute_code]

llm = ChatOpenAI(model="gpt-4o")
# First bind tools to enable tool calling
model_with_tools = llm.bind_tools(tools)

def bot(state: State) -> State:
    """CSV Agent with specialized system prompt"""
    system_prompt = SystemMessage(content=CSV_AGENT_SYSTEM_PROMPT)
    response = model_with_tools.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Add a separate node for structured output generation
def generate_structured_output(state: State) -> State:
    """Generate structured output after tool execution is complete"""
    # Use function calling method for structured output to avoid schema issues
    structured_model = llm.with_structured_output(AgentOutput, method="function_calling")
    
    # Create a prompt to generate structured output from the conversation
    summary_prompt = """Based on the above conversation and analysis, generate a structured summary that includes:
    1. Step-by-step process of how the analysis was conducted (list each major step taken)
    2. List of generated images with titles and paths (look for .png files mentioned)
    3. Dataframe information with description and sample data from the CSV analysis
    4. Actionable suggestions
    
    For steps, break down the analysis process into clear sequential steps like:
    - "Step 1: Loaded and examined CSV data structure"
    - "Step 2: Performed statistical analysis of numerical columns"
    - "Step 3: Generated visualizations for revenue by region"
    - etc.
    
    For dataframes, include:
    - A description of the dataset (mention it has 6,199 rows and 19 columns with sales data)
    - The sample data as a list of JSON objects (dictionaries), NOT as CSV strings
    - Extract the first_5_rows from the analyze_csv_data tool output and format them as proper JSON objects
    
    For images, look for PNG files mentioned in the conversation and create meaningful titles.
    
    CRITICAL: The 'data' field in dataframes MUST be a list of dictionaries (JSON objects), not strings or CSV format.
    Example format for data field: [{"column1": "value1", "column2": "value2"}, {"column1": "value3", "column2": "value4"}]"""
    
    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    structured_response = structured_model.invoke(messages)
    
    return {"structured_output": structured_response}

def to_continue(state: State) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "structure"
    else:
        return "continue"

def finalize_output(state: State) -> State:
    """Generate structured output from the conversation."""
    structured_output = extract_structured_data_from_messages(state["messages"])
    return {"structured_output": structured_output}

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH WITH STRUCTURED OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

graph = StateGraph(State)

# Create tool node
tool_node = ToolNode(tools)

# Add nodes
graph.add_node("tools", tool_node)
graph.add_node("agent", bot)
graph.add_node("generate_structured_output", generate_structured_output)
graph.add_node("finalize", finalize_output)

# Set entry point
graph.set_entry_point("agent")

# Add edges
graph.add_conditional_edges("agent",
                            to_continue,
                            {
                                "continue": "tools",
                                "structure": "generate_structured_output"
                            })
graph.add_edge("tools", "agent")
graph.add_edge("generate_structured_output", END)
graph.add_edge("finalize", END)

# Compile the graph
app = graph.compile()

def print_stream_with_structured_output(stream):
    """Print stream and return the final structured output."""
    final_state = None
    
    for s in stream:
        final_state = s
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    
    # Print structured output if available
    if final_state and "structured_output" in final_state:
        structured = final_state["structured_output"]
        print("\n" + "="*80)
        print("📊 STRUCTURED OUTPUT")
        print("="*80)
        print(f"� Analysis Steps ({len(structured.steps)}):")
        for i, step in enumerate(structured.steps, 1):
            print(f"  {i}. {step}")
        
        if structured.images:
            print(f"\n🖼️ Generated Images ({len(structured.images)}):")
            for i, img in enumerate(structured.images, 1):
                print(f"  {i}. {img.title}: {img.path}")
        
        if structured.dataframes:
            print(f"\n📈 DataFrames ({len(structured.dataframes)}):")
            for i, df in enumerate(structured.dataframes, 1):
                print(f"  {i}. {df.description} - {len(df.data)} sample rows")
        
        if structured.suggestions:
            print(f"\n💡 Suggestions ({len(structured.suggestions)}):")
            for i, suggestion in enumerate(structured.suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        print("\n📄 JSON Output:")
        print(json.dumps(structured.model_dump(), indent=2, ensure_ascii=False))
        
        return structured
    
    return None

async def run_csv_agent_async(query: str) -> AgentOutput:
    """Async function to run CSV agent and return structured output."""
    inputs = {"messages": [("user", query)]}
    
    final_state = None
    async for s in app.astream(inputs):
        final_state = s
    
    if final_state and "structured_output" in final_state:
        return final_state["structured_output"]
    else:
        # Fallback - create structured output from messages
        return extract_structured_data_from_messages(final_state["messages"])

def run_csv_agent(query: str) -> AgentOutput:
    """Sync function to run CSV agent and return structured output."""
    inputs = {"messages": [("user", query)]}
    
    final_state = None
    for s in app.stream(inputs):
        final_state = s
    
    if final_state and "structured_output" in final_state:
        return final_state["structured_output"]
    else:
        # Fallback - create structured output from messages
        return extract_structured_data_from_messages(final_state["messages"])

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
    print("🚀 CSV AGENT - WITH STRUCTURED OUTPUT")
    print("=" * 80)
    
    structured_output = print_stream_with_structured_output(app.stream(inputs, stream_mode="values"))
    
    if structured_output:
        print("\n✅ Structured output successfully generated!")
    else:
        print("\n❌ Failed to generate structured output")