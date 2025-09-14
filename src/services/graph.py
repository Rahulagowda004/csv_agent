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
    """Extract structured data from the conversation messages with improved parsing."""
    
    # Extract all content from messages for analysis
    all_content = ""
    ai_messages = []
    tool_outputs = []
    
    for message in messages:
        if hasattr(message, 'content') and message.content:
            content = str(message.content)
            all_content += content + "\n"
            
            # Categorize messages
            if hasattr(message, 'name'):  # Tool message
                tool_outputs.append((message.name, content))
            elif not hasattr(message, 'tool_calls'):  # AI message without tool calls
                ai_messages.append(content)
    
    # EXTRACT STEPS - look for step-by-step patterns in AI responses
    steps = []
    step_patterns = [
        r'step\s*\d+[:\-\.]?\s*([^\n]+)',
        r'\d+\.\s*([^\n]+)',
        r'first[,\s]+([^\n]+)',
        r'then[,\s]+([^\n]+)',
        r'next[,\s]+([^\n]+)',
        r'finally[,\s]+([^\n]+)'
    ]
    
    for ai_content in ai_messages:
        for pattern in step_patterns:
            matches = re.findall(pattern, ai_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                clean_step = match.strip().rstrip('.,')
                if len(clean_step) > 10 and clean_step not in steps:  # Avoid duplicates and too short steps
                    steps.append(f"Step {len(steps) + 1}: {clean_step}")
    
    # If no steps found, generate from tool usage
    if not steps:
        step_count = 1
        for tool_name, tool_output in tool_outputs:
            if tool_name == 'analyze_csv_data':
                steps.append(f"Step {step_count}: Analyzed CSV data structure and performed comprehensive data profiling")
                step_count += 1
            elif tool_name == 'execute_code':
                if 'plt.savefig' in tool_output or 'saved to' in tool_output.lower():
                    steps.append(f"Step {step_count}: Generated data visualizations and saved plots")
                else:
                    steps.append(f"Step {step_count}: Executed data analysis code")
                step_count += 1
        
        if not steps:
            steps = ["Step 1: Performed comprehensive CSV data analysis"]
    
    # EXTRACT IMAGES - improved pattern matching for saved files
    images = []
    # Enhanced pattern to catch various file path formats
    image_patterns = [
        r'(?:saved to:|saved as:|saved at:|plot_path\s*=|file:)\s*["\']([^"\']+\.(?:png|jpg|jpeg|svg|gif))["\']',
        r'([./]*data/plots/[^"\'\s\n`]+\.(?:png|jpg|jpeg|svg|gif))',
        r'([./]*plots/[^"\'\s\n`]+\.(?:png|jpg|jpeg|svg|gif))'
    ]
    
    seen_paths = set()
    for pattern in image_patterns:
        matches = re.findall(pattern, all_content, re.IGNORECASE)
        for match in matches:
            clean_path = match.strip().strip('"\'`').strip()
            # Skip empty, very short paths, or malformed paths
            if len(clean_path) < 8 or clean_path in seen_paths or clean_path.startswith('`'):
                continue
            # Skip paths that look malformed
            if '```' in clean_path or clean_path.count('`') > 0:
                continue
            
            seen_paths.add(clean_path)
            
            # Generate meaningful title from path and context
            title = "Data Visualization"
            filename = os.path.basename(clean_path).lower()
            
            # More specific title generation based on filename
            if 'top_5_products' in filename:
                title = "Top 5 Products Analysis"
            elif 'revenue' in filename:
                if 'region' in filename:
                    title = "Revenue by Region Analysis"
                elif 'time' in filename or 'trend' in filename:
                    title = "Revenue Trend Analysis"
                else:
                    title = "Revenue Analysis Chart"
            elif 'sales' in filename:
                if 'distribution' in filename:
                    title = "Sales Distribution Chart"
                elif 'channel' in filename:
                    title = "Sales by Channel Analysis"
                else:
                    title = "Sales Analysis Chart"
            elif 'distribution' in filename:
                title = "Distribution Analysis"
            elif 'correlation' in filename:
                title = "Correlation Matrix"
            elif 'trend' in filename:
                title = "Trend Analysis"
            elif 'category' in filename:
                title = "Category Analysis"
            elif 'region' in filename:
                title = "Regional Analysis"
            elif 'channel' in filename:
                title = "Channel Analysis"
            elif 'matrix' in filename:
                title = "Analysis Matrix"
            elif 'chart' in filename or 'plot' in filename:
                title = "Data Chart"
            
            images.append(ImageInfo(title=title, path=clean_path))
    
    # EXTRACT DATAFRAMES - improved JSON parsing
    dataframes = []
    
    # First, try to get CSV analysis data
    for tool_name, tool_output in tool_outputs:
        if tool_name == 'analyze_csv_data':
            try:
                # Find JSON content in tool output
                json_start = tool_output.find('{')
                json_end = tool_output.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = tool_output[json_start:json_end]
                    data = json.loads(json_str)
                    
                    # Extract data information
                    data_info = data.get('data_info', {})
                    sample_data = data.get('sample_data', {})
                    file_info = data.get('file_info', {})
                    
                    rows = data_info.get('rows', 'unknown')
                    cols = data_info.get('columns', 'unknown')
                    filename = file_info.get('filename', 'data.csv')
                    
                    description = f"CSV Dataset Analysis: {filename} with {rows} rows and {cols} columns"
                    
                    # Get sample rows as proper JSON objects
                    sample_rows = sample_data.get('first_5_rows', [])
                    if not sample_rows:
                        sample_rows = sample_data.get('last_5_rows', [])
                    
                    # Ensure sample_rows is a list of dictionaries
                    if sample_rows and isinstance(sample_rows, list):
                        # Clean up the data to ensure JSON serializable
                        clean_sample_rows = []
                        for row in sample_rows[:5]:  # Limit to 5 rows
                            if isinstance(row, dict):
                                clean_row = {}
                                for k, v in row.items():
                                    # Convert numpy types to native Python types
                                    if hasattr(v, 'item'):  # numpy scalar
                                        clean_row[k] = v.item()
                                    elif pd.isna(v):  # Handle NaN values
                                        clean_row[k] = None
                                    else:
                                        clean_row[k] = v
                                clean_sample_rows.append(clean_row)
                        
                        dataframes.append(DataFrameInfo(
                            description=description,
                            data=clean_sample_rows
                        ))
                    else:
                        dataframes.append(DataFrameInfo(
                            description=description,
                            data=[]
                        ))
                        
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Fallback with minimal info
                dataframes.append(DataFrameInfo(
                    description="CSV Dataset Analysis - Unable to parse sample data",
                    data=[]
                ))
    
    # Second, try to extract analysis results from AI messages (like top 5 products)
    for ai_content in ai_messages:
        # Look for JSON data in AI responses - flexible patterns
        json_patterns = [
            r'```json\s*(\[.*?\])\s*```',  # Array format
            r'```json\s*(\{.*?\})\s*```',  # Object format
        ]
        
        for pattern in json_patterns:
            json_matches = re.findall(pattern, ai_content, re.DOTALL)
            
            for json_match in json_matches:
                try:
                    result_data = json.loads(json_match)
                    
                    if isinstance(result_data, list) and len(result_data) > 0:
                        # Determine description based on content
                        first_item = result_data[0]
                        if isinstance(first_item, dict):
                            keys = list(first_item.keys())
                            if 'product_name' in keys and 'units_sold' in keys:
                                description = f"Top {len(result_data)} Products by Units Sold Analysis"
                                dataframes.append(DataFrameInfo(
                                    description=description,
                                    data=result_data
                                ))
                            elif 'revenue' in str(keys).lower():
                                description = f"Revenue Analysis Results ({len(result_data)} items)"
                                dataframes.append(DataFrameInfo(
                                    description=description,
                                    data=result_data
                                ))
                            elif 'region' in str(keys).lower():
                                description = f"Regional Analysis Results ({len(result_data)} items)"
                                dataframes.append(DataFrameInfo(
                                    description=description,
                                    data=result_data
                                ))
                except json.JSONDecodeError:
                    continue
    
    # Ensure at least one dataframe entry
    if not dataframes:
        dataframes.append(DataFrameInfo(
            description="CSV Data Analysis Results",
            data=[]
        ))
    
    # EXTRACT SUGGESTIONS - context-aware generation
    suggestions = []
    
    # Analyze the content to generate relevant suggestions
    content_lower = all_content.lower()
    
    # Column-specific suggestions based on actual data
    if 'revenue' in content_lower or 'sales' in content_lower:
        suggestions.extend([
            "Analyze revenue trends over time periods",
            "Identify top-performing products or regions",
            "Create customer segmentation analysis"
        ])
    
    if 'distribution' in content_lower:
        suggestions.extend([
            "Examine outliers and anomalies in distributions",
            "Perform normality tests on key metrics",
            "Create box plots for detailed distribution analysis"
        ])
    
    if 'correlation' in content_lower:
        suggestions.extend([
            "Investigate strong correlations for causal relationships",
            "Create scatter plots for key variable pairs",
            "Perform multivariate regression analysis"
        ])
    
    if 'category' in content_lower or 'region' in content_lower:
        suggestions.extend([
            "Compare performance across different categories/regions",
            "Analyze market share by segment",
            "Create geographic heat maps"
        ])
    
    if 'time' in content_lower or 'date' in content_lower:
        suggestions.extend([
            "Analyze seasonal patterns and trends",
            "Create time series forecasting models",
            "Examine year-over-year growth rates"
        ])
    
    # Quality-based suggestions
    if 'missing' in content_lower or 'null' in content_lower:
        suggestions.extend([
            "Investigate patterns in missing data",
            "Implement data imputation strategies",
            "Assess impact of missing data on analysis"
        ])
    
    # Default suggestions if none generated
    if not suggestions:
        suggestions = [
            "Perform detailed statistical analysis on numeric columns",
            "Create additional visualizations for key insights",
            "Analyze data quality and handle outliers",
            "Explore relationships between variables",
            "Generate predictive models for forecasting"
        ]
    
    # Remove duplicates and limit to 5
    unique_suggestions = []
    for suggestion in suggestions:
        if suggestion not in unique_suggestions:
            unique_suggestions.append(suggestion)
    
    return AgentOutput(
        steps=steps[:10],  # Limit to 10 steps
        images=images,
        dataframes=dataframes,
        suggestions=unique_suggestions[:5]  # Limit to 5 suggestions
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
    """Generate structured output after tool execution is complete using improved extraction."""
    # Use the improved extraction function instead of LLM generation
    structured_output = extract_structured_data_from_messages(state["messages"])
    return {"structured_output": structured_output}

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
    """Print stream and return the final structured output with enhanced display."""
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
        print("📊 STRUCTURED OUTPUT SUMMARY")
        print("="*80)
        
        # Display steps with better formatting
        print(f"🔄 Analysis Process ({len(structured.steps)} steps):")
        for i, step in enumerate(structured.steps, 1):
            print(f"  {i:2d}. {step}")
        
        # Display images with validation
        if structured.images:
            print(f"\n🖼️ Generated Visualizations ({len(structured.images)} files):")
            for i, img in enumerate(structured.images, 1):
                # Check if file exists
                file_exists = "✅" if os.path.exists(img.path) else "❌"
                print(f"  {i:2d}. {img.title}")
                print(f"      Path: {img.path} {file_exists}")
        else:
            print(f"\n🖼️ Generated Visualizations: None")
        
        # Display dataframes with sample preview
        if structured.dataframes:
            print(f"\n📈 DataFrames ({len(structured.dataframes)} datasets):")
            for i, df in enumerate(structured.dataframes, 1):
                print(f"  {i:2d}. {df.description}")
                if df.data and len(df.data) > 0:
                    print(f"      Sample Data ({len(df.data)} rows):")
                    # Show first row as example
                    if df.data[0]:
                        sample_row = df.data[0]
                        for k, v in list(sample_row.items())[:3]:  # Show first 3 columns
                            print(f"        {k}: {v}")
                        if len(sample_row) > 3:
                            print(f"        ... and {len(sample_row) - 3} more columns")
                else:
                    print(f"      Sample Data: No data available")
        else:
            print(f"\n📈 DataFrames: None")
        
        # Display suggestions
        if structured.suggestions:
            print(f"\n💡 Actionable Suggestions ({len(structured.suggestions)} items):")
            for i, suggestion in enumerate(structured.suggestions, 1):
                print(f"  {i:2d}. {suggestion}")
        else:
            print(f"\n� Actionable Suggestions: None")
        
        # Display compact JSON for API usage
        print("\n�📄 Structured JSON Output:")
        json_output = structured.model_dump()
        # Truncate long data arrays for display
        if json_output.get('dataframes'):
            for df in json_output['dataframes']:
                if df.get('data') and len(df['data']) > 2:
                    df['data'] = df['data'][:2] + [f"... {len(df['data']) - 2} more rows truncated for display"]
        
        print(json.dumps(json_output, indent=2, ensure_ascii=False))
        
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
    inputs = {"messages": [("user", "Show me the top 5 products by sales volume and create a visualization.")]}
    
    print("=" * 80)
    print("🚀 CSV AGENT - WITH STRUCTURED OUTPUT")
    print("=" * 80)
    
    structured_output = print_stream_with_structured_output(app.stream(inputs, stream_mode="values"))
    
    if structured_output:
        print("\n✅ Structured output successfully generated!")
    else:
        print("\n❌ Failed to generate structured output")