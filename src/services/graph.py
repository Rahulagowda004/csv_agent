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
from langchain.tools import tool
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from src.constants.prompts import CSV_AGENT_SYSTEM_PROMPT

load_dotenv()

df = pd.read_csv("data/data.csv")
llm = ChatOpenAI(model="gpt-5", temperature=0)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    df: pd.DataFrame

def extract_dataframe_from_result(intermediate_steps):
    """
    Extract the dataframe result from agent intermediate steps.
    Returns the actual pandas DataFrame object when possible.
    """
    if not intermediate_steps:
        return None
    
    last_result = intermediate_steps[-1][1]
    
    # If it's already a DataFrame, return it
    if isinstance(last_result, pd.DataFrame):
        return last_result
    
    # If it's a string representation, try to re-execute
    action = intermediate_steps[-1][0]
    if hasattr(action, 'tool_input') and 'query' in action.tool_input:
        query = action.tool_input['query']
        try:
            result_df = eval(query.split('\n')[-1])
            if isinstance(result_df, pd.DataFrame):
                return result_df
        except:
            pass
    return last_result

@tool("csv_analysis_agent", return_direct=True)
def analysis_agent_tool(query: str, state: State = State) -> str:
    """
    Run a Pandas DataFrame analysis agent on a CSV file.
    
    Args:
        path (str): Path to the CSV file
        query (str): Natural language query about the dataset
    
    Returns:
        str: A string summary plus any DataFrame rows if available
    """
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        return_intermediate_steps=True
    )

    result = agent_executor.invoke({"input": query})

    extracted_df = None
    
    state['df'] = extracted_df if isinstance(extracted_df, pd.DataFrame) else state.get('df', None)
    
    return result['output']
    
@tool("visualization_tool", return_direct=True)
def execute_code(script: str) -> str:
    """Execute Python script for visualization with comprehensive data science libraries."""
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

tools = [analysis_agent_tool, execute_code]

llm = ChatOpenAI(model="gpt-5", temperature=0)
def bot(state: State) -> State:
    """a simple chatbot"""
    system_prompt = SystemMessage(content = """You are an advanced CSV Data Analysis AI Assistant specializing in data exploration, analysis, and visualization.

    **Your Capabilities:**
    - Analyze CSV datasets using pandas operations
    - Generate insights, statistics, and summaries
    - Create visualizations (matplotlib, seaborn, plotly)
    - Perform data cleaning and transformation
    - Answer business questions about data

    **Available Tools:**
    1. `csv_analysis_agent`: For pandas DataFrame operations, filtering, grouping, calculations
    2. `visualization_tool`: For creating charts, graphs, and visual analysis

    **Instructions:**
    - Always use the csv_analysis_agent tool first to understand the data structure
    - For data queries, filtering, or calculations, use csv_analysis_agent
    - For creating visualizations, use visualization_tool with complete Python code
    - Provide clear explanations of findings and insights
    - Include relevant statistics and context in your responses
    - When creating visualizations, ensure they are properly labeled and formatted
    - Save plots with descriptive filenames (e.g., 'sales_by_category.png')

    **Response Format:**
    1. Analyze the data using appropriate tools
    2. Provide key insights and findings
    3. Create visualizations when helpful
    4. Summarize conclusions and recommendations

    Always be thorough, accurate, and provide actionable insights from the data analysis.""")
    response = llm.invoke([system_prompt]+state["messages"])
    return {"messages":[response]}

def to_continue(state: State)->State:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(State)

tool_node = ToolNode(tools)

graph.add_node("tools",tool_node)
graph.add_node("agent",bot)

graph.set_entry_point("agent")

graph.add_conditional_edges("agent",
                            to_continue,
                            {
                                "continue":"tools",
                                "end":END
                            })
graph.add_edge("tools","agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Filter products in the Software Subscription category.")]}
print_stream(app.stream(inputs, stream_mode="values"))