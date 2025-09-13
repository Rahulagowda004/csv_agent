from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,SystemMessage,BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters, stdio_client
import os

from src.constants.prompts import CSV_AGENT_SYSTEM_PROMPT

server_params = StdioServerParameters(
    command="./.venv/scripts/python",
    args=["src/core/csv_server.py"],
    env={"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") }
)

# Initialize tools synchronously
with stdio_client(server_params) as (read, write):
    with ClientSession(read, write) as session:
        # Initialize the connection
        session.initialize()
        # Get tools
        tools = load_mcp_tools(session)

load_dotenv()

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    
llm = ChatOpenAI(model="gpt-4o")

def bot(state: State) -> State:
    """a simple chatbot"""
    system_prompt = SystemMessage(content = "You are my AI assistant, please answer my query to the best of your ability.")
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

inputs = {"messages": [("user","""Please analyze the CSV data in the ./data folder. 
            After analyzing the data, create meaningful visualizations (charts, plots) based on the data patterns you find. 
            Save all generated visualizations as image files (PNG format) in the same data directory. 
            Provide a comprehensive analysis report including:
            1. Data overview and structure
            2. Statistical insights
            3. Data quality assessment
            4. Key findings and patterns
            5. Recommendations based on the analysis

            Make sure to save any plots or charts you create to help visualize the data insights.""" )]}
print_stream(app.stream(inputs, stream_mode="values"))