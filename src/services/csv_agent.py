# Create server parameters for stdio connection
import os
import json
from tabnanny import verbose
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any, Dict

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Import system prompt
from src.constants.prompts import CSV_AGENT_SYSTEM_PROMPT

# Load environment variables from .env file in project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

os.environ["OPENAI_API_KEY"] = openai_api_key

class CSVAgentResponse(BaseModel):
    """Structured response from CSV analysis agent."""
    text: str = Field(description="The main text content from the agent message")
    tool_calls: Optional[List[str]] = Field(
        default=None,
        description="Readable format of tool messages and their outputs"
    )
    image_paths: Optional[List[str]] = Field(
        default=None, 
        description="List of image file paths when data visualization tasks are present"
    )
    table_visualization: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON data for table visualization output from tools"
    )
    suggested_next_steps: Optional[List[str]] = Field(
        default=None,
        description="List of suggested queries when user options are vague"
    )


def extract_tool_calls_readable(messages: List[Any]) -> List[str]:
    """Extract tool calls from messages and format them in a readable way."""
    tool_calls_readable = []
    
    for message in messages:
        # Check for tool calls in AI messages
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', 'Unknown Tool')
                tool_args = tool_call.get('args', {})
                readable_call = f"🔧 Used {tool_name}"
                if tool_args:
                    # Format arguments in a readable way
                    args_str = ", ".join([f"{k}: {v}" for k, v in tool_args.items() if v])
                    readable_call += f" with parameters: {args_str}"
                tool_calls_readable.append(readable_call)
        
        # Check for tool messages (responses from tools)
        if hasattr(message, 'name') and hasattr(message, 'content'):
            tool_name = message.name
            content = str(message.content)
            # Truncate very long content
            if len(content) > 200:
                content = content[:200] + "..."
            readable_response = f"📊 {tool_name} returned: {content}"
            tool_calls_readable.append(readable_response)
    
    return tool_calls_readable


async def create_structured_output_from_response(agent_response: Dict[str, Any]) -> CSVAgentResponse:
    """
    Main function to create structured output from agent response.
    This is the function you should use to format any agent response.
    """
    return await format_agent_response_structured(agent_response)


async def format_agent_response_structured(agent_response: Dict[str, Any]) -> CSVAgentResponse:
    """
    Takes the entire agent_response and uses an LLM to create structured output.
    
    Args:
        agent_response: The complete response from the react agent
        
    Returns:
        CSVAgentResponse: Structured output with text, tool_calls, image_paths, 
                         table_visualization, and suggested_next_steps
    """
    # Initialize the structured LLM
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = model.with_structured_output(CSVAgentResponse)
    
    # Extract messages from agent response
    messages = agent_response.get('messages', [])
    
    # Extract main text content from the final AI message
    main_text = ""
    for message in reversed(messages):  # Start from the last message
        if hasattr(message, 'content') and message.content and not hasattr(message, 'name'):
            main_text = str(message.content)
            break
    
    # Extract tool calls in readable format
    tool_calls_readable = extract_tool_calls_readable(messages)
    
    # Create a comprehensive prompt for the LLM to structure the response
    structure_prompt = f"""
    Please analyze this agent response and extract structured information:
    
    MAIN CONTENT:
    {main_text}
    
    TOOL INTERACTIONS:
    {chr(10).join(tool_calls_readable) if tool_calls_readable else "No tool calls"}
    
    FULL MESSAGES CONTEXT:
    {str(messages)}...
    
    Extract and structure this information into:
    1. text: The main readable content/analysis from the agent
    2. tool_calls: List of readable descriptions of what tools were used and their key outputs. Display the steps taken by the agent, explain in simple terms what the agent did.
    3. image_paths: Any file paths to images/plots that were created 
    4. table_visualization: Any JSON/dict data that represents tabular data for visualization
    5. suggested_next_steps: If the query was vague, suggest specific follow-up questions/analyses
    
    Focus on extracting actionable, useful information for the user.
    """
    
    try:
        structured_response = await structured_llm.ainvoke(structure_prompt)
        return structured_response
    except Exception as e:
        # Fallback: create a basic structured response if LLM fails
        return CSVAgentResponse(
            text=main_text or "Analysis completed",
            tool_calls=tool_calls_readable if tool_calls_readable else None,
            image_paths=None,
            table_visualization=None,
            suggested_next_steps=None
        )


server_params = StdioServerParameters(
    command="venv/bin/python",
    args=["src/core/csv_server.py"],
    env={"OPENAI_API_KEY": openai_api_key}
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent (without structured output at this level)
            model = ChatOpenAI(model="gpt-4o", temperature=0)
            agent = create_react_agent(
                model=model,
                tools=tools,
                prompt=SystemMessage(content=CSV_AGENT_SYSTEM_PROMPT)
            )
            
            # Test the CSV analysis functionality with visualization
            test_message = """Please analyze the CSV data in the ./data folder. 
            After analyzing the data, create meaningful visualizations (charts, plots) based on the data patterns you find. 
            Save all generated visualizations as image files (PNG format) in the same data directory. 
            Provide a comprehensive analysis report including:
            1. Data overview and structure
            2. Statistical insights
            3. Data quality assessment
            4. Key findings and patterns
            5. Recommendations based on the analysis
            
            Make sure to save any plots or charts you create to help visualize the data insights."""
            
            print("=" * 80)
            print("🚀 TESTING CSV DATA ANALYSIS")
            print("=" * 80)
            print(f"📝 Query: {test_message}")
            print("-" * 80)
            
            agent_response = await agent.ainvoke({"messages": test_message})
            
            print("\n🤖 AGENT RESPONSE:")
            print("=" * 80)
            
            # Use the new structured output function
            structured_data = await create_structured_output_from_response(agent_response)
            
            print(f"\n📊 STRUCTURED RESPONSE:")
            print("-" * 40)
            print(f"📝 Text: {structured_data.text}")
            
            if structured_data.tool_calls:
                print(f"\n🔧 Tool Calls ({len(structured_data.tool_calls)}):")
                for i, tool_call in enumerate(structured_data.tool_calls, 1):
                    print(f"  {i}. {tool_call}")
            
            if structured_data.image_paths:
                print(f"\n🖼️ Image Paths ({len(structured_data.image_paths)}):")
                for i, path in enumerate(structured_data.image_paths, 1):
                    print(f"  {i}. {path}")
            
            if structured_data.table_visualization:
                print(f"\n📈 Table Visualization:")
                print(f"  Data: {structured_data.table_visualization}")
            
            if structured_data.suggested_next_steps:
                print(f"\n💡 Suggested Next Steps ({len(structured_data.suggested_next_steps)}):")
                for i, step in enumerate(structured_data.suggested_next_steps, 1):
                    print(f"  {i}. {step}")
            print("\n\nENTIRE JSON STRUCTURED DATA:")
            print(json.dumps(structured_data.model_dump(), indent=2, ensure_ascii=False))
            # Also show raw messages for debugging
            print(f"\n🔧 RAW MESSAGES (for debugging):")
            print("-" * 40)
            if 'messages' in agent_response:
                for i, message in enumerate(agent_response['messages']):
                    print(f"\n📨 Message {i+1}: {type(message).__name__}")
                    if hasattr(message, 'content'):
                        print(f"Content: {message.content}")
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        print(f"Tool Calls: {len(message.tool_calls)}")
                        for j, tool_call in enumerate(message.tool_calls):
                            print(f"  🔧 Tool {j+1}: {tool_call.get('name', 'Unknown')}")
                    if hasattr(message, 'name'):
                        print(f"Tool: {message.name}")
            
            print("=" * 80)
            print("✅ TESTING COMPLETED")
            print("=" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())