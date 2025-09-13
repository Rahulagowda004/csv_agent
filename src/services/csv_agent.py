# Create server parameters for stdio connection
import os
from tabnanny import verbose
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
# from langchain.agents.structured_output import ToolStrategy

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
    image_paths: Optional[List[str]] = Field(
        default=None, 
        description="List of image file paths when data visualization tasks are present"
    )
    table_visualization: Optional[dict] = Field(
        default=None,
        description="JSON data for table visualization output from tools"
    )
    suggested_next_steps: Optional[List[str]] = Field(
        default=None,
        description="List of suggested queries when user options are vague"
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

            # Create and run the agent with structured output
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
            
            # Extract structured response
            if 'structured_response' in agent_response:
                structured_data = agent_response['structured_response']
                print(f"\n📊 STRUCTURED RESPONSE:")
                print("-" * 40)
                print(f"📝 Text: {structured_data.text}")
                
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