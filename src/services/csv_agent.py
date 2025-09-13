# Create server parameters for stdio connection
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

os.environ["OPENAI_API_KEY"] = openai_api_key

server_params = StdioServerParameters(
    command=".venv/Scripts/python",
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

            # Create and run the agent
            agent = create_react_agent(
                model="gpt-4.1",
                tools=tools
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
            
            # Extract and format the response messages
            if 'messages' in agent_response:
                for i, message in enumerate(agent_response['messages']):
                    print(f"\n📨 Message {i+1}: {type(message).__name__}")
                    print("-" * 40)
                    if hasattr(message, 'content'):
                        print(f"Content: {message.content}")
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        print(f"Tool Calls: {len(message.tool_calls)}")
                        for j, tool_call in enumerate(message.tool_calls):
                            print(f"  🔧 Tool {j+1}: {tool_call.get('name', 'Unknown')}")
                            print(f"     Args: {tool_call.get('args', {})}")
                    if hasattr(message, 'name'):
                        print(f"Tool: {message.name}")
                    print()
            
            print("=" * 80)
            print("✅ TESTING COMPLETED")
            print("=" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())