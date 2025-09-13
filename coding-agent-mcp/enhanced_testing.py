# Enhanced testing.py with structured output integration
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy

# Import our structured functions
from structured_functions import CSVAnalysisResponse, create_structured_response

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

os.environ["OPENAI_API_KEY"] = openai_api_key

server_params = StdioServerParameters(
    command="./venv/bin/python",
    # Path to the coding agent MCP server
    args=["./server.py"],
    env={"OPENAI_API_KEY": openai_api_key}
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create agent with structured output using ToolStrategy
            agent = create_react_agent(
                model=ChatOpenAI(model="gpt-4o", temperature=0),
                tools=tools,
                response_format=ToolStrategy(CSVAnalysisResponse)
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
            print("🚀 TESTING CSV DATA ANALYSIS WITH STRUCTURED OUTPUT")
            print("=" * 80)
            print(f"📝 Query: {test_message}")
            print("-" * 80)
            
            agent_response = await agent.ainvoke({"messages": [{"role": "user", "content": test_message}]})
            
            print("\n🤖 PROCESSING AGENT RESPONSE...")
            print("=" * 80)
            
            # Try to get structured response first
            structured_result = None
            
            if 'structured_response' in agent_response:
                print("✅ Native structured response found!")
                structured_result = agent_response['structured_response']
            else:
                print("⚠️  No native structured response. Creating structured output...")
                structured_result = await create_structured_response(agent_response, "./data")
            
            # Display the structured output
            print("\n📊 STRUCTURED OUTPUT:")
            print("=" * 80)
            
            # Pretty print the structured response
            structured_dict = structured_result.dict() if hasattr(structured_result, 'dict') else structured_result
            print(json.dumps(structured_dict, indent=2, default=str))
            
            # Show summary statistics
            print(f"\n📈 RESPONSE SUMMARY:")
            print("-" * 40)
            print(f"• Final Response Length: {len(structured_result.final_response)} characters")
            print(f"• Table Data Available: {'Yes' if structured_result.table_data else 'No'}")
            print(f"• Images Generated: {len(structured_result.image_url_list)}")
            print(f"• Execution Steps: {len(structured_result.steps_taken)}")
            print(f"• Suggested Questions: {len(structured_result.suggested_questions)}")
            
            # Show the LLM-generated questions
            if structured_result.suggested_questions:
                print(f"\n❓ LLM-GENERATED FOLLOW-UP QUESTIONS:")
                print("-" * 40)
                for i, question in enumerate(structured_result.suggested_questions, 1):
                    print(f"{i}. {question}")
            
            # Show generated images
            if structured_result.image_url_list:
                print(f"\n🖼️  GENERATED VISUALIZATIONS:")
                print("-" * 40)
                for img_path in structured_result.image_url_list:
                    print(f"• {img_path}")
            
            # Show execution steps
            if structured_result.steps_taken:
                print(f"\n🔧 EXECUTION STEPS:")
                print("-" * 40)
                for i, step in enumerate(structured_result.steps_taken, 1):
                    print(f"{i}. {step['tool_name']}: {step['description'][:80]}...")
            
            # Save structured output to file
            output_file = "structured_analysis_result.json"
            with open(output_file, 'w') as f:
                json.dump(structured_dict, f, indent=2, default=str)
            print(f"\n💾 Structured output saved to: {output_file}")
            
            # Also show raw messages for debugging if needed
            print(f"\n🔍 RAW RESPONSE DEBUG INFO:")
            print("-" * 40)
            print(f"Response keys: {list(agent_response.keys())}")
            print(f"Message count: {len(agent_response.get('messages', []))}")
            
            print("\n" + "=" * 80)
            print("✅ STRUCTURED TESTING COMPLETED SUCCESSFULLY")
            print("=" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
