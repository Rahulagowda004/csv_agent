# Enhanced CSV Agent with Structured Output
import os
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class CSVAnalysisResponse(BaseModel):
    """Structured response schema for CSV data analysis."""
    final_response: str = Field(
        description="The comprehensive final analysis response and summary"
    )
    table_data: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Any tabular data extracted or generated during analysis (JSON format)"
    )
    image_url_list: List[str] = Field(
        default_factory=list, 
        description="List of image file paths/URLs generated during analysis"
    )
    steps_taken: List[Dict[str, str]] = Field(
        description="List of tools used and steps taken by the agent with tool names and descriptions"
    )
    suggested_questions: List[str] = Field(
        description="Three relevant follow-up questions based on the analysis"
    )

async def generate_suggested_questions(
    final_response: str, 
    data_context: str, 
    analysis_type: str = "general"
) -> List[str]:
    """
    Generate 3 contextual follow-up questions using LLM.
    
    Args:
        final_response: The analysis results and findings
        data_context: Brief description of the data (columns, size, type)
        analysis_type: Type of analysis performed (general, statistical, visualization, etc.)
        
    Returns:
        List of exactly 3 relevant follow-up questions
    """
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        # Fallback to default questions if no API key
        return [
            "Can you create additional visualizations to explore different aspects of this data?",
            "What are the strongest correlations between different variables in the dataset?", 
            "What data quality issues should I be aware of and how can I address them?"
        ]
    
    # Create LLM instance for question generation
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=openai_api_key)
    
    # Create context-aware prompt
    prompt = f"""Based on the following CSV data analysis results, generate exactly 3 relevant and insightful follow-up questions that would help the user dive deeper into their data.

Data Context: {data_context}
Analysis Type: {analysis_type}

Analysis Results:
{final_response[:1500]}...

Requirements for the questions:
1. Be specific and actionable
2. Focus on different aspects (statistical, visual, business insights)
3. Consider the data characteristics mentioned in the context
4. Avoid generic questions - make them relevant to this specific dataset
5. Help the user discover new insights or validate findings

Return exactly 3 questions, one per line, without numbering or bullet points."""

    try:
        response = await llm.ainvoke(prompt)
        questions = response.content.strip().split('\n')
        
        # Clean and validate questions
        clean_questions = []
        for q in questions:
            q = q.strip()
            if q and not q.startswith(('1.', '2.', '3.', '-', '*')):
                if not q.endswith('?'):
                    q += '?'
                clean_questions.append(q)
        
        # Ensure we have exactly 3 questions
        if len(clean_questions) >= 3:
            return clean_questions[:3]
        else:
            # Add fallback questions if needed
            fallback_questions = [
                "What additional patterns or trends can be explored in this dataset?",
                "How can this analysis be extended to provide more business value?",
                "What predictive models would be most suitable for this data?"
            ]
            while len(clean_questions) < 3:
                clean_questions.append(fallback_questions[len(clean_questions)])
            return clean_questions
            
    except Exception as e:
        print(f"Error generating questions with LLM: {e}")
        # Return default questions on error
        return [
            "Can you create additional visualizations to explore different aspects of this data?",
            "What are the strongest correlations between different variables in the dataset?",
            "What data quality issues should I be aware of and how can I address them?"
        ]

def find_generated_images(data_folder: str) -> List[str]:
    """
    Find all PNG image files in the specified folder.
    
    Args:
        data_folder: Path to search for generated images
        
    Returns:
        List of absolute paths to PNG files
    """
    try:
        data_path = Path(data_folder)
        if not data_path.exists():
            return []
        
        png_files = list(data_path.glob("*.png"))
        return [str(file.absolute()) for file in png_files]
        
    except Exception as e:
        print(f"Error finding images: {e}")
        return []

def extract_table_data(agent_messages) -> Optional[Dict[str, Any]]:
    """
    Extract structured table data from agent response messages.
    
    Args:
        agent_messages: List of messages from agent response
        
    Returns:
        Dictionary containing structured data or None
    """
    try:
        for message in agent_messages:
            if hasattr(message, 'content') and message.content:
                content = str(message.content)
                
                # Look for JSON-like structures in the content
                import re
                
                # Try to find data structures that look like analysis results
                if any(keyword in content.lower() for keyword in ['statistics', 'summary', 'data_info', 'sample_data']):
                    # Try to extract JSON objects
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, content)
                    
                    for match in matches:
                        try:
                            parsed = json.loads(match)
                            if isinstance(parsed, dict) and len(parsed) > 2:  # Substantial data
                                return parsed
                        except json.JSONDecodeError:
                            continue
                
                # Look for tabular data patterns
                if 'rows' in content.lower() and 'columns' in content.lower():
                    lines = content.split('\n')
                    data_info = {}
                    for line in lines:
                        if ':' in line and any(keyword in line.lower() for keyword in ['rows', 'columns', 'shape']):
                            parts = line.split(':')
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip()
                                try:
                                    # Try to convert to number if possible
                                    data_info[key] = int(value) if value.isdigit() else value
                                except:
                                    data_info[key] = value
                    
                    if data_info:
                        return data_info
        
        return None
        
    except Exception as e:
        print(f"Error extracting table data: {e}")
        return None

def extract_execution_steps(agent_messages) -> List[Dict[str, str]]:
    """
    Extract execution steps from agent response messages.
    
    Args:
        agent_messages: List of messages from agent response
        
    Returns:
        List of dictionaries containing tool names and descriptions
    """
    steps = []
    
    try:
        for message in agent_messages:
            # Extract tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', 'Unknown Tool')
                    args = tool_call.get('args', {})
                    
                    # Create descriptive step
                    if tool_name == 'analyze_csv_data':
                        description = f"Analyzed CSV data in folder: {args.get('folder', 'unknown')}"
                    elif tool_name == 'execute_code':
                        script_preview = str(args.get('script', ''))[:100] + "..." if len(str(args.get('script', ''))) > 100 else str(args.get('script', ''))
                        description = f"Executed Python code: {script_preview}"
                    else:
                        description = f"Used {tool_name} with parameters: {args}"
                    
                    steps.append({
                        "tool_name": tool_name,
                        "description": description
                    })
            
            # Extract tool results/responses
            elif hasattr(message, 'name') and message.name:
                steps.append({
                    "tool_name": message.name,
                    "description": f"Received results from {message.name}"
                })
        
        return steps
        
    except Exception as e:
        print(f"Error extracting execution steps: {e}")
        return []

async def create_structured_response(
    agent_response: dict, 
    data_folder: str = "./data"
) -> CSVAnalysisResponse:
    """
    Create structured response from raw agent output.
    
    Args:
        agent_response: Raw response from the agent
        data_folder: Path to data directory for finding images
        
    Returns:
        CSVAnalysisResponse with all required fields populated
    """
    messages = agent_response.get('messages', [])
    
    # Extract final response (last meaningful content)
    final_response = ""
    for message in reversed(messages):
        if hasattr(message, 'content') and message.content:
            content = str(message.content)
            if len(content.strip()) > 50:  # Meaningful content
                final_response = content
                break
    
    if not final_response:
        final_response = "Analysis completed successfully."
    
    # Extract other components
    table_data = extract_table_data(messages)
    image_urls = find_generated_images(data_folder)
    steps_taken = extract_execution_steps(messages)
    
    # Create data context for question generation
    data_context = f"Dataset with {len(image_urls)} visualizations generated"
    if table_data:
        data_context += f", structured data available"
    
    # Generate contextual questions using LLM
    suggested_questions = await generate_suggested_questions(
        final_response=final_response,
        data_context=data_context,
        analysis_type="comprehensive"
    )
    
    return CSVAnalysisResponse(
        final_response=final_response,
        table_data=table_data,
        image_url_list=image_urls,
        steps_taken=steps_taken,
        suggested_questions=suggested_questions
    )

async def run_structured_csv_analysis(
    query: str, 
    data_folder: str = "./data"
) -> CSVAnalysisResponse:
    """
    Run CSV analysis with structured output.
    
    Args:
        query: The analysis query/request
        data_folder: Path to folder containing CSV data
        
    Returns:
        CSVAnalysisResponse with all required fields
    """
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Server parameters
    server_params = StdioServerParameters(
        command="./venv/bin/python",
        args=["./server.py"],
        env={"OPENAI_API_KEY": openai_api_key}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            await session.initialize()

            # Load tools
            tools = await load_mcp_tools(session)

            # Create agent with structured output
            agent = create_react_agent(
                model=ChatOpenAI(model="gpt-4o", temperature=0),
                tools=tools,
                response_format=ToolStrategy(CSVAnalysisResponse)
            )
            
            # Run the analysis
            response = await agent.ainvoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            # Check if we got native structured response
            if 'structured_response' in response:
                return response['structured_response']
            else:
                # Create structured response using our helper functions
                return await create_structured_response(response, data_folder)

async def main():
    """Example usage of the structured CSV analysis."""
    
    print("🚀 CSV Agent with Structured Output")
    print("=" * 50)
    
    # Example query
    query = """
    Please analyze the CSV data in the ./data folder. 
    Create comprehensive visualizations showing:
    1. Data distribution patterns
    2. Key relationships and correlations  
    3. Any interesting trends or outliers
    
    Save all plots as PNG files and provide detailed insights.
    """
    
    try:
        print("📊 Running structured CSV analysis...")
        result = await run_structured_csv_analysis(query)
        
        print("\n✅ Analysis Complete!")
        print("=" * 50)
        
        # Display structured results
        print("\n📋 STRUCTURED RESPONSE:")
        print(json.dumps(result.dict(), indent=2, default=str))
        
        print(f"\n📊 Summary:")
        print(f"• Final Response: {len(result.final_response)} characters")
        print(f"• Table Data: {'Yes' if result.table_data else 'No'}")
        print(f"• Images Generated: {len(result.image_url_list)}")
        print(f"• Steps Taken: {len(result.steps_taken)}")
        print(f"• Suggested Questions: {len(result.suggested_questions)}")
        
        if result.suggested_questions:
            print(f"\n❓ Follow-up Questions:")
            for i, q in enumerate(result.suggested_questions, 1):
                print(f"   {i}. {q}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
