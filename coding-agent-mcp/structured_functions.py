"""
Clean function-based structured output utilities for CSV Agent
"""

import os
import json
import re
from pathlib import Path
from typing import List, Optional, Any, Dict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class CSVAnalysisResponse(BaseModel):
    """Structured response schema for CSV data analysis."""
    final_response: str = Field(description="Comprehensive analysis summary")
    table_data: Optional[Dict[str, Any]] = Field(default=None, description="Tabular data in JSON format")  
    image_url_list: List[str] = Field(default_factory=list, description="Generated image file paths")
    steps_taken: List[Dict[str, str]] = Field(description="Tools used and execution steps")
    suggested_questions: List[str] = Field(description="Three LLM-generated follow-up questions")

# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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
        analysis_type: Type of analysis performed
        
    Returns:
        List of exactly 3 relevant follow-up questions
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return _get_fallback_questions()
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=openai_api_key)
    
    prompt = f"""Based on the CSV data analysis results below, generate exactly 3 specific and actionable follow-up questions.

Data Context: {data_context}
Analysis Type: {analysis_type}

Analysis Results:
{final_response[:1500]}

Requirements:
1. Make questions specific to this dataset and analysis
2. Focus on different aspects (statistical, visual, business insights)
3. Be actionable and help discover new insights
4. Avoid generic questions

Return exactly 3 questions, one per line, without numbering."""

    try:
        response = await llm.ainvoke(prompt)
        questions = _parse_questions(response.content)
        return questions if len(questions) == 3 else _get_fallback_questions()
        
    except Exception as e:
        print(f"LLM question generation failed: {e}")
        return _get_fallback_questions()

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
        
    except Exception:
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
            if not (hasattr(message, 'content') and message.content):
                continue
                
            content = str(message.content)
            
            # Look for analysis results with statistical data
            if _contains_analysis_keywords(content):
                table_data = _extract_json_data(content)
                if table_data:
                    return table_data
                
                # Try extracting structured info from text
                table_data = _extract_text_data(content)
                if table_data:
                    return table_data
        
        return None
        
    except Exception:
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
                    step = _create_tool_step(tool_call)
                    steps.append(step)
            
            # Extract tool results
            elif hasattr(message, 'name') and message.name:
                steps.append({
                    "tool_name": message.name,
                    "description": f"Received results from {message.name}"
                })
        
        return steps
        
    except Exception:
        return []

async def create_structured_response(
    agent_response: dict, 
    data_folder: str = "./data"
) -> CSVAnalysisResponse:
    """
    Create structured response from raw agent output.
    
    Args:
        agent_response: Raw response from the agent
        data_folder: Path to data directory
        
    Returns:
        CSVAnalysisResponse with all required fields populated
    """
    messages = agent_response.get('messages', [])
    
    # Extract components
    final_response = _extract_final_response(messages)
    table_data = extract_table_data(messages)
    image_urls = find_generated_images(data_folder)
    steps_taken = extract_execution_steps(messages)
    
    # Create context for LLM question generation
    data_context = _create_data_context(image_urls, table_data, steps_taken)
    
    # Generate contextual questions
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

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_fallback_questions() -> List[str]:
    """Return default questions when LLM generation fails."""
    return [
        "Can you create additional visualizations to explore different aspects of this data?",
        "What are the strongest correlations between different variables in the dataset?",
        "What data quality issues should I be aware of and how can I address them?"
    ]

def _parse_questions(content: str) -> List[str]:
    """Parse and clean questions from LLM response."""
    questions = content.strip().split('\n')
    clean_questions = []
    
    for q in questions:
        q = q.strip()
        # Remove numbering and bullet points
        q = re.sub(r'^[\d\.\-\*\s]+', '', q)
        if q and len(q) > 10:  # Meaningful question
            if not q.endswith('?'):
                q += '?'
            clean_questions.append(q)
    
    return clean_questions

def _contains_analysis_keywords(content: str) -> bool:
    """Check if content contains analysis-related keywords."""
    keywords = ['statistics', 'summary', 'data_info', 'sample_data', 'analysis', 'rows', 'columns']
    return any(keyword in content.lower() for keyword in keywords)

def _extract_json_data(content: str) -> Optional[Dict[str, Any]]:
    """Try to extract JSON data from content."""
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, content)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and len(parsed) > 2:
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None

def _extract_text_data(content: str) -> Optional[Dict[str, Any]]:
    """Extract structured data from text format."""
    lines = content.split('\n')
    data_info = {}
    
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Try to convert to appropriate type
                if value.isdigit():
                    data_info[key] = int(value)
                elif _is_float(value):
                    data_info[key] = float(value)
                else:
                    data_info[key] = value
    
    return data_info if len(data_info) > 2 else None

def _is_float(value: str) -> bool:
    """Check if string can be converted to float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def _create_tool_step(tool_call: dict) -> Dict[str, str]:
    """Create a structured step from tool call."""
    tool_name = tool_call.get('name', 'Unknown Tool')
    args = tool_call.get('args', {})
    
    # Create specific descriptions based on tool type
    if tool_name == 'analyze_csv_data':
        description = f"Analyzed CSV data in folder: {args.get('folder', 'unknown')}"
    elif tool_name == 'execute_code':
        script_preview = str(args.get('script', ''))[:100]
        if len(str(args.get('script', ''))) > 100:
            script_preview += "..."
        description = f"Executed Python code: {script_preview}"
    else:
        description = f"Used {tool_name} with parameters: {args}"
    
    return {
        "tool_name": tool_name,
        "description": description
    }

def _extract_final_response(messages) -> str:
    """Extract the final meaningful response from messages."""
    for message in reversed(messages):
        if hasattr(message, 'content') and message.content:
            content = str(message.content).strip()
            if len(content) > 50:  # Meaningful content
                return content
    
    return "Analysis completed successfully."

def _create_data_context(image_urls: List[str], table_data: Optional[dict], steps_taken: List[dict]) -> str:
    """Create context string for LLM question generation."""
    context_parts = []
    
    if image_urls:
        context_parts.append(f"{len(image_urls)} visualizations generated")
    
    if table_data:
        context_parts.append("structured data analysis available")
    
    if steps_taken:
        tools_used = set(step['tool_name'] for step in steps_taken)
        context_parts.append(f"tools used: {', '.join(tools_used)}")
    
    return "Dataset with " + ", ".join(context_parts) if context_parts else "Dataset analyzed"

# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

async def example_usage():
    """Example of how to use the structured functions."""
    
    # Mock agent response for demonstration
    mock_response = {
        'messages': [
            type('MockMessage', (), {
                'content': 'Analysis complete. Found 1000 rows, 5 columns. Generated visualizations showing age distribution and salary patterns.',
                'tool_calls': [
                    {'name': 'analyze_csv_data', 'args': {'folder': './data'}},
                    {'name': 'execute_code', 'args': {'script': 'plt.hist(df["age"]); plt.savefig("age_dist.png")'}}
                ]
            })()
        ]
    }
    
    # Create structured response
    result = await create_structured_response(mock_response, "./data")
    
    print("📊 Structured Response Created:")
    print(f"✓ Final Response: {len(result.final_response)} chars")
    print(f"✓ Images: {len(result.image_url_list)}")
    print(f"✓ Steps: {len(result.steps_taken)}")
    print(f"✓ Questions: {len(result.suggested_questions)}")
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
