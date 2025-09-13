# CSV Agent with Structured Output

This implementation provides a clean, function-based structured output system for the CSV analysis agent with LLM-generated follow-up questions.

## 🎯 Features

- **5-Field Structured Output**: `final_response`, `table_data`, `image_url_list`, `steps_taken`, `suggested_questions`
- **LLM-Generated Questions**: Uses GPT-4o to generate contextual follow-up questions
- **Clean Function Architecture**: Modular, testable functions
- **Fallback Support**: Works even when native structured output fails
- **Type Safety**: Full Pydantic validation

## 📁 Files Overview

### Core Implementation
- **`structured_functions.py`** - Clean function-based utilities (recommended)
- **`structured_agent.py`** - Complete agent implementation with structured output
- **`enhanced_testing.py`** - Integration with existing testing framework

### Testing & Examples
- **`test_structured.py`** - Standalone test script
- **`README_structured_output.md`** - This documentation

## 🚀 Quick Start

### Option 1: Using Clean Functions (Recommended)

```python
from structured_functions import create_structured_response, CSVAnalysisResponse

# After getting agent response
structured_result = await create_structured_response(agent_response, "./data")
print(structured_result.dict())
```

### Option 2: Using Complete Structured Agent

```python
from structured_agent import run_structured_csv_analysis

result = await run_structured_csv_analysis(
    "Analyze the CSV data and create visualizations", 
    "./data"
)
```

### Option 3: Integration with Existing Code

```python
# In your existing agent setup
from langchain.agents.structured_output import ToolStrategy
from structured_functions import CSVAnalysisResponse

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=tools,
    response_format=ToolStrategy(CSVAnalysisResponse)
)
```

## 📊 Output Schema

```python
class CSVAnalysisResponse(BaseModel):
    final_response: str              # Comprehensive analysis summary
    table_data: Optional[Dict]       # Structured data (JSON format)
    image_url_list: List[str]       # Generated image file paths
    steps_taken: List[Dict]         # Tool usage and execution steps
    suggested_questions: List[str]   # 3 LLM-generated follow-up questions
```

## 🔧 Key Functions

### `generate_suggested_questions(final_response, data_context, analysis_type)`
- **Purpose**: Generate 3 contextual follow-up questions using LLM
- **Input**: Analysis results, data context, analysis type
- **Output**: List of 3 relevant questions
- **LLM**: Uses GPT-4o with temperature 0.3 for creativity

### `find_generated_images(data_folder)`
- **Purpose**: Find all PNG files in the data directory
- **Input**: Path to data folder
- **Output**: List of absolute image file paths

### `extract_table_data(agent_messages)`
- **Purpose**: Extract structured data from agent responses
- **Input**: List of agent messages
- **Output**: Dictionary with structured data or None

### `extract_execution_steps(agent_messages)`
- **Purpose**: Extract tool usage steps from agent responses
- **Input**: List of agent messages
- **Output**: List of tool names and descriptions

### `create_structured_response(agent_response, data_folder)`
- **Purpose**: Orchestrate all functions to create complete structured output
- **Input**: Raw agent response and data folder path
- **Output**: Complete CSVAnalysisResponse object

## 🧪 Testing

Run the test script:
```bash
cd coding-agent-mcp
python test_structured.py
```

Or use the enhanced testing framework:
```bash
python enhanced_testing.py
```

## 💡 Usage Examples

### Basic Usage
```python
import asyncio
from structured_agent import run_structured_csv_analysis

async def analyze_data():
    result = await run_structured_csv_analysis(
        "Create age distribution and salary analysis charts",
        "./data"
    )
    
    print(f"Generated {len(result.image_url_list)} visualizations")
    print(f"Follow-up questions: {result.suggested_questions}")

asyncio.run(analyze_data())
```

### Integration with Existing Agent
```python
# Your existing agent setup
agent_response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

# Add structured output
from structured_functions import create_structured_response
structured_result = await create_structured_response(agent_response, "./data")

# Access structured fields
print(structured_result.final_response)
print(structured_result.suggested_questions)
```

## 🎨 LLM Question Generation

The `generate_suggested_questions()` function uses GPT-4o to create contextual questions:

- **Context Aware**: Considers data characteristics and analysis results
- **Specific**: Avoids generic questions, focuses on the actual dataset
- **Actionable**: Questions help users discover new insights
- **Diverse**: Covers statistical, visual, and business aspects
- **Fallback**: Provides sensible defaults if LLM fails

Example generated questions:
- "What seasonal patterns can you identify in the sales data across different regions?"
- "How do employee performance metrics correlate with tenure and department?"
- "What outliers in the customer data might indicate data quality issues or special cases?"

## 🔄 Error Handling

- **LLM Failures**: Fallback to default questions if API fails
- **Missing Data**: Graceful handling of missing files or empty responses
- **Validation**: Pydantic ensures all required fields are present
- **Type Safety**: Automatic type conversion and validation

## 📋 Requirements

- `langchain >= 0.3.27`
- `langchain-openai >= 0.3.33`
- `pydantic`
- `pathlib`
- OpenAI API key in environment variables

## 🎯 Benefits

1. **Clean Architecture**: Function-based design is easy to test and maintain
2. **LLM-Powered**: Contextual questions enhance user experience
3. **Flexible Integration**: Works with existing code or as standalone system
4. **Type Safety**: Full Pydantic validation prevents runtime errors
5. **Comprehensive Output**: All required fields automatically populated
