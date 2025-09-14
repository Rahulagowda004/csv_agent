from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

load_dotenv()

llm = ChatOpenAI(model="gpt-5",temperature=0)

def extract_dataframe_from_result(intermediate_steps):
    """
    Extract the dataframe result from agent intermediate steps.
    Returns the actual pandas DataFrame object when possible.
    """
    if not intermediate_steps:
        return None
    
    # Get the result from the last tool execution
    last_result = intermediate_steps[-1][1]
    
    # If it's already a DataFrame, return it
    if isinstance(last_result, pd.DataFrame):
        return last_result
    
    # If it's a string representation, try to extract the query and re-execute
    action = intermediate_steps[-1][0]
    if hasattr(action, 'tool_input') and 'query' in action.tool_input:
        query = action.tool_input['query']
        print(f"Executing query: {query}")
        try:
            result_df = eval(query.split('\n')[-1])
            if isinstance(result_df, pd.DataFrame):
                return result_df
        except:
            pass
    return last_result
