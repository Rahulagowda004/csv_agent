"""
FastAPI Chat Endpoint for CSV Agent
Main entry point that provides a REST API interface for the CSV analysis agent.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import shutil
import time
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from src.utils.utils import extract_dataframe_from_result

from src.services.csv_agent import CSVAgentService

# Load environment variables
load_dotenv()

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

# Initialize FastAPI app
app = FastAPI(
    title="CSV Agent Chat API",
    description="A REST API interface for CSV data analysis using OpenAI Agents SDK",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the CSV agent service
csv_agent_service = CSVAgentService()

# Initialize LLM for analysis agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    user_id: str  # Required user ID for folder management
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    text: str
    steps: Optional[List[str]] = None
    image_paths: Optional[List[str]] = None
    table_visualization: Optional[List[Dict[str, Any]]] = None
    suggested_next_steps: Optional[List[str]] = None
    session_id: str
    user_id: str  # Include user_id in response
    response_time_seconds: float  # Response time in seconds

class UploadResponse(BaseModel):
    message: str
    user_id: str
    filename: str
    file_path: str
    file_size: int

class AnalysisRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    text: str
    steps: Optional[List[str]] = None
    image_paths: Optional[List[str]] = None
    table_visualization: Optional[List[Dict[str, Any]]] = None
    suggested_next_steps: Optional[List[str]] = None
    session_id: str
    user_id: str  # Include user_id in response
    response_time_seconds: float  # Response time in seconds
    csv_file_path: Optional[str] = None  # Path to saved CSV file for large datasets

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that processes user messages through the CSV agent.
    
    Args:
        request: ChatRequest containing the user message and optional session_id
        
    Returns:
        ChatResponse with the agent's structured response
    """
    try:
        # Start timing the request
        start_time = time.time()
        
        # Call the CSV agent service
        response = await csv_agent_service.process_message(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Add response time to the response
        response["response_time_seconds"] = round(response_time, 2)
        
        return ChatResponse(**response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_csv(user_id: str, file: UploadFile = File(...)):
    """
    Upload a CSV file to the user's folder.
    
    Args:
        user_id: User ID for folder management
        file: CSV file to upload
        
    Returns:
        UploadResponse with upload details
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Validate file size (limit to 100MB)
        # max_size = 100 * 1024 * 1024  # 100MB
        # file_content = await file.read()
        # if len(file_content) > max_size:
        #     raise HTTPException(status_code=400, detail="File too large. Maximum size is 100MB")
        
        # Reset file pointer
        await file.seek(0)
        
        # Create a temporary file to save the uploaded content
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        try:
            # Use CSVAgentService to standardize the filename to "data.csv"
            print(f"🔍 DEBUG: Uploading file for user_id: {user_id}")
            standardized_path = csv_agent_service.upload_csv_file(temp_file_path, user_id)
            print(f"🔍 DEBUG: File saved to: {standardized_path}")
            
            # Verify the file exists
            if not os.path.exists(standardized_path):
                raise HTTPException(status_code=500, detail=f"File was not saved correctly to {standardized_path}")
            
            # Get file size
            file_size = os.path.getsize(standardized_path)
            print(f"🔍 DEBUG: File size: {file_size} bytes")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        return UploadResponse(
            message="File uploaded successfully",
            user_id=user_id,
            filename="data.csv",  # Standardized filename
            file_path=standardized_path,
            file_size=file_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/analyze", response_model=AnalysisResponse)
async def csv_analysis_agent(request: AnalysisRequest):
    """
    CSV Analysis Agent endpoint that processes natural language queries on CSV data.
    
    Args:
        request: AnalysisRequest containing the query and user_id
        
    Returns:
        AnalysisResponse with the analysis result
    """
    try:
        # Start timing the request
        start_time = time.time()
        
        # Get user's CSV file path
        user_csv_folder = csv_agent_service._create_user_folder(request.user_id)
        
        # Find the CSV file in user's folder
        csv_files = [f for f in os.listdir(user_csv_folder) if f.lower().endswith('.csv')]
        if not csv_files:
            raise HTTPException(status_code=404, detail=f"No CSV file found for user {request.user_id}. Please upload a CSV file first.")
        
        # Use the first CSV file found
        csv_file_path = os.path.join(user_csv_folder, csv_files[0])
        
        # Load the CSV data
        df = pd.read_csv(csv_file_path)
        
        # Create pandas DataFrame agent
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            return_intermediate_steps=True
        )

        # Execute the query
        result = agent_executor.invoke({"input": request.query})
        
        # Extract DataFrame from intermediate steps - simplified approach
        table_visualization = None
        saved_csv_path = None
        
        if "intermediate_steps" in result and result["intermediate_steps"]:
            for step in result["intermediate_steps"]:
                tool_result = step[1]
                
                # Check if the tool result is a DataFrame
                if isinstance(tool_result, pd.DataFrame):
                    # If more than 100 rows, save to CSV and limit table_visualization
                    if len(tool_result) > 100:
                        # Create user's plots folder to save the CSV
                        user_plots_folder = os.path.join("data", "plots", request.user_id)
                        os.makedirs(user_plots_folder, exist_ok=True)
                        
                        # Save the full DataFrame as CSV
                        csv_filename = f"query_result_{int(time.time())}.csv"
                        csv_path = os.path.join(user_plots_folder, csv_filename)
                        tool_result.to_csv(csv_path, index=False)
                        saved_csv_path = csv_path
                        
                        # Return only first 50 rows for table_visualization
                        table_visualization = tool_result.head(50).to_dict('records')
                        print(f"🔍 DEBUG: Large DataFrame ({len(tool_result)} rows) saved to {csv_path}, showing first 50 rows")
                    else:
                        table_visualization = tool_result.to_dict('records')
                        print(f"🔍 DEBUG: DataFrame with {len(tool_result)} rows returned in full")
                    break
                # Check if the tool result is a Series
                elif isinstance(tool_result, pd.Series):
                    df_from_series = tool_result.reset_index()
                    
                    # If more than 100 rows, save to CSV and limit table_visualization
                    if len(df_from_series) > 100:
                        # Create user's plots folder to save the CSV
                        user_plots_folder = os.path.join("data", "plots", request.user_id)
                        os.makedirs(user_plots_folder, exist_ok=True)
                        
                        # Save the full DataFrame as CSV
                        csv_filename = f"query_result_{int(time.time())}.csv"
                        csv_path = os.path.join(user_plots_folder, csv_filename)
                        df_from_series.to_csv(csv_path, index=False)
                        saved_csv_path = csv_path
                        
                        # Return only first 50 rows for table_visualization
                        table_visualization = df_from_series.head(50).to_dict('records')
                        print(f"🔍 DEBUG: Large Series ({len(df_from_series)} rows) saved to {csv_path}, showing first 50 rows")
                    else:
                        table_visualization = df_from_series.to_dict('records')
                        print(f"🔍 DEBUG: Series with {len(df_from_series)} rows returned in full")
                    break
        
        # Generate session_id if not provided
        session_id = request.session_id or f"analysis_{int(time.time())}_{request.user_id}"
        
        # Calculate response time
        end_time = time.time()
        response_time = end_time - start_time
        
        return AnalysisResponse(
            text=result['output'],
            steps=None,  # Could be populated with intermediate steps if needed
            image_paths=None,  # Could be populated with generated chart paths
            table_visualization=table_visualization,
            suggested_next_steps=None,  # Could be populated with suggestions
            session_id=session_id,
            user_id=request.user_id,
            response_time_seconds=round(response_time, 2),
            csv_file_path=saved_csv_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing CSV: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "CSV Agent Chat API"}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CSV Agent Chat API",
        "endpoints": {
            "chat": "/chat",
            "upload": "/upload", 
            "analyze": "/analyze",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
