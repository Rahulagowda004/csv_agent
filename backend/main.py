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

from src.services.csv_agent import CSVAgentService

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
