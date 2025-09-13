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
    table_visualization: Optional[Dict[str, Any]] = None
    suggested_next_steps: Optional[List[str]] = None
    session_id: str
    user_id: str  # Include user_id in response

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
        # Call the CSV agent service
        response = await csv_agent_service.process_message(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
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
        
        # Create user CSV folder using the same logic as CSVAgentService
        # This returns data/csv/user_id/ folder path
        user_csv_folder = csv_agent_service._create_user_folder(user_id)
        
        # Remove any existing CSV files in the user's CSV folder (only one CSV per user)
        for existing_file in os.listdir(user_csv_folder):
            if existing_file.lower().endswith('.csv'):
                os.remove(os.path.join(user_csv_folder, existing_file))
        
        # Save the uploaded file
        file_path = os.path.join(user_csv_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return UploadResponse(
            message="File uploaded successfully",
            user_id=user_id,
            filename=file.filename,
            file_path=file_path,
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
