"""
API Client for communicating with the FastAPI backend.
"""

import requests
from typing import Dict, Any, Optional
import streamlit as st

class APIClient:
    """Client for interacting with the CSV Agent FastAPI backend."""
    
    def __init__(self, base_url: str):
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the FastAPI server
        """
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/chat"
        self.upload_endpoint = f"{base_url}/upload"
        self.health_endpoint = f"{base_url}/health"
    
    def check_health(self) -> bool:
        """Check if the FastAPI server is running.
        
        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def upload_csv_file(self, file, user_id: str) -> Dict[str, Any]:
        """Upload a CSV file to the FastAPI server.
        
        Args:
            file: File object to upload
            user_id: User ID for folder management
            
        Returns:
            Dict containing success status and response data or error
        """
        print(f"📤 Frontend: Starting file upload for user: {user_id}")
        print(f"📤 Frontend: File name: {file.name}")
        print(f"📤 Frontend: Upload endpoint: {self.upload_endpoint}")
        
        try:
            # Reset file pointer to beginning
            file.seek(0)
            
            # Prepare files for multipart form
            files = {"file": (file.name, file.getvalue(), "text/csv")}
            
            # Send user_id as query parameter
            params = {"user_id": user_id}
            
            print(f"📤 Frontend: Sending POST request with params: {params}")
            response = requests.post(self.upload_endpoint, files=files, params=params)
            print(f"📤 Frontend: Response status: {response.status_code}")
            
            response.raise_for_status()
            response_data = response.json()
            print(f"📤 Frontend: Upload successful: {response_data}")
            return {"success": True, "data": response_data}
        except requests.RequestException as e:
            print(f"❌ Frontend: Upload failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def send_chat_message(self, message: str, user_id: str, session_id: str) -> Dict[str, Any]:
        """Send a message to the chat endpoint.
        
        Args:
            message: User message to send
            user_id: User ID
            session_id: Session ID for conversation continuity
            
        Returns:
            Dict containing success status and response data or error
        """
        print(f"💬 Frontend: Sending chat message for user: {user_id}")
        print(f"💬 Frontend: Session ID: {session_id}")
        print(f"💬 Frontend: Message: {message[:100]}...")
        print(f"💬 Frontend: Chat endpoint: {self.chat_endpoint}")
        
        try:
            payload = {
                "message": message,
                "user_id": user_id,
                "session_id": session_id
            }
            
            print(f"💬 Frontend: Sending POST request with payload: {payload}")
            response = requests.post(self.chat_endpoint, json=payload)
            print(f"💬 Frontend: Response status: {response.status_code}")
            
            response.raise_for_status()
            response_data = response.json()
            print(f"💬 Frontend: Chat response received")
            print(f"💬 Frontend: Response keys: {list(response_data.keys())}")
            if response_data.get("image_paths"):
                print(f"💬 Frontend: Image paths: {response_data['image_paths']}")
            return {"success": True, "data": response_data}
        except requests.RequestException as e:
            print(f"❌ Frontend: Chat request failed: {str(e)}")
            return {"success": False, "error": str(e)}
