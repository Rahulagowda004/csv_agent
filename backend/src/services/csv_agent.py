# OpenAI Agents SDK CSV Analysis Agent Service
import asyncio
import os
import json
import time
from pathlib import Path
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv

from agents import Agent, Runner, AgentOutputSchema, enable_verbose_stdout_logging
from agents.model_settings import ModelSettings
from agents.memory import SQLiteSession

# Import system prompt, memory config, and models
from src.constants.prompts import CSV_AGENT_SYSTEM_PROMPT
from src.constants.memory_config import setup_memory_db, get_session_id
from src.constants.model_properties import CSVAgentResponse
# Import the tools directly from csv_server
from src.core.csv_server import analyze_csv_data, manipulate_table, create_visualization, execute_code

# Load environment variables from .env file in project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")


class CSVAgentService:
    """Service class that handles CSV agent business logic."""
    
    def __init__(self):
        """Initialize the CSV agent service."""
        self.openai_api_key = openai_api_key
        self.db_path = None
        self.sessions = {}  # Store active sessions
        
    def _create_user_folder(self, user_id: str) -> str:
        """
        Create user-specific folders for CSV data and visualizations.
        
        Args:
            user_id: User identifier
            
        Returns:
            Path to the user's CSV folder (data/csv/user_id/)
        """
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        print(f"🔍 DEBUG: Project root: {project_root}")
        print(f"🔍 DEBUG: Original user_id: {user_id}")
        
        # Create user folder path (sanitize user_id for filesystem)
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_user_id:
            safe_user_id = "default_user"
        
        print(f"🔍 DEBUG: Sanitized user_id: {safe_user_id}")
        
        # Create separate folders for CSV files and plots
        csv_folder = os.path.join(project_root, "data", "csv", safe_user_id)
        plots_folder = os.path.join(project_root, "data", "plots", safe_user_id)
        
        print(f"🔍 DEBUG: CSV folder path: {csv_folder}")
        print(f"🔍 DEBUG: Plots folder path: {plots_folder}")
        
        # Create both folders if they don't exist with proper permissions
        os.makedirs(csv_folder, mode=0o755, exist_ok=True)
        os.makedirs(plots_folder, mode=0o755, exist_ok=True)
        
        # Verify folders were created
        if os.path.exists(csv_folder):
            print(f"✅ CSV folder created successfully: {csv_folder}")
        else:
            print(f"❌ Failed to create CSV folder: {csv_folder}")
            
        if os.path.exists(plots_folder):
            print(f"✅ Plots folder created successfully: {plots_folder}")
        else:
            print(f"❌ Failed to create plots folder: {plots_folder}")
        
        return csv_folder
    
    def upload_csv_file(self, file_path: str, user_id: str) -> str:
        """
        Upload and standardize CSV file to user's folder as 'data.csv'.
        
        Args:
            file_path: Path to the uploaded CSV file
            user_id: User identifier
            
        Returns:
            Path to the standardized CSV file (data/csv/user_id/data.csv)
        """
        # Create user folder
        csv_folder = self._create_user_folder(user_id)
        
        # Standardized filename
        standardized_path = os.path.join(csv_folder, "data.csv")
        
        # Copy/rename the file to standardized name
        import shutil
        shutil.copy2(file_path, standardized_path)
        
        print(f"📄 Uploaded CSV file: {file_path} → {standardized_path}")
        return standardized_path
        
    async def process_message(self, message: str, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user message through the CSV agent.
        
        Args:
            message: User's message/question
            user_id: User ID for folder management and data isolation
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Dict containing the agent's structured response
        """
        print(f"🔄 CSVAgentService: Processing message: '{message[:100]}{'...' if len(message) > 100 else ''}'")
        print(f"👤 CSVAgentService: User ID: {user_id}")
        
        try:
            # Ensure user folders exist
            self._create_user_folder(user_id)
            print(f"📁 CSVAgentService: User folders ready for user: {user_id}")
            # Set up memory database if not already done
            if not self.db_path:
                print("📂 CSVAgentService: Setting up memory database...")
                self.db_path = setup_memory_db()
                print(f"✅ CSVAgentService: Memory database initialized at: {self.db_path}")
            
            # Generate session ID if not provided
            if not session_id:
                session_id = get_session_id()
                print(f"🆔 CSVAgentService: Generated new session ID: {session_id}")
            else:
                print(f"🆔 CSVAgentService: Using existing session ID: {session_id}")
            
            # Get or create memory session
            if session_id not in self.sessions:
                print(f"🧠 CSVAgentService: Creating new memory session for: {session_id}")
                self.sessions[session_id] = SQLiteSession(
                    session_id=session_id,
                    db_path=str(self.db_path)
                )
            else:
                print(f"🧠 CSVAgentService: Using existing memory session for: {session_id}")
            
            memory_session = self.sessions[session_id]
            
            print("🤖 CSVAgentService: Creating CSV analysis agent...")
            # Create the agent with structured output and memory using tools from csv_server
            # Add folder paths to system prompt
            agent_instructions = f"""{CSV_AGENT_SYSTEM_PROMPT}

**IMPORTANT - Your user folder: {user_id}**
- CSV data: `data/csv/{user_id}/data.csv`
- Save plots: `data/plots/{user_id}/`"""
            
            # Create agent with tools from csv_server
            agent = Agent(
                name="CSV Analysis Agent",
                instructions=agent_instructions,
                tools=[analyze_csv_data, manipulate_table, create_visualization, execute_code],  # Use tools directly from csv_server
                output_type=AgentOutputSchema(CSVAgentResponse, strict_json_schema=False),
                model_settings=ModelSettings(
                    model="gpt-5",
                    temperature=0,
                    tool_choice="auto"
                ),
            )
            print("✅ CSVAgentService: Agent created successfully")
                
            print("🚀 CSVAgentService: Running agent with user message...")
            # Run agent with persistent memory
            result = await Runner.run(
                agent, 
                input=message,
                session=memory_session,
                max_turns=10
            )
            print("✅ CSVAgentService: Agent execution completed")
                
       
            print("📊 CSVAgentService: Processing agent response...")
            # Extract structured response
            if isinstance(result.final_output, CSVAgentResponse):
                response_data = result.final_output.model_dump()
                print("✅ CSVAgentService: Structured response extracted successfully")
            else:
                print("⚠️ CSVAgentService: Falling back to unstructured response")
                # Fallback if structured output fails
                response_data = {
                    "text": str(result.final_output),
                    "steps": None,
                    "image_paths": None,
                    "table_visualization": None,
                    "suggested_next_steps": None
                }
            
            # Add session ID and user ID to response
            response_data["session_id"] = session_id
            response_data["user_id"] = user_id
            
            # Show memory stats
            try:
                memory_items = await memory_session.get_items()
                total_items = len(memory_items) if memory_items else 0
                print(f"💭 CSVAgentService: Conversation memory contains {total_items} items")
            except Exception as mem_error:
                print(f"⚠️ CSVAgentService: Could not retrieve memory stats: {mem_error}")
            
            print("✅ CSVAgentService: Message processing completed successfully")
            return response_data
                
        except Exception as e:
            print(f"❌ CSVAgentService: Error processing message: {str(e)}")
            print(f"❌ CSVAgentService: Exception type: {type(e).__name__}")
            # Return error in structured format
            return {
                "text": f"Error processing request: {str(e)}",
                "steps": None,
                "image_paths": None,
                "table_visualization": None,
                "suggested_next_steps": None,
                "session_id": session_id or get_session_id(),
                "user_id": user_id
            }
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear a specific session's memory."""
        try:
            if session_id in self.sessions:
                await self.sessions[session_id].clear_session()
                self.sessions[session_id].close()
                del self.sessions[session_id]
                return True
            return False
        except Exception:
            return False
    
    def close_all_sessions(self):
        """Close all active sessions."""
        for session in self.sessions.values():
            try:
                session.close()
            except Exception:
                pass
        self.sessions.clear()
