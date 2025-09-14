# OpenAI Agents SDK CSV Analysis Agent Service
import asyncio
import os
import json
import time
from pathlib import Path
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv

from agents import Agent, Runner, AgentOutputSchema, enable_verbose_stdout_logging
from agents.mcp import MCPServerStdio
from agents.model_settings import ModelSettings
from agents.memory import SQLiteSession

# Import system prompt, memory config, and models
from src.constants.prompts import CSV_AGENT_SYSTEM_PROMPT
from src.constants.memory_config import setup_memory_db, get_session_id
from src.constants.model_properties import CSVAgentResponse

# Load environment variables from .env file in project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

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
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Create user folder path (sanitize user_id for filesystem)
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_')).rstrip()
        if not safe_user_id:
            safe_user_id = "default_user"
        
        # Create separate folders for CSV files and plots
        csv_folder = os.path.join(project_root, "data", "csv", safe_user_id)
        plots_folder = os.path.join(project_root, "data", "plots", safe_user_id)
        
        # Create both folders if they don't exist with proper permissions
        os.makedirs(csv_folder, mode=0o755, exist_ok=True)
        os.makedirs(plots_folder, mode=0o755, exist_ok=True)
        
        print(f"📁 Created CSV folder: {csv_folder}")
        print(f"📊 Created plots folder: {plots_folder}")
        
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
            
            print("🔗 CSVAgentService: Connecting to MCP server...")
            # Set up MCP server connection for CSV tools (stdio)
            async with MCPServerStdio(
                name="CSV Analysis Server",
                params={
                    "command": "venv/bin/python",
                    "args": ["src/core/csv_server.py"],
                    "env": {"OPENAI_API_KEY": self.openai_api_key},
                },
            ) as mcp_server:
                print("✅ CSVAgentService: MCP server connected successfully")
                
                print("🤖 CSVAgentService: Creating CSV analysis agent...")
                # Create the agent with structured output and memory
                # Add folder paths to system prompt
                agent_instructions = f"""{CSV_AGENT_SYSTEM_PROMPT}

**IMPORTANT - Your user folder: {user_id}**
- CSV data: `data/csv/{user_id}/data.csv`
- Save plots: `data/plots/{user_id}/`"""
                
                agent = Agent(
                    name="CSV Analysis Agent",
                    instructions=agent_instructions,
                    mcp_servers=[mcp_server],
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
                
                # Log tool calls if available
                try:
                    if hasattr(result, 'turns') and result.turns:
                        print("🔧 CSVAgentService: Tool calls made:")
                        for i, turn in enumerate(result.turns):
                            if hasattr(turn, 'messages') and turn.messages:
                                for msg in turn.messages:
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        for tool_call in msg.tool_calls:
                                            print(f"  📞 Turn {i+1}: {tool_call.function.name}({tool_call.function.arguments[:100]}{'...' if len(tool_call.function.arguments) > 100 else ''})")
                    else:
                        print("🔧 CSVAgentService: No tool call information available")
                except Exception as tool_log_error:
                    print(f"⚠️ CSVAgentService: Could not log tool calls: {tool_log_error}")
                
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




# COMMENTED OUT - Main function moved to separate CLI script if needed
# This was the original CLI chat interface, now replaced by FastAPI endpoint

# async def main():
#     """Main function to run the CSV analysis agent with persistent memory."""
#     
#     # Enable verbose logging for debugging
#     enable_verbose_stdout_logging()
#     
#     # Set up memory database
#     db_path = setup_memory_db()
#     
#     # Create unique session ID for this chat session
#     session_id = get_session_id()
#     
#     # Initialize persistent memory session
#     memory_session = SQLiteSession(
#         session_id=session_id,
#         db_path=str(db_path)
#     )
#     
#     print("=" * 80)
#     print("🤖 CSV DATA ANALYSIS CHAT (OpenAI Agents SDK)")
#     print("=" * 80)
#     print("Welcome! I'm your CSV analysis assistant powered by OpenAI Agents SDK.")
#     print("I can help you analyze data, create visualizations, and answer questions about your CSV files.")
#     print("Type 'stop' to end the chat.")
#     print(f"💾 Session ID: {session_id}")
#     print(f"📂 Memory Database: {db_path}")
#     print("-" * 80)
#     
#     try:
#         # Set up MCP server connection for CSV tools (stdio)
#         async with MCPServerStdio(
#             name="CSV Analysis Server",
#             params={
#                 "command": "venv/bin/python",
#                 "args": ["src/core/csv_server.py"],
#                 "env": {"OPENAI_API_KEY": openai_api_key},
#             },
#         ) as mcp_server:
#             
#             # Create the agent with structured output and memory
#             agent = Agent(
#                 name="CSV Analysis Agent",
#                 instructions=CSV_AGENT_SYSTEM_PROMPT,
#                 mcp_servers=[mcp_server],
#                 output_type=AgentOutputSchema(CSVAgentResponse, strict_json_schema=False),  # Wrapped for compatibility
#                 model_settings=ModelSettings(
#                     model="gpt-5",
#                     temperature=0,
#                     tool_choice="auto"
#                 ),
#             )
#             
#             print("✅ Agent initialized successfully!")
#             print("🔗 MCP server connected")
#             print("🧠 Memory system ready")
#             print("-" * 80)
#             
#             # Interactive chat loop
#             while True:
#                 try:
#                     # Get user input
#                     user_input = input("\n💬 You: ").strip()
#                     
#                     # Check for exit condition
#                     if user_input.lower() == 'stop':
#                         print("\n👋 Goodbye! Thanks for using the CSV Analysis Chat!")
#                         break
#                     
#                     # Skip empty inputs
#                     if not user_input:
#                         print("Please enter a question or type 'stop' to exit.")
#                         continue
#                     
#                     print(f"\n🔄 Processing your request...")
#                     
#                     # Run agent with persistent memory
#                     result = await Runner.run(
#                         agent, 
#                         input=user_input,
#                         session=memory_session,  # Persistent conversation memory
#                         max_turns=10
#                     )
#                     
#                     # Extract structured response
#                     if isinstance(result.final_output, CSVAgentResponse):
#                         response_json = result.final_output.model_dump()
#                     else:
#                         # Fallback if structured output fails
#                         response_json = {
#                             "text": str(result.final_output),
#                             "steps": None,
#                             "image_paths": None,
#                             "table_visualization": None,
#                             "suggested_next_steps": None
#                         }
#                     
#                     # Display structured JSON response
#                     print(f"\n🤖 Assistant JSON Response:")
#                     print("-" * 50)
#                     print(json.dumps(response_json, indent=2, ensure_ascii=False))
#                     print("-" * 50)
#                     
#                     # Show memory stats
#                     memory_items = await memory_session.get_items(limit=1)
#                     total_items = len(await memory_session.get_items())
#                     print(f"💭 Conversation memory: {total_items} items stored")
#                     
#                 except KeyboardInterrupt:
#                     print("\n\n👋 Chat interrupted. Goodbye!")
#                     break
#                 except Exception as e:
#                     print(f"\n❌ Error: {str(e)}")
#                     print("Please try again or type 'stop' to exit.")
#                     
#                     # Optional: Clear corrupted session on repeated errors
#                     error_count = getattr(main, 'error_count', 0) + 1
#                     main.error_count = error_count
#                     if error_count >= 3:
#                         print("🔧 Multiple errors detected. Clearing session memory...")
#                         await memory_session.clear_session()
#                         main.error_count = 0
#     
#     except Exception as e:
#         print(f"\n💥 Critical error during setup: {str(e)}")
#         print("Please check your MCP server is running and OpenAI API key is valid.")
#     
#     finally:
#         # Clean up memory session
#         try:
#             memory_session.close()
#             print("💾 Memory session closed")
#         except Exception as e:
#             print(f"Warning: Error closing memory session: {e}")
#         
#         print("\n" + "=" * 80)
#         print("✅ CHAT SESSION ENDED")
#         print("=" * 80)
#             

# if __name__ == "__main__":
#     # Run the main chat application
#     asyncio.run(main())
#     
#     # Uncomment to test memory operations
#     # asyncio.run(test_memory_operations())