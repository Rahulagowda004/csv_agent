"""
Chat Interface for managing conversation state and rendering chat messages.
"""

import streamlit as st
import uuid
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from .api_client import APIClient
import os
from PIL import Image

class ChatInterface:
    """Manages chat interface state and interactions."""
    
    def __init__(self, api_client: APIClient):
        """Initialize the chat interface.
        
        Args:
            api_client: API client for backend communication
        """
        self.api_client = api_client
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_response' not in st.session_state:
            st.session_state.current_response = None
        if 'username' not in st.session_state:
            st.session_state.username = ""
    
    def generate_user_id(self, username: str) -> str:
        """Generate a unique user ID based on username and UUID.
        
        Args:
            username: User's display name
            
        Returns:
            str: Unique user ID
        """
        unique_id = str(uuid.uuid4())[:8]
        return f"{username.lower().replace(' ', '_')}_{unique_id}"
    
    def set_user(self, username: str):
        """Set the current user and generate user ID.
        
        Args:
            username: User's display name
        """
        st.session_state.username = username.strip()
        st.session_state.user_id = self.generate_user_id(username)
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated.
        
        Returns:
            bool: True if user is authenticated
        """
        return st.session_state.user_id is not None
    
    def get_user_info(self) -> Dict[str, str]:
        """Get current user information.
        
        Returns:
            Dict containing username and user_id
        """
        return {
            "username": st.session_state.username,
            "user_id": st.session_state.user_id,
            "session_id": st.session_state.session_id
        }
    
    def upload_file(self, file) -> Dict[str, Any]:
        """Upload a CSV file using the API client.
        
        Args:
            file: File object to upload
            
        Returns:
            Dict containing upload result
        """
        return self.api_client.upload_csv_file(file, st.session_state.user_id)
    
    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a chat message using the API client.
        
        Args:
            message: Message to send
            
        Returns:
            Dict containing chat response
        """
        print(f"💬 Frontend: ChatInterface sending message: {message[:50]}...")
        print(f"💬 Frontend: User ID: {st.session_state.user_id}")
        print(f"💬 Frontend: Session ID: {st.session_state.session_id}")
        
        result = self.api_client.send_chat_message(
            message, 
            st.session_state.user_id, 
            st.session_state.session_id
        )
        
        print(f"💬 Frontend: ChatInterface received result: {result.get('success', False)}")
        return result
    
    def add_to_history(self, user_message: str, bot_response: Dict[str, Any]):
        """Add a message exchange to chat history.
        
        Args:
            user_message: User's message
            bot_response: Bot's response data
        """
        print(f"📝 Frontend: Adding to chat history")
        print(f"📝 Frontend: User message: {user_message[:50]}...")
        print(f"📝 Frontend: Bot response keys: {list(bot_response.keys())}")
        if bot_response.get("image_paths"):
            print(f"📝 Frontend: Bot response has {len(bot_response['image_paths'])} images")
        
        st.session_state.chat_history.append((user_message, bot_response))
        st.session_state.current_response = bot_response
        
        # Update session ID if provided
        if bot_response.get("session_id"):
            old_session_id = st.session_state.session_id
            st.session_state.session_id = bot_response["session_id"]
            print(f"📝 Frontend: Updated session ID: {old_session_id} -> {st.session_state.session_id}")
    
    def get_chat_history(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get current chat history.
        
        Returns:
            List of tuples containing (user_message, bot_response)
        """
        return st.session_state.chat_history
    
    def get_current_response(self) -> Optional[Dict[str, Any]]:
        """Get the current bot response for sidebar display.
        
        Returns:
            Current response data or None
        """
        return st.session_state.current_response
    
    def clear_session(self):
        """Clear all session data."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    def display_table(self, table_data: List[Dict[str, Any]]):
        """Display table data from JSON structure.
        
        Args:
            table_data: List of dictionaries containing table data
        """
        print(f"📊 Frontend: Attempting to display table data")
        print(f"📊 Frontend: Table data type: {type(table_data)}")
        print(f"📊 Frontend: Table data length: {len(table_data) if table_data else 0}")
        
        try:
            # Handle list of dictionaries (new format)
            if isinstance(table_data, list) and len(table_data) > 0:
                # Convert list of dictionaries to DataFrame
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
                print(f"📊 Frontend: Successfully displayed DataFrame with {len(df)} rows")
                return
            
            # Handle different table data structures (legacy support)
            elif isinstance(table_data, dict):
                # Check if it's a pandas-style data structure
                if 'data' in table_data and 'columns' in table_data:
                    # DataFrame-like structure
                    df = pd.DataFrame(table_data['data'], columns=table_data['columns'])
                    st.dataframe(df, use_container_width=True)
                    print(f"📊 Frontend: Successfully displayed DataFrame with {len(df)} rows")
                
                # Check if it's a list of records (common format)
                elif isinstance(list(table_data.values())[0], list) and len(table_data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True)
                    print(f"📊 Frontend: Successfully displayed DataFrame with {len(df)} rows")
                
                # Check if it's a nested structure with sample data
                elif 'sample_data' in table_data:
                    sample_data = table_data['sample_data']
                    if 'first_5_rows' in sample_data:
                        df = pd.DataFrame(sample_data['first_5_rows'])
                        st.subheader("📋 Sample Data (First 5 rows)")
                        st.dataframe(df, use_container_width=True)
                        print(f"📊 Frontend: Successfully displayed sample data with {len(df)} rows")
                    
                    if 'last_5_rows' in sample_data:
                        df = pd.DataFrame(sample_data['last_5_rows'])
                        st.subheader("📋 Sample Data (Last 5 rows)")
                        st.dataframe(df, use_container_width=True)
                        print(f"📊 Frontend: Successfully displayed last 5 rows with {len(df)} rows")
                
                # Check if it's statistics data
                elif 'numeric_statistics' in table_data or 'categorical_statistics' in table_data:
                    # Display statistics in a more readable format
                    if 'numeric_statistics' in table_data:
                        st.subheader("📈 Numeric Statistics")
                        numeric_stats = table_data['numeric_statistics']
                        if numeric_stats:
                            df = pd.DataFrame(numeric_stats).T
                            st.dataframe(df, use_container_width=True)
                            print(f"📊 Frontend: Successfully displayed numeric statistics")
                    
                    if 'categorical_statistics' in table_data:
                        st.subheader("📊 Categorical Statistics")
                        cat_stats = table_data['categorical_statistics']
                        if cat_stats:
                            # Display categorical stats in a more readable format
                            for col, stats in cat_stats.items():
                                with st.expander(f"Column: {col}"):
                                    st.write(f"**Unique values:** {stats.get('unique_values', 'N/A')}")
                                    st.write(f"**Most frequent:** {stats.get('most_frequent', 'N/A')}")
                                    if 'frequency' in stats and stats['frequency']:
                                        freq_df = pd.DataFrame(list(stats['frequency'].items()), 
                                                             columns=['Value', 'Count'])
                                        st.dataframe(freq_df, use_container_width=True)
                            print(f"📊 Frontend: Successfully displayed categorical statistics")
                
                # Check if it's a simple key-value structure
                elif all(isinstance(v, (str, int, float, bool)) for v in table_data.values()):
                    # Simple key-value pairs
                    df = pd.DataFrame(list(table_data.items()), columns=['Key', 'Value'])
                    st.dataframe(df, use_container_width=True)
                    print(f"📊 Frontend: Successfully displayed key-value table")
                
                else:
                    # Try to display as JSON for debugging
                    st.json(table_data)
                    print(f"📊 Frontend: Displayed as JSON (fallback)")
            
            else:
                st.error("Unsupported table data format")
                print(f"❌ Frontend: Unsupported table data format: {type(table_data)}")
                
        except Exception as e:
            st.error(f"Error displaying table: {str(e)}")
            print(f"❌ Frontend: Error displaying table: {str(e)}")
            # Fallback to JSON display
            st.json(table_data)

    def display_image(self, image_path: str):
        """Display an image from the given path.
        
        Args:
            image_path: Path to the image file (relative to project root)
        """
        print(f"🖼️ Frontend: Attempting to display image: {image_path}")
        try:
            # Convert relative path to absolute path from project root
            if not os.path.isabs(image_path):
                # Get project root (go up from frontend/src/ to project root)
                current_file_dir = os.path.dirname(os.path.abspath(__file__))  # frontend/src/
                frontend_dir = os.path.dirname(current_file_dir)  # frontend/
                project_root = os.path.dirname(frontend_dir)  # project root
                full_image_path = os.path.join(project_root, image_path)
            else:
                full_image_path = image_path
            
            print(f"🖼️ Frontend: Full image path: {full_image_path}")
            print(f"🖼️ Frontend: Image exists: {os.path.exists(full_image_path)}")
            
            if os.path.exists(full_image_path):
                image = Image.open(full_image_path)
                clean_name = os.path.splitext(os.path.basename(image_path))[0]
                st.image(image, caption=clean_name, use_container_width=True)
                print(f"🖼️ Frontend: Successfully displayed image: {os.path.basename(image_path)}")
            else:
                st.error(f"Image not found: {full_image_path}")
                print(f"❌ Frontend: Image not found at: {full_image_path}")
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            print(f"❌ Frontend: Error displaying image: {str(e)}")
    
    def display_images_grid(self, image_paths: List[str]):
        """Display multiple images in a compact grid format.
        
        Args:
            image_paths: List of image paths to display
        """
        if not image_paths:
            return
            
        print(f"🖼️ Frontend: Displaying {len(image_paths)} images in grid format")
        
        # Display images in rows of 4
        for i in range(0, len(image_paths), 4):
            row_images = image_paths[i:i+4]
            cols = st.columns(4)
            
            for j, image_path in enumerate(row_images):
                with cols[j]:
                    try:
                        # Convert relative path to absolute path from project root
                        if not os.path.isabs(image_path):
                            current_file_dir = os.path.dirname(os.path.abspath(__file__))
                            frontend_dir = os.path.dirname(current_file_dir)
                            project_root = os.path.dirname(frontend_dir)
                            full_image_path = os.path.join(project_root, image_path)
                        else:
                            full_image_path = image_path
                        
                        if os.path.exists(full_image_path):
                            image = Image.open(full_image_path)
                            # Resize image to be smaller (max width 200px)
                            image.thumbnail((200, 200), Image.Resampling.LANCZOS)
                            
                            # Show thumbnail in grid with clickable button
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                clean_name = os.path.splitext(os.path.basename(image_path))[0]
                                st.image(image, caption=f"{clean_name} (Click 🔍 to enlarge)", use_container_width=True)
                            with col2:
                                # Create unique key for chat history images
                                unique_key = f"chat_history_zoom_{i}_{j}_{hash(image_path)}"
                                if st.button("🔍", key=unique_key, help="Click to enlarge image"):
                                    # Store the full image path in session state for popup
                                    st.session_state["popup_image"] = full_image_path
                                    st.session_state["popup_image_name"] = os.path.basename(image_path)
                                    st.rerun()
                            print(f"🖼️ Frontend: Successfully displayed grid image: {os.path.basename(image_path)}")
                        else:
                            st.error(f"Image not found: {full_image_path}")
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
                        print(f"❌ Frontend: Error displaying grid image: {str(e)}")
    
    def display_new_message_images(self, image_paths: List[str]):
        """Display multiple images for new messages with unique keys.
        
        Args:
            image_paths: List of image paths to display
        """
        if not image_paths:
            return
            
        print(f"🖼️ Frontend: Displaying {len(image_paths)} new message images")
        
        # Display images in rows of 4
        for i in range(0, len(image_paths), 4):
            row_images = image_paths[i:i+4]
            cols = st.columns(4)
            
            for j, image_path in enumerate(row_images):
                with cols[j]:
                    try:
                        # Convert relative path to absolute path from project root
                        if not os.path.isabs(image_path):
                            current_file_dir = os.path.dirname(os.path.abspath(__file__))
                            frontend_dir = os.path.dirname(current_file_dir)
                            project_root = os.path.dirname(frontend_dir)
                            full_image_path = os.path.join(project_root, image_path)
                        else:
                            full_image_path = image_path
                        
                        if os.path.exists(full_image_path):
                            image = Image.open(full_image_path)
                            # Resize image to be smaller (max width 200px)
                            image.thumbnail((200, 200), Image.Resampling.LANCZOS)
                            
                            # Show thumbnail in grid with clickable button
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                clean_name = os.path.splitext(os.path.basename(image_path))[0]
                                st.image(image, caption=f"{clean_name} (Click 🔍 to enlarge)", use_container_width=True)
                            with col2:
                                # Create unique key for new message images
                                unique_key = f"new_message_zoom_{i}_{j}_{hash(image_path)}"
                                if st.button("🔍", key=unique_key, help="Click to enlarge"):
                                    # Store the full image path in session state for popup
                                    st.session_state["popup_image"] = full_image_path
                                    st.session_state["popup_image_name"] = os.path.basename(image_path)
                                    st.rerun()
                            print(f"🖼️ Frontend: Successfully displayed new message image: {os.path.basename(image_path)}")
                        else:
                            st.error(f"Image not found: {full_image_path}")
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
    
    def render_chat(self):
        """Render the main chat interface."""
        st.subheader("💬 Chat History")
        
        # Display chat history
        chat_history = self.get_chat_history()
        for i, (user_msg, bot_response) in enumerate(chat_history):
            # User message
            with st.chat_message("user"):
                st.write(user_msg)
            
            # Bot response
            with st.chat_message("assistant"):
                # Display images first if any
                if bot_response.get("image_paths"):
                    self.display_images_grid(bot_response["image_paths"])
                
                # Display table data if any
                if bot_response.get("table_visualization"):
                    self.display_table(bot_response["table_visualization"])
                
                # Then display the text response
                st.write(bot_response.get("text", ""))
                
                # Display response time in bottom right corner (skip for most recent message to avoid duplication)
                is_most_recent = (i == len(chat_history) - 1)
                if bot_response.get("response_time_seconds") and not is_most_recent:
                    response_time = bot_response["response_time_seconds"]
                    col1, col2 = st.columns([1, 1])
                    with col2:
                        st.caption(f"⏱️ {response_time}s", help="Response time")
        
        # Chat input
        message_input = st.chat_input("Ask me anything about your CSV data...")
        
        # Handle suggested message from sidebar
        if hasattr(st.session_state, 'suggested_message'):
            message_input = st.session_state.suggested_message
            delattr(st.session_state, 'suggested_message')
        
        # Process new message
        if message_input:
            # Add user message to chat history display
            with st.chat_message("user"):
                st.write(message_input)
            
            # Send message to API
            with st.spinner("Thinking..."):
                chat_result = self.send_message(message_input)
            
            if chat_result["success"]:
                response_data = chat_result["data"]
                
                # Display bot response
                with st.chat_message("assistant"):
                    # Display images first if any
                    if response_data.get("image_paths"):
                        self.display_new_message_images(response_data["image_paths"])
                    
                    # Display table data if any
                    if response_data.get("table_visualization"):
                        self.display_table(response_data["table_visualization"])
                    
                    # Then display the text response
                    st.write(response_data.get("text", ""))
                    
                    # Display response time immediately for new messages
                    if response_data.get("response_time_seconds"):
                        response_time = response_data["response_time_seconds"]
                        col1, col2 = st.columns([1, 1])
                        with col2:
                            st.caption(f"⏱️ {response_time}s", help="Response time")
                
                # Add to chat history
                self.add_to_history(message_input, response_data)
            else:
                st.error(f"❌ Error: {chat_result['error']}")
