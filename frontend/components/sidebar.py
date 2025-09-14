"""
Sidebar component for displaying response details and navigation.
"""

import streamlit as st
import os
from typing import TYPE_CHECKING, Dict, Any, List
from PIL import Image

if TYPE_CHECKING:
    from src.chat_interface import ChatInterface

def display_image_in_sidebar(image_path: str):
    """Display an image in the sidebar.
    
    Args:
        image_path: Path to the image file (as sent by backend)
    """
    try:
        # Get project root (go up from frontend/components/ to project root)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))  # frontend/components/
        frontend_dir = os.path.dirname(current_file_dir)  # frontend/
        project_root = os.path.dirname(frontend_dir)  # project root
        
        # Handle different path formats from backend
        if os.path.isabs(image_path):
            # Absolute path - use as is
            full_image_path = image_path
        elif image_path.startswith('data/plots/'):
            # Backend sends paths like "data/plots/user_id/image.png" - prepend project root
            full_image_path = os.path.join(project_root, image_path)
        elif image_path.startswith('backend/data/plots/'):
            # Backend might send paths like "backend/data/plots/user_id/image.png" - prepend project root
            full_image_path = os.path.join(project_root, image_path)
        else:
            # Fallback - treat as relative to project root
            full_image_path = os.path.join(project_root, image_path)
        
        if os.path.exists(full_image_path):
            image = Image.open(full_image_path)
            # Resize image to be smaller for sidebar (max width 150px)
            image.thumbnail((150, 150), Image.Resampling.LANCZOS)
            st.image(image, use_container_width=True)
        else:
            st.error(f"Image not found: {full_image_path}")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def display_images_grid_in_sidebar(image_paths: List[str]):
    """Display multiple images in a compact grid format in the sidebar.
    
    Args:
        image_paths: List of image paths to display
    """
    if not image_paths:
        return
    
    # Display images in rows of 2 for sidebar (smaller space)
    for i in range(0, len(image_paths), 2):
        row_images = image_paths[i:i+2]
        cols = st.columns(2)
        
        for j, image_path in enumerate(row_images):
            with cols[j]:
                try:
                    # Get project root
                    current_file_dir = os.path.dirname(os.path.abspath(__file__))
                    frontend_dir = os.path.dirname(current_file_dir)
                    project_root = os.path.dirname(frontend_dir)
                    
                    # Handle different path formats from backend
                    if os.path.isabs(image_path):
                        # Absolute path - use as is
                        full_image_path = image_path
                    elif image_path.startswith('data/plots/'):
                        # Backend sends paths like "data/plots/user_id/image.png" - prepend project root
                        full_image_path = os.path.join(project_root, image_path)
                    elif image_path.startswith('backend/data/plots/'):
                        # Backend might send paths like "backend/data/plots/user_id/image.png" - prepend project root
                        full_image_path = os.path.join(project_root, image_path)
                    else:
                        # Fallback - treat as relative to project root
                        full_image_path = os.path.join(project_root, image_path)
                    
                    if os.path.exists(full_image_path):
                        image = Image.open(full_image_path)
                        # Resize image to be smaller for sidebar (max width 120px)
                        image.thumbnail((120, 120), Image.Resampling.LANCZOS)
                        
                        # Create a unique key for this image
                        image_key = f"sidebar_image_{i}_{j}"
                        
                        # Show thumbnail in grid with clickable button
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.image(image, use_container_width=True)
                        with col2:
                            if st.button("🔍", key=f"zoom_{image_key}", help="Click to enlarge"):
                                # Store the full image path in session state for popup
                                st.session_state["popup_image"] = full_image_path
                                st.session_state["popup_image_name"] = os.path.basename(image_path)
                                st.rerun()
                    else:
                        st.error(f"Image not found: {full_image_path}")
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")

def render_sidebar_content(chat_interface: 'ChatInterface'):
    """Render expandable content in the sidebar.
    
    Args:
        chat_interface: ChatInterface instance for accessing response data
    """
    with st.sidebar:
        st.header("🛠️ CSV Analysis Agent")
        st.write("Upload your CSV file and start asking questions about your data!")
        
        # User info and session management
        if chat_interface.is_authenticated():
            user_info = chat_interface.get_user_info()
            st.write(f"**Logged in as:** {user_info['username']}")
            if st.button("🔄 Reset Session"):
                chat_interface.clear_session()
                st.rerun()
        
        st.markdown("---")
        
        # Response details section
        response_data = chat_interface.get_current_response()
        if response_data:
            st.header("Response Details")
            
            # Steps section
            if response_data.get("steps"):
                with st.expander("🔄 Processing Steps", expanded=False):
                    for i, step in enumerate(response_data["steps"], 1):
                        st.write(f"{i}. {step}")
            
            # Suggested next steps
            if response_data.get("suggested_next_steps"):
                with st.expander("💡 Suggested Next Steps", expanded=False):
                    for i, suggestion in enumerate(response_data["suggested_next_steps"], 1):
                        if st.button(f"{i}. {suggestion}", key=f"suggestion_{i}"):
                            # Add the suggestion to the chat input
                            st.session_state.suggested_message = suggestion
                            st.rerun()
            
            # Table visualization
            if response_data.get("table_visualization"):
                with st.expander("📊 Table Data", expanded=False):
                    table_data = response_data["table_visualization"]
                    if isinstance(table_data, list):
                        # Convert list of dictionaries to DataFrame for better display
                        import pandas as pd
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)
                    elif isinstance(table_data, dict):
                        st.json(table_data)
                    else:
                        st.write(table_data)
            
            # Images section
            if response_data.get("image_paths"):
                with st.expander("📈 Generated Visualizations", expanded=True):
                    display_images_grid_in_sidebar(response_data["image_paths"])
            
            # Session info
            with st.expander("ℹ️ Session Info", expanded=False):
                st.write(f"**User ID:** {response_data.get('user_id', 'N/A')}")
                st.write(f"**Session ID:** {response_data.get('session_id', 'N/A')}")
        
        st.markdown("---")
        
        # Usage instructions
        st.markdown("### 📋 How to use:")
        st.markdown("""
        1. Enter your name to start
        2. Upload a CSV file
        3. Ask questions about your data
        4. View detailed responses in the sidebar
        5. Click on suggested questions for more insights
        """)
        
        # API connection status
        st.markdown("### 🔗 Connection Status:")
        if hasattr(chat_interface, 'api_client') and chat_interface.api_client.check_health():
            st.success("✅ Connected to API")
        else:
            st.error("❌ API Disconnected")
    
