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
        
        # Backend now sends absolute paths, so use directly
        full_image_path = image_path
        
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
                    
                    # Backend now sends absolute paths, so use directly
                    full_image_path = image_path
                    
                    if os.path.exists(full_image_path):
                        image = Image.open(full_image_path)
                        # Resize image to be smaller for sidebar (max width 100px)
                        image.thumbnail((100, 100), Image.Resampling.LANCZOS)
                        
                        # Create a unique key for this image
                        image_key = f"sidebar_image_{i}_{j}"
                        
                        # Show thumbnail with sleek styling
                        st.markdown(f"""
                        <div style='border: 1px solid #e0e0e0; border-radius: 8px; padding: 0.3rem; margin: 0.2rem 0; background-color: white;'>
                        """, unsafe_allow_html=True)
                        
                        st.image(image, use_container_width=True)
                        
                        # Add zoom button below image
                        if st.button("🔍 Enlarge", key=f"zoom_{image_key}", help="Click to enlarge"):
                            # Store the full image path in session state for popup
                            st.session_state["popup_image"] = full_image_path
                            st.session_state["popup_image_name"] = os.path.basename(image_path)
                            st.rerun()
                        
                        st.markdown("</div>", unsafe_allow_html=True)
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
        # Add custom styling for sidebar
        st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: white !important;
        }
        .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
            color: black;
        }
        .sidebar .sidebar-content .stButton > button {
            background-color: black;
            color: white;
            border: 2px solid black;
        }
        .sidebar .sidebar-content .stButton > button:hover {
            background-color: #333333;
            border-color: #333333;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.header("CSV Analysis Agent")
        
        # User info
        if chat_interface.is_authenticated():
            user_info = chat_interface.get_user_info()
            st.write(f"**Logged in as:** {user_info['username']}")
        
        # Response details section
        response_data = chat_interface.get_current_response()
        if response_data:
            st.markdown("### 📋 Response Details")
            
            # Steps section - sleek display
            if response_data.get("steps"):
                st.markdown("**🔄 Processing Steps:**")
                for i, step in enumerate(response_data["steps"], 1):
                    st.markdown(f"<div style='background-color: #f8f9fa; padding: 0.5rem; margin: 0.3rem 0; border-radius: 6px; border-left: 3px solid #007bff;'><strong>{i}.</strong> {step}</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Table visualization - improved display
            if response_data.get("table_visualization"):
                st.markdown("**📊 Table Data:**")
                table_data = response_data["table_visualization"]
                if isinstance(table_data, list):
                    # Convert list of dictionaries to DataFrame for better display
                    import pandas as pd
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True, height=200)
                elif isinstance(table_data, dict):
                    st.json(table_data)
                else:
                    st.write(table_data)
                st.markdown("<br>", unsafe_allow_html=True)
            
            # Images section - sleek display
            if response_data.get("image_paths"):
                st.markdown("**📈 Generated Visualizations:**")
                display_images_grid_in_sidebar(response_data["image_paths"])
            
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
    
