"""
Streamlit Chat App for CSV Agent
A user-friendly interface for the CSV analysis agent with chat functionality.
"""

import streamlit as st
from src.chat_interface import ChatInterface
from src.api_client import APIClient
from components.sidebar import render_sidebar_content
from components.auth import render_auth_section
from components.file_upload import render_file_upload_section
import os
from PIL import Image

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def main():
    """Main Streamlit app entry point."""
    # Page configuration
    st.set_page_config(
        page_title="CSV Analysis Chat Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize API client
    api_client = APIClient(API_BASE_URL)
    
    # Initialize chat interface
    chat_interface = ChatInterface(api_client)
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.title("🤖 CSV Analysis Chat Agent")
    
    # Check API health
    print(f"🔍 Frontend: Checking API health at {API_BASE_URL}")
    if not api_client.check_health():
        print("❌ Frontend: API server not responding")
        st.error("⚠️ API server is not running. Please start the FastAPI server first.")
        st.code("python main.py")
        return
    
    print("✅ Frontend: API server is healthy")
    st.success("✅ Connected to API server")
    
    # Render authentication section
    if not chat_interface.is_authenticated():
        print("🔐 Frontend: User not authenticated, showing auth section")
        render_auth_section(chat_interface)
        return
    
    print(f"👤 Frontend: User authenticated: {chat_interface.get_user_info()['username']}")
    
    # Render file upload section
    render_file_upload_section(chat_interface)
    
    # Render main chat interface
    chat_interface.render_chat()
    
    # Render sidebar content
    render_sidebar_content(chat_interface)
    
    # Image popup modal for main chat
    if "popup_image" in st.session_state and st.session_state["popup_image"]:
        # Create a prominent modal section
        st.markdown("---")
        st.markdown("### 🔍 Enlarged Image View")
        
        # Close button positioned at the top right
        col1, col2, col3 = st.columns([4, 1, 1])
        with col3:
            if st.button("❌ Close", key="close_main_popup", help="Close enlarged view", type="primary"):
                del st.session_state["popup_image"]
                del st.session_state["popup_image_name"]
                st.rerun()
        
        # Image display
        try:
            full_image = Image.open(st.session_state["popup_image"])
            image_name = st.session_state.get("popup_image_name", "Image")
            
            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Center the image with maximum width
            col1, col2, col3 = st.columns([0.5, 4, 0.5])
            with col2:
                st.image(full_image, caption=image_name, use_container_width=True)
            
            # Add spacing after image
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("---")
                
        except Exception as e:
            st.error(f"Error displaying enlarged image: {str(e)}")
            if st.button("❌ Close", key="close_main_popup_error"):
                if "popup_image" in st.session_state:
                    del st.session_state["popup_image"]
                if "popup_image_name" in st.session_state:
                    del st.session_state["popup_image_name"]
                st.rerun()

if __name__ == "__main__":
    main()
