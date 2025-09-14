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
    
    # Add custom CSS for white, black, and orange color scheme
    st.markdown("""
    <style>
    /* Main layout */
    .main > div {
        padding-top: 2rem;
        background-color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb, .stSidebar {
        background-color: white !important;
        border-right: 2px solid #e0e0e0;
    }
    
    /* Sidebar content */
    .css-1d391kg .css-1cypcdb, .stSidebar .sidebar-content {
        background-color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background-color: black;
        color: white;
        border: 2px solid black;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #333333;
        border-color: #333333;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .user-message {
        background-color: black;
        color: white;
        border-color: black;
    }
    
    .bot-message {
        background-color: white;
        color: black;
        border-color: #e0e0e0;
    }
    
    /* Upload section - simplified */
    .upload-section {
        margin-bottom: 2rem;
    }
    
    /* Titles and headers */
    h1 {
        text-align: center;
        margin-bottom: 2rem;
        margin-top: 1rem;
        color: black;
        font-weight: 700;
    }
    
    h2 {
        color: black;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: black;
        text-align: center;
    }
    
    /* Spacing */
    .element-container:has(h1) {
        margin-bottom: 3rem;
    }
    
    .element-container:has(h3) {
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .stTextInput {
        margin-bottom: 1rem;
    }
    
    .stButton {
        margin-top: 1rem;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: black;
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Chat input */
    .stChatInput > div > div > div > div {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: black;
        color: white;
        border-radius: 8px 8px 0 0;
    }
    
    .streamlit-expanderContent {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-top: none;
        border-radius: 0 0 8px 8px;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    
    /* Caption styling */
    .stCaption {
        color: #666;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title removed for cleaner interface
    
    # Check API health
    print(f"🔍 Frontend: Checking API health at {API_BASE_URL}")
    if not api_client.check_health():
        print("❌ Frontend: API server not responding")
        st.error("⚠️ API server is not running. Please start the FastAPI server first.")
        st.code("python main.py")
        return
    
    print("✅ Frontend: API server is healthy")
    
    # Render authentication section
    if not chat_interface.is_authenticated():
        print("🔐 Frontend: User not authenticated, showing auth section")
        render_auth_section(chat_interface)
        return
    
    print(f"👤 Frontend: User authenticated: {chat_interface.get_user_info()['username']}")
    
    # Render file upload section
    render_file_upload_section(chat_interface)
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
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
