"""
File upload component for CSV files.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chat_interface import ChatInterface

def render_file_upload_section(chat_interface: 'ChatInterface'):
    """Render the file upload section.
    
    Args:
        chat_interface: ChatInterface instance for handling file uploads
    """
    user_info = chat_interface.get_user_info()
    
    # Simple welcome message
    st.markdown(f"### Hello, {user_info['username']}! 👋")
    
    # Clean file uploader
    uploaded_file = st.file_uploader(
        "Upload a CSV file to analyze", 
        type=['csv'],
        help="Upload your CSV file to start analyzing your data"
    )
    
    if uploaded_file is not None:
        # Show selected file info
        st.info(f"📄 Selected file: **{uploaded_file.name}** ({uploaded_file.size} bytes)")
        
        if st.button("📁 Upload CSV", type="primary"):
            with st.spinner("Uploading file..."):
                upload_result = chat_interface.upload_file(uploaded_file)
                
                if upload_result["success"]:
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                    # Store upload status in session state
                    st.session_state.file_uploaded = True
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.rerun()  # Refresh to show the chat interface
                else:
                    st.error(f"❌ Upload failed: {upload_result['error']}")
    else:
        st.info("👆 Please select a CSV file to upload")
