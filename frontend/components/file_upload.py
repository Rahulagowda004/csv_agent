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
    
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader(f"Hello, {user_info['username']}! 📁")
        
        uploaded_file = st.file_uploader("Upload a CSV file to analyze", type=['csv'])
        
        if uploaded_file is not None:
            if st.button("Upload CSV"):
                with st.spinner("Uploading file..."):
                    upload_result = chat_interface.upload_file(uploaded_file)
                    
                    if upload_result["success"]:
                        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                        
                        # Display upload details in an expander
                        with st.expander("Upload Details"):
                            st.json(upload_result["data"])
                    else:
                        st.error(f"❌ Upload failed: {upload_result['error']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
