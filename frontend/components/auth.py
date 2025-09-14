"""
Authentication component for user login.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chat_interface import ChatInterface

def render_auth_section(chat_interface: 'ChatInterface'):
    """Render the user authentication section.
    
    Args:
        chat_interface: ChatInterface instance for managing authentication
    """
    # Add centered title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-family: 'Times New Roman', 'Georgia', serif; font-weight: 300; color: #333; margin: 0;">CSV Analysis Agent</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a centered layout for better positioning
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username = st.text_input("Enter your name:", placeholder="e.g., John Doe", key="username_input")
        
        # Add some spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Button with same width as text input
        if st.button("Start Chat", disabled=not username.strip(), key="start_chat_btn", type="primary"):
            chat_interface.set_user(username)
            st.rerun()
