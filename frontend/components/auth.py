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
    st.subheader("👋 Welcome! Please enter your name to get started")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        username = st.text_input("Enter your name:", placeholder="e.g., John Doe")
    with col2:
        if st.button("Start Chat", disabled=not username.strip()):
            chat_interface.set_user(username)
            st.rerun()
