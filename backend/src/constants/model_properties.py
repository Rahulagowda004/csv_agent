"""Model definitions for the CSV Agent."""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class CSVAgentResponse(BaseModel):
    """Structured response from CSV analysis agent."""
    text: str = Field(description="The main text content from the agent message")
    steps: Optional[List[str]] = Field(
        default=None,
        description="Readable format of steps taken and their outputs"
    )
    image_paths: Optional[List[str]] = Field(
        default=None, 
        description="List of image file paths when data visualization tasks are present"
    )
    table_visualization: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of dictionaries containing table data for visualization output from tools"
    )
    suggested_next_steps: Optional[List[str]] = Field(
        default=None,
        description="List of suggested queries when user options are vague"
    )
