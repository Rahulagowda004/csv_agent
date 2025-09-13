# CSV Agent Frontend

A Streamlit-based web interface for the CSV Analysis Agent that provides an intuitive chat interface for data analysis.

## Features

- **User Authentication**: Simple name-based authentication with unique user ID generation
- **File Upload**: Easy CSV file upload with validation
- **Interactive Chat**: Real-time chat interface with the CSV analysis agent
- **Rich Responses**: Support for text, images, tables, and step-by-step analysis
- **Expandable Sidebar**: Detailed response information including:
  - Processing steps
  - Suggested next questions
  - Generated visualizations
  - Table data
  - Session information

## Project Structure

```
frontend/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Frontend dependencies
├── README.md                # This file
├── src/                     # Core modules
│   ├── __init__.py
│   ├── api_client.py        # API communication
│   └── chat_interface.py    # Chat state management
├── components/              # UI components
│   ├── __init__.py
│   ├── auth.py              # Authentication component
│   ├── file_upload.py       # File upload component
│   └── sidebar.py           # Sidebar component
└── assets/                  # Static assets (if needed)
```

## Setup

### Prerequisites

- Python 3.8+
- The CSV Agent FastAPI backend running on `http://localhost:8000`

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Set the API base URL (optional):
```bash
export API_BASE_URL=http://localhost:8000  # Default value
```

## Running the Application

1. Make sure the FastAPI backend is running:
   ```bash
   # In the main project directory
   python main.py
   ```

2. Start the Streamlit frontend:
   ```bash
   # In the frontend directory
   streamlit run app.py
   ```

3. Open your browser and go to `http://localhost:8501`

## Usage

1. **Enter Your Name**: Start by entering your name to create a unique session
2. **Upload CSV File**: Upload a CSV file you want to analyze
3. **Ask Questions**: Use the chat interface to ask questions about your data
4. **Explore Results**: 
   - View responses in the main chat area
   - Check the sidebar for detailed information
   - Click on suggested questions for more insights
   - View generated visualizations

## Features in Detail

### Authentication
- Simple name-based authentication
- Automatic unique user ID generation using UUID
- Session management across conversations

### File Upload
- Drag-and-drop CSV file upload
- File validation and size checking
- Automatic cleanup of previous uploads per user

### Chat Interface
- Real-time messaging with the analysis agent
- Message history preservation
- Support for suggested questions from the sidebar

### Sidebar Features
- **Processing Steps**: See how your query was processed
- **Suggested Questions**: Click to automatically ask follow-up questions
- **Visualizations**: View generated charts and graphs
- **Table Data**: Inspect structured data results
- **Session Info**: Track your user and session IDs

## Troubleshooting

### Common Issues

1. **API Connection Error**
   - Ensure the FastAPI backend is running on the correct port
   - Check the API_BASE_URL configuration
   - Verify network connectivity

2. **File Upload Issues**
   - Ensure the file is a valid CSV format
   - Check file size limitations
   - Verify proper file permissions

3. **Missing Visualizations**
   - Check that image files exist in the expected location
   - Verify file path permissions
   - Ensure PIL/Pillow is properly installed

### Development

To run in development mode with auto-reload:
```bash
streamlit run app.py --server.runOnSave true
```

## API Integration

The frontend communicates with the FastAPI backend through two main endpoints:

- `POST /chat`: Send messages and receive analysis results
- `POST /upload`: Upload CSV files for analysis

The API client handles all communication and error handling automatically.

## Customization

### Styling
- Custom CSS is included in `app.py` for basic styling
- Modify the CSS in the `st.markdown()` sections to customize appearance

### Components
- Each UI component is modular and can be customized independently
- Add new components in the `components/` directory

### Configuration
- Environment variables can be used for configuration
- API endpoints and other settings can be modified in `src/api_client.py`
