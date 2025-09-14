# CSV Agent

Conversational data analysis for your CSV files. Upload a CSV, ask questions in natural language, and get summaries, stats, and plots. Built with FastAPI (backend), Streamlit (frontend), and OpenAI Agent SDK.

## Features
- **CSV Upload**: Per-session isolation with secure file handling (200MB file limit)
- **Natural Language Queries**: Chat interface to query your data and receive text answers and visualizations
- **Automatic EDA**: Summary statistics, column profiling, filtering, grouping, and correlations
- **Dynamic Visualizations**: Plot generation using Matplotlib, Seaborn, and Plotly
- **Intelligent Sidebar**: 
  - **Processing Steps**: Detailed breakdown of how results were achieved
  - **Suggested Next Steps**: AI-powered recommendations for further analysis
  - **Response Details**: Transparent explanation of the analysis workflow
- **Smart Error Handling**: Graceful handling of visualization errors with alternative suggestions
- **Persistent Memory**: Conversational context and chat history stored in SQLite database
- **Export Capabilities**: Download processed CSV files and generated plots
- **REST API**: Full API documentation with Swagger/Redoc interface

## Tech Stack
- **Backend**: FastAPI, OpenAI Agents SDK, Uvicorn, Pandas, NumPy, Scikit-learn, SciPy
- **Visualization**: Seaborn, Matplotlib, Plotly
- **Frontend**: Streamlit
- **AI/LLM**: OpenAI Agents SDK
- **Database**: SQLite (for conversation memory)
- **Configuration**: python-dotenv
- **Python Version**: 3.12+

## Repository Structure
```
csv_agent/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── data/
│   │   ├── csv/               # Uploaded CSVs organized by session
│   │   └── plots/             # Generated visualizations by session
│   ├── memory/                # SQLite conversation database
│   ├── src/                   # Core application modules
│   │   ├── agent/             # AI agent logic
│   │   ├── services/          # Business logic services
│   │   ├── constants/         # Application constants
│   │   └── core/              # Core utilities
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── app.py                 # Main Streamlit application
│   ├── run.py                 # Frontend runner script
│   ├── components/            # Reusable UI components
│   │   ├── auth/              # Authentication components
│   │   ├── file_upload/       # File upload interface
│   │   └── sidebar/           # Navigation sidebar
│   ├── src/                   # Frontend application logic
│   │   ├── api_client/        # Backend API communication
│   │   └── chat_interface/    # Chat UI components
│   └── requirements.txt
└── README.md
```

## Prerequisites
- **Operating System**: macOS (recommended)
  - *Note: Some dependencies may have compatibility issues on Windows*
- **Python**: 3.12 or higher
- **OpenAI API Key**: Required for AI agent functionality

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd csv_agent
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python3.12 -m venv .venv

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Verify activation (you should see (.venv) in your prompt)
which python  # Should show path to .venv/bin/python
```

### 3. Environment Configuration
```bash
# Create environment file
cp .env.example .env

# Edit .env file with your configuration
nano .env
```

Required environment variables:
```bash
 OPENAI_API_KEY="key"
CSV_DATA_FOLDER="Users/abc/Desktop/csv_agent/backend/data/csv"
PLOTS_FOLDER="/Users/abc/Desktop/csv_agent/backend/data/plots"
VENV_DIR="/Users/abc/Desktop/csv_agent/backend/venv"
```

### 4. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 5. Install Frontend Dependencies
```bash
cd frontend
pip install -r requirements.txt
cd ..
```

### 6. Initialize Database
```bash
cd backend
python -c "from src.core.database import init_db; init_db()"
cd ..
```

## Running the Application

### Start Backend Server
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs` (Swagger)
- Alternative docs: `http://localhost:8000/redoc`

### Start Frontend (New Terminal)
```bash
# Activate virtual environment in new terminal
source .venv/bin/activate

cd frontend
streamlit run app.py
```
The frontend will be available at `http://localhost:8501`


## API Usage

### Upload CSV
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_file.csv"
```

### Query Data
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me the summary statistics", "session_id": "your-session-id"}'
```

## Troubleshooting

### Common Issues
1. **Python Version**: Ensure you're using Python 3.12+
   ```bash
   python --version  # Should show 3.12.x
   ```

2. **Virtual Environment**: Make sure it's activated
   ```bash
   which python  # Should point to .venv/bin/python
   ```

3. **Port Conflicts**: If ports 8000 or 8501 are in use:
   ```bash
   # Backend on different port
   uvicorn main:app --port 8001
   
   # Frontend on different port
   streamlit run app.py --server.port 8502
   ```

4. **OpenAI API Key**: Verify your API key is valid and has sufficient credits

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.