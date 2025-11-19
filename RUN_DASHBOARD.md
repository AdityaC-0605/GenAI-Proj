# Running the Streamlit Dashboard

## Quick Start

### 1. Install Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate  # On macOS/Linux

# Install Streamlit and dependencies
pip install -r streamlit-requirements.txt
```

### 2. Start the API Server (Required!)

**Important**: The dashboard needs the API server running to answer questions.

```bash
# In Terminal 1: Start the API server
./start_api.sh

# Or manually:
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Wait for the message: "Application startup complete."

### 3. Run the Dashboard

```bash
# In Terminal 2: Start the dashboard
./start_dashboard.sh

# Or manually:
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

### What Was Fixed

Previously, the dashboard was showing mock data (always returning "Paris" as the answer). Now it:
- ✅ Connects to the real API server
- ✅ Uses actual mBERT/mT5 models for predictions
- ✅ Shows real confidence scores and processing times
- ✅ Displays actual API server status

## Dashboard Features

### 🏠 Home Page
- System overview and status
- Supported languages
- Available models
- Recent activity and logs

### ❓ Ask Questions
- Interactive question answering interface
- Support for all language pairs
- Real-time model inference
- Confidence scores and processing time

### 📊 Model Comparison
- Performance heatmaps across language pairs
- Side-by-side model metrics
- Category-based analysis
- Visual comparisons

### 📈 Training Monitor
- Real-time training progress
- Loss curves (training & validation)
- Few-shot learning curves
- Current training status

### 📁 Dataset Explorer
- Dataset statistics and distributions
- Language distribution charts
- Question type analysis
- Sample examples viewer

### ⚙️ Settings
- General system configuration
- Model management
- API configuration
- Device selection

## Customization

### API Connection

The dashboard is now configured to connect to the API server automatically. Make sure:

1. API server is running on `http://localhost:8000`
2. Check the sidebar for "API Server: Online" status
3. If offline, start the API server:
   ```bash
   ./start_api.sh
   ```

The dashboard will show:
- ✅ Green "Online" if API is reachable
- ❌ Red "Offline" if API is not running

### Loading Real Data

To display actual experiment results:

```python
import json

# Load experiment results
with open('experiments/evaluations/results.json', 'r') as f:
    results = json.load(f)

# Display in dashboard
st.json(results)
```

### Custom Visualizations

Add your own visualizations:

```python
import plotly.graph_objects as go

# Create custom plot
fig = go.Figure(data=[...])
st.plotly_chart(fig, use_container_width=True)
```

## Deployment

### Local Network Access

To access the dashboard from other devices on your network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

### Cloud Deployment

Deploy to Streamlit Cloud:

1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Deploy!

## Troubleshooting

### Port Already in Use

If port 8501 is already in use:

```bash
streamlit run app.py --server.port 8502
```

### Module Not Found

Make sure you're in the correct directory and virtual environment:

```bash
cd /path/to/Bert_VS_T5
source venv/bin/activate
pip install -r streamlit-requirements.txt
```

### Slow Performance

For better performance:

1. Reduce batch size in settings
2. Use CPU mode if MPS is slow
3. Enable model caching
4. Limit dataset size for exploration

## Tips

- Use the sidebar for quick navigation
- Refresh the page to reload data
- Check the system status in the sidebar
- Export visualizations using Plotly's built-in tools
- Use the settings page to configure the system

## Support

For issues or questions:
- Check the main README.md
- Review the API documentation at `/docs`
- Check logs in `experiments/tracking/`
