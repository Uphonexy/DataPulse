# Data Analysis Bot 🧠📊

A comprehensive Streamlit-based data analysis tool that automates the entire workflow from data ingestion to insightful reporting.

## Features

### 🚀 Linear Workflow
- **File Upload and Ingestion**: Drag-and-drop CSV/Excel files, automatic merging for multiple files
- **Automated Preprocessing**: Handles missing values, outliers, scaling, and anonymization
- **Domain Detection**: Intelligent suggestions for analyses based on data structure
- **Parallel Execution**: Runs analyses concurrently with progress tracking
- **Interactive Dashboard**: Multi-column layout with filters and insights
- **Export and Reporting**: PDF reports with embedded visualizations

### 🎯 Key Capabilities
- Multi-file upload and auto-merge
- Sensitive data anonymization (email/SSN/phone via hashing)
- Comprehensive preprocessing pipeline
- Multiple analysis types: Descriptive stats, correlations, regressions, time series, anomaly detection, clustering
- AI-powered insights using Google AI
- Professional PDF reports with charts and summaries
- Demo mode with sample data

### 🛡️ Privacy and Security
- Client-side processing for small datasets
- Automatic anonymization of PII fields
- No data persistence (in-memory only)
- Secure API key management via `st.secrets`

## Installation

1. **Clone the repository**:
   ```bash
   cd /path/to/your/desktop/data_analysis_bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key**:
   Create a `.streamlit/secrets.toml` file:
   ```toml
   [google_ai_api_key]
   google_ai_api_key = "your_google_ai_api_key_here"
   ```
   Sign up for Google AI and get your key from https://makersuite.google.com/app/apikey

4. **Run the app**:
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Upload Data**: Choose files or use demo mode
2. **Preprocessing**: Automatic cleaning and transformation
3. **Select Analyses**: Choose from suggested analyses
4. **Execute**: Parallel processing of selected analyses
5. **Dashboard**: Explore interactive visualizations and AI insights
6. **Export**: Generate professional PDF reports

## Requirements

- Python 3.8+
- Streamlit
- Pandas, NumPy, Scikit-learn
- Plotly, statsmodels
- ReportLab for PDFs
- Google Generative AI

## Deployment

For cloud deployment, use:
- Streamlit Cloud
- Heroku
- AWS/Azure with secrets management

## Support

Built for data analysis workflows emphasizing ease of use, automation, and professional outputs.

---

🤖 *Powered by Streamlit and AI for automated insights*
