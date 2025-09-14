import streamlit as st
import pandas as pd
import warnings
import os

# Suppress ALL warnings to keep logs clean
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Configure Streamlit for clean logging
os.environ['STREAMLIT_LOG_LEVEL'] = 'ERROR'  # Only show errors, no info/warnings

# Note: AI connection test moved to dashboard page to avoid startup delays

st.set_page_config(page_title="Data Analysis Bot", layout="wide", page_icon="📊")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'step' not in st.session_state:
    st.session_state.step = 'upload'

# Import modules
from data_upload import data_upload
from preprocessing import preprocess_data
from analysis import suggest_analyses, execute_analyses
from enhanced_dashboard import enhanced_dashboard
from export import export_report

# Main app flow
if st.session_state.step == 'upload':
    data_upload()
elif st.session_state.step == 'preprocess':
    preprocess_data()
elif st.session_state.step == 'suggest':
    suggest_analyses()
elif st.session_state.step == 'execute':
    execute_analyses()
elif st.session_state.step == 'dashboard':
    enhanced_dashboard()
elif st.session_state.step == 'export':
    export_report()

# Sidebar navigation
st.sidebar.title("Navigation")
steps = ["Upload Files", "Preprocessing", "Suggestions", "Execution", "Dashboard", "Export"]
step_keys = ["upload", "preprocess", "suggest", "execute", "dashboard", "export"]
current_step_idx = step_keys.index(st.session_state.step)

def set_step(step):
    st.session_state.step = step
    st.rerun()

for i, step_name in enumerate(steps):
    disabled = i > current_step_idx
    if not disabled:
        st.sidebar.button(step_name, on_click=set_step, args=(step_keys[i],))
    else:
        st.sidebar.button(step_name, disabled=True)
