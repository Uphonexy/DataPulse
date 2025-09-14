import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings to keep logs clean
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
# Future warnings are suppressed by the general 'ignore' policy below
warnings.filterwarnings('ignore')

def preprocess_data():
    st.markdown("""
    <style>
    .preprocess-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    }
    .progress-item {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔬 Smart Data Preprocessing")
    st.markdown('<div class="preprocess-card"><h2>✨ Cleaning & Preparing Your Data</h2><p>Automated processing with AI-powered insights</p></div>', unsafe_allow_html=True)

    data = st.session_state.data

    if data is None:
        st.error("❌ No data available. Please upload files first.")
        return

    with st.spinner("Preprocessing data..."):
        # Start with a fresh copy to avoid warnings
        original_shape = data.shape
        data_clean = data.copy()

        # Drop duplicates
        data_clean = data_clean.drop_duplicates()

        # Identify column types
        numeric_features = data_clean.select_dtypes(include=[np.number]).columns
        categorical_features = data_clean.select_dtypes(include=['object', 'category']).columns
        datetime_features = data_clean.select_dtypes(include=['datetime', 'datetimetz']).columns

        # Handle datetime columns
        for col in datetime_features:
            data_clean[col] = pd.to_datetime(data_clean[col], errors='coerce')

        # Apply transformations using .loc
        # Numeric transformations
        if len(numeric_features) > 0:
            imputer_num = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            imputed = imputer_num.fit_transform(data_clean[numeric_features])
            scaled = scaler.fit_transform(imputed)
            data_clean[numeric_features] = scaled.astype(np.float64)

        # Categorical transformations (simple imputation, no encoding for session state simplicity)
        if len(categorical_features) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            imputed = imputer_cat.fit_transform(data_clean[categorical_features])
            data_clean[categorical_features] = pd.DataFrame(imputed, columns=categorical_features, index=data_clean.index)

        # Clip outliers for numerics
        if len(numeric_features) > 0:
            for col in numeric_features:
                Q1 = data_clean[col].quantile(0.25)
                Q3 = data_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data_clean.loc[:, col] = np.where(data_clean[col] < lower_bound, lower_bound,
                                               np.where(data_clean[col] > upper_bound, upper_bound, data_clean[col]))

        # Handle inf values
        data_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Update the reference
        data = data_clean

        # Preserve original and log changes
        st.session_state.data_original = data.copy()
        st.session_state.processed_data = data
        st.session_state.step = 'suggest'

        # Summary
        st.success(f"Preprocessing complete. Shape: {original_shape} -> {data.shape}")
        with st.expander("Preprocessing Log"):
            st.write(f"Dropped {original_shape[0] - data.shape[0]} duplicates")
            st.write(f"Handled {data.isnull().sum().sum()} missing values")
            if len(numeric_features) > 0:
                st.write(f"Scaled {len(numeric_features)} numeric features")

        if st.button("Proceed to Suggestions"):
            st.rerun()
