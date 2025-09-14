import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures

def suggest_analyses():
    st.title("💡 Suggested Analyses")
    data = st.session_state.processed_data

    if data is None:
        st.error("No processed data available.")
        return

    # Lightweight domain detection
    suggestions = []

    # Check for time-based data
    datetime_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
    if datetime_cols:
        suggestions.append("Explore trends over time")
        suggestions.append("Time series forecasting")

    # Check for numeric data
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        suggestions.append("Correlation analysis")
        suggestions.append("Regression analysis")

    # Check for categorical data
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        suggestions.append("Clustering analysis")
        suggestions.append("Category distribution")

    # General suggestions
    suggestions += ["Descriptive statistics", "Anomaly detection", "Data distribution visualizations"]

    st.subheader("Suggested Analyses")
    selected_analyses = st.multiselect("Choose analyses to run:", suggestions)

    if st.button("Run Selected Analyses"):
        st.session_state.selected_analyses = selected_analyses
        st.session_state.step = 'execute'
        st.rerun()

def execute_analyses():
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .analysis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .success-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("⚡ AI Analysis Engine")
    st.markdown('<div class="analysis-card"><h2>🧠 Processing Your Data</h2></div>', unsafe_allow_html=True)

    data = st.session_state.processed_data
    analyses = st.session_state.get('selected_analyses', [])

    if not analyses:
        st.warning("No analyses selected. Running defaults.")
        analyses = ["Descriptive statistics"]

    if not analyses or data is None:
        st.error("❌ No analyses selected or data available.")
        st.error("Please go back and select analyses to run.")
        return

    # Show selected analyses with icons
    st.subheader("🎯 Selected Analyses")
    analysis_icons = {
        "Descriptive statistics": "📊",
        "Correlation analysis": "🔗",
        "Regression analysis": "📈",
        "Explore trends over time": "🕐",
        "Time series forecasting": "🔮",
        "Anomaly detection": "🚨",
        "Data distribution visualizations": "📊",
        "Clustering analysis": "🎯",
        "Category distribution": "🏷️"
    }

    cols = st.columns(3)
    for i, analysis in enumerate(analyses):
        with cols[i % 3]:
            icon = analysis_icons.get(analysis, "📋")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 10px; margin: 5px 0; color: white; text-align: center;'>
                {icon} {analysis}
            </div>
            """, unsafe_allow_html=True)

    results = {}

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    datetime_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]

    def run_analysis(analysis):
        try:
            if analysis == "Descriptive statistics":
                desc = data.describe().T
                fig = px.scatter_matrix(data[numeric_cols].head(500), title="Scatter Matrix") if len(numeric_cols) > 1 else None
                return {"desc": desc, "plot": fig}

            if analysis == "Correlation analysis":
                if len(numeric_cols) > 1:
                    corr = data[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix", color_continuous_scale='RdBu_r')
                    return {"corr": corr, "plot": fig}
                return {"corr": "Not enough numeric columns"}

            if analysis == "Regression analysis":
                if len(numeric_cols) >= 2:
                    try:
                        X_col = numeric_cols[0]
                        y_col = numeric_cols[1]
                        X = data[X_col]
                        y = data[y_col]
                        model = LinearRegression().fit(X.values.reshape(-1, 1), y)
                        fig = px.scatter(data, x=X_col, y=y_col, trendline="ols", title="Regression")
                        return {"model": model, "plot": fig}
                    except Exception as e:
                        return {"regression": f"Failed to run regression: {str(e)}"}
                return {"regression": "Need at least 2 numeric columns"}

            if analysis == "Explore trends over time":
                if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                    try:
                        trend_data = data.set_index(datetime_cols[0])
                        fig = px.line(trend_data, x=trend_data.index, y=numeric_cols[0], title="Trend Over Time")
                        return {"plot": fig}
                    except Exception as e:
                        return {"trends": f"Failed to create trends visualization: {str(e)}"}

            if analysis == "Time series forecasting":
                if len(datetime_cols) == 0 or len(numeric_cols) == 0:
                    return {"forecast": "Insufficient data for forecasting - need datetime and numeric columns"}
                if len(data) < 10:
                    return {"forecast": "Need minimum 10 data points for forecasting"}

                try:
                    ts_data = data.set_index(datetime_cols[0])[numeric_cols[0]]
                    # Check if data has proper datetime index
                    if not pd.api.types.is_datetime64_any_dtype(ts_data.index):
                        return {"forecast": "Datetime column must be properly formatted for forecasting"}

                    model = sm.tsa.arima.ARIMA(ts_data, order=(1,1,1)).fit()
                    forecast = model.forecast(steps=min(5, len(ts_data)))  # Don't forecast more than data length

                    # Use proper frequency detection
                    freq = pd.infer_freq(ts_data.index.head(20))
                    if freq is None:
                        freq = 'D'  # Default to daily if can't infer

                    forecast_idx = pd.date_range(start=ts_data.index[-1], periods=len(forecast)+1, freq=freq)[1:]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Actual'))
                    fig.add_trace(go.Scatter(x=forecast_idx, y=forecast, mode='lines', name='Forecast'))
                    fig.update_layout(title="Time Series Forecast")
                    return {"forecast": forecast, "plot": fig}
                except Exception as e:
                    return {"forecast": f"Time series modeling failed: {str(e)}"}

            if analysis == "Anomaly detection":
                if len(numeric_cols) > 0:
                    iso_forest = IsolationForest(random_state=42)
                    outliers = iso_forest.fit_predict(data[numeric_cols])
                    local_data = data.copy()
                    local_data['anomaly'] = outliers
                    fig = px.scatter_matrix(local_data, color='anomaly', title="Anomaly Detection (Outliers highlighted)")
                    return {"anomalies": local_data[local_data['anomaly'] == -1], "plot": fig}

            if analysis == "Data distribution visualizations":
                figs = []
                for col in numeric_cols[:5]:
                    fig = px.histogram(data, x=col, title=f"Distribution of {col}")
                    figs.append(fig)
                if figs:
                    return {"plots": figs}

            if analysis == "Clustering analysis":
                if len(numeric_cols) > 1:
                    kmeans = KMeans(n_clusters=min(3, len(data)), random_state=42)
                    clusters = kmeans.fit_predict(data[numeric_cols])
                    local_data = data.copy()
                    local_data['cluster'] = clusters
                    fig = px.scatter_matrix(local_data, color='cluster', title="Clustering Results")
                    return {"clusters": clusters, "plot": fig}

            if analysis == "Category distribution":
                if len(categorical_cols) > 0:
                    # Take the first categorical column for distribution
                    col = categorical_cols[0]
                    fig = px.bar(data[col].value_counts(), title=f"Distribution of {col}")
                    return {"plot": fig}

            return {analysis: "Analysis not implemented yet"}
        except Exception as e:
            return {analysis: f"Error: {str(e)}"}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_analysis, analysis) for analysis in analyses]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.update(result)

    st.session_state.analysis_results = results

    # Success animation and confirmation
    st.markdown('<div class="success-card"><h2>🎉 Analysis Complete!</h2><p>Your data has been processed successfully!</p></div>', unsafe_allow_html=True)
    st.balloons()

    # Progress completion
    completed_analyses = len(results)
    total_analyses = len(analyses)
    st.progress(min(completed_analyses / total_analyses, 1.0))
    st.info(f"✅ Completed {completed_analyses}/{total_analyses} analyses")

    # Proceed button with animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 View Dynamic Dashboard", type="primary", use_container_width=True):
            st.session_state.step = 'dashboard'
            st.rerun()
