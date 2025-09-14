import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib import colors
import concurrent.futures
import re
from datetime import datetime
import hashlib
import openai

# Set OpenAI base URL for OpenRouter
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = st.secrets["openrouter_api_key"]

st.set_page_config(page_title="Data Analysis Bot", layout="wide")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'step' not in st.session_state:
    st.session_state.step = 'upload'

def upload_data():
    st.title("📊 Data Analysis Bot")
    st.header("Step 1: File Upload and Ingestion")

    st.subheader("Upload Data")
    uploaded_files = st.file_uploader("Upload CSV or Excel files", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)

    st.subheader("Or Try with Sample Data")
    if st.button("Load Sample Sales Data"):
        # Generate sample sales data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        sales = np.random.normal(5000, 1000, 100)
        categories = np.random.choice(['A', 'B', 'C'], 100)
        data = pd.DataFrame({'date': dates, 'sales': sales, 'category': categories})
        st.session_state.data = data
        st.success("Sample data loaded.")
        # Preview
        st.write(f"Shape: {data.shape}")
        st.write(data.dtypes)
        st.write(data.head(10))
        return

    if not uploaded_files:
        return

    if uploaded_files:
        data_frame_list = []
        merge_column = None

        for file in uploaded_files:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-8', errors='replace')
            else:
                df = pd.read_excel(file)
            data_frame_list.append(df)

        if len(data_frame_list) == 1:
            data = data_frame_list[0]
        else:
            st.subheader("Merge Multiple Files")
            # Auto-detect common columns
            common_columns = set(data_frame_list[0].columns)
            for df in data_frame_list[1:]:
                common_columns = common_columns.intersection(set(df.columns))

            if common_columns:
                merge_options = ['Auto-detect'] + list(common_columns)
                merge_choice = st.selectbox("Choose merge column or auto-detect", merge_options, index=0)
                if merge_choice == 'Auto-detect':
                    merge_column = list(common_columns)[0] if common_columns else None
                else:
                    merge_column = merge_choice

                if merge_column:
                    data = data_frame_list[0].set_index(merge_column).combine_first(
                        pd.concat(data_frame_list[1:], ignore_index=True).set_index(merge_column)
                    ).reset_index()
                else:
                    data = pd.concat(data_frame_list, ignore_index=True)
            else:
                data = pd.concat(data_frame_list, ignore_index=True)

        # Basic validation
        if data.empty:
            st.error("Error: No data found in files.")
            return

        # Remove empty files check
        if len(data) == 0:
            st.error("Error: Files are empty.")
            return

        # Anonymize sensitive fields
        for col in data.columns:
            if re.search(r'(email|ssn|phone)', col.lower()):
                data[col] = data[col].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else x)

        # Preview
        st.subheader("Data Preview (First 100 rows)")
        st.write(f"Shape: {data.shape}")
        st.write(f"Column Types: {data.dtypes.to_dict()}")
        st.write(data.head(100))

        # Sample values and issues
        missing = data.isnull().sum()
        issues = [col for col in data.columns if missing[col] > 0]
        if issues:
            st.warning(f"Missing values in columns: {', '.join(issues)}")

        if st.button("Proceed to Preprocessing"):
            st.session_state.data = data
            st.session_state.step = 'preprocess'
            st.rerun()

def preprocess_data():
    st.title("🔄 Preprocessing Data")
    data = st.session_state.data

    if data is None:
        st.error("No data available. Please upload files first.")
        return

    with st.spinner("Preprocessing data..."):
        # Drop duplicates
        original_shape = data.shape
        data = data.drop_duplicates()

        # Identify column types
        numeric_features = data.select_dtypes(include=[np.number]).columns
        categorical_features = data.select_dtypes(include=['object', 'category']).columns
        datetime_features = data.select_dtypes(include=['datetime', 'datetimetz']).columns

        # Handle datetime columns
        for col in datetime_features:
            data[col] = pd.to_datetime(data[col], errors='coerce')

        # Apply transformations in place
        # Numeric transformations
        imputer_num = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        for col in numeric_features:
            data[col] = imputer_num.fit_transform(data[[col]])
            data[col] = scaler.fit_transform(data[[col]])

        # Categorical transformations
        imputer_cat = SimpleImputer(strategy='most_frequent')
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        for col in categorical_features:
            data[col] = imputer_cat.fit_transform(data[[col]].values.reshape(-1, 1)).flatten()

        # Clip outliers for numerics
        for col in numeric_features:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = np.where(data[col] < lower_bound, lower_bound,
                                np.where(data[col] > upper_bound, upper_bound, data[col]))

        # Handle inf values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Preserve original and log changes
        st.session_state.data_original = data.copy()
        st.session_state.processed_data = data
        st.session_state.step = 'suggest'

        # Summary
        st.success(f"Preprocessing complete. Shape: {original_shape} -> {data.shape}")
        with st.expander("Preprocessing Log"):
            st.write(f"Duplicated removed: {original_shape[0] - data.shape[0]}")
            # Add more logs as needed

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
    if categorical_cols:
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
    st.title("⚙️ Executing Analyses")
    data = st.session_state.processed_data
    analyses = st.session_state.get('selected_analyses', [])

    if not analyses or data is None:
        st.error("No analyses selected or data available.")
        return

    results = {}

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    datetime_cols = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]

    def run_analysis(analysis):
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
                X = data[numeric_cols[0]]
                y = data[numeric_cols[1]]
                model = LinearRegression().fit(X.values.reshape(-1, 1), y)
                fig = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1], trendline="ols", title="Regression")
                return {"model": model, "plot": fig}
            return {"regression": "Need at least 2 numeric columns"}

        if analysis == "Explore trends over time":
            if datetime_cols and numeric_cols:
                trend_data = data.set_index(datetime_cols[0])
                fig = px.line(trend_data, x=trend_data.index, y=numeric_cols[0], title="Trend Over Time")
                return {"plot": fig}

        if analysis == "Time series forecasting":
            if datetime_cols and numeric_cols:
                ts_data = data.set_index(datetime_cols[0])[numeric_cols[0]]
                try:
                    model = sm.tsa.arima.ARIMA(ts_data, order=(1,1,1)).fit()
                    forecast = model.forecast(steps=5)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Actual'))
                    fig.add_trace(go.Scatter(x=pd.date_range(ts_data.index[-1], periods=6, freq='D')[1:], y=forecast, mode='lines', name='Forecast'))
                    fig.update_layout(title="Time Series Forecast")
                    return {"forecast": forecast, "plot": fig}
                except:
                    return {"forecast": "Time series modeling failed"}

        if analysis == "Anomaly detection":
            if numeric_cols:
                iso_forest = IsolationForest(random_state=42)
                outliers = iso_forest.fit_predict(data[numeric_cols])
                data['anomaly'] = outliers
                fig = px.scatter_matrix(data, color='anomaly', title="Anomaly Detection (Outliers highlighted)")
                return {"anomalies": data[data['anomaly'] == -1], "plot": fig}

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
                data['cluster'] = clusters
                fig = px.scatter_matrix(data, color='cluster', title="Clustering Results")
                return {"clusters": clusters, "plot": fig}

        if analysis == "Category distribution":
            if categorical_cols:
                fig = px.bar(data[categorical_cols].value_counts(), title="Category Distribution")
                return {"plot": fig}

        return {analysis: "Analysis not implemented yet"}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_analysis, analysis) for analysis in analyses]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.update(result)

    st.session_state.analysis_results = results
    st.session_state.step = 'dashboard'

    st.success("Analyses executed.")

def dashboard():
    st.title("📈 Interactive Dashboard")
    results = st.session_state.analysis_results
    data = st.session_state.processed_data

    if not results:
        st.error("No analysis results available.")
        return

    # Sidebar filters
    st.sidebar.subheader("Filters")
    if data is not None and 'date' in data.columns:
        date_range = st.sidebar.date_input("Date Range", value=(data['date'].min(), data['date'].max()))
    if st.sidebar.button("Apply Filters"):
        # Apply filters (simple placeholder)
        pass

    # Display results
    cols = st.columns(2)
    chart_idx = 0
    for key, result in results.items():
        if isinstance(result, go.Figure):
            with cols[chart_idx % 2]:
                st.plotly_chart(result, use_container_width=True)
            chart_idx += 1
        elif isinstance(result, list) and result:  # For histograms, etc.
            for fig in result if key in results and 'plots' in str(result) else []:
                with cols[chart_idx % 2]:
                    st.plotly_chart(fig, use_container_width=True)
                    chart_idx += 1

    # Summary insights
    st.header("Automated Insights")
    if 'insights' not in st.session_state:
        st.session_state.insights = get_insights(results)

    insights = st.session_state.insights
    for insight in insights:
        with st.expander(f"📊 {insight['title']}"):
            st.write(insight['content'])

    # Overall summary
    if st.button("Generate Overall Summary"):
        overall_prompt = "Provide an overall summary of all analysis results, highlighting key trends and recommendations."
        try:
            overall_response = openai.Completion.create(
                engine="deepseek/deepseek-r1:free",
                prompt=overall_prompt,
                max_tokens=200
            )
            st.success("Overall Summary:")
            st.write(overall_response.choices[0].text.strip())
        except:
            st.error("Failed to generate summary.")

    if st.button("Proceed to Export"):
        st.session_state.step = 'export'
        st.rerun()

def get_insights(results):
    insights = []
    for analysis, result in results.items():
        prompt = f"Summarize these {analysis} findings from [data snippet] in simple, engaging English for a non-expert. Avoid jargon; suggest 2-3 actions. Keep under 150 words."
        try:
            response = openai.Completion.create(
                engine="deepseek/deepseek-r1:free",
                prompt=prompt,
                max_tokens=150
            )
            insight = {
                "title": analysis,
                "content": response.choices[0].text.strip()
            }
            insights.append(insight)
        except:
            insight = {
                "title": analysis,
                "content": "Insight generation failed. Please check later."
            }
            insights.append(insight)
    return insights

def export_report():
    st.title("📥 Export Report")
    results = st.session_state.analysis_results
    data = st.session_state.processed_data

    if not results:
        st.error("No analysis results to export.")
        return

    if st.button("Generate PDF Report"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("Data Analysis Report", styles['Title']))

        # Executive Summary
        try:
            summary_prompt = "Generate an executive summary for this data analysis."
            summary_response = openai.Completion.create(
                engine="deepseek/deepseek-r1:free",
                prompt=summary_prompt,
                max_tokens=100
            )
            elements.append(Paragraph("Executive Summary", styles['Heading1']))
            elements.append(Paragraph(summary_response.choices[0].text.strip(), styles['Normal']))
        except:
            pass

        insights = st.session_state.get('insights', [])
        for insight in insights:
            elements.append(Paragraph(f"Insights for {insight['title']}", styles['Heading2']))
            elements.append(Paragraph(insight['content'], styles['Normal']))

        # Data sample
        elements.append(Paragraph("Data Sample", styles['Heading1']))
        sample_data = data.head(10).to_string()
        table_data = [list(data.columns)] + data.head(10).values.tolist()
        t = Table(table_data)
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige)]))
        elements.append(t)

        # Add images for plots (if any)
        for key, result in results.items():
            if isinstance(result, go.Figure):
                img_bytes = pio.to_image(result, format='png', engine='kaleido')
                from reportlab.lib.utils import ImageReader
                img = ImageReader(BytesIO(img_bytes))
                elements.append(Paragraph(f"Visualization: {key}", styles['Heading2']))
                elements.append(img)

        doc.build(elements)
        buffer.seek(0)

        st.download_button("Download PDF", buffer, file_name="report.pdf", mime="application/pdf")

    # Export options
    st.header("Additional Exports")
    if st.button("Export Raw Data as CSV"):
        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_data, file_name="processed_data.csv", mime="text/csv")

# Main app flow
if st.session_state.step == 'upload':
    upload_data()
elif st.session_state.step == 'preprocess':
    preprocess_data()
elif st.session_state.step == 'suggest':
    suggest_analyses()
elif st.session_state.step == 'execute':
    execute_analyses()
elif st.session_state.step == 'dashboard':
    dashboard()
elif st.session_state.step == 'export':
    export_report()

# Sidebar navigation
st.sidebar.title("Navigation")
steps = ["Upload Files", "Preprocessing", "Suggestions", "Execution", "Dashboard", "Export"]
current_step_idx = ["upload", "preprocess", "suggest", "execute", "dashboard", "export"].index(st.session_state.step)
for i, step_name in enumerate(steps):
    if i <= current_step_idx:
        st.sidebar.button(step_name, disabled=i < current_step_idx)
    else:
        st.sidebar.button(step_name, disabled=True)
