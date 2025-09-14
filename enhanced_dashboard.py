import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ollama
import time
import concurrent.futures

def enhanced_dashboard():
    st.title("📈 Dynamic Analytics Dashboard")

    # Move AI test to expander with button
    with st.expander("🧠 AI Status Test"):
        if st.button("🔍 Test DataPulse AI Connection"):
            try:
                response = ollama.generate(model='llama3.2:3b', prompt='Say "Local AI is ready" in exactly those words.')
                if 'ready' in response['response'].lower():
                    st.success(f"✅ DataPulse Connected - {response['response']}")
                else:
                    st.warning("🤖 Connected but response unexpected")
            except Exception as e:
                st.error(f"❌ Failed: {str(e)}")
                st.info("💡 Ensure AI service is running.")
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #ff6b6b;
    }
    </style>
    """, unsafe_allow_html=True)

    results = st.session_state.analysis_results
    data = st.session_state.processed_data

    if not results:
        st.error("📊 No analysis results available.")
        return

    # 1. Downsample data for plots to improve performance
    if data.shape[0] > 500:
        data_display = data.sample(n=500, random_state=42).reset_index(drop=True)
        st.info(f"📊 Dashboard using 500-row sample for performance (full data: {data.shape[0]} rows)")
    else:
        data_display = data

    # 2. Limit charts to improve rendering speed
    display_results = dict(list(results.items())[:4]) if len(results) > 4 else results

    # Header with metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Total Rows", data.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 Columns", data.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("🔢 Numeric Columns", numeric_cols)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        cat_cols = len(data.select_dtypes(include=['object']).columns)
        st.metric("🏷️ Categorical Columns", cat_cols)
        st.markdown('</div>', unsafe_allow_html=True)

    # Dynamic sidebar with advanced filters
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        if 'date' in data.columns:
            date_range = st.date_input("📅 Date Range", value=(data['date'].min(), data['date'].max()))
        if st.button("🔄 Refresh Dashboard", type="secondary"):
            st.rerun()

        # Add column selection
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect("📊 Select Numeric Columns", numeric_cols, numeric_cols[:3])

    st.progress(1.0)

    # Animated charts with tabs for better organization
    tabs = st.tabs(["📊 Overview", "📈 Trends", "🎯 Correlations", "🔍 Anomalies", "🎨 Distributions"])

    with tabs[0]:
        st.subheader("🔍 Data Overview")
        cols = st.columns(3)
        chart_idx = 0
        for key, result in display_results.items():
            if isinstance(result, go.Figure):
                with cols[chart_idx % 3]:
                    st.plotly_chart(result, use_container_width=True)
                chart_idx += 1

    with tabs[1]:
        st.subheader("📈 Time Series & Trends")
        datetime_cols = [col for col in data_display.columns if pd.api.types.is_datetime64_any_dtype(data_display[col])]
        if datetime_cols and len(selected_cols) > 0:
            trend_data = data_display.set_index(datetime_cols[0])
            fig = px.line(trend_data[selected_cols[:3]], title="📈 Multi-Variable Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("🎯 Correlation Matrix")
        if len(selected_cols) > 1:
            corr = data_display[selected_cols].corr()
            fig = px.imshow(corr, text_auto=True,
                          title="🔗 Correlation Heatmap",
                          color_continuous_scale='RdYlBu',
                          zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("🔍 Anomaly Detection")
        if "Anomaly detection" in results:
            st.info("🎯 Anomalies detected and highlighted in scatter plots")
            anomaly_fig = results.get("Anomaly detection", {}).get("plot")
            if anomaly_fig:
                st.plotly_chart(anomaly_fig, use_container_width=True)

    with tabs[4]:
        st.subheader("🎨 Data Distributions")
        cols = st.columns(2)
        for i, col in enumerate(selected_cols[:4]):
            with cols[i % 2]:
                fig = px.histogram(data_display, x=col, marginal="box",
                                 title=f"📊 Distribution: {col}",
                                 color_discrete_sequence=['#636efa'])
                st.plotly_chart(fig, use_container_width=True)

    # Enhanced AI Insights with animations
    st.header("🧠 AI-Powered Insights")
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.write("✨ **AI Analysis**: Our advanced AI has analyzed your data patterns and generated actionable insights.")
    st.markdown('</div>', unsafe_allow_html=True)

    if 'insights' not in st.session_state or st.session_state.insights is None or st.button("🔄 Refresh Insights", key="refresh_insights"):
        with st.spinner("🤖 DataPulse AI processing insights..."):
            try:
                st.session_state.insights = get_enhanced_insights(display_results)
                st.success("✅ AI Insights generated successfully!")
            except Exception as e:
                st.error(f"❌ AI Insights failed: {str(e)}")
                # Create dummy insights
                st.session_state.insights = [{"title": "🚨 AI Error", "content": f"AI insights unavailable: {str(e)}. Please check your API key."}]
                st.info("💡 Using fallback insights")

    insights = st.session_state.insights[:3]  # Limit to top 3 insights for performance
    for i, insight in enumerate(insights):
        with st.expander(f"📋 Insight #{i+1}: {insight['title']}"):
            st.markdown(f"💡 **{insight['content']}**")

    # Overall summary with dynamic content
    if st.button("🔮 Generate Overall Summary", type="primary"):
        with st.spinner("🔮 DataPulse AI generating comprehensive summary..."):
            overall_prompt = f"""
            Generate an executive summary of all analysis results, highlighting key trends and recommendations.
            Analysis completed: {list(results.keys())[:3]}

            Be concise, focus on key insights and actionable items.
            Keep summary under 100 words.
            """

            try:
                response = ollama.generate(model='llama3.2:3b', prompt=overall_prompt)

                st.success("📈 **Overall Summary Generated:**")
                st.markdown(f"🌟 {response['response'].strip()}")
            except Exception as e:
                st.error(f"❌ Failed to generate summary: {str(e)}")

    # Interactive export button
    if st.button("🚀 Export to Report Generator", type="secondary"):
        st.session_state.step = 'export'
        st.balloons()  # Celebration animation
        st.success("🎯 Moving to Export Section!")
        time.sleep(1)
        st.rerun()

@st.cache_data(ttl=3600)
def get_enhanced_insights(results):
    insights = [None] * len(results)
    icons = ["🚀", "💰", "📊", "🎯", "📈", "⚡"]

    def generate_insight(idx, analysis, result):
        prompt = f"""
        Summarize these {analysis} findings from data analysis in simple, engaging English for a non-expert.
        Use emojis where appropriate, be concise, and suggest 1-2 actionable insights.
        <100 words.

        Data context: {list(results.keys())[:3]}
        """

        try:
            response = ollama.generate(model='llama3.2:3b', prompt=prompt)

            insight = {
                "title": f"{icons[idx % len(icons)]} {analysis.replace('_', ' ').title()}",
                "content": response['response'].strip()
            }
        except Exception as e:
            insight = {
                "title": f"{icons[idx % len(icons)]} {analysis.replace('_', ' ').title()}",
                "content": f"AI analysis temporarily unavailable. Error: {str(e)}. Data processed successfully."
            }
        return insight

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures_with_idx = [(idx, executor.submit(generate_insight, idx, analysis, result)) for idx, (analysis, result) in enumerate(results.items())]
        for idx, future in futures_with_idx:
            try:
                insight = future.result(timeout=20)
                insights[idx] = insight
            except Exception as e:
                insights[idx] = {
                    "title": f"{icons[idx % len(icons)]} {list(results.keys())[idx].replace('_', ' ').title()}",
                    "content": f"AI analysis timed out or failed: {str(e)}"
                }
    return insights
