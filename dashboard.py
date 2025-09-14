import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import openai

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
        elif key in results and 'plots' in str(result) and isinstance(result, list):
            for fig in result:
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
