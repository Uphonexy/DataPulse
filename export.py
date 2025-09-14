import streamlit as st
import pandas as pd
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.platypus import Image  # Import Image for embedded plots
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import plotly.io as pio
import ollama

def export_report():
    st.markdown("""
    <style>
    .export-card {
        background: linear-gradient(135deg, #4421af 0%, #069fad 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    .feature-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        text-align: center;
        border-left: 5px solid #ff6b6b;
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-10px);}
        60% {transform: translateY(-5px);}
    }
    .bounce {
        animation: bounce 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 Professional Report Generator")
    st.markdown('<div class="export-card"><h1>📄 Create Your Analysis Report</h1><p>Generate comprehensive PDF reports with beautiful visualizations</p></div>', unsafe_allow_html=True)

    results = st.session_state.analysis_results
    data = st.session_state.processed_data

    if not results:
        st.error("❌ No analysis results available.")
        st.info("💡 Please complete the analysis process first to generate reports.")
        return

    # Move AI test to expander with button
    with st.expander("🧠 AI Status Test"):
        if st.button("Test DataPulse AI", key="test_ai_export"):
            try:
                response = ollama.generate(model='llama3.2:3b', prompt='Say "AI Ready" if you can understand this message.')
                if 'ready' in response['response'].lower():
                    st.success(f"🧠 DataPulse AI Status: Connected ✅ - {response['response']}")
                else:
                    st.warning("🧠 Connected but response unclear")
            except Exception as e:
                st.error(f"🧠 DataPulse AI Status: Failed ❌ - {str(e)}")

    st.write("---")

    if st.button("Generate PDF Report"):
        # Show progress for PDF generation
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("📄 Initializing PDF generation...")
        progress_bar.progress(20)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("Data Analysis Report", styles['Title']))
        progress_bar.progress(40)
        status_text.text("📝 Adding executive summary...")

        # Executive Summary with fallback
        if 'pdf_summary' not in st.session_state:
            with st.spinner("Generating AI summary..."):
                try:
                    summary_prompt = f"""
                    Summarize analyses: {list(results.keys())[:3]}
                    Professional, concise, <100 words.
                    """

                    summary_response = ollama.generate(model='llama3.2:3b', prompt=summary_prompt)

                    st.session_state.pdf_summary = summary_response['response'].strip()
                    progress_bar.progress(60)
                    status_text.text("✅ AI summary generated successfully!")
                except Exception as e:
                    st.session_state.pdf_summary = "Summary unavailable."
                    st.info(f"AI summary failed: {e}. Using fallback summary.")
        
        elements.append(Paragraph("Executive Summary", styles['Heading1']))
        elements.append(Paragraph(st.session_state.pdf_summary, styles['Normal']))

        insights = st.session_state.get('insights', [])
        progress_bar.progress(80)
        status_text.text("📊 Compiling insights...")

        for insight in insights:
            elements.append(Paragraph(f"Insights for {insight['title']}", styles['Heading2']))
            elements.append(Paragraph(insight['content'], styles['Normal']))

        progress_bar.progress(90)
        status_text.text("📄 Finalizing PDF...")

        # Data sample
        elements.append(Paragraph("Data Sample", styles['Heading1']))
        sample_data = data.head(10)
        table_data = [list(sample_data.columns)] + sample_data.values.tolist()
        t = Table(table_data)
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige)]))
        elements.append(t)

        # Add embedded charts
        chart_count = 0
        for analysis, result in list(results.items())[:2]:
            if 'plot' in result and result['plot']:
                fig = result['plot']
                try:
                    img_bytes = pio.to_image(fig, format='png', width=400, height=300, scale=1, engine='kaleido')
                    img = ImageReader(BytesIO(img_bytes))
                    elements.append(Paragraph(f"Visualization: {analysis}", styles['Heading2']))
                    elements.append(img)
                    chart_count += 1
                except Exception as e:
                    elements.append(Paragraph(f"Chart for {analysis} could not be embedded: {str(e)}", styles['Italic']))

        total_charts = len([r for r in results.values() if 'plot' in r and r['plot']])
        if total_charts > 2:
            elements.append(Paragraph("Additional visualizations are available in the dashboard view.", styles['Italic']))

        doc.build(elements)
        progress_bar.progress(100)

        buffer.seek(0)

        st.download_button("Download PDF", buffer, file_name="report.pdf", mime="application/pdf",
                          help="Download the comprehensive report as PDF")

    # Export options
    st.header("Additional Exports")
    if st.button("Export Raw Data as CSV"):
        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv_data, file_name="processed_data.csv", mime="text/csv",
                          help="Download the processed data as CSV")
