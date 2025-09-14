import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import re

def data_upload():
    st.title("🚀 Data Analysis Bot")
    st.markdown("""
    <div style='text-align: center;'>
        <h3 style='color: #636efa;'>✨ Upload & Explore Your Data ✨</h3>
    </div>
    """, unsafe_allow_html=True)

    # Animated header
    st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 1s ease-in-out;
    }
    .upload-box {
        border: 3px dashed #636efa;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
        background: linear-gradient(135deg, rgba(99, 110, 250, 0.1) 0%, rgba(0, 0, 0, 0.01) 100%);
    }
    .demo-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="upload-box fade-in">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "📁 Drop CSV or Excel files here (Single or Multiple)",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload files and let AI analyze them automatically!"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced demo section
    st.markdown("---")
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.write("🎯 **Try with Demo Data**")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if st.button("🚀 Load Sample Sales Dataset", key="demo_btn", use_container_width=True):
            with st.spinner("🎉 Generating sample data..."):
                # Generate sample sales data with proper types
                np.random.seed(42)
                dates = pd.date_range('2020-01-01', periods=100, freq='D')
                sales = np.random.normal(5000, 1000, 100).astype(float)  # Ensure float type
                categories = np.random.choice(['A', 'B', 'C'], 100, dtype=object)  # Ensure object type
                data = pd.DataFrame({
                    'date': dates,
                    'sales': sales,
                    'category': categories
                })

                st.session_state.data = data
                st.success("✅ Sample data loaded successfully!")
                st.balloons()

                # Preview with enhanced styling
                st.markdown("### 📊 Data Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📋 Rows", data.shape[0])
                with col2:
                    st.metric("📈 Columns", data.shape[1])
                with col3:
                    st.metric("📅 Date Range", f"{data['date'].min().strftime('%Y-%m')} to {data['date'].max().strftime('%Y-%m')}")
                with col4:
                    st.metric("🔢 Avg Sales", f"${data['sales'].mean():.0f}")

                # Enhanced data preview
                st.markdown("### 🔍 Data Preview")
                st.dataframe(data.head(10), use_container_width=True)

                # Auto proceed button
                if st.button("➡️ Proceed to Next Step", type="primary"):
                    st.rerun()
            return
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded_files:
        return

    data_frame_list = []
    merge_column = None

    for file in uploaded_files:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', encoding_errors='replace')
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
