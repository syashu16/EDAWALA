"""
EDAwala Streamlit Application - Simplified UI Version
"""
import streamlit as st
import pandas as pd
import os
import base64
from io import BytesIO
import time
import numpy as np

# Import internal modules
from edawala.core.data_loader import load_data
from edawala.auto_eda.report_generator import generate_eda_report
from edawala.storytelling.insights import generate_insights, generate_story, get_executive_summary

# Set page configuration
st.set_page_config(
    page_title="EDAwala - Advanced EDA Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add minimal CSS - remove complex styling that might cause rendering issues
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .sub-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.title("EDAwala - Advanced EDA Tool")
st.write("Upload your dataset and quickly generate comprehensive EDA reports or get data insights.")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'report_path' not in st.session_state:
    st.session_state.report_path = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'executive_summary' not in st.session_state:
    st.session_state.executive_summary = None
if 'data_story' not in st.session_state:
    st.session_state.data_story = None

# Sidebar for data loading
with st.sidebar:
    st.header("Data Input")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    # Load example data
    st.markdown("### Or load an example dataset")
    example_data = st.selectbox(
        "Select example dataset",
        ["None", "Iris", "Titanic", "Diamonds", "House Prices"]
    )
    
    if example_data != "None":
        if example_data == "Iris":
            from sklearn.datasets import load_iris
            data = load_iris(as_frame=True)
            df = data.data
            df['target'] = data.target
            file_name = "iris_dataset.csv"
        elif example_data == "Titanic":
            df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
            file_name = "titanic_dataset.csv"
        elif example_data == "Diamonds":
            df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")
            file_name = "diamonds_dataset.csv"
        elif example_data == "House Prices":
            df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
            file_name = "house_prices_dataset.csv"
        
        st.session_state.df = df
        st.session_state.file_name = file_name
        st.success(f"Loaded example dataset: {example_data}")
    
    # Load uploaded file
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            
            if df is not None:
                st.session_state.df = df
                st.session_state.file_name = uploaded_file.name
                st.success(f"Successfully loaded {uploaded_file.name}")
            else:
                st.error("Failed to load the file. Please check the format.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Display dataset info
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### Dataset Information")
        st.markdown(f"**Rows:** {df.shape[0]}")
        st.markdown(f"**Columns:** {df.shape[1]}")
        
        # Display memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        if memory_usage < 1024:
            memory_str = f"{memory_usage} bytes"
        elif memory_usage < 1024 ** 2:
            memory_str = f"{memory_usage / 1024:.2f} KB"
        else:
            memory_str = f"{memory_usage / (1024 ** 2):.2f} MB"
        
        st.markdown(f"**Memory Usage:** {memory_str}")

# Main area content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Create simple tabs without emojis
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Auto-EDA Report", "Storytelling EDA"])
    
    # Data Preview Tab
    with tab1:
        st.header("Data Preview")
        
        # Display basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", df.shape[0])
            
        with col2:
            st.metric("Columns", df.shape[1])
            
        with col3:
            missing = df.isna().sum().sum()
            missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
            st.metric("Missing Values", f"{missing_pct:.1f}%")
            
        with col4:
            duplicates = df.duplicated().sum()
            duplicate_pct = (duplicates / df.shape[0]) * 100 if df.shape[0] > 0 else 0
            st.metric("Duplicated Rows", f"{duplicate_pct:.1f}%")
        
        st.divider()
        
        # Display the dataframe
        st.dataframe(df.head(10), use_container_width=True)
        
        # Display dtypes
        st.subheader("Data Types")
        
        # Format the dtypes as a table
        dtypes_data = []
        for col, dtype in zip(df.columns, df.dtypes):
            dtypes_data.append([col, str(dtype)])
            
        # Show dtypes as dataframe
        st.dataframe(pd.DataFrame(dtypes_data, columns=["Column", "Data Type"]), use_container_width=True)
    
    # Auto-EDA Report Tab
    with tab2:
        st.header("Automatic EDA Report")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            report_format = st.selectbox(
                "Report Format",
                ["HTML", "PDF", "Jupyter Notebook"],
                index=0
            )
            
        with col2:
            generate_btn = st.button("Generate Report", type="primary", use_container_width=True)
        
        if generate_btn:
            with st.spinner("Generating EDA report..."):
                try:
                    # Map user-friendly names to actual format strings
                    format_map = {
                        "HTML": "html",
                        "PDF": "pdf",
                        "Jupyter Notebook": "notebook"
                    }
                    
                    # Generate the report
                    report_path = generate_eda_report(
                        df, 
                        format=format_map[report_format],
                        output_path=None
                    )
                    
                    st.session_state.report_path = report_path
                    st.success(f"Report successfully generated: {os.path.basename(report_path)}")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
        
        # Display download button if report exists
        if st.session_state.report_path and os.path.exists(st.session_state.report_path):
            st.divider()
            
            # Read the file contents
            with open(st.session_state.report_path, "rb") as f:
                report_bytes = f.read()
            
            # Get file extension
            _, file_extension = os.path.splitext(st.session_state.report_path)
            
            # Determine mime type
            mime_types = {
                ".html": "text/html",
                ".pdf": "application/pdf",
                ".ipynb": "application/x-ipynb+json"
            }
            mime_type = mime_types.get(file_extension, "application/octet-stream")
            
            # Show download button
            st.download_button(
                label="Download Report",
                data=report_bytes,
                file_name=os.path.basename(st.session_state.report_path),
                mime=mime_type
            )
            
            # For HTML reports, display an iframe preview
            if file_extension == ".html":
                st.subheader("Report Preview")
                
                # Create a base64 encoded data URL for the HTML content
                report_b64 = base64.b64encode(report_bytes).decode()
                iframe_html = f"""
                <div style="width:100%; height:600px; border:none; overflow:hidden;">
                    <iframe src="data:text/html;base64,{report_b64}" width="100%" height="600px" style="border:none;"></iframe>
                </div>
                """
                
                st.markdown(iframe_html, unsafe_allow_html=True)
    
    # Storytelling EDA Tab
    with tab3:
        st.header("Data Storytelling")
        
        # Options for insights
        st.subheader("Options")
        
        # AI provider options
        use_ai = st.checkbox("Use AI-powered enhanced insights", value=False)
        
        if use_ai:
            ai_col1, ai_col2 = st.columns(2)
            
            with ai_col1:
                llm_provider = st.radio(
                    "AI Provider",
                    ["Gemini (Free)", "OpenAI (Paid)"],
                    index=0
                )
            
            with ai_col2:
                if llm_provider == "Gemini (Free)":
                    api_key = st.text_input("Google Gemini API Key", type="password")
                    st.markdown("""
                    <small>Get your free API key from <a href="https://makersuite.google.com/" target="_blank">Google AI Studio</a></small>
                    """, unsafe_allow_html=True)
                else:
                    api_key = st.text_input("OpenAI API Key", type="password")
                    st.markdown("""
                    <small>Get your API key from <a href="https://platform.openai.com/" target="_blank">OpenAI</a></small>
                    """, unsafe_allow_html=True)
        
        # More options for insights
        options_col1, options_col2 = st.columns(2)
        
        with options_col1:
            max_insights = st.slider("Maximum number of insights", 5, 30, 15)
            
        with options_col2:
            # Types of insights to include
            insight_types = st.multiselect(
                "Insight types to include",
                ["Basic Statistics", "Correlations", "Distributions", "Outliers", "Patterns", "Recommendations"],
                default=["Basic Statistics", "Correlations", "Distributions", "Outliers"]
            )
        
        # Generate insights button
        gen_insights_btn = st.button("Generate Data Story", type="primary", use_container_width=True)
        
        if gen_insights_btn:
            # Check if API key is provided when use_ai is enabled
            if use_ai and not api_key:
                st.warning("Please provide an API key to use AI-powered insights. For now, generating basic statistical insights.")
                use_ai = False
            
            with st.spinner("Generating data insights..."):
                try:
                    # Map the provider name to the expected parameter
                    provider = "gemini" if llm_provider == "Gemini (Free)" else "openai"
                    
                    # Set the API key in environment variables
                    if use_ai and api_key:
                        if provider == "gemini":
                            os.environ["GOOGLE_API_KEY"] = api_key
                        else:
                            os.environ["OPENAI_API_KEY"] = api_key
                    
                    # Generate insights with or without LLM
                    insights = generate_insights(
                        df, 
                        use_llm=use_ai,
                        llm_provider=provider if use_ai else None,
                        max_insights=max_insights
                    )
                    
                    # Get executive summary and story
                    executive_summary = get_executive_summary(df, insights, use_llm=use_ai, llm_provider=provider if use_ai else None)
                    data_story = generate_story(df, insights, use_llm=use_ai, llm_provider=provider if use_ai else None)
                    
                    # Store in session state
                    st.session_state.insights = insights
                    st.session_state.executive_summary = executive_summary
                    st.session_state.data_story = data_story
                    
                    st.success("Data story successfully generated!")
                except Exception as e:
                    st.error(f"Error generating insights: {e}")
                    st.session_state.insights = None
                    st.session_state.executive_summary = None
                    st.session_state.data_story = None
        
        # If insights are available, display them
        if st.session_state.insights is not None and st.session_state.data_story is not None:
            st.divider()
            
            # Display executive summary
            if st.session_state.executive_summary:
                st.subheader("Executive Summary")
                st.info(st.session_state.executive_summary)
            
            # Display the data story
            st.subheader("Data Story")
            st.write(st.session_state.data_story)
            
            # Show all raw insights (in an expandable section)
            with st.expander("View All Insights"):
                for idx, insight in enumerate(st.session_state.insights):
                    st.markdown(f"### Insight {idx+1}")
                    st.markdown(f"**{insight['title']}**")
                    st.markdown(insight['description'])
                    if "recommendation" in insight:
                        st.markdown(f"**Recommendation:** {insight['recommendation']}")
                    st.markdown("---")
            
            # Add download options
            st.divider()
            
            # Prepare content for download
            markdown_content = f"""# Data Story: {st.session_state.file_name}\n\n"""
            
            if st.session_state.executive_summary:
                markdown_content += f"""## Executive Summary\n\n{st.session_state.executive_summary}\n\n"""
                
            markdown_content += f"""## Detailed Data Story\n\n{st.session_state.data_story}\n\n"""
            
            markdown_content += """## All Insights\n\n"""
            
            for idx, insight in enumerate(st.session_state.insights):
                markdown_content += f"""### Insight {idx+1}: {insight['title']}\n\n"""
                markdown_content += f"""{insight['description']}\n\n"""
                if "recommendation" in insight:
                    markdown_content += f"""**Recommendation:** {insight['recommendation']}\n\n"""
            
            # Download button
            st.download_button(
                label="Download Data Story as Markdown",
                data=markdown_content,
                file_name=f"data_story_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
else:
    # Display welcome message
    st.info("Please upload a dataset or select an example dataset from the sidebar to get started.")
    
    st.write("""
    ### EDAwala Features:
    
    * Automatic EDA report generation
    * Data storytelling with insights
    * Interactive data exploration
    * Multiple export formats
    """)