"""
EDAwala - Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
from datetime import datetime

# Import our features
from edawala.core.data_loader import DataLoader
from edawala.auto_eda.report_generator import generate_eda_report
from edawala.storytelling.insights import generate_insights, generate_story, get_executive_summary

# Set page configuration
st.set_page_config(
    page_title="EDAwala - Advanced EDA Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .feature-header {
        font-size: 1.2rem;
        color: #3498db;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-bottom: 1rem;
    }
    .stTabs {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
    }
    .stTextInput, .stFileUploader {
        margin-bottom: 20px;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #7f8c8d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>🔍 EDAwala - Advanced EDA Tool</h1>", unsafe_allow_html=True)
st.markdown("<p class='feature-description'>Upload your dataset and let AI-powered tools help you analyze it efficiently</p>", unsafe_allow_html=True)

# Create sidebar for navigation
st.sidebar.image("https://via.placeholder.com/150x80?text=EDAwala", width=150)
st.sidebar.title("Navigation")

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Function to load data
def load_data(uploaded_file):
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.csv':
            df = DataLoader.load_csv(uploaded_file)
        elif file_extension in ['.xls', '.xlsx']:
            df = DataLoader.load_excel(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
        st.session_state.df = df
        st.session_state.filename = uploaded_file.name
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Data upload section
with st.sidebar.expander("Data Upload", expanded=True):
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None and (st.session_state.uploaded_file != uploaded_file):
        st.session_state.uploaded_file = uploaded_file
        df = load_data(uploaded_file)
        if df is not None:
            st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Example datasets
with st.sidebar.expander("Example Datasets"):
    example_datasets = {
        "Iris Flower Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "Titanic Passenger Data": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "Housing Prices": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/housing.csv"
    }
    
    selected_example = st.selectbox("Select an example dataset", list(example_datasets.keys()))
    
    if st.button("Load Example"):
        with st.spinner(f"Loading {selected_example}..."):
            url = example_datasets[selected_example]
            df = pd.read_csv(url)
            df.name = selected_example
            st.session_state.df = df
            st.session_state.filename = f"{selected_example}.csv"
            st.success(f"Loaded {selected_example} with {df.shape[0]} rows and {df.shape[1]} columns")

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Display basic info
    st.markdown(f"<h2 class='sub-header'>Dataset: {st.session_state.filename}</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())
    
    # Create tabs for different features
    tab1, tab2 = st.tabs([
        "Auto-EDA Report", 
        "Storytelling EDA"
    ])
    
    # Tab 1: Auto-EDA Report
    with tab1:
        st.markdown("<h3 class='feature-header'>1️⃣ Auto-EDA Report Generator</h3>", unsafe_allow_html=True)
        st.markdown("<p class='feature-description'>Generate a comprehensive EDA report with a single click</p>", unsafe_allow_html=True)
        
        # Report options
        col1, col2 = st.columns(2)
        
        with col1:
            report_format = st.selectbox("Report Format", ["HTML", "PDF", "Jupyter Notebook"], index=0)
            report_format = report_format.lower()
            
        with col2:
            report_name = st.text_input("Report Name (optional)", f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Generate report button
        if st.button("Generate EDA Report"):
            try:
                with st.spinner("Generating comprehensive EDA report..."):
                    # Create reports directory if it doesn't exist
                    os.makedirs("reports", exist_ok=True)
                    
                    # Generate the report
                    output_path = f"reports/{report_name}.{report_format}"
                    report_path = generate_eda_report(df, format=report_format, output_path=output_path)
                    
                    # Provide download link
                    if os.path.exists(report_path):
                        with open(report_path, "rb") as file:
                            btn = st.download_button(
                                label=f"Download {report_format.upper()} Report",
                                data=file,
                                file_name=os.path.basename(report_path),
                                mime=f"{'application/pdf' if report_format == 'pdf' else 'text/html'}"
                            )
                        
                        st.success(f"Report generated successfully! Click the button above to download.")
                        
                        # Show preview for HTML reports
                        if report_format == 'html':
                            with open(report_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.subheader("Report Preview")
                            st.components.v1.html(html_content, height=500, scrolling=True)
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
        
        # Data preview
        with st.expander("Preview Data"):
            st.dataframe(df.head(10))
    
    # Tab 2: Data Storytelling
    with tab2:
        st.markdown("<h3 class='feature-header'>2️⃣ Storytelling EDA (Auto Insights)</h3>", unsafe_allow_html=True)
        st.markdown("<p class='feature-description'>Let AI tell the story of your data with automated insights and narratives</p>", unsafe_allow_html=True)
        
        # Options for insight generation
        use_llm = st.checkbox("Use AI-powered enhanced insights", value=False)
        
        if use_llm:
            llm_provider = st.radio(
                "Select AI Provider:",
                ["Gemini (Free)", "OpenAI (Requires paid API)"],
                index=0
            )
            
            provider = "gemini" if "Gemini" in llm_provider else "openai"
            
            api_key = st.text_input(
                "API Key (leave blank to use environment variable)", 
                type="password",
                help="For Gemini, get a free API key from https://makersuite.google.com/"
            )
            
            st.caption("Your API key is not stored and is only used for this session.")
        else:
            provider = "gemini"
            api_key = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_insights = st.slider("Maximum number of insights", min_value=5, max_value=30, value=15)
        
        with col2:
            insight_types = st.multiselect(
                "Include insight types",
                ["Correlations", "Distributions", "Outliers", "Missing Values", "Categorical", "Time Series"],
                ["Correlations", "Distributions", "Outliers", "Missing Values", "Categorical", "Time Series"]
            )
        
        # Generate insights button
        if st.button("Generate Data Story"):
            try:
                with st.spinner("Analyzing data and generating insights..."):
                    # Generate insights
                    insights = generate_insights(
                        df, 
                        use_llm=use_llm, 
                        api_key=api_key, 
                        llm_provider=provider, 
                        max_insights=max_insights
                    )
                    
                    # Generate story
                    story = generate_story(
                        df, 
                        insights, 
                        use_llm=use_llm, 
                        api_key=api_key, 
                        llm_provider=provider
                    )
                    
                    # Display results
                    st.subheader("Executive Summary")
                    summary = get_executive_summary(df, use_llm=use_llm, api_key=api_key, llm_provider=provider)
                    st.info(summary)
                    
                    st.subheader("Data Story")
                    st.markdown(story)
                    
                    # Display raw insights in an expander
                    with st.expander("Raw Insights"):
                        for i, insight in enumerate(insights, 1):
                            st.markdown(f"**{i}. {insight.get('description', '')}**")
                            st.text(f"Type: {insight.get('type', 'N/A')} | Importance: {insight.get('importance', 0):.2f}")
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")

else:
    # Welcome message when no data is loaded
    st.info("👈 Please upload a dataset or select an example dataset from the sidebar to get started.")
    
    # Feature showcase
    st.markdown("<h2 class='sub-header'>EDAwala Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 class='feature-header'>1️⃣ Auto-EDA Report Generator</h3>", unsafe_allow_html=True)
        st.markdown("<p>Single-click solution to generate comprehensive EDA reports in multiple formats</p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 class='feature-header'>2️⃣ Storytelling EDA (Auto Insights)</h3>", unsafe_allow_html=True)
        st.markdown("<p>AI-generated insights and narratives that explain your data in plain language</p>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Created by Yashu • © 2025 EDAwala</div>", unsafe_allow_html=True)