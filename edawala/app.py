"""
EDAwala Streamlit Application - Dark Theme with Purple Accents
"""
import streamlit as st
import pandas as pd
import os
import base64
from io import BytesIO
import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Define the dark theme with purple accents
st.markdown("""
<style>
    /* Global theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --main-bg-color: #121212;
        --secondary-bg-color: #1e1e2e;
        --accent-color: #8A2BE2;  /* Purple accent */
        --accent-color-light: #9d4edd;
        --text-color: #E0E0E0;
        --text-color-secondary: #B0B0B0;
        --card-bg-color: #2d2d3a;
        --border-color: #3d3d50;
        --success-color: #4CAF50;
        --info-color: #2196F3;
        --warning-color: #FF9800;
        --error-color: #F44336;
    }
    
    /* Global styles */
    body {
        background-color: var(--main-bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
        font-weight: 600;
    }
    
    p, span, div {
        color: var(--text-color);
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(90deg, #3a0647, #8A2BE2);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.3);
    }
    
    .header-banner h1 {
        margin: 0;
        color: white;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    .header-banner p {
        color: rgba(255,255,255,0.85);
        margin-top: 8px;
    }
    
    /* Cards */
    .card {
        background-color: var(--card-bg-color);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        border-top: 3px solid var(--accent-color);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #2d2d3a, #3a3a4a);
        padding: 12px 15px;
        border-radius: 6px;
        margin-bottom: 20px;
        border-left: 4px solid var(--accent-color);
    }
    
    .section-header h2 {
        margin: 0;
        color: white;
    }
    
    .section-header p {
        color: var(--text-color-secondary);
        margin-top: 5px;
        margin-bottom: 0;
    }
    
    /* Buttons */
    div[data-testid="stButton"] > button {
        background: linear-gradient(45deg, var(--accent-color), var(--accent-color-light));
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(138, 43, 226, 0.4);
        background: linear-gradient(45deg, var(--accent-color-light), var(--accent-color));
        cursor: pointer;
    }
    
    /* Example dataset buttons */
    .example-button {
        background-color: #2d2d3a;
        border: 1px solid #4b4b63;
        color: var(--text-color);
        border-radius: 4px;
        padding: 8px 12px;
        transition: all 0.3s ease;
        text-align: center;
        cursor: pointer;
    }
    
    .example-button:hover {
        background-color: #3d3d50;
        border-color: var(--accent-color);
        box-shadow: 0 0 8px rgba(138, 43, 226, 0.4);
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #2d2d3a;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    div[data-testid="stMetric"] > div:first-child {
        color: var(--accent-color-light) !important;
    }
    
    div[data-testid="stMetric"] label {
        color: var(--text-color) !important;
    }
    
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: #2d2d3a;
        border-radius: 4px 4px 0 0;
        font-weight: 500;
        color: var(--text-color-secondary);
        margin-right: 4px;
        font-size: 15px;
        padding: 10px 16px;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, #3a3a4a, #2d2d3a);
        color: white;
        border-bottom: 2px solid var(--accent-color);
    }
    
    div[data-testid="stTabContent"] {
        background-color: var(--secondary-bg-color);
        border-radius: 0 8px 8px 8px;
        padding: 15px;
        border: 1px solid #3d3d50;
    }
    
    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    div[data-testid="stDataFrame"] th {
        background-color: #3d3d50 !important;
        color: white !important;
    }
    
    div[data-testid="stDataFrame"] td {
        background-color: #2d2d3a !important;
        color: var(--text-color) !important;
        border-bottom: 1px solid #3d3d50 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-bg-color);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: white;
    }
    
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: var(--text-color) !important;
    }
    
    /* File uploader */
    div.stFileUploader > div {
        background-color: #2d2d3a;
        border: 1px dashed #6a5acd;
        padding: 15px;
        border-radius: 8px;
    }
    
    div.stFileUploader > div:hover {
        border-color: var(--accent-color);
    }
    
    .css-nps9tx, .stFileUploader > label {
        color: var(--text-color) !important;
    }
    
    div.stFileUploader > div > div > button {
        background-color: var(--accent-color);
        color: white;
        border-radius: 4px;
        border: none;
        padding: 8px 12px;
        transition: all 0.3s;
    }
    
    div.stFileUploader > div > div > button:hover {
        background-color: var(--accent-color-light);
        box-shadow: 0 0 8px rgba(138, 43, 226, 0.4);
    }
    
    /* Message containers */
    div[data-testid="stInfo"], 
    div[data-testid="stSuccess"],
    div[data-testid="stWarning"],
    div[data-testid="stError"] {
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    div[data-testid="stInfo"] {
        background-color: rgba(33, 150, 243, 0.15);
        border-left: 4px solid var(--info-color);
    }
    
    div[data-testid="stSuccess"] {
        background-color: rgba(76, 175, 80, 0.15);
        border-left: 4px solid var(--success-color);
    }
    
    div[data-testid="stWarning"] {
        background-color: rgba(255, 152, 0, 0.15);
        border-left: 4px solid var(--warning-color);
    }
    
    div[data-testid="stError"] {
        background-color: rgba(244, 67, 54, 0.15);
        border-left: 4px solid var(--error-color);
    }
    
    /* Form inputs */
    input, select, textarea, div[role="listbox"] {
        background-color: #2d2d3a !important;
        border: 1px solid #3d3d50 !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
    }
    
    input:focus, select:focus, textarea:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 2px rgba(138, 43, 226, 0.2) !important;
    }
    
    div.stTextInput > div > div > input,
    div.stSelectbox > div > div > select,
    div.stTextArea > div > div > textarea {
        background-color: #2d2d3a !important;
        color: var(--text-color) !important;
        border: 1px solid #3d3d50 !important;
    }
    
    /* Checkbox */
    .stCheckbox > div[role="checkbox"] {
        background-color: #2d2d3a !important;
        border-color: #3d3d50 !important;
    }
    
    .stCheckbox > div[role="checkbox"][aria-checked="true"] {
        background-color: var(--accent-color) !important;
        border-color: var(--accent-color) !important;
    }
    
    /* Expander */
    details {
        background-color: #2d2d3a;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 15px;
    }
    
    details summary {
        padding: 10px 15px;
        background-color: #3d3d50;
        color: white;
        cursor: pointer;
        font-weight: 500;
    }
    
    details > div {
        padding: 15px;
    }
    
    /* Slider */
    div.stSlider > div > div {
        background-color: #3d3d50 !important;
    }
    
    div.stSlider > div > div > div {
        background-color: var(--accent-color) !important;
    }
    
    div.stSlider > div > div > div > div {
        background-color: var(--accent-color-light) !important;
        border: 2px solid var(--accent-color-light) !important;
    }

    /* Success message with better visibility */
    .success-message {
        background-color: rgba(76, 175, 80, 0.2); 
        color: #d4edda; 
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 15px 0;
    }
    
    /* Download button styling */
    .download-button {
        background: linear-gradient(45deg, #7B2CBF, #9D4EDD);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 15px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .download-button:hover {
        background: linear-gradient(45deg, #9D4EDD, #C77DFF);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(138, 43, 226, 0.4);
    }
    
    .download-button:disabled {
        background: #4b4b63;
        opacity: 0.7;
        border: 1px solid #6a6a8a;
        cursor: not-allowed;
    }
    
    .download-button:disabled:hover {
        transform: none;
        box-shadow: none;
    }
    
    /* Feature cards on welcome screen */
    .feature-card {
        background-color: #2d2d3a;
        border-radius: 10px;
        padding: 25px;
        margin: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        border-top: 3px solid var(--accent-color);
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 15px;
        background: -webkit-linear-gradient(#9D4EDD, #7B2CBF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-title {
        color: white;
        font-size: 20px;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        color: #B0B0B0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #3d3d50;
        color: #B0B0B0;
        font-size: 12px;
    }
    
    /* Override any remaining white backgrounds */
    div.css-1kyxreq, div.css-12oz5g7 {
        background-color: var(--secondary-bg-color) !important;
    }
    
    /* Fix for markdown text */
    .element-container div.markdown-text-container p,
    .element-container div.markdown-text-container li {
        color: var(--text-color) !important;
    }
    
    /* Data story display area */
    .story-container {
        background-color: #2d2d3a;
        border-radius: 8px;
        padding: 20px;
        border-left: 3px solid var(--accent-color);
        margin: 15px 0;
    }
    
    .executive-summary {
        background-color: rgba(157, 78, 221, 0.1);
        border-radius: 8px;
        padding: 20px;
        border-left: 3px solid #9D4EDD;
        margin: 15px 0;
    }
    
    /* Make SVG and canvas elements compatible with dark theme */
    .stMarkdown svg, .element-container svg, canvas {
        filter: hue-rotate(180deg) invert(92%) !important;
    }
</style>
""", unsafe_allow_html=True)

# App title with corporate banner
st.markdown("""
<div class="header-banner">
    <h1>EDAwala - Advanced EDA Tool</h1>
    <p>Uncover insights and patterns in your data with comprehensive exploratory analysis.</p>
</div>
""", unsafe_allow_html=True)

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
    st.markdown("""
    <div class="section-header">
        <h2>Data Input</h2>
        <p>Upload your data or try one of our examples.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader with enhanced UI
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    # Divider
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Load example data with thumbnails
    st.markdown("""
    <div style="margin: 10px 0;">
        <h3 style="font-size: 1.1em; font-weight: 600; color: white;">Or try an example dataset</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Example datasets with icons
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="example-button">', unsafe_allow_html=True)
        iris_btn = st.button("🌸 Iris", help="Iris flower dataset - great for classification", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="example-button">', unsafe_allow_html=True)
        titanic_btn = st.button("🚢 Titanic", help="Titanic survival dataset", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="example-button">', unsafe_allow_html=True)
        diamonds_btn = st.button("💎 Diamonds", help="Diamonds price dataset", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="example-button">', unsafe_allow_html=True)
        housing_btn = st.button("🏠 Housing", help="Housing prices dataset", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data based on button clicks
    if iris_btn:
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.data
        df['target'] = data.target
        file_name = "iris_dataset.csv"
        st.session_state.df = df
        st.session_state.file_name = file_name
        st.success(f"Loaded example dataset: Iris")
        
    elif titanic_btn:
        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        file_name = "titanic_dataset.csv"
        st.session_state.df = df
        st.session_state.file_name = file_name
        st.success(f"Loaded example dataset: Titanic")
        
    elif diamonds_btn:
        df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")
        file_name = "diamonds_dataset.csv"
        st.session_state.df = df
        st.session_state.file_name = file_name
        st.success(f"Loaded example dataset: Diamonds")
        
    elif housing_btn:
        df = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
        file_name = "house_prices_dataset.csv"
        st.session_state.df = df
        st.session_state.file_name = file_name
        st.success(f"Loaded example dataset: Housing")
    
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
            st.error(f"Error loading file: {str(e)}")
    
    # Display dataset info with improved styling
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(90deg, #2d3a4a, #3d4a5a); border-left: 4px solid #9D4EDD;">
            <h3 style="margin: 0; color: white;">Dataset Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Format metrics with improved styling
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div style='font-weight:600; color:#B0B0B0;'>Rows</div><div style='font-size:20px;color:#9D4EDD;'>{df.shape[0]:,}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='font-weight:600; color:#B0B0B0;'>Columns</div><div style='font-size:20px;color:#9D4EDD;'>{df.shape[1]}</div>", unsafe_allow_html=True)
        
        # Memory usage with nice formatting
        memory_usage = df.memory_usage(deep=True).sum()
        if memory_usage < 1024:
            memory_str = f"{memory_usage} bytes"
        elif memory_usage < 1024 ** 2:
            memory_str = f"{memory_usage / 1024:.2f} KB"
        else:
            memory_str = f"{memory_usage / (1024 ** 2):.2f} MB"
        
        st.markdown(f"<div style='font-weight:600;margin-top:10px;color:#B0B0B0;'>Memory Usage</div><div style='font-size:16px;color:#FF5252;'>{memory_str}</div>", unsafe_allow_html=True)
        
        # Add column type counts
        num_cols = len(df.select_dtypes(include=['number']).columns)
        cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"<div style='font-weight:600;margin-top:10px;color:#B0B0B0;'>Numeric Columns</div><div style='font-size:16px;color:#BB86FC;'>{num_cols}</div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div style='font-weight:600;margin-top:10px;color:#B0B0B0;'>Categorical Columns</div><div style='font-size:16px;color:#03DAC5;'>{cat_cols}</div>", unsafe_allow_html=True)

# Main area content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Create enhanced tabs with icons
    tabs = st.tabs([
        "📊 Data Preview", 
        "📈 Auto-EDA Report", 
        "📝 Storytelling EDA"
    ])
    
    # Data Preview Tab
    with tabs[0]:
        st.markdown("""
        <div class="section-header">
            <h2>Data Preview</h2>
            <p>Examine your dataset structure and basic statistics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display basic stats with enhanced cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}", delta=None)
            
        with col2:
            st.metric("Total Columns", df.shape[1], delta=None)
            
        with col3:
            missing = df.isna().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            missing_pct = (missing / total_cells) * 100 if total_cells > 0 else 0
            st.metric("Missing Values", f"{missing_pct:.1f}%", f"{missing:,} cells" if missing > 0 else "None")
            
        with col4:
            duplicates = df.duplicated().sum()
            duplicate_pct = (duplicates / df.shape[0]) * 100 if df.shape[0] > 0 else 0
            st.metric("Duplicated Rows", f"{duplicate_pct:.1f}%", f"{duplicates:,} rows" if duplicates > 0 else "None")
        
        # Add enhanced section divider
        st.markdown("""
        <div style="margin: 30px 0; height: 3px; background-image: linear-gradient(to right, rgba(157, 78, 221, 0), rgba(157, 78, 221, 0.5), rgba(157, 78, 221, 0));"></div>
        """, unsafe_allow_html=True)
        
        # Display the dataframe with improved UI
        st.markdown("<h3>Data Sample</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Add quick data exploration options
        st.markdown("""
        <div style="margin: 20px 0;">
            <h3>Quick Exploration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        explore_tab1, explore_tab2, explore_tab3 = st.tabs(["Data Types", "Statistics", "Missing Values"])
        
        with explore_tab1:
            # Format the dtypes as a table with enhanced styling
            dtypes_data = []
            for col, dtype in zip(df.columns, df.dtypes):
                dtypes_data.append([col, str(dtype)])
                
            # Show dtypes as dataframe
            dtypes_df = pd.DataFrame(dtypes_data, columns=["Column", "Data Type"])
            st.dataframe(dtypes_df, use_container_width=True)
        
        with explore_tab2:
            # Display descriptive statistics
            st.dataframe(df.describe().T, use_container_width=True)
        
        with explore_tab3:
            # Display missing values analysis
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': df.isna().sum(),
                'Missing %': df.isna().mean() * 100
            }).sort_values('Missing %', ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)
            
            # Add missing values visualization if there are any missing values
            if df.isna().sum().sum() > 0:
                top_missing = missing_df[missing_df['Missing Values'] > 0]
                if not top_missing.empty:
                    st.markdown("<h3>Missing Values by Column</h3>", unsafe_allow_html=True)
                    
                    # Create a simple horizontal bar chart for missing values
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    # Configure the dark theme for matplotlib
                    plt.style.use('dark_background')
                    
                    # Only show top 10 columns with missing values
                    top_missing = top_missing.head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.set_facecolor('#2d2d3a')
                    fig.patch.set_facecolor('#2d2d3a')
                    
                    bars = sns.barplot(
                        x='Missing %', 
                        y='Column', 
                        data=top_missing,
                        palette='viridis',
                        ax=ax
                    )
                    
                    # Add percentage labels
                    for i, p in enumerate(bars.patches):
                        percentage = top_missing.iloc[i]['Missing %']
                        bars.annotate(
                            f'{percentage:.1f}%', 
                            (p.get_width() + 1, p.get_y() + p.get_height()/2),
                            ha = 'left', va = 'center', 
                            fontsize=9, color='white'
                        )
                    
                    ax.set_title('Columns with Highest Missing Values', color='white', fontsize=14)
                    ax.set_xlabel('Missing Values (%)', color='white')
                    ax.set_ylabel('Column Name', color='white')
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('#3d3d50')
                    ax.spines['top'].set_color('#3d3d50') 
                    ax.spines['right'].set_color('#3d3d50')
                    ax.spines['left'].set_color('#3d3d50')
                    ax.grid(axis='x', linestyle='--', alpha=0.6, color='#3d3d50')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # Auto-EDA Report Tab
    with tabs[1]:
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(90deg, #2d3a2d, #3a4a3a); border-left: 4px solid #00C853;">
            <h2>Automatic EDA Report</h2>
            <p>Generate comprehensive exploratory data analysis reports.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced report generation options
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Report Options</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            report_format = st.selectbox(
                "Report Format",
                ["HTML", "PDF", "Jupyter Notebook"],
                index=0
            )
            
        with col2:
            generate_btn = st.button("Generate Report", type="primary", use_container_width=True)
        
        # Display report generation time estimate
        report_time_estimate = {
            "HTML": "~30 seconds",
            "PDF": "~40 seconds",
            "Jupyter Notebook": "~20 seconds"
        }
        
        st.markdown(f"""
        <div style="font-size:0.9em; color:#B0B0B0; margin-top:10px;">
            <strong>Estimated time:</strong> {report_time_estimate[report_format]} 
            <span style="margin-left:10px;">•</span>
            <strong>Last updated:</strong> {time.strftime("%Y-%m-%d %H:%M")}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
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
                    
                    # Enhanced success message
                    st.markdown(f"""
                    <div class="success-message">
                        <h4 style="margin-top:0">Report successfully generated! 🎉</h4>
                        <p><strong>Filename:</strong> {os.path.basename(report_path)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Display download button if report exists
        if st.session_state.report_path and os.path.exists(st.session_state.report_path):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Download Report</h3>", unsafe_allow_html=True)
            
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
            
            # Show download button with file size info
            file_size = len(report_bytes)
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 ** 2:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 ** 2):.1f} MB"
                
            st.markdown(f"""
            <div style="font-size:0.9em; color:#B0B0B0; margin-bottom:15px;">
                <strong>File size:</strong> {size_str} 
                <span style="margin-left:10px;">•</span>
                <strong>Created:</strong> {time.strftime("%Y-%m-%d %H:%M")}
            </div>
            """, unsafe_allow_html=True)
            
            # Custom download button
            col1, col2 = st.columns([3, 1])
            with col2:
                st.download_button(
                    label="Download Report",
                    data=report_bytes,
                    file_name=os.path.basename(st.session_state.report_path),
                    mime=mime_type,
                    use_container_width=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # For HTML reports, display an iframe preview with enhanced styling
            if file_extension == ".html":
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Report Preview</h3>", unsafe_allow_html=True)
                
                # Create a base64 encoded data URL for the HTML content
                report_b64 = base64.b64encode(report_bytes).decode()
                iframe_html = f"""
                <div style="width:100%; height:600px; border:1px solid #3d3d50; border-radius:8px; overflow:hidden;">
                    <iframe src="data:text/html;base64,{report_b64}" width="100%" height="600px" style="border:none;"></iframe>
                </div>
                """
                
                st.markdown(iframe_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Storytelling EDA Tab
    with tabs[2]:
        st.markdown("""
        <div class="section-header" style="background: linear-gradient(90deg, #3a2d3a, #4a3a4a); border-left: 4px solid #FF4081;">
            <h2>Data Storytelling</h2>
            <p>Generate insights and data stories from your dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced options for insights
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Insight Options</h3>", unsafe_allow_html=True)
        
        # AI provider options with improved styling
        use_ai = st.checkbox("Use AI-powered enhanced insights", value=False)
        
        if use_ai:
            api_key = st.text_input("Google Gemini API Key", type="password")
            st.markdown("""
            <div style="background-color: #3d3d50; padding: 10px; border-radius: 6px; margin: 10px 0; border-left: 3px solid #9D4EDD;">
                <small style="color: #B0B0B0;">Get your free API key from <a href="https://aistudio.google.com/" target="_blank" style="color: #BB86FC;">Google AI Studio</a></small>
            </div>
            """, unsafe_allow_html=True)
            
            # Add this notice about API rate limits
            st.info("""
            **Note about AI features:** The free tier of the Gemini API has usage limits. 
            If you encounter rate limit errors, the app will automatically fall back to statistical insights.
            """)
        
        # More options for insights with better UI organization
        col1, col2 = st.columns(2)
        
        with col1:
            max_insights = st.slider("Maximum number of insights", 5, 30, 15)
            
        with col2:
            # Types of insights to include
            insight_types = st.multiselect(
                "Insight types to include",
                ["Basic Statistics", "Correlations", "Distributions", "Outliers", "Patterns", "Recommendations"],
                default=["Basic Statistics", "Correlations", "Distributions", "Outliers"]
            )
        
        # Generate insights button with loading animation
        gen_insights_btn = st.button("Generate Data Story", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if gen_insights_btn:
            # Check if API key is provided when use_ai is enabled
            if use_ai and not api_key:
                st.warning("Please provide an API key to use AI-powered insights. For now, generating basic statistical insights.")
                use_ai = False
            
            with st.spinner("Generating data insights..."):
                try:
                    # Set the API key in environment variables
                    if use_ai and api_key:
                        os.environ["GOOGLE_API_KEY"] = api_key
                    
                    # Generate insights with or without LLM
                    insights = generate_insights(
                        df, 
                        use_llm=use_ai,
                        max_insights=max_insights
                    )
                    
                    # Get executive summary and story
                    executive_summary = get_executive_summary(df, insights, use_llm=use_ai)
                    data_story = generate_story(df, insights, use_llm=use_ai)
                    
                    # Store in session state
                    st.session_state.insights = insights
                    st.session_state.executive_summary = executive_summary
                    st.session_state.data_story = data_story
                    
                    st.markdown("""
                    <div class="success-message">
                        <h4 style="margin-top:0">Data story successfully generated! 🎉</h4>
                        <p>Your insights are ready below.</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating insights: {str(e)}")
                    st.session_state.insights = None
                    st.session_state.executive_summary = None
                    st.session_state.data_story = None
        
        # If insights are available, display them with enhanced styling
        if st.session_state.insights is not None and st.session_state.data_story is not None:
            # Enhanced section divider
            st.markdown("""
            <div style="margin: 30px 0; height: 3px; background-image: linear-gradient(to right, rgba(255, 64, 129, 0), rgba(255, 64, 129, 0.5), rgba(255, 64, 129, 0));"></div>
            """, unsafe_allow_html=True)
            
            # Display executive summary with improved styling
            if st.session_state.executive_summary:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h3>Executive Summary</h3>", unsafe_allow_html=True)
                
                summary_html = st.session_state.executive_summary.replace("\n", "<br>")
                st.markdown(f"""
                <div class="executive-summary">
                    <p>{summary_html}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display the data story with improved styling
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Data Story</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div class="story-container">
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.data_story)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show all raw insights with enhanced UI
            with st.expander("View All Insights"):
                for idx, insight in enumerate(st.session_state.insights):
                    st.markdown(f"""
                    <div style="margin: 15px 0; padding: 15px; background-color: #3d3d50; border-radius: 8px; border-left: 4px solid #9D4EDD;">
                        <h4 style="margin: 0 0 10px 0; color: #BB86FC;">Insight {idx+1}: {insight['title']}</h4>
                        <p style="margin: 0 0 10px 0; color: #E0E0E0;">{insight['description']}</p>
                        {f'<p style="margin: 0; font-weight: 600; color: #FFB2E6;">Recommendation: {insight["recommendation"]}</p>' if "recommendation" in insight else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add download options with enhanced styling
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Export Options</h3>", unsafe_allow_html=True)
            
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
            
            # Add timestamp to filename
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            # Download options in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Download button for Markdown
                st.download_button(
                    label="Download as Markdown",
                    data=markdown_content,
                    file_name=f"data_story_{timestamp}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                # Generate simple HTML version
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Data Story: {st.session_state.file_name}</title>
                    <style>
                        body {{ font-family: 'Inter', sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }}
                        h1, h2, h3 {{ color: #8A2BE2; }}
                        .summary {{ background-color: #f5f0ff; padding: 15px; border-radius: 8px; border-left: 4px solid #9D4EDD; }}
                        .insight {{ margin: 15px 0; padding: 15px; background-color: #f9f9fb; border-radius: 8px; border-left: 4px solid #8A2BE2; }}
                        .recommendation {{ font-weight: bold; color: #8A2BE2; }}
                        .footer {{ margin-top: 30px; text-align: center; font-size: 0.8em; color: #666; }}
                        code {{ background-color: #f5f0ff; padding: 2px 4px; border-radius: 3px; }}
                    </style>
                </head>
                <body>
                    <h1>Data Story: {st.session_state.file_name}</h1>
                    <h2>Executive Summary</h2>
                    <div class="summary">{st.session_state.executive_summary.replace('\n', '<br>')}</div>
                    <h2>Detailed Data Story</h2>
                    <div>{st.session_state.data_story.replace('\n', '<br>')}</div>
                    <h2>All Insights</h2>
                """
                
                for idx, insight in enumerate(st.session_state.insights):
                    html_content += f"""
                    <div class="insight">
                        <h3>Insight {idx+1}: {insight['title']}</h3>
                        <p>{insight['description']}</p>
                        {f'<p class="recommendation">Recommendation: {insight["recommendation"]}</p>' if "recommendation" in insight else ''}
                    </div>
                    """
                
                html_content += f"""
                    <div class="footer">
                        <p>Generated with EDAwala on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                </body>
                </html>
                """
                
                # Download button for HTML
                st.download_button(
                    label="Download as HTML",
                    data=html_content,
                    file_name=f"data_story_{timestamp}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
else:
    # Welcome screen with enhanced styling and animations
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <div style="font-size: 70px; margin-bottom: 20px;">📊</div>
        <h2 style="color: #9D4EDD; margin-bottom: 30px; font-size: 28px;">Welcome to EDAwala</h2>
        <p style="max-width: 600px; margin: 0 auto; font-size: 18px; color: #B0B0B0;">
            Please upload a dataset or select an example dataset from the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
   
    
    # Add version and copyright info
    st.markdown("""
    <div class="footer">
        <p>EDAwala v1.0.0 | © 2025 EDAwala</p>
        <p style="margin-top: 5px;">Made with ❤️ by Yashu </p>
    </div>
    """, unsafe_allow_html=True)