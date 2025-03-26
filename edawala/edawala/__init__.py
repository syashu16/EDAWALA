"""
EDAwala: Advanced Exploratory Data Analysis toolkit with AI-powered features
"""

__version__ = "0.1.0"

# Import main functionality for easy access
from edawala.auto_eda.report_generator import generate_eda_report
from edawala.storytelling.insights import generate_insights, generate_story, get_executive_summary

def run_app():
    """Launch the EDAwala Streamlit application"""
    import streamlit.web.bootstrap as bootstrap
    import sys
    import os
    
    # Get the path to app.py
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Run the Streamlit app with the correct parameters
    sys.argv = ["streamlit", "run", app_path]
    bootstrap.run(
        app_path,
        False,  # is_hello
        [],     # args
        {}      # flag_options
    )