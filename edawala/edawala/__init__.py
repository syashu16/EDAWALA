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