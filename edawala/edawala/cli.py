"""
Command Line Interface for EDAwala
"""
import argparse
import sys
import pandas as pd
from edawala import __version__

def main():
    """Main CLI entrypoint"""
    parser = argparse.ArgumentParser(description="EDAwala: Advanced Exploratory Data Analysis tool")
    parser.add_argument('--version', action='version', version=f'EDAwala {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # App command
    app_parser = subparsers.add_parser('app', help='Launch the EDAwala interactive app')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate an EDA report')
    report_parser.add_argument('file', help='Path to the CSV or Excel file')
    report_parser.add_argument('--output', '-o', help='Output file path')
    report_parser.add_argument('--format', '-f', choices=['html', 'pdf', 'notebook'], 
                              default='html', help='Output format (default: html)')
    
    # Insights command
    insights_parser = subparsers.add_parser('insights', help='Generate insights from data')
    insights_parser.add_argument('file', help='Path to the CSV or Excel file')
    insights_parser.add_argument('--use-llm', action='store_true', help='Use LLM for enhanced insights')
    insights_parser.add_argument('--provider', choices=['gemini', 'openai'], 
                                default='gemini', help='LLM provider (default: gemini)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == "app":
        # Launch the Streamlit app
        try:
            import streamlit.web.bootstrap as bootstrap
            import os
            
            # Get path to app.py
            from edawala import run_app
            run_app()
            return 0
        except Exception as e:
            print(f"Error launching Streamlit app: {e}", file=sys.stderr)
            return 1
    
    elif args.command == "report":
        # Generate report
        try:
            from edawala.auto_eda.report_generator import generate_eda_report
            from edawala.core.data_loader import load_data
            
            df = load_data(args.file)
            if df is None:
                print(f"Error: Could not load data from {args.file}", file=sys.stderr)
                return 1
                
            output_path = generate_eda_report(df, format=args.format, output_path=args.output)
            print(f"Report generated successfully: {output_path}")
            return 0
        except Exception as e:
            print(f"Error generating report: {e}", file=sys.stderr)
            return 1
    
    elif args.command == "insights":
        # Generate insights
        try:
            from edawala.storytelling.insights import generate_story
            from edawala.core.data_loader import load_data
            
            df = load_data(args.file)
            if df is None:
                print(f"Error: Could not load data from {args.file}", file=sys.stderr)
                return 1
                
            story = generate_story(df, use_llm=args.use_llm, llm_provider=args.provider)
            print(story)
            return 0
        except Exception as e:
            print(f"Error generating insights: {e}", file=sys.stderr)
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())