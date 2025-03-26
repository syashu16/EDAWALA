"""
Command-line interface for EDAwala
"""
import argparse
import sys
import os
import pandas as pd


def main():
    """Main entry point for the EDAwala CLI"""
    parser = argparse.ArgumentParser(
        description="EDAwala - Advanced Exploratory Data Analysis toolkit"
    )
    
    parser.add_argument(
        "--version", 
        action="store_true", 
        help="Show the EDAwala version"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # App command
    app_parser = subparsers.add_parser("app", help="Run the Streamlit app")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate an EDA report")
    report_parser.add_argument("file", help="Path to the CSV file to analyze")
    report_parser.add_argument(
        "--format", 
        choices=["html", "pdf", "notebook"], 
        default="html",
        help="Output format for the report"
    )
    report_parser.add_argument(
        "--output", 
        help="Output path for the report"
    )
    
    # Insights command
    insights_parser = subparsers.add_parser("insights", help="Generate insights from data")
    insights_parser.add_argument("file", help="Path to the CSV file to analyze")
    insights_parser.add_argument(
        "--use-llm", 
        action="store_true",
        help="Use LLM for enhanced insights"
    )
    insights_parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="LLM provider to use"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.version:
        from edawala import __version__
        print(f"EDAwala version {__version__}")
        return 0
        
    if args.command == "app":
        # Launch the Streamlit app
        try:
            from edawala import run_app
            run_app()
            return 0
        except Exception as e:
            print(f"Error launching Streamlit app: {e}", file=sys.stderr)
            return 1
        
    elif args.command == "report":
        try:
            from edawala.auto_eda.report_generator import generate_eda_report
            from edawala.core.data_loader import DataLoader
            
            print(f"Analyzing file: {args.file}")
            
            # Determine file type and load accordingly
            file_ext = os.path.splitext(args.file)[1].lower()
            if file_ext == '.csv':
                df = DataLoader.load_csv(args.file)
            elif file_ext in ['.xls', '.xlsx']:
                df = DataLoader.load_excel(args.file)
            else:
                print(f"Unsupported file format: {file_ext}", file=sys.stderr)
                return 1
            
            print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Generate the report
            output_path = args.output
            if not output_path:
                output_dir = "reports"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"eda_report_{os.path.basename(args.file).split('.')[0]}.{args.format}")
            
            print(f"Generating {args.format.upper()} report...")
            report_path = generate_eda_report(
                df, 
                format=args.format,
                output_path=output_path
            )
            print(f"Report generated successfully: {report_path}")
            return 0
        except Exception as e:
            print(f"Error generating report: {e}", file=sys.stderr)
            return 1
            
    elif args.command == "insights":
        try:
            from edawala.storytelling.insights import generate_insights
            from edawala.core.data_loader import DataLoader
            
            print(f"Analyzing file: {args.file}")
            
            # Load the file
            file_ext = os.path.splitext(args.file)[1].lower()
            if file_ext == '.csv':
                df = DataLoader.load_csv(args.file)
            elif file_ext in ['.xls', '.xlsx']:
                df = DataLoader.load_excel(args.file)
            else:
                print(f"Unsupported file format: {file_ext}", file=sys.stderr)
                return 1
            
            print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Generate insights
            print("Generating insights...")
            insights = generate_insights(
                df, 
                use_llm=args.use_llm,
                llm_provider=args.provider
            )
            
            # Print insights
            print("\n===== INSIGHTS =====\n")
            for i, insight in enumerate(insights, 1):
                print(f"{i}. {insight['description']}")
                print()
                
            return 0
        except Exception as e:
            print(f"Error generating insights: {e}", file=sys.stderr)
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())