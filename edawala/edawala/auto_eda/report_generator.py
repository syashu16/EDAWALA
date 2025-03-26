"""
Auto-EDA report generator for EDAwala
"""
import pandas as pd
import numpy as np
import os
import datetime
import logging

# Import the simple report generator
from .simple_report import generate_simple_report

# Add the following import at the top of your report_generator.py file:

from edawala.utils.chart_themes import (
    apply_edawala_theme, 
    EDAWALA_PALETTE, 
    CORRELATION_CMAP,
    CATEGORICAL_PALETTE
)

# Then in the chart generation parts of the file, add this line before creating any chart:
apply_edawala_theme()
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_eda_report(
    df,
    format='html',
    output_path=None,
    include_sections=None
):
    """
    Generate a comprehensive EDA report
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    format : str, optional
        Output format: 'html', 'pdf', or 'notebook' (default: 'html')
    output_path : str, optional
        Path where the report will be saved (default: auto-generated)
    include_sections : List[str], optional
        Specific sections to include in the report
        
    Returns:
    --------
    str
        Path to the generated report
    """
    try:
        # Determine output path if not provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"eda_report_{timestamp}.{format}"
        
        # Use the simplified report generator for HTML reports
        if format.lower() == 'html':
            logger.info("Generating HTML report...")
            report_path = generate_simple_report(df, output_path)
            logger.info(f"HTML report generated at: {report_path}")
            return report_path
        
        # For PDF, generate HTML first then try to convert
        elif format.lower() == 'pdf':
            logger.info("Generating HTML report first...")
            html_path = output_path.replace('.pdf', '.html')
            report_path = generate_simple_report(df, html_path)
            
            logger.info("Trying to convert HTML to PDF...")
            try:
                # Try pdfkit (wkhtmltopdf)
                try:
                    import pdfkit
                    pdfkit.from_file(html_path, output_path)
                    logger.info(f"PDF report generated at: {output_path}")
                    return output_path
                except Exception as e:
                    logger.warning(f"pdfkit error: {e}")
                    logger.info("Trying alternative PDF conversion method...")
                
                # Try WeasyPrint
                try:
                    import weasyprint
                    weasyprint.HTML(html_path).write_pdf(output_path)
                    logger.info(f"PDF report generated with WeasyPrint at: {output_path}")
                    return output_path
                except Exception as e:
                    logger.warning(f"WeasyPrint error: {e}")
                    logger.info("Trying final PDF conversion method...")
                
                # Try ReportLab
                try:
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.pagesizes import letter
                    
                    c = canvas.Canvas(output_path, pagesize=letter)
                    c.drawString(100, 750, f"EDA Report")
                    c.drawString(100, 735, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    c.drawString(100, 720, "HTML version available with all visualizations.")
                    c.drawString(100, 705, f"See: {html_path}")
                    c.save()
                    
                    logger.info(f"Simple PDF report generated at: {output_path}")
                    logger.info(f"For full visualizations, see the HTML report: {html_path}")
                    return output_path
                except Exception as e:
                    logger.warning(f"ReportLab error: {e}")
            
            except Exception as e:
                logger.error(f"All PDF generation methods failed: {e}")
                
            logger.warning("PDF generation failed. Using HTML format instead.")
            logger.info(f"HTML report generated at: {html_path}")
            return html_path
        
        # For notebook, create a simple Jupyter notebook
        elif format.lower() == 'notebook':
            try:
                import nbformat
                from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
                
                # Create notebook cells
                cells = []
                
                # Title
                cells.append(new_markdown_cell(f"# EDA Report"))
                
                # Metadata
                metadata_md = f"""
## Dataset Information
- **Report Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Rows:** {df.shape[0]}
- **Columns:** {df.shape[1]}
                """
                cells.append(new_markdown_cell(metadata_md))
                
                # Setup code
                setup_code = """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the dataset
# Replace with your actual file path
df = pd.read_csv('your_dataset.csv')
                """
                cells.append(new_code_cell(setup_code))
                
                # Data overview
                cells.append(new_markdown_cell("## Data Overview"))
                cells.append(new_code_cell("df.head()"))
                cells.append(new_code_cell("df.info()"))
                cells.append(new_code_cell("df.describe()"))
                
                # Create notebook
                nb = new_notebook(cells=cells)
                
                # Write notebook to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                
                logger.info(f"Notebook report generated at: {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Notebook generation failed: {e}")
                logger.info("Falling back to HTML format.")
                
                html_path = output_path.replace('.ipynb', '.html')
                return generate_simple_report(df, html_path)
        
        # Default to HTML for any other format
        else:
            logger.warning(f"Unsupported format: {format}. Using HTML instead.")
            html_path = output_path.replace(f".{format}", ".html")
            return generate_simple_report(df, html_path)
        
    except Exception as e:
        logger.error(f"Error in report generation: {e}")
        
        # Last resort - create an extremely basic HTML report
        try:
            if output_path is None or not output_path.endswith('.html'):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"basic_report_{timestamp}.html"
                
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <html>
                <head><title>Basic EDA Report</title></head>
                <body>
                    <h1>Basic EDA Report</h1>
                    <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <h2>Data Sample</h2>
                    {df.head().to_html()}
                    <h2>Descriptive Statistics</h2>
                    {df.describe().to_html()}
                </body>
                </html>
                """)
                
            logger.info(f"Basic HTML report generated at: {output_path}")
            return output_path
        except Exception as final_e:
            logger.error(f"Failed to generate even basic report: {final_e}")
            return "Report generation failed completely."