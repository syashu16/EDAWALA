from typing import Dict, List, Any, Union, Optional, Tuple
def generate_report(
    self, 
    format: str = 'html', 
    output_path: Optional[str] = None,
    include_sections: Optional[List[str]] = None
) -> str:
    """
    Generate a comprehensive EDA report
    
    Parameters:
    -----------
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
    # Get stats and visualizations
    basic_stats = self._generate_basic_stats()
    column_stats = self._generate_column_stats()
    visualizations, numeric_columns, categorical_columns = self._generate_visualizations()
    
    # Prepare data for template
    report_data = {
        'report_title': f"Exploratory Data Analysis Report: {self.df_name}",
        'dataset_name': self.df_name,
        'report_date': self.report_date,
        'stats': basic_stats,
        'column_stats': column_stats,
        'visualizations': visualizations,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'descriptive_stats': self.df.describe().to_html(classes='dataframe'),
        'sample_data': self.df.head().to_html(classes='dataframe'),
    }
    
    # Load and render template
    template_loader = jinja2.FileSystemLoader(searchpath=self.template_dir)
    template_env = jinja2.Environment(loader=template_loader)
    
    # Always use HTML template for initial rendering
    template_file = "eda_report.html.j2"
    template = template_env.get_template(template_file)
    html_content = template.render(**report_data)
    
    # Determine output path
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"eda_report_{timestamp}.{format}"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Generate report in the requested format
    if format.lower() == 'pdf':
        try:
            import weasyprint
            
            # Create a temporary HTML file
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Convert HTML to PDF
            weasyprint.HTML(html_path).write_pdf(output_path)
            
            # Optionally remove the temporary HTML file
            try:
                os.remove(html_path)
            except:
                pass
            
            print(f"PDF report generated at: {output_path}")
            
        except ImportError:
            print("WeasyPrint not installed. Please install it with: pip install weasyprint")
            print("Falling back to HTML format.")
            
            # Fallback to HTML if WeasyPrint is not available
            if not output_path.endswith('.html'):
                output_path = output_path.replace('.pdf', '.html')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
    
    elif format.lower() == 'notebook':
        try:
            # Generate a Jupyter notebook
            import nbformat
            from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
            
            # Create notebook cells
            cells = []
            
            # Title
            cells.append(new_markdown_cell(f"# {report_data['report_title']}"))
            
            # Metadata
            metadata_md = f"""
## Dataset Information
- **Dataset:** {report_data['dataset_name']}
- **Report Generated:** {report_data['report_date']}
- **Rows:** {report_data['stats']['rows']}
- **Columns:** {report_data['stats']['columns']}
- **Missing Values:** {report_data['stats']['missing_cells']} ({report_data['stats']['missing_percent']}%)
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
            
            # Missing values
            cells.append(new_markdown_cell("## Missing Values Analysis"))
            cells.append(new_code_cell("""
# Missing values heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isna(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()
            """))
            
            # Add cells for numeric columns
            if numeric_columns:
                cells.append(new_markdown_cell("## Numeric Distributions"))
                for col in numeric_columns:
                    cells.append(new_code_cell(f"""
# Distribution of {col}
plt.figure(figsize=(10, 6))
sns.histplot(df['{col}'].dropna(), kde=True)
plt.axvline(df['{col}'].mean(), color='red', linestyle='--', label=f'Mean: {{df['{col}'].mean():.2f}}')
plt.axvline(df['{col}'].median(), color='green', linestyle=':', label=f'Median: {{df['{col}'].median():.2f}}')
plt.legend()
plt.title('Distribution of {col}')
plt.show()
                    """))
            
            # Correlation matrix
            if len(numeric_columns) > 1:
                cells.append(new_markdown_cell("## Correlation Analysis"))
                cells.append(new_code_cell(f"""
# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df[{numeric_columns}].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
                """))
            
            # Create notebook
            nb = new_notebook(cells=cells)
            
            # Write notebook to file
            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
                
        except ImportError:
            print("nbformat not installed. Please install it with: pip install nbformat")
            print("Falling back to HTML format.")
            
            # Fallback to HTML
            if not output_path.endswith('.html'):
                output_path = output_path.replace('.ipynb', '.html')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
    
    else:  # Default to HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    return output_path