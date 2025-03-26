"""
Auto-EDA report generator for EDAwala
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import io
import base64
import jinja2
from typing import Dict, List, Any, Union, Optional, Tuple
import datetime
import json
from pathlib import Path
import warnings

# Suppress the matplotlib thread warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                        message="Starting a Matplotlib GUI outside of the main thread will likely fail.")

# Import custom plotting functions
from edawala.utils.plot_utils import (
    create_missing_values_heatmap,
    create_correlation_matrix,
    create_distribution_plot,
    create_categorical_plot,
    create_pairplot
)

# Set plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

class EDAReportGenerator:
    """
    Generates comprehensive EDA reports from pandas DataFrames
    """
    
    def __init__(self, df: pd.DataFrame, df_name: str = "dataset"):
        """
        Initialize the EDA report generator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to analyze
        df_name : str, optional
            Name of the dataset (default: "dataset")
        """
        self.df = df
        self.df_name = df_name
        self.report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get the package directory for templates
        module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.template_dir = os.path.join(module_path, "templates")
        
        # Create template dir if it doesn't exist
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir)
            # Create a basic template if one doesn't exist
            default_template_path = os.path.join(self.template_dir, "eda_report.html.j2")
            if not os.path.exists(default_template_path):
                with open(default_template_path, 'w') as f:
                    f.write(self._get_default_template())
        
    def _get_default_template(self):
        """Return a basic template if one doesn't exist"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 40px;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metadata {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 30px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .metadata-item {
            text-align: center;
            flex: 1 0 150px;
            margin: 10px;
        }
        .metadata-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .metadata-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        .dataframe {
            width: 100%;
            overflow-x: auto;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-title {
            font-size: 18px;
            margin-bottom: 15px;
            color: #2c3e50;
        }
        .chart {
            max-width: 100%;
            height: auto;
            margin: 0 auto;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e5e5e5;
            color: #7f8c8d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>{{ report_title }}</h1>
    
    <div class="section">
        <h2>Dataset Overview</h2>
        
        <div class="metadata">
            <div class="metadata-item">
                <div class="metadata-value">{{ stats.rows }}</div>
                <div class="metadata-label">Rows</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-value">{{ stats.columns }}</div>
                <div class="metadata-label">Columns</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-value">{{ stats.missing_percent }}%</div>
                <div class="metadata-label">Missing Values</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-value">{{ stats.duplicate_rows }}</div>
                <div class="metadata-label">Duplicate Rows</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-value">{{ stats.numeric_columns }}</div>
                <div class="metadata-label">Numeric Columns</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-value">{{ stats.categorical_columns }}</div>
                <div class="metadata-label">Categorical Columns</div>
            </div>
        </div>
        
        <h3>Sample Data</h3>
        {{ sample_data | safe }}
    </div>
    
    <div class="section">
        <h2>Data Visualizations</h2>
        
        {% if 'missing_values' in visualizations %}
        <div class="chart-container">
            <div class="chart-title">{{ visualizations.missing_values.title }}</div>
            <img class="chart" src="data:image/png;base64,{{ visualizations.missing_values.data }}" alt="Missing Values Heatmap">
        </div>
        {% endif %}
        
        {% if 'correlation' in visualizations %}
        <div class="chart-container">
            <div class="chart-title">{{ visualizations.correlation.title }}</div>
            <img class="chart" src="data:image/png;base64,{{ visualizations.correlation.data }}" alt="Correlation Matrix">
        </div>
        {% endif %}
        
        <h3>Distribution Plots</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
            {% for col in numeric_columns[:6] %}
                {% if 'dist_' + col in visualizations %}
                <div style="flex: 0 0 45%; max-width: 500px;">
                    <div class="chart-container">
                        <div class="chart-title">{{ visualizations['dist_' + col].title }}</div>
                        <img class="chart" src="data:image/png;base64,{{ visualizations['dist_' + col].data }}" alt="Distribution of {{ col }}">
                    </div>
                </div>
                {% endif %}
            {% endfor %}
        </div>
        
        <h3>Categorical Plots</h3>
        <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
            {% for col in categorical_columns[:6] %}
                {% if 'count_' + col in visualizations %}
                <div style="flex: 0 0 45%; max-width: 500px;">
                    <div class="chart-container">
                        <div class="chart-title">{{ visualizations['count_' + col].title }}</div>
                        <img class="chart" src="data:image/png;base64,{{ visualizations['count_' + col].data }}" alt="Count of {{ col }}">
                    </div>
                </div>
                {% endif %}
            {% endfor %}
        </div>
        
        {% if 'pairplot' in visualizations %}
        <div class="chart-container">
            <div class="chart-title">{{ visualizations.pairplot.title }}</div>
            <img class="chart" src="data:image/png;base64,{{ visualizations.pairplot.data }}" alt="Pairplot">
        </div>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Descriptive Statistics</h2>
        {{ descriptive_stats | safe }}
    </div>
    
    <div class="section">
        <h2>Column Analysis</h2>
        
        <div style="display: flex; flex-wrap: wrap; gap: 15px;">
            {% for col in column_stats %}
            <div style="flex: 1 0 300px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
                <h3>{{ col.name }}</h3>
                <p><strong>Type:</strong> {{ col.dtype }}</p>
                <p><strong>Missing:</strong> {{ col.missing }} ({{ col.missing_percent }}%)</p>
                <p><strong>Unique Values:</strong> {{ col.unique_values }}</p>
                
                {% if 'min' in col and col.min is not none %}
                <p><strong>Min:</strong> {{ col.min }}</p>
                <p><strong>Max:</strong> {{ col.max }}</p>
                <p><strong>Mean:</strong> {{ col.mean }}</p>
                <p><strong>Median:</strong> {{ col.median }}</p>
                <p><strong>Std Dev:</strong> {{ col.std }}</p>
                {% endif %}
                
                {% if 'skewness' in col %}
                <p><strong>Skewness:</strong> {{ col.skewness }}</p>
                {% endif %}
                
                {% if 'kurtosis' in col %}
                <p><strong>Kurtosis:</strong> {{ col.kurtosis }}</p>
                {% endif %}
                
                {% if 'top_values' in col %}
                <p><strong>Top Values:</strong></p>
                <ul>
                    {% for val in col.top_values %}
                    <li>{{ val.value }}: {{ val.count }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="footer">
        <p>Report generated with EDAwala on {{ report_date }}</p>
    </div>
</body>
</html>"""
    
    def _generate_basic_stats(self) -> Dict[str, Any]:
        """
        Generate basic statistics for the dataset.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary of basic statistics
        """
        # Calculate basic stats
        rows, cols = self.df.shape
        missing_values = self.df.isna().sum().sum()
        missing_percent = round((missing_values / (rows * cols)) * 100, 2) if rows * cols > 0 else 0
        duplicates = self.df.duplicated().sum()
        
        # Count data types
        dtype_counts = self.df.dtypes.value_counts().to_dict()
        numeric_count = dtype_counts.get('int64', 0) + dtype_counts.get('float64', 0) + dtype_counts.get('int32', 0) + dtype_counts.get('float32', 0)
        categorical_count = dtype_counts.get('object', 0) + dtype_counts.get('category', 0)
        datetime_count = dtype_counts.get('datetime64[ns]', 0)
        boolean_count = dtype_counts.get('bool', 0)
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum()
        memory_usage_mb = round(memory_usage / 1024 / 1024, 2)
        
        return {
            'rows': rows,
            'columns': cols,
            'missing_cells': int(missing_values),
            'missing_percent': missing_percent,
            'duplicate_rows': int(duplicates),
            'numeric_columns': numeric_count,
            'categorical_columns': categorical_count,
            'datetime_columns': datetime_count,
            'boolean_columns': boolean_count,
            'memory_usage_mb': memory_usage_mb
        }
    
    def _generate_column_stats(self) -> List[Dict[str, Any]]:
        """
        Generate statistics for each column.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of column statistics
        """
        column_stats = []
        
        for col in self.df.columns:
            # Skip columns that cause errors
            try:
                # Basic info
                dtype = str(self.df[col].dtype)
                missing = self.df[col].isna().sum()
                missing_percent = round((missing / len(self.df)) * 100, 2) if len(self.df) > 0 else 0
                unique_values = self.df[col].nunique()
                
                col_stats = {
                    'name': col,
                    'dtype': dtype,
                    'missing': int(missing),
                    'missing_percent': missing_percent,
                    'unique_values': int(unique_values)
                }
                
                # Numeric column stats
                if np.issubdtype(self.df[col].dtype, np.number):
                    # Check if there's any data to analyze
                    if self.df[col].count() > 0:
                        col_stats.update({
                            'min': float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                            'max': float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                            'mean': float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None,
                            'median': float(self.df[col].median()) if not pd.isna(self.df[col].median()) else None,
                            'std': float(self.df[col].std()) if not pd.isna(self.df[col].std()) else None
                        })
                        
                        # Check for zero variance
                        if col_stats['std'] == 0:
                            col_stats['zero_variance'] = True
                        
                        # Calculate skewness if available
                        try:
                            col_stats['skewness'] = float(self.df[col].skew())
                        except:
                            pass
                            
                        # Calculate kurtosis if available
                        try:
                            col_stats['kurtosis'] = float(self.df[col].kurtosis())
                        except:
                            pass
                
                # Categorical column stats
                elif self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                    if unique_values <= 20:  # Only if not too many unique values
                        value_counts = self.df[col].value_counts().head(5).to_dict()
                        col_stats['top_values'] = [{'value': str(k), 'count': int(v)} for k, v in value_counts.items()]
                
                column_stats.append(col_stats)
            except Exception as e:
                # Skip columns that cause errors
                print(f"Error processing column {col}: {e}")
                continue
            
        return column_stats
    
    def _generate_visualizations(self) -> Tuple[Dict[str, Dict[str, str]], List[str], List[str]]:
        """
        Generate visualizations for the dataset.
        
        Returns:
        --------
        Tuple[Dict[str, Dict[str, str]], List[str], List[str]]
            Tuple containing:
            - Dictionary of visualizations
            - List of numeric column names
            - List of categorical column names
        """
        visualizations = {}
        numeric_columns = []
        categorical_columns = []
        
        # Identify numeric and categorical columns
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                numeric_columns.append(col)
            elif self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                if self.df[col].nunique() <= 20:  # Only include if not too many categories
                    categorical_columns.append(col)
        
        # Generate missing values heatmap with the thread-safe function
        try:
            visualizations['missing_values'] = {
                'title': 'Missing Values Heatmap',
                'data': create_missing_values_heatmap(self.df)
            }
        except Exception as e:
            print(f"Error creating missing values heatmap: {e}")
        
        # Generate correlation matrix if there are numeric columns
        if len(numeric_columns) > 1:
            try:
                visualizations['correlation'] = {
                    'title': 'Correlation Matrix',
                    'data': create_correlation_matrix(self.df, numeric_columns)
                }
            except Exception as e:
                print(f"Error creating correlation matrix: {e}")
        
        # Generate distribution plots for numeric columns
        for col in numeric_columns[:10]:  # Limit to first 10 columns
            try:
                visualizations[f'dist_{col}'] = {
                    'title': f'Distribution of {col}',
                    'data': create_distribution_plot(self.df, col)
                }
            except Exception as e:
                print(f"Error creating distribution plot for {col}: {e}")
        
        # Generate count plots for categorical columns
        for col in categorical_columns[:10]:  # Limit to first 10 columns
            try:
                visualizations[f'count_{col}'] = {
                    'title': f'Count of {col}',
                    'data': create_categorical_plot(self.df, col)
                }
            except Exception as e:
                print(f"Error creating categorical plot for {col}: {e}")
        
        # Generate pairplots for numeric columns if there are at least 2
        if len(numeric_columns) >= 2 and len(numeric_columns) <= 5:
            try:
                visualizations['pairplot'] = {
                    'title': 'Pairplot of Numeric Variables',
                    'data': create_pairplot(self.df, numeric_columns)
                }
            except Exception as e:
                print(f"Error creating pairplot: {e}")
        
        return visualizations, numeric_columns, categorical_columns
    
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
        try:
            template_loader = jinja2.FileSystemLoader(searchpath=self.template_dir)
            template_env = jinja2.Environment(loader=template_loader)
            
            # Always use HTML template for initial rendering
            template_file = "eda_report.html.j2"
            template = template_env.get_template(template_file)
            html_content = template.render(**report_data)
        except Exception as e:
            print(f"Error rendering template: {e}")
            # Create a very simple HTML if template fails
            html_content = f"""
            <html>
            <head><title>EDA Report: {self.df_name}</title></head>
            <body>
                <h1>EDA Report: {self.df_name}</h1>
                <p>Generated on: {self.report_date}</p>
                <h2>Sample Data</h2>
                {self.df.head().to_html()}
                <h2>Descriptive Statistics</h2>
                {self.df.describe().to_html()}
            </body>
            </html>
            """
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"eda_report_{timestamp}.{format}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Generate report in the requested format
        if format.lower() == 'pdf':
            # Always create an HTML version first
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            pdf_generated = False
            
            # Try multiple PDF generation methods
            try:
                # 1. Try pdfkit first (wkhtmltopdf-based) as it's more Windows-friendly
                try:
                    import pdfkit
                    pdfkit_config = None
                    
                    # Check if wkhtmltopdf is in PATH or needs to be specified
                    try:
                        pdfkit.from_file(html_path, output_path)
                        pdf_generated = True
                        print(f"PDF report generated at: {output_path}")
                    except OSError:
                        # wkhtmltopdf might not be in PATH, inform the user
                        print("Warning: wkhtmltopdf not found in PATH")
                        print("Download wkhtmltopdf from: https://wkhtmltopdf.org/downloads.html")
                        print("And add it to your PATH or specify its location")
                except ImportError:
                    print("pdfkit not installed. To use it: pip install pdfkit")
                    
                # 2. Try WeasyPrint as fallback
                if not pdf_generated:
                    try:
                        import weasyprint
                        weasyprint.HTML(html_path).write_pdf(output_path)
                        pdf_generated = True
                        print(f"PDF report generated at: {output_path}")
                    except ImportError:
                        print("WeasyPrint not installed. To use it: pip install weasyprint")
                    except Exception as e:
                        print(f"WeasyPrint error: {e}")
                        print("WeasyPrint requires GTK libraries on Windows.")
                        print("See: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows")
                
                # 3. Last resort - try using reportlab directly
                if not pdf_generated:
                    try:
                        from reportlab.pdfgen import canvas
                        from reportlab.lib.pagesizes import letter
                        
                        c = canvas.Canvas(output_path, pagesize=letter)
                        c.drawString(100, 750, f"EDA Report: {self.df_name}")
                        c.drawString(100, 735, f"Generated on: {self.report_date}")
                        c.drawString(100, 720, "HTML version available with all visualizations.")
                        c.drawString(100, 705, f"See: {html_path}")
                        c.save()
                        
                        pdf_generated = True
                        print(f"Simple PDF report generated at: {output_path}")
                        print(f"For full visualizations, see the HTML report: {html_path}")
                    except ImportError:
                        print("ReportLab not installed. To use it: pip install reportlab")
                    
            except Exception as e:
                print(f"All PDF generation methods failed: {e}")
            
            # If all methods failed, inform the user about the HTML alternative
            if not pdf_generated:
                print("PDF generation failed. Using HTML format instead.")
                print(f"HTML report generated at: {html_path}")
                output_path = html_path
        
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

def generate_eda_report(
    df: pd.DataFrame,
    format: str = 'html',
    output_path: Optional[str] = None,
    include_sections: Optional[List[str]] = None
) -> str:
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
    generator = EDAReportGenerator(df)
    return generator.generate_report(format, output_path, include_sections)