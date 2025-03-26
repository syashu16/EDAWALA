"""
Auto-EDA report generator for EDAwala
"""
import pandas as pd
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
        self.template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
        
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
        missing_percent = round((missing_values / (rows * cols)) * 100, 2)
        duplicates = self.df.duplicated().sum()
        
        # Count data types
        dtype_counts = self.df.dtypes.value_counts().to_dict()
        numeric_count = dtype_counts.get('int64', 0) + dtype_counts.get('float64', 0)
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
            # Basic info
            dtype = str(self.df[col].dtype)
            missing = self.df[col].isna().sum()
            missing_percent = round((missing / len(self.df)) * 100, 2)
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
        
        # Generate missing values heatmap
        visualizations['missing_values'] = {
            'title': 'Missing Values Heatmap',
            'data': self._plot_missing_values()
        }
        
        # Generate correlation matrix if there are numeric columns
        if len(numeric_columns) > 1:
            visualizations['correlation'] = {
                'title': 'Correlation Matrix',
                'data': self._plot_correlation_matrix(numeric_columns)
            }
        
        # Generate distribution plots for numeric columns
        for col in numeric_columns[:10]:  # Limit to first 10 columns
            visualizations[f'dist_{col}'] = {
                'title': f'Distribution of {col}',
                'data': self._plot_distribution(col)
            }
        
        # Generate count plots for categorical columns
        for col in categorical_columns[:10]:  # Limit to first 10 columns
            visualizations[f'count_{col}'] = {
                'title': f'Count of {col}',
                'data': self._plot_categorical(col)
            }
        
        # Generate pairplots for numeric columns if there are at least 2
        if len(numeric_columns) >= 2 and len(numeric_columns) <= 5:
            visualizations['pairplot'] = {
                'title': 'Pairplot of Numeric Variables',
                'data': self._plot_pairplot(numeric_columns)
            }
        
        return visualizations, numeric_columns, categorical_columns
    
    def _plot_to_base64(self, plt_func) -> str:
        """
        Convert a matplotlib plot to a base64 encoded string.
        
        Parameters:
        -----------
        plt_func : Callable
            Function that creates a matplotlib plot
            
        Returns:
        --------
        str
            Base64 encoded string of the plot
        """
        # Create a BytesIO object
        buf = io.BytesIO()
        
        # Call the plot function
        plt_func()
        
        # Save the plot to the BytesIO object
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        
        # Get the image as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
    
    def _plot_missing_values(self) -> str:
        """
        Plot missing values heatmap.
        
        Returns:
        --------
        str
            Base64 encoded string of the plot
        """
        def plot_func():
            plt.figure(figsize=(10, 6))
            
            # Check if the dataframe is too large, if so, take a sample
            if self.df.shape[0] > 100:
                df_sample = self.df.sample(n=100, random_state=42)
            else:
                df_sample = self.df
                
            sns.heatmap(df_sample.isna(), cbar=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.xlabel('Features')
            plt.ylabel('Observations')
            
        return self._plot_to_base64(plot_func)
    
    def _plot_correlation_matrix(self, numeric_columns: List[str]) -> str:
        """
        Plot correlation matrix.
        
        Parameters:
        -----------
        numeric_columns : List[str]
            List of numeric column names
            
        Returns:
        --------
        str
            Base64 encoded string of the plot
        """
        def plot_func():
            plt.figure(figsize=(10, 8))
            
            # Calculate correlation matrix
            corr = self.df[numeric_columns].corr()
            
            # Generate mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Plot heatmap
            sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
        return self._plot_to_base64(plot_func)
    
    def _plot_distribution(self, column: str) -> str:
        """
        Plot distribution of a numeric column.
        
        Parameters:
        -----------
        column : str
            Name of the column to plot
            
        Returns:
        --------
        str
            Base64 encoded string of the plot
        """
        def plot_func():
            plt.figure(figsize=(10, 6))
            
            # Plot histogram with KDE
            sns.histplot(self.df[column].dropna(), kde=True)
            
            # Add mean and median lines
            plt.axvline(self.df[column].mean(), color='red', linestyle='--', 
                       label=f'Mean: {self.df[column].mean():.2f}')
            plt.axvline(self.df[column].median(), color='green', linestyle='-', 
                       label=f'Median: {self.df[column].median():.2f}')
            
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend()
            
        return self._plot_to_base64(plot_func)
    
    def _plot_categorical(self, column: str) -> str:
        """
        Plot counts for a categorical column.
        
        Parameters:
        -----------
        column : str
            Name of the column to plot
            
        Returns:
        --------
        str
            Base64 encoded string of the plot
        """
        def plot_func():
            plt.figure(figsize=(10, 6))
            
            # Get counts
            value_counts = self.df[column].value_counts()
            
            # If too many categories, take top ones
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)
                plt.title(f'Top 10 Values for {column}')
            else:
                plt.title(f'Values for {column}')
                
            # Plot barplot
            sns.barplot(x=value_counts.index, y=value_counts.values)
            
            plt.xticks(rotation=45, ha='right')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.tight_layout()
            
        return self._plot_to_base64(plot_func)
    
    def _plot_pairplot(self, numeric_columns: List[str]) -> str:
        """
        Create a pairplot of numeric variables.
        
        Parameters:
        -----------
        numeric_columns : List[str]
            List of numeric column names
            
        Returns:
        --------
        str
            Base64 encoded string of the plot
        """
        def plot_func():
            # Take a sample if dataframe is large
            if self.df.shape[0] > 1000:
                df_sample = self.df[numeric_columns].sample(n=1000, random_state=42)
            else:
                df_sample = self.df[numeric_columns]
                
            # Create pairplot
            sns.pairplot(df_sample)
            plt.suptitle('Pairplot of Numeric Variables', y=1.02)
            
        return self._plot_to_base64(plot_func)
    
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