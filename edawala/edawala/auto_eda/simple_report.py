"""
Simplified EDA report generator with thread-safe plotting
"""
import pandas as pd
import numpy as np
import os
import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def generate_simple_report(df, output_path=None):
    """
    Generate a simple EDA report with basic charts.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataset to analyze
    output_path : str, optional
        Path to save the report (default: auto-generated)
    
    Returns:
    --------
    str
        Path to the generated HTML report
    """
    # Create output path if not provided
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"simple_eda_report_{timestamp}.html"
    
    # Get basic stats
    rows, cols = df.shape
    missing_values = df.isna().sum().sum()
    missing_percent = round((missing_values / (rows * cols)) * 100, 2) if rows * cols > 0 else 0
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Start building HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EDAwala Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; line-height: 1.6; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .stats {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }}
            .stat-card {{ 
                flex: 1 0 200px; padding: 15px; background-color: #f8f9fa; 
                border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; 
            }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            .chart {{ margin: 30px 0; text-align: center; background-color: white; padding: 15px;
                    border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .chart img {{ max-width: 100%; height: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ margin: 30px 0; padding: 20px; background-color: white; 
                     border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>EDAwala: Exploratory Data Analysis Report</h1>
            <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{rows}</div>
                        <div class="stat-label">Rows</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{cols}</div>
                        <div class="stat-label">Columns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{missing_percent}%</div>
                        <div class="stat-label">Missing Values</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(numeric_cols)}</div>
                        <div class="stat-label">Numeric Columns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(categorical_cols)}</div>
                        <div class="stat-label">Categorical Columns</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Data Sample</h2>
                {df.head().to_html(classes="dataframe")}
            </div>
            
            <div class="section">
                <h2>Descriptive Statistics</h2>
                {df.describe().to_html(classes="dataframe")}
            </div>
    """
    
    # Add visualizations
    html += '<div class="section"><h2>Visualizations</h2>'
    
    # Function to convert plot to base64
    def get_plot_as_base64(plot_func):
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Execute the plotting function
        plot_func()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close('all')  # Close all figures
        
        # Get as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        return img_str
    
    # 1. Missing values heatmap
    if missing_values > 0:
        try:
            def plot_missing():
                # Take a sample if dataframe is too large
                if df.shape[0] > 100:
                    sample_df = df.sample(n=100, random_state=42)
                else:
                    sample_df = df
                    
                sns.heatmap(sample_df.isna(), cbar=False, cmap='viridis')
                plt.title('Missing Values Heatmap')
                plt.xlabel('Features')
                plt.ylabel('Observations')
                plt.tight_layout()
            
            img_str = get_plot_as_base64(plot_missing)
            html += f"""
            <div class="chart">
                <h3>Missing Values Heatmap</h3>
                <img src="data:image/png;base64,{img_str}" alt="Missing Values Heatmap">
            </div>
            """
        except Exception as e:
            html += f"<p>Error generating missing values heatmap: {str(e)}</p>"
    
    # 2. Correlation matrix for numeric columns
    if len(numeric_cols) > 1:
        try:
            def plot_correlation():
                # Calculate correlation matrix
                corr = df[numeric_cols].corr()
                
                # Generate mask for the upper triangle
                mask = np.triu(np.ones_like(corr, dtype=bool))
                
                # Plot heatmap
                sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
                plt.title('Correlation Matrix')
                plt.tight_layout()
            
            img_str = get_plot_as_base64(plot_correlation)
            html += f"""
            <div class="chart">
                <h3>Correlation Matrix</h3>
                <img src="data:image/png;base64,{img_str}" alt="Correlation Matrix">
            </div>
            """
        except Exception as e:
            html += f"<p>Error generating correlation matrix: {str(e)}</p>"
    
    # 3. Distribution plots for numeric columns (limited to first 5)
    html += '<h3>Distribution Plots</h3><div style="display: flex; flex-wrap: wrap; justify-content: space-between;">'
    for col in numeric_cols[:5]:
        try:
            def plot_distribution(column=col):
                # Plot histogram with KDE
                sns.histplot(df[column].dropna(), kde=True)
                
                # Add mean and median lines if there's data
                if df[column].count() > 0:
                    plt.axvline(df[column].mean(), color='red', linestyle='--', 
                               label=f'Mean: {df[column].mean():.2f}')
                    plt.axvline(df[column].median(), color='green', linestyle='-', 
                               label=f'Median: {df[column].median():.2f}')
                    plt.legend()
                    
                plt.title(f'Distribution of {column}')
                plt.tight_layout()
            
            img_str = get_plot_as_base64(lambda: plot_distribution(col))
            html += f"""
            <div class="chart" style="flex: 0 0 48%;">
                <h3>Distribution of {col}</h3>
                <img src="data:image/png;base64,{img_str}" alt="Distribution of {col}">
            </div>
            """
        except Exception as e:
            html += f"<p>Error generating distribution plot for {col}: {str(e)}</p>"
    html += '</div>'
    
    # 4. Count plots for categorical columns (limited to first 5)
    if categorical_cols:
        html += '<h3>Categorical Plots</h3><div style="display: flex; flex-wrap: wrap; justify-content: space-between;">'
        for col in categorical_cols[:5]:
            try:
                # Skip if too many unique values
                if df[col].nunique() > 20:
                    continue
                    
                def plot_categorical(column=col):
                    # Get counts
                    value_counts = df[column].value_counts()
                    
                    # If too many categories, take top ones
                    if len(value_counts) > 10:
                        value_counts = value_counts.head(10)
                        plt.title(f'Top 10 Values for {column}')
                    else:
                        plt.title(f'Values for {column}')
                        
                    # Plot barplot
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                
                img_str = get_plot_as_base64(lambda: plot_categorical(col))
                html += f"""
                <div class="chart" style="flex: 0 0 48%;">
                    <h3>Count plot for {col}</h3>
                    <img src="data:image/png;base64,{img_str}" alt="Count plot for {col}">
                </div>
                """
            except Exception as e:
                html += f"<p>Error generating count plot for {col}: {str(e)}</p>"
        html += '</div>'
    
    # Close the visualization section
    html += '</div>'
    
    # Column details section
    html += '<div class="section"><h2>Column Details</h2>'
    
    # Add column-by-column analysis
    for col in df.columns:
        html += f'<h3>{col}</h3>'
        html += '<table>'
        
        # Type and basic info
        html += f'<tr><td>Type</td><td>{df[col].dtype}</td></tr>'
        html += f'<tr><td>Missing Values</td><td>{df[col].isna().sum()} ({df[col].isna().mean() * 100:.1f}%)</td></tr>'
        html += f'<tr><td>Unique Values</td><td>{df[col].nunique()}</td></tr>'
        
        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(df[col]):
            html += f'<tr><td>Min</td><td>{df[col].min()}</td></tr>'
            html += f'<tr><td>Max</td><td>{df[col].max()}</td></tr>'
            html += f'<tr><td>Mean</td><td>{df[col].mean()}</td></tr>'
            html += f'<tr><td>Median</td><td>{df[col].median()}</td></tr>'
            html += f'<tr><td>Standard Deviation</td><td>{df[col].std()}</td></tr>'
            
            # Add skewness and kurtosis if possible
            try:
                html += f'<tr><td>Skewness</td><td>{df[col].skew():.2f}</td></tr>'
                html += f'<tr><td>Kurtosis</td><td>{df[col].kurtosis():.2f}</td></tr>'
            except:
                pass
        
        # Add top values for categorical columns
        elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
            if df[col].nunique() <= 20:
                html += '<tr><td>Top Values</td><td>'
                for val, count in df[col].value_counts().head(5).items():
                    percent = count / len(df) * 100
                    html += f'{val}: {count} ({percent:.1f}%)<br>'
                html += '</td></tr>'
            else:
                html += '<tr><td>Top Values</td><td>(Too many unique values to display)</td></tr>'
        
        html += '</table>'
    
    html += '</div>'
    
    # Add footer
    html += """
            <div class="footer">
                <p>Report generated with EDAwala</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path