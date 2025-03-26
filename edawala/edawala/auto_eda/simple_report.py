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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Set modern styling
plt.style.use('ggplot')
sns.set(style="whitegrid", palette="muted", color_codes=True)

# Custom color palettes
CORRELATION_CMAP = LinearSegmentedColormap.from_list("custom_coolwarm", 
                                                    ["#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027"])
CATEGORICAL_PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#34495e", "#e67e22", "#c0392b", "#16a085"]
DISTRIBUTION_COLOR = "#3498db"
CUSTOM_THEME = {
    'axes.facecolor': '#f8f9fa',
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.color': '#dddddd',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'axes.titleweight': 'bold',
    'axes.titlecolor': '#2c3e50',
    'axes.titlesize': 14,
    'axes.labelweight': 'bold'
}

# Apply custom theme
plt.rcParams.update(CUSTOM_THEME)

def generate_simple_report(df, output_path=None):
    """
    Generate a simple EDA report with enhanced, visually appealing charts.
    
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
        output_path = f"eda_report_{timestamp}.html"
    
    # Get basic stats
    rows, cols = df.shape
    missing_values = df.isna().sum().sum()
    missing_percent = round((missing_values / (rows * cols)) * 100, 2) if rows * cols > 0 else 0
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Start building HTML with enhanced styling
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EDAwala Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            :root {{
                --primary-color: #3498db;
                --secondary-color: #2ecc71;
                --accent-color: #e74c3c;
                --text-color: #333333;
                --light-bg: #f8f9fa;
                --border-color: #e0e0e0;
                --shadow: 0 4px 6px rgba(0,0,0,0.05);
                --hover-shadow: 0 6px 12px rgba(0,0,0,0.1);
            }}
            
            body {{
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 20px;
                color: var(--text-color);
                line-height: 1.6;
                background-color: #f5f7fa;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: var(--shadow);
                padding: 30px;
            }}
            
            h1, h2, h3 {{
                color: #2c3e50;
                font-weight: 600;
                margin-top: 1.2em;
                margin-bottom: 0.8em;
            }}
            
            h1 {{
                font-size: 2.2rem;
                text-align: center;
                margin-bottom: 1.5rem;
                color: var(--primary-color);
                border-bottom: 2px solid var(--border-color);
                padding-bottom: 15px;
            }}
            
            h2 {{
                font-size: 1.8rem;
                border-left: 4px solid var(--primary-color);
                padding-left: 15px;
                margin-top: 2rem;
            }}
            
            h3 {{
                font-size: 1.4rem;
                color: #34495e;
            }}
            
            .stats {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                flex: 1 0 200px;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
                box-shadow: var(--shadow);
                text-align: center;
                transition: transform 0.3s, box-shadow 0.3s;
                border-top: 4px solid var(--primary-color);
            }}
            
            .stat-card:nth-child(2) {{
                border-top-color: var(--secondary-color);
            }}
            
            .stat-card:nth-child(3) {{
                border-top-color: var(--accent-color);
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: var(--hover-shadow);
            }}
            
            .stat-value {{
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }}
            
            .stat-label {{
                font-size: 14px;
                color: #7f8c8d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .chart {{
                margin: 30px 0;
                text-align: center;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: var(--shadow);
                transition: transform 0.3s;
            }}
            
            .chart:hover {{
                transform: translateY(-5px);
                box-shadow: var(--hover-shadow);
            }}
            
            .chart img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                box-shadow: var(--shadow);
                border-radius: 8px;
                overflow: hidden;
            }}
            
            th, td {{
                border: 1px solid var(--border-color);
                padding: 12px;
                text-align: left;
            }}
            
            th {{
                background-color: #f1f5f9;
                font-weight: 600;
                color: #2c3e50;
            }}
            
            tr:nth-child(even) {{
                background-color: #f9fafb;
            }}
            
            tr:hover {{
                background-color: #f0f7ff;
            }}
            
            .section {{
                margin: 40px 0;
                padding: 25px;
                background-color: white;
                border-radius: 10px;
                box-shadow: var(--shadow);
            }}
            
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding: 20px;
                color: #7f8c8d;
                font-size: 14px;
                border-top: 1px solid var(--border-color);
            }}
            
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(450px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .insight-card {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: var(--shadow);
                border-left: 4px solid var(--primary-color);
            }}
            
            .insight-card h4 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            
            .column-details {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            
            .column-card {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: var(--shadow);
            }}
            
            .column-card h4 {{
                margin-top: 0;
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 10px;
                color: var(--primary-color);
            }}
            
            .numeric-column {{
                border-top: 3px solid var(--primary-color);
            }}
            
            .categorical-column {{
                border-top: 3px solid var(--secondary-color);
            }}
            
            .data-table {{
                overflow-x: auto;
                margin: 20px 0;
                box-shadow: var(--shadow);
                border-radius: 8px;
            }}
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {{
                .chart-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .stats {{
                    flex-direction: column;
                }}
                
                .column-details {{
                    grid-template-columns: 1fr;
                }}
            }}
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
                        <div class="stat-value">{rows:,}</div>
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
                <div class="data-table">
                    {df.head().to_html(classes="dataframe", index=False)}
                </div>
            </div>
            
            <div class="section">
                <h2>Descriptive Statistics</h2>
                <div class="data-table">
                    {df.describe().to_html(classes="dataframe")}
                </div>
            </div>
    """
    
    # Add visualizations with enhanced styling
    html += '<div class="section"><h2>Visualizations</h2>'
    
    # Function to convert plot to base64
    def get_plot_as_base64(plot_func):
        # Create a new figure
        plt.figure(figsize=(10, 6), dpi=100)
        
        # Execute the plotting function
        plot_func()
        
        # Add watermark
        plt.figtext(0.5, 0.01, "Generated with EDAwala", 
                   ha="center", fontsize=8, color="#999999")
        
        # Improve layout
        plt.tight_layout(pad=3.0)
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close('all')  # Close all figures
        
        # Get as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        return img_str
    
    # 1. Enhanced Missing values heatmap with better styling
    if missing_values > 0:
        try:
            def plot_missing():
                # Take a sample if dataframe is too large
                if df.shape[0] > 100:
                    sample_df = df.sample(n=100, random_state=42)
                else:
                    sample_df = df
                    
                # Enhanced heatmap with better color palette
                ax = sns.heatmap(
                    sample_df.isna(), 
                    cbar=False, 
                    cmap='viridis',
                    yticklabels=False
                )
                plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Features', fontsize=12, fontweight='bold')
                plt.ylabel('Observations', fontsize=12, fontweight='bold')
                
                # Add annotations for columns with most missing values
                missing_cols = df.isna().sum().sort_values(ascending=False)
                missing_cols = missing_cols[missing_cols > 0]
                if len(missing_cols) > 0:
                    for i, (col, count) in enumerate(missing_cols.items()[:3]):
                        percent = 100 * count / len(df)
                        plt.annotate(
                            f"{col}: {percent:.1f}% missing",
                            xy=(0.5, 0.97 - i*0.05),
                            xycoords='figure fraction',
                            ha='center',
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                        )
            
            img_str = get_plot_as_base64(plot_missing)
            html += f"""
            <div class="chart">
                <h3>Missing Values Heatmap</h3>
                <img src="data:image/png;base64,{img_str}" alt="Missing Values Heatmap">
                <p>Columns with darker colors have more missing values.</p>
            </div>
            """
        except Exception as e:
            html += f"<p>Error generating missing values heatmap: {str(e)}</p>"
    
    # 2. Enhanced Correlation matrix for numeric columns
    if len(numeric_cols) > 1:
        try:
            def plot_correlation():
                # Calculate correlation matrix
                corr = df[numeric_cols].corr()
                
                # Generate mask for the upper triangle
                mask = np.triu(np.ones_like(corr, dtype=bool))
                
                # Plot enhanced heatmap
                sns.heatmap(
                    corr, 
                    mask=mask, 
                    cmap=CORRELATION_CMAP,
                    annot=True, 
                    fmt='.2f', 
                    linewidths=0.5,
                    cbar_kws={"shrink": .8},
                    annot_kws={"size": 8}
                )
                plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
                
                # Add annotation for highest correlation
                if corr.shape[0] > 1:  # If there's more than one variable
                    # Get highest correlation (excluding self-correlations)
                    corr_no_diag = corr.copy()
                    np.fill_diagonal(corr_no_diag.values, 0)
                    max_corr = corr_no_diag.max().max()
                    if max_corr > 0.5:  # Only annotate strong correlations
                        max_idx = np.where(corr_no_diag.values == max_corr)
                        if len(max_idx[0]) > 0:
                            var1, var2 = corr_no_diag.index[max_idx[0][0]], corr_no_diag.columns[max_idx[1][0]]
                            plt.annotate(
                                f"Strongest correlation: {var1} & {var2} ({max_corr:.2f})",
                                xy=(0.5, 0.01),
                                xycoords='figure fraction',
                                ha='center',
                                fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                            )
            
            img_str = get_plot_as_base64(plot_correlation)
            html += f"""
            <div class="chart">
                <h3>Correlation Matrix</h3>
                <img src="data:image/png;base64,{img_str}" alt="Correlation Matrix">
                <p>Blue indicates positive correlation, red indicates negative correlation. Darker colors indicate stronger relationships.</p>
            </div>
            """
        except Exception as e:
            html += f"<p>Error generating correlation matrix: {str(e)}</p>"
    
    # 3. Distribution plots for numeric columns (with enhanced styling)
    html += '<h3>Distribution Plots</h3><div class="chart-grid">'
    for col in numeric_cols[:5]:
        try:
            def plot_distribution(column=col):
                plt.figure(figsize=(10, 6))
                
                # Create a GridSpec layout for combining histogram and boxplot
                gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
                
                # Top plot: Histogram with KDE
                ax_hist = plt.subplot(gs[0])
                sns.histplot(df[column].dropna(), kde=True, ax=ax_hist, color=DISTRIBUTION_COLOR, 
                           alpha=0.7, edgecolor='white', linewidth=0.5)
                
                # Add mean and median lines
                if df[column].count() > 0:
                    mean_val = df[column].mean()
                    median_val = df[column].median()
                    
                    ax_hist.axvline(mean_val, color='#e74c3c', linestyle='--', 
                                  linewidth=2, label=f'Mean: {mean_val:.2f}')
                    ax_hist.axvline(median_val, color='#2ecc71', linestyle='-', 
                                  linewidth=2, label=f'Median: {median_val:.2f}')
                    ax_hist.legend(frameon=True, facecolor='white', edgecolor='#dddddd')
                
                # Bottom plot: Boxplot
                ax_box = plt.subplot(gs[1])
                sns.boxplot(x=df[column].dropna(), ax=ax_box, color=DISTRIBUTION_COLOR, 
                          flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
                ax_box.set(yticks=[])  # Remove y ticks
                ax_box.set_xlabel('')  # Remove x label from boxplot
                
                # Set titles and labels
                ax_hist.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold', pad=15)
                ax_hist.set_ylabel('Frequency', fontweight='bold')
                ax_hist.set_xlabel('')  # Remove x label from histogram
                
                # Add distribution stats
                stats_text = (f"Mean: {df[column].mean():.2f}\n"
                             f"Median: {df[column].median():.2f}\n"
                             f"Std Dev: {df[column].std():.2f}\n"
                             f"Range: [{df[column].min():.2f}, {df[column].max():.2f}]")
                
                bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='#dddddd')
                plt.figtext(0.95, 0.7, stats_text, fontsize=9,
                           bbox=bbox_props, verticalalignment='top', horizontalalignment='right')
                
                # Adjust spacing between subplots
                plt.tight_layout()
                
            img_str = get_plot_as_base64(lambda: plot_distribution(col))
            html += f"""
            <div class="chart">
                <h3>Distribution of {col}</h3>
                <img src="data:image/png;base64,{img_str}" alt="Distribution of {col}">
                <p>The histogram shows the frequency distribution, while the box plot shows median, quartiles, and outliers.</p>
            </div>
            """
        except Exception as e:
            html += f"<p>Error generating distribution plot for {col}: {str(e)}</p>"
    html += '</div>'
    
    # 4. Count plots for categorical columns (with enhanced styling)
    if categorical_cols:
        html += '<h3>Categorical Plots</h3><div class="chart-grid">'
        for col in categorical_cols[:5]:
            try:
                # Skip if too many unique values
                if df[col].nunique() > 20:
                    continue
                    
                def plot_categorical(column=col):
                    # Get counts
                    value_counts = df[column].value_counts()
                    
                    # Determine if we need to limit categories
                    if len(value_counts) > 10:
                        value_counts = value_counts.head(10)
                        plt.title(f'Top 10 Values for {column}', fontsize=14, fontweight='bold', pad=15)
                        show_others = True
                    else:
                        plt.title(f'Values for {column}', fontsize=14, fontweight='bold', pad=15)
                        show_others = False
                        
                    # Calculate percentages
                    total = value_counts.sum()
                    pcts = [f"{x/total*100:.1f}%" for x in value_counts.values]
                    
                    # Plot with enhanced styling
                    ax = sns.barplot(
                        x=value_counts.index, 
                        y=value_counts.values,
                        palette=CATEGORICAL_PALETTE[:len(value_counts)],
                        edgecolor='white',
                        linewidth=1
                    )
                    
                    # Add percentage labels to each bar
                    for i, (p, pct) in enumerate(zip(ax.patches, pcts)):
                        ax.annotate(
                            pct, 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'bottom', 
                            fontsize=9, fontweight='bold', color='#333333',
                            xytext=(0, 5), textcoords='offset points'
                        )
                    
                    # Add note if showing truncated results
                    if show_others:
                        other_count = df[column].nunique() - 10
                        plt.figtext(
                            0.5, 0.01, 
                            f"+ {other_count} more categories not shown", 
                            ha='center', fontsize=9, color='#777'
                        )
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.xlabel(column, fontweight='bold')
                    plt.ylabel('Count', fontweight='bold')
                    plt.tight_layout()
                
                img_str = get_plot_as_base64(lambda: plot_categorical(col))
                html += f"""
                <div class="chart">
                    <h3>Distribution of {col}</h3>
                    <img src="data:image/png;base64,{img_str}" alt="Count plot for {col}">
                    <p>This chart shows the count distribution across different categories.</p>
                </div>
                """
            except Exception as e:
                html += f"<p>Error generating count plot for {col}: {str(e)}</p>"
        html += '</div>'
    
    # Add pair plot for numerical variables (if present)
    if len(numeric_cols) >= 2 and len(numeric_cols) <= 5:
        try:
            def plot_pairplot():
                # Take a sample if dataframe is large
                if df.shape[0] > 1000:
                    sample_df = df[numeric_cols].sample(n=1000, random_state=42)
                else:
                    sample_df = df[numeric_cols]
                
                # Create enhanced pairplot
                g = sns.pairplot(
                    sample_df,
                    diag_kind="kde",
                    plot_kws=dict(alpha=0.6, edgecolor="w", linewidth=0.5),
                    diag_kws=dict(shade=True, color=DISTRIBUTION_COLOR)
                )
                g.fig.suptitle('Relationships Between Numeric Variables', y=1.02, fontsize=16, fontweight='bold')
                g.fig.tight_layout()
                
                return g.fig
                
            # Save pairplot manually since it creates its own figure
            plt.figure(figsize=(12, 10))
            fig = plot_pairplot()
            
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            
            html += f"""
            <div class="chart">
                <h3>Pair Plot of Numeric Variables</h3>
                <img src="data:image/png;base64,{img_str}" alt="Pair Plot">
                <p>This matrix shows relationships between all pairs of numeric variables. Diagonal plots show distributions of individual variables.</p>
            </div>
            """
        except Exception as e:
            html += f"<p>Error generating pair plot: {str(e)}</p>"
    
    # Close the visualization section
    html += '</div>'
    
    # Column details section
    html += '<div class="section"><h2>Column Details</h2>'
    html += '<div class="column-details">'
    
    # Add column-by-column analysis
    for col in df.columns:
        try:
            # Skip if the column causes errors
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_categorical = df[col].dtype == 'object' or df[col].dtype.name == 'category'
            
            card_class = "numeric-column" if is_numeric else "categorical-column"
            
            html += f'<div class="column-card {card_class}">'
            html += f'<h4>{col}</h4>'
            
            # Type and basic info
            missing = df[col].isna().sum()
            missing_pct = round(100 * missing / len(df), 1) if len(df) > 0 else 0
            unique = df[col].nunique()
            
            html += f'<p><strong>Type:</strong> {df[col].dtype}</p>'
            html += f'<p><strong>Missing Values:</strong> {missing:,} ({missing_pct}%)</p>'
            html += f'<p><strong>Unique Values:</strong> {unique:,}</p>'
            
            # Add numeric stats if applicable
            if is_numeric:
                # Try to add visualization for numeric
                try:
                    def mini_hist():
                        plt.figure(figsize=(5, 2))
                        sns.histplot(df[col].dropna(), color=DISTRIBUTION_COLOR, kde=True)
                        plt.title(f"Distribution", fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                        plt.tight_layout()
                        
                    mini_img = get_plot_as_base64(mini_hist)
                    html += f'<img src="data:image/png;base64,{mini_img}" alt="Mini histogram" style="width:100%; margin:10px 0;">'
                except:
                    pass
                    
                # Add numeric stats
                html += f'<p><strong>Min:</strong> {df[col].min():,.2f}</p>'
                html += f'<p><strong>Max:</strong> {df[col].max():,.2f}</p>'
                html += f'<p><strong>Mean:</strong> {df[col].mean():,.2f}</p>'
                html += f'<p><strong>Median:</strong> {df[col].median():,.2f}</p>'
                html += f'<p><strong>Std Dev:</strong> {df[col].std():,.2f}</p>'
                
                # Add skewness and kurtosis if possible
                try:
                    skew = df[col].skew()
                    kurt = df[col].kurtosis()
                    html += f'<p><strong>Skewness:</strong> {skew:.2f}</p>'
                    html += f'<p><strong>Kurtosis:</strong> {kurt:.2f}</p>'
                except:
                    pass
            
            # Add top values for categorical columns
            elif is_categorical:
                # Try to add visualization for categorical
                try:
                    if df[col].nunique() <= 10:
                        def mini_bar():
                            plt.figure(figsize=(5, 2))
                            top_vals = df[col].value_counts().head(5)
                            sns.barplot(x=top_vals.index, y=top_vals.values, palette=CATEGORICAL_PALETTE)
                            plt.title(f"Top Values", fontsize=10)
                            plt.xticks(fontsize=8, rotation=45, ha="right")
                            plt.yticks(fontsize=8)
                            plt.tight_layout()
                            
                        mini_img = get_plot_as_base64(mini_bar)
                        html += f'<img src="data:image/png;base64,{mini_img}" alt="Mini bar chart" style="width:100%; margin:10px 0;">'
                except:
                    pass
                
                if df[col].nunique() <= 20:
                    html += '<p><strong>Top Values:</strong></p>'
                    html += '<ul style="padding-left: 20px; margin-top: 5px;">'
                    for val, count in df[col].value_counts().head(5).items():
                        percent = 100 * count / len(df)
                        html += f'<li>{val}: {count:,} ({percent:.1f}%)</li>'
                    html += '</ul>'
                else:
                    html += '<p><strong>Top Values:</strong> (Too many unique values to display)</p>'
            
            html += '</div>'
        except Exception as e:
            # Skip columns that cause errors
            continue
    
    html += '</div>'
    html += '</div>'
    
    # Add insights section (statistical observations)
    html += '<div class="section"><h2>Statistical Insights</h2>'
    
    # Add some automatic insights based on the data
    insights = []
    
    # Insight 1: Missing values
    if missing_values > 0:
        missing_cols = df.isna().sum()
        top_missing = missing_cols[missing_cols > 0].sort_values(ascending=False).head(3)
        if len(top_missing) > 0:
            missing_text = "The following columns have missing values:<ul>"
            for col, count in top_missing.items():
                pct = 100 * count / len(df)
                missing_text += f"<li><strong>{col}</strong>: {count:,} values ({pct:.1f}%)</li>"
            missing_text += "</ul>"
            insights.append({
                "title": "Missing Values Analysis",
                "content": missing_text,
                "type": "info"
            })
    
    # Insight 2: Correlation insights
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr()
            # Get top 3 highest absolute correlations (excluding self-correlations)
            corr_no_diag = corr.copy()
            np.fill_diagonal(corr_no_diag.values, 0)
            corr_abs = corr_no_diag.abs()
            top_corr = []
            for _ in range(min(3, len(numeric_cols))):
                max_idx = corr_abs.values.argmax()
                i, j = max_idx // corr_abs.shape[1], max_idx % corr_abs.shape[1]
                if corr_abs.iloc[i, j] > 0.5:  # Only include if correlation is meaningful
                    col1, col2 = corr_abs.index[i], corr_abs.columns[j]
                    corr_val = corr.loc[col1, col2]
                    top_corr.append((col1, col2, corr_val))
                    # Set this value to 0 to find the next highest
                    corr_abs.iloc[i, j] = 0
                else:
                    break
            
            if top_corr:
                corr_text = "Notable correlations between variables:<ul>"
                for col1, col2, val in top_corr:
                    corr_type = "positive" if val > 0 else "negative"
                    corr_text += f"<li><strong>{col1}</strong> and <strong>{col2}</strong>: {val:.2f} ({corr_type})</li>"
                corr_text += "</ul>"
                insights.append({
                    "title": "Correlation Findings",
                    "content": corr_text,
                    "type": "primary"
                })
        except:
            pass
    
    # Insight 3: Distribution insights
    skewed_cols = []
    for col in numeric_cols:
        try:
            skew = df[col].skew()
            if abs(skew) > 1.5:
                direction = "right" if skew > 0 else "left"
                skewed_cols.append((col, skew, direction))
        except:
            continue
    
    if skewed_cols:
        skew_text = "The following numeric columns have skewed distributions:<ul>"
        for col, skew, direction in skewed_cols[:3]:
            skew_text += f"<li><strong>{col}</strong>: {skew:.2f} ({direction}-skewed)</li>"
        skew_text += "</ul>"
        insights.append({
            "title": "Distribution Analysis",
            "content": skew_text,
            "type": "warning"
        })
    
    # Insight 4: Categorical imbalance
    imbalanced_cols = []
    for col in categorical_cols:
        try:
            if df[col].nunique() > 1:
                top_val = df[col].value_counts().iloc[0]
                top_pct = 100 * top_val / df[col].count()
                if top_pct > 75:
                    imbalanced_cols.append((col, top_pct, df[col].value_counts().index[0]))
        except:
            continue
    
    if imbalanced_cols:
        imb_text = "The following categorical columns show significant imbalance:<ul>"
        for col, pct, val in imbalanced_cols[:3]:
            imb_text += f"<li><strong>{col}</strong>: '{val}' accounts for {pct:.1f}% of values</li>"
        imb_text += "</ul>"
        insights.append({
            "title": "Category Imbalance",
            "content": imb_text,
            "type": "warning"
        })
    
    # Add insights to the report
    if insights:
        for insight in insights:
            type_class = {
                "primary": "style='border-left-color: #3498db;'",
                "info": "style='border-left-color: #2ecc71;'",
                "warning": "style='border-left-color: #e74c3c;'"
            }
            html += f"""
            <div class="insight-card" {type_class.get(insight.get('type'), '')}>
                <h4>{insight['title']}</h4>
                <div>{insight['content']}</div>
            </div>
            """
    else:
        html += "<p>No significant statistical insights found in the data.</p>"
    
    html += '</div>'
    
    # Add footer
    html += """
            <div class="footer">
                <p>Report generated with EDAwala - Advanced Exploratory Data Analysis Tool</p>
                <p>Copyright © 2025 EDAwala</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path