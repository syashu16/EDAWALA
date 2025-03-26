"""
Custom chart themes and styling utilities for EDAwala
"""
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Define color palettes
EDAWALA_PALETTE = {
    'primary': '#3498db',     # Blue
    'secondary': '#2ecc71',   # Green
    'accent': '#e74c3c',      # Red
    'neutral': '#7f8c8d',     # Gray
    'warning': '#f39c12',     # Orange
    'info': '#9b59b6',        # Purple
    'dark': '#2c3e50',        # Dark blue
    'light': '#ecf0f1',       # Light gray
}

# Custom color palettes
CORRELATION_CMAP = LinearSegmentedColormap.from_list("custom_coolwarm", 
                                                    ["#4575b4", "#91bfdb", "#e0f3f8", "#ffffbf", "#fee090", "#fc8d59", "#d73027"])

DISTRIBUTION_CMAP = LinearSegmentedColormap.from_list("custom_viridis", 
                                                     ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"])

CATEGORICAL_PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#34495e", "#e67e22", "#c0392b", "#16a085"]

def apply_edawala_theme():
    """Apply the EDAwala custom theme to matplotlib plots"""
    plt.style.use('ggplot')
    
    # Custom theme parameters
    custom_theme = {
        'axes.facecolor': '#f8f9fa',
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.color': '#dddddd',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'axes.titleweight': 'bold',
        'axes.titlecolor': '#2c3e50',
        'axes.titlesize': 14,
        'axes.labelweight': 'medium',
        'axes.edgecolor': '#dddddd',
        'xtick.color': '#666666',
        'ytick.color': '#666666',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'figure.titleweight': 'bold',
        'figure.titlesize': 16,
    }
    
    plt.rcParams.update(custom_theme)
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.color': '#dddddd'})
    sns.set_context("notebook", font_scale=1.1)
    
    # Set the default color palette
    sns.set_palette(CATEGORICAL_PALETTE)

def create_distribution_plot(ax, data, column, title=None, kde=True, add_stats=True):
    """
    Create a styled distribution plot on the given axis
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    data : pandas.DataFrame
        Data to plot
    column : str
        Column name to plot distribution
    title : str, optional
        Plot title
    kde : bool, optional
        Whether to include KDE plot
    add_stats : bool, optional
        Whether to add statistics
    """
    # Create histplot
    sns.histplot(
        data[column].dropna(),
        kde=kde,
        color=EDAWALA_PALETTE['primary'],
        alpha=0.7,
        edgecolor='white',
        linewidth=0.5,
        ax=ax
    )
    
    # Add mean and median lines
    if data[column].count() > 0:
        mean_val = data[column].mean()
        median_val = data[column].median()
        
        ax.axvline(mean_val, color=EDAWALA_PALETTE['accent'], linestyle='--', 
                 linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color=EDAWALA_PALETTE['secondary'], linestyle='-', 
                 linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend(frameon=True, facecolor='white', edgecolor='#dddddd')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold', pad=15)
    
    # Label axes
    ax.set_xlabel(column, fontweight='medium')
    ax.set_ylabel('Frequency', fontweight='medium')
    
    # Add statistics annotation
    if add_stats and data[column].count() > 0:
        stats_text = (f"Mean: {data[column].mean():.2f}\n"
                     f"Median: {data[column].median():.2f}\n"
                     f"Std Dev: {data[column].std():.2f}\n"
                     f"Range: [{data[column].min():.2f}, {data[column].max():.2f}]")
        
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='#dddddd')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='top', horizontalalignment='right', bbox=bbox_props)

def create_correlation_matrix(ax, data, numeric_columns=None, annot=True, mask_upper=True):
    """
    Create a styled correlation matrix on the given axis
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    data : pandas.DataFrame
        Data to plot
    numeric_columns : list, optional
        List of numeric columns to include
    annot : bool, optional
        Whether to annotate values
    mask_upper : bool, optional
        Whether to mask upper triangle
    """
    # Select numeric columns if not specified
    if numeric_columns is None:
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate correlation
    corr = data[numeric_columns].corr()
    
    # Generate mask for the upper triangle if requested
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=CORRELATION_CMAP,
        annot=annot, 
        fmt='.2f', 
        linewidths=0.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 8},
        ax=ax
    )
    
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

def create_categorical_plot(ax, data, column, title=None, limit=15, add_stats=True):
    """
    Create a styled categorical plot on the given axis
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    data : pandas.DataFrame
        Data to plot
    column : str
        Column name to plot
    title : str, optional
        Plot title
    limit : int, optional
        Limit number of categories to show
    add_stats : bool, optional
        Whether to add statistics
    """
    # Get value counts
    value_counts = data[column].value_counts()
    
    # Limit categories if needed
    show_others = False
    if len(value_counts) > limit:
        others_sum = value_counts.iloc[limit:].sum()
        value_counts = value_counts.iloc[:limit]
        value_counts['Other Categories'] = others_sum
        show_others = True
    
    # Calculate percentages
    total = value_counts.sum()
    pcts = [(x/total*100) for x in value_counts.values]
    
    # Plot barplot
    bars = sns.barplot(
        x=value_counts.index, 
        y=value_counts.values, 
        palette=CATEGORICAL_PALETTE[:len(value_counts)],
        ax=ax
    )
    
    # Add percentage labels
    for i, p in enumerate(bars.patches):
        percentage = pcts[i]
        bars.annotate(
            f'{percentage:.1f}%', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha = 'center', va = 'bottom', 
            fontsize=8, fontweight='bold', color='#333333',
            xytext=(0, 5), textcoords='offset points'
        )
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        title_suffix = " (Top Categories)" if show_others else ""
        ax.set_title(f'Distribution of {column}{title_suffix}', fontsize=14, fontweight='bold', pad=15)
    
    # Formatting
    ax.set_xlabel(column, fontweight='medium')
    ax.set_ylabel('Count', fontweight='medium')
    plt.xticks(rotation=45, ha='right')
    
    # Add stats annotation
    if add_stats:
        most_common = value_counts.index[0]
        most_common_pct = (value_counts.iloc[0] / total) * 100
        
        stats_text = (f"Total: {total}\n"
                     f"Unique: {data[column].nunique()}\n"
                     f"Most Common: {most_common}\n"
                     f"(Represents {most_common_pct:.1f}%)")
        
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='#dddddd')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
              verticalalignment='top', horizontalalignment='right', bbox=bbox_props)