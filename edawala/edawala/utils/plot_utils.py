"""
Plotting utilities with thread-safety measures for EDAwala
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64

def plot_to_base64(plot_function, *args, **kwargs):
    """
    Convert a plotting function to a base64 encoded string in a thread-safe way.
    
    Parameters:
    -----------
    plot_function : callable
        Function that creates a matplotlib plot
    args, kwargs : 
        Arguments to pass to the plot function
        
    Returns:
    --------
    str
        Base64 encoded string of the plot
    """
    # Create a new figure
    plt.figure()
    
    # Create a BytesIO object
    buf = io.BytesIO()
    
    # Call the plot function
    plot_function(*args, **kwargs)
    
    # Save the plot to the BytesIO object
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close('all')  # Close all figures to avoid memory leaks
    
    # Get the image as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def create_missing_values_heatmap(df):
    """Thread-safe missing values heatmap"""
    def plot_func():
        # Check if the dataframe is too large, if so, take a sample
        if df.shape[0] > 100:
            df_sample = df.sample(n=100, random_state=42)
        else:
            df_sample = df
            
        sns.heatmap(df_sample.isna(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Observations')
    
    return plot_to_base64(plot_func)

def create_correlation_matrix(df, numeric_columns):
    """Thread-safe correlation matrix plot"""
    def plot_func():
        # Calculate correlation matrix
        corr = df[numeric_columns].corr()
        
        # Generate mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
    
    return plot_to_base64(plot_func)

def create_distribution_plot(df, column):
    """Thread-safe distribution plot"""
    def plot_func():
        # Plot histogram with KDE
        sns.histplot(df[column].dropna(), kde=True)
        
        # Add mean and median lines
        plt.axvline(df[column].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[column].mean():.2f}')
        plt.axvline(df[column].median(), color='green', linestyle='-', 
                   label=f'Median: {df[column].median():.2f}')
        
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
    
    return plot_to_base64(plot_func)

def create_categorical_plot(df, column):
    """Thread-safe categorical plot"""
    def plot_func():
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
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.tight_layout()
    
    return plot_to_base64(plot_func)

def create_pairplot(df, numeric_columns):
    """Thread-safe pairplot"""
    def plot_func():
        # Take a sample if dataframe is large
        if df.shape[0] > 1000:
            df_sample = df[numeric_columns].sample(n=1000, random_state=42)
        else:
            df_sample = df[numeric_columns]
            
        # Create pairplot
        sns.pairplot(df_sample)
        plt.suptitle('Pairplot of Numeric Variables', y=1.02)
    
    return plot_to_base64(plot_func)