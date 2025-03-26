"""
Interactive visualization functions for EDAwala using Plotly
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, List, Optional

def create_interactive_histograms(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create interactive histograms for selected columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to visualize
    columns : List[str], optional
        List of column names to visualize. If None, numeric columns will be used
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary of plotly figure JSON for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns[:5].tolist()
    
    figures = {}
    
    for col in columns:
        try:
            # Create figure with plotly
            fig = make_subplots(rows=2, cols=1, 
                              row_heights=[0.8, 0.2],
                              shared_xaxes=True,
                              vertical_spacing=0.05)
            
            # Add histogram to top subplot
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(), 
                    histnorm='percent',
                    name=col,
                    marker=dict(
                        color='rgba(52, 152, 219, 0.7)',
                        line=dict(
                            color='rgba(52, 152, 219, 1)',
                            width=1
                        )
                    ),
                    hovertemplate=f"{col}: %{{x}}<br>Percent: %{{y:.2f}}%<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Add box plot to bottom subplot
            fig.add_trace(
                go.Box(
                    x=df[col].dropna(),
                    name=col,
                    marker_color='rgba(52, 152, 219, 0.7)',
                    line=dict(color='rgba(52, 152, 219, 1)'),
                    boxmean=True,
                    hovertemplate=f"{col}<br>Min: %{{customdata[0]:.2f}}<br>Q1: %{{customdata[1]:.2f}}<br>Median: %{{customdata[2]:.2f}}<br>Q3: %{{customdata[3]:.2f}}<br>Max: %{{customdata[4]:.2f}}<extra></extra>",
                    customdata=np.array([[
                        df[col].min(),
                        df[col].quantile(0.25),
                        df[col].median(),
                        df[col].quantile(0.75),
                        df[col].max()
                    ]])
                ),
                row=2, col=1
            )
            
            # Add vertical lines for mean and median in the histogram
            fig.add_vline(
                x=df[col].mean(), 
                line_dash="dash", 
                line_color="#e74c3c",
                annotation_text=f"Mean: {df[col].mean():.2f}",
                annotation_position="top right",
                annotation_font_color="#e74c3c",
                row=1, col=1
            )
            
            fig.add_vline(
                x=df[col].median(), 
                line_dash="solid", 
                line_color="#2ecc71",
                annotation_text=f"Median: {df[col].median():.2f}",
                annotation_position="top left",
                annotation_font_color="#2ecc71",
                row=1, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'Distribution of {col}',
                title_font=dict(size=16, family="Arial", color="#2c3e50"),
                template="simple_white",
                showlegend=False,
                margin=dict(l=40, r=40, t=60, b=40),
                height=500,
                hovermode="x",
                xaxis_title=None,
                xaxis2_title=col,
                yaxis_title="Percent (%)",
                yaxis2_showticklabels=False
            )
            
            # Add statistics annotation
            stats_text = (f"<b>Statistics:</b><br>" +
                         f"Count: {df[col].count():,}<br>" +
                         f"Mean: {df[col].mean():.2f}<br>" +
                         f"Median: {df[col].median():.2f}<br>" +
                         f"Std Dev: {df[col].std():.2f}<br>" +
                         f"Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.99, y=0.97,
                text=stats_text,
                align="right",
                showarrow=False,
                bgcolor="white",
                bordercolor="#dddddd",
                borderwidth=1,
                borderpad=6,
                font=dict(size=10, family="Arial", color="#333333")
            )
            
            # Convert to JSON for embedding
            figures[col] = json.dumps(fig.to_dict())
        except Exception as e:
            print(f"Error creating interactive histogram for {col}: {e}")
            continue
    
    return figures

def create_interactive_correlation_matrix(df: pd.DataFrame) -> Optional[str]:
    """
    Create an interactive correlation matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    Optional[str]
        JSON string of plotly figure or None if error
    """
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return None
            
        # Calculate correlation
        corr = df[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        # Create heatmap
        fig = go.Figure()
        
        # Add main heatmap trace
        heatmap = go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1, zmax=1,
            colorscale=[
                [0.0, "#d73027"],
                [0.25, "#fc8d59"],
                [0.5, "#ffffbf"],
                [0.75, "#91bfdb"],
                [1.0, "#4575b4"]
            ],
            colorbar=dict(
                title="Correlation",
                thickness=15,
                titleside="right",
                titlefont=dict(size=12)
            ),
            hovertemplate="<b>%{y}</b> & <b>%{x}</b><br>Correlation: %{z:.2f}<extra></extra>"
        )
        
        # Create a custom template for text formatting
        text_template = np.zeros_like(corr, dtype=object)
        for i in range(len(corr)):
            for j in range(len(corr)):
                text_template[i, j] = f"{corr.iloc[i, j]:.2f}"
        
        # Add text annotations
        annotations = go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            text=text_template,
            hoverinfo="none",
            showscale=False,
            colorscale="Viridis",
            zmin=-1, zmax=1,
            texttemplate="%{text}",
            textfont={"color": "black", "size": 10}
        )
        
        # Add both traces
        fig.add_trace(heatmap)
        fig.add_trace(annotations)
        
        # Update layout
        fig.update_layout(
            title="Correlation Matrix",
            title_font=dict(size=16, family="Arial", color="#2c3e50"),
            template="simple_white",
            height=600,
            width=800,
            xaxis=dict(
                title=None,
                side="bottom",
                tickangle=-45,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title=None,
                autorange="reversed",
                tickfont=dict(size=11)
            ),
            margin=dict(l=60, r=30, t=60, b=60)
        )
        
        # Find highest correlation values
        corr_no_diag = corr.copy()
        np.fill_diagonal(corr_no_diag.values, 0)
        max_corr = corr_no_diag.max().max()
        max_idx = np.where(corr_no_diag.values == max_corr)
        if len(max_idx[0]) > 0:
            var1, var2 = corr.index[max_idx[0][0]], corr.columns[max_idx[1][0]]
            
            # Add annotation
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                text=f"Strongest correlation: <b>{var1}</b> & <b>{var2}</b> ({max_corr:.2f})",
                showarrow=False,
                font=dict(size=12)
            )
        
        # Convert to JSON for embedding
        return json.dumps(fig.to_dict())
    except Exception as e:
        print(f"Error creating interactive correlation matrix: {e}")
        return None

def create_interactive_categorical_chart(df: pd.DataFrame, column: str) -> Optional[str]:
    """
    Create an interactive categorical chart
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to visualize
    column : str
        Categorical column to visualize
        
    Returns:
    --------
    Optional[str]
        JSON string of plotly figure or None if error
    """
    try:
        # Check if column is valid
        if column not in df.columns or df[column].nunique() > 30:
            return None
            
        # Calculate value counts
        value_counts = df[column].value_counts()
        total = value_counts.sum()
        
        # Limit to top 15 categories if needed
        if len(value_counts) > 15:
            others_count = value_counts.iloc[15:].sum()
            value_counts = value_counts.iloc[:15]
            value_counts["Others"] = others_count
        
        # Calculate percentages
        percentages = [f"{count/total*100:.1f}%" for count in value_counts]
        
        # Create figure
        fig = go.Figure()
        
        # Add bar trace
        fig.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            marker_color='rgba(52, 152, 219, 0.7)',
            marker_line=dict(
                color='rgba(52, 152, 219, 1)',
                width=1
            ),
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Percentage: %{customdata}<extra></extra>",
            customdata=percentages,
            texttemplate="%{customdata}",
            textposition="outside"
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Distribution of {column}',
            title_font=dict(size=16, family="Arial", color="#2c3e50"),
            template="simple_white",
            height=500,
            xaxis=dict(
                title=column,
                tickangle=-45,
                categoryorder='total descending'
            ),
            yaxis=dict(
                title="Count",
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            margin=dict(l=40, r=40, t=60, b=100)
        )
        
        # Add stats annotation
        stats_text = (f"<b>Statistics:</b><br>" +
                     f"Total: {total:,}<br>" +
                     f"Unique: {df[column].nunique():,}<br>" +
                     f"Most Common: {value_counts.index[0]} ({value_counts.iloc[0]:,})<br>" +
                     f"% of Most Common: {value_counts.iloc[0]/total*100:.1f}%")
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.99, y=0.97,
            text=stats_text,
            align="right",
            showarrow=False,
            bgcolor="white",
            bordercolor="#dddddd",
            borderwidth=1,
            borderpad=6,
            font=dict(size=10, family="Arial", color="#333333")
        )
        
        # Convert to JSON for embedding
        return json.dumps(fig.to_dict())
    except Exception as e:
        print(f"Error creating interactive categorical chart for {column}: {e}")
        return None