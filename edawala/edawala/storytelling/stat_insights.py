"""
Statistical insights generator for EDAwala
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_statistical_insights(df: pd.DataFrame, max_insights: int = 15) -> List[Dict[str, Any]]:
    """
    Generate statistical insights from a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    max_insights : int, optional
        Maximum number of insights to generate (default: 15)
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of insights, each as a dictionary
    """
    insights = []
    
    # 1. Basic dataset statistics
    rows, cols = df.shape
    missing_values = df.isna().sum().sum()
    missing_percent = round((missing_values / (rows * cols)) * 100, 2) if rows * cols > 0 else 0
    
    insights.append({
        'title': f"Dataset contains {rows} rows and {cols} columns",
        'description': f"The dataset has {rows} observations across {cols} variables. "
                     f"There are {missing_values} missing values ({missing_percent}% of the data)."
    })
    
    # 2. Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 3. Correlation insights for numeric columns
    if len(numeric_cols) >= 2:
        # Calculate correlation while handling missing values
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Find the strongest positive correlation
            pos_corr = corr_matrix.unstack().sort_values(ascending=False)
            pos_corr = pos_corr[pos_corr < 1.0]  # Remove self-correlations
            
            if not pos_corr.empty:
                top_pos = pos_corr.index[0]
                top_pos_val = pos_corr.iloc[0]
                
                if abs(top_pos_val) > 0.5:  # Only if correlation is meaningful
                    insights.append({
                        'title': f"Strong positive correlation between {top_pos[0]} and {top_pos[1]}",
                        'description': f"There is a strong positive correlation of {top_pos_val:.2f} between "
                                    f"{top_pos[0]} and {top_pos[1]}. As one increases, the other tends to increase as well."
                    })
            
            # Find the strongest negative correlation
            neg_corr = corr_matrix.unstack().sort_values()
            
            if not neg_corr.empty:
                top_neg = neg_corr.index[0]
                top_neg_val = neg_corr.iloc[0]
                
                if top_neg_val < -0.5:  # Only if correlation is meaningful
                    insights.append({
                        'title': f"Strong negative correlation between {top_neg[0]} and {top_neg[1]}",
                        'description': f"There is a strong negative correlation of {top_neg_val:.2f} between "
                                    f"{top_neg[0]} and {top_neg[1]}. As one increases, the other tends to decrease."
                    })
        except Exception as e:
            logger.warning(f"Error calculating correlations: {e}")
    
    # 4. Distribution insights for numeric columns
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        # Skip columns with no data
        if df[col].count() == 0:
            continue
            
        try:
            # Check for skewness
            skewness = df[col].skew()
            
            if abs(skewness) > 1:
                skew_direction = "right" if skewness > 0 else "left"
                insights.append({
                    'title': f"The distribution of {col} is skewed to the {skew_direction}",
                    'description': f"{col} has a skewness of {skewness:.2f}, indicating a {skew_direction}-skewed distribution. "
                                f"This means the {'right' if skewness > 0 else 'left'} tail is longer."
                })
            
            # Check for outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero or negative IQR
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outlier_percent = (len(outliers) / df[col].count()) * 100 if df[col].count() > 0 else 0
                
                if outlier_percent > 5:
                    insights.append({
                        'title': f"Significant outliers detected in {col}",
                        'description': f"{outlier_percent:.1f}% of the values in {col} are outliers (outside the range of "
                                    f"{lower_bound:.2f} to {upper_bound:.2f}). These may influence statistical analyses."
                    })
        except Exception as e:
            logger.warning(f"Error analyzing distribution of {col}: {e}")
            continue
    
    # 5. Categorical column insights
    for col in cat_cols[:5]:  # Limit to first 5 categorical columns
        try:
            value_counts = df[col].value_counts(normalize=True)
            
            if len(value_counts) > 0:
                # Check for imbalance
                most_common = value_counts.index[0]
                most_common_pct = value_counts.iloc[0] * 100
                
                if most_common_pct > 75:
                    insights.append({
                        'title': f"Imbalanced distribution in {col}",
                        'description': f"The category '{most_common}' dominates the {col} variable, "
                                    f"representing {most_common_pct:.1f}% of the values. This imbalance may "
                                    f"affect models trained on this data."
                    })
                
                # Check for high cardinality
                if len(value_counts) > 20:
                    insights.append({
                        'title': f"High cardinality detected in {col}",
                        'description': f"The column {col} has {len(value_counts)} unique values, which may be too many "
                                    f"for effective one-hot encoding. Consider grouping less frequent categories."
                    })
        except Exception as e:
            logger.warning(f"Error analyzing categorical column {col}: {e}")
            continue
    
    # 6. Missing value patterns
    if missing_values > 0:
        try:
            missing_by_col = df.isna().mean() * 100
            highest_missing = missing_by_col.sort_values(ascending=False).head(1)
            
            if not highest_missing.empty and highest_missing.iloc[0] > 10:
                col = highest_missing.index[0]
                pct = highest_missing.iloc[0]
                
                insights.append({
                    'title': f"High missingness in {col}",
                    'description': f"The column {col} is missing {pct:.1f}% of its values. "
                                f"Consider imputation strategies or evaluating if this column can be excluded."
                })
        except Exception as e:
            logger.warning(f"Error analyzing missing values: {e}")
    
    # Limit to max_insights
    return insights[:max_insights]