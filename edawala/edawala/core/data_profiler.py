"""
Data profiling utilities for EDAwala
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple


class DataProfiler:
    """
    Utility class for profiling datasets
    """
    
    @staticmethod
    def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic statistics about the dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to profile
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of basic statistics
        """
        stats = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "total_cells": df.size,
            "missing_cells": df.isna().sum().sum(),
            "missing_percent": round((df.isna().sum().sum() / df.size) * 100, 2),
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percent": round((df.duplicated().sum() / df.shape[0]) * 100, 2),
            "memory_usage": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),  # in MB
            "dtypes": df.dtypes.value_counts().to_dict()
        }
        
        return stats
    
    @staticmethod
    def get_column_stats(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed statistics for each column
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to profile
            
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Dictionary of column statistics
        """
        column_stats = {}
        
        for column in df.columns:
            col_data = df[column]
            dtype = str(col_data.dtype)
            
            # Common stats for all types
            stats = {
                "dtype": dtype,
                "count": len(col_data),
                "missing": col_data.isna().sum(),
                "missing_percent": round((col_data.isna().sum() / len(col_data)) * 100, 2),
                "unique_values": col_data.nunique(),
            }
            
            # Additional stats based on data type
            if np.issubdtype(col_data.dtype, np.number):
                # Numeric columns
                stats.update({
                    "min": col_data.min() if not pd.isna(col_data).all() else None,
                    "max": col_data.max() if not pd.isna(col_data).all() else None,
                    "mean": col_data.mean() if not pd.isna(col_data).all() else None,
                    "median": col_data.median() if not pd.isna(col_data).all() else None,
                    "std": col_data.std() if not pd.isna(col_data).all() else None,
                    "zeros": (col_data == 0).sum(),
                    "zeros_percent": round(((col_data == 0).sum() / len(col_data)) * 100, 2),
                })
                
                # Check for potential outliers using IQR method
                if not pd.isna(col_data).all():
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    stats["potential_outliers"] = len(outliers)
                    stats["potential_outliers_percent"] = round((len(outliers) / len(col_data)) * 100, 2)
            
            elif col_data.dtype == 'object' or col_data.dtype.name == 'category':
                # String or categorical columns
                value_counts = col_data.value_counts()
                if not value_counts.empty:
                    stats.update({
                        "most_common_value": value_counts.index[0],
                        "most_common_count": value_counts.iloc[0],
                        "most_common_percent": round((value_counts.iloc[0] / len(col_data)) * 100, 2),
                    })
                
                # Check if it might be a datetime
                if col_data.dtype == 'object':
                    try:
                        # Try to convert a sample to datetime
                        sample = col_data.dropna().iloc[0] if not col_data.dropna().empty else None
                        if sample and pd.to_datetime(sample, errors='coerce') is not pd.NaT:
                            stats["might_be_datetime"] = True
                    except (IndexError, ValueError):
                        stats["might_be_datetime"] = False
            
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                # Datetime columns
                if not pd.isna(col_data).all():
                    stats.update({
                        "min_date": col_data.min(),
                        "max_date": col_data.max(),
                        "range_days": (col_data.max() - col_data.min()).days if not pd.isna(col_data).all() else None,
                    })
            
            column_stats[column] = stats
        
        return column_stats
    
    @staticmethod
    def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect and categorize columns by their type
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to analyze
            
        Returns:
        --------
        Dict[str, List[str]]
            Dictionary of column types and their corresponding columns
        """
        column_types = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": [],
            "binary": [],
            "id_like": [],
            "constant": [],
        }
        
        for column in df.columns:
            col_data = df[column]
            
            # Check for constant columns
            if col_data.nunique() <= 1:
                column_types["constant"].append(column)
                continue
                
            # Check for binary columns
            if col_data.nunique() == 2:
                column_types["binary"].append(column)
                continue
                
            # Check for ID-like columns (unique values > 80% of rows)
            if col_data.nunique() / len(col_data) > 0.8:
                column_types["id_like"].append(column)
                continue
                
            # Identify datetime columns
            if pd.api.types.is_datetime64_any_dtype(col_data):
                column_types["datetime"].append(column)
                continue
                
            # Try to convert string columns to datetime
            if col_data.dtype == 'object':
                try:
                    if pd.to_datetime(col_data, errors='coerce').notna().sum() > 0.7 * len(col_data):
                        column_types["datetime"].append(column)
                        continue
                except (TypeError, ValueError):
                    pass
                    
            # Identify numeric columns
            if np.issubdtype(col_data.dtype, np.number):
                column_types["numeric"].append(column)
                continue
                
            # Identify categorical columns (fewer than 20 unique values)
            if col_data.nunique() < 20:
                column_types["categorical"].append(column)
                continue
                
            # Default to text for other string columns
            if col_data.dtype == 'object':
                column_types["text"].append(column)
                
        return column_types