"""
Data loading utilities for EDAwala
"""
import pandas as pd
from typing import Optional, Union
import io
import os

def load_data(file_path_or_buffer: Union[str, io.BytesIO]) -> Optional[pd.DataFrame]:
    """
    Load data from a file path or uploaded file buffer.
    
    Parameters:
    -----------
    file_path_or_buffer : str or BytesIO
        Path to the file or an uploaded file buffer
        
    Returns:
    --------
    pd.DataFrame or None
        The loaded DataFrame, or None if loading failed
    """
    try:
        # Get file extension
        if isinstance(file_path_or_buffer, str):
            _, ext = os.path.splitext(file_path_or_buffer)
        else:
            # For uploaded files, get name from the buffer
            if hasattr(file_path_or_buffer, 'name'):
                _, ext = os.path.splitext(file_path_or_buffer.name)
            else:
                # Try to infer the format
                try:
                    # Try CSV first
                    return pd.read_csv(file_path_or_buffer)
                except:
                    # Then try Excel
                    try:
                        return pd.read_excel(file_path_or_buffer)
                    except:
                        return None
        
        # Load based on extension
        ext = ext.lower()
        
        if ext == '.csv':
            return pd.read_csv(file_path_or_buffer)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path_or_buffer)
        elif ext == '.json':
            return pd.read_json(file_path_or_buffer)
        elif ext == '.pkl':
            return pd.read_pickle(file_path_or_buffer)
        elif ext == '.parquet':
            return pd.read_parquet(file_path_or_buffer)
        elif ext == '.feather':
            return pd.read_feather(file_path_or_buffer)
        elif ext == '.h5':
            return pd.read_hdf(file_path_or_buffer)
        elif ext == '.sas7bdat':
            return pd.read_sas(file_path_or_buffer)
        elif ext == '.dta':
            return pd.read_stata(file_path_or_buffer)
        elif ext == '.sav':
            return pd.read_spss(file_path_or_buffer)
        elif ext in ['.tsv', '.txt', '.dat']:
            return pd.read_csv(file_path_or_buffer, sep='\t')
        else:
            # Try to infer the format
            try:
                # Try CSV first
                return pd.read_csv(file_path_or_buffer)
            except:
                # Then try Excel
                try:
                    return pd.read_excel(file_path_or_buffer)
                except:
                    return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_data(df: pd.DataFrame, file_path: str) -> bool:
    """
    Save DataFrame to a file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Path where the file will be saved
        
    Returns:
    --------
    bool
        True if saving was successful, False otherwise
    """
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Save based on extension
        if ext == '.csv':
            df.to_csv(file_path, index=False)
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(file_path, index=False)
        elif ext == '.json':
            df.to_json(file_path, orient='records')
        elif ext == '.pkl':
            df.to_pickle(file_path)
        elif ext == '.parquet':
            df.to_parquet(file_path, index=False)
        elif ext == '.feather':
            df.to_feather(file_path)
        elif ext == '.h5':
            df.to_hdf(file_path, key='data', index=False)
        elif ext in ['.tsv', '.txt', '.dat']:
            df.to_csv(file_path, sep='\t', index=False)
        else:
            # Default to CSV
            df.to_csv(file_path, index=False)
        
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def preview_data(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """
    Get a preview of the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to preview
    n_rows : int, optional
        Number of rows to preview (default: 5)
        
    Returns:
    --------
    pd.DataFrame
        Preview of the DataFrame
    """
    return df.head(n_rows)

def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    dict
        Dictionary with basic information
    """
    rows, cols = df.shape
    missing_values = df.isna().sum().sum()
    missing_percent = round((missing_values / (rows * cols)) * 100, 2)
    duplicates = df.duplicated().sum()
    
    # Count data types
    dtype_counts = df.dtypes.value_counts().to_dict()
    numeric_count = dtype_counts.get('int64', 0) + dtype_counts.get('float64', 0)
    categorical_count = dtype_counts.get('object', 0) + dtype_counts.get('category', 0)
    datetime_count = dtype_counts.get('datetime64[ns]', 0)
    boolean_count = dtype_counts.get('bool', 0)
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum()
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