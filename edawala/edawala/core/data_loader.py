"""
Data loading utilities for EDAwala
"""
import pandas as pd
from typing import Union, Optional, Dict, Any
import io
import os


class DataLoader:
    """
    Utility class for loading data from various sources
    """
    
    @staticmethod
    def load_csv(
        file_path_or_buffer: Union[str, io.BytesIO],
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from a CSV file or buffer
        
        Parameters:
        -----------
        file_path_or_buffer : str or BytesIO
            Path to CSV file or BytesIO object
        **kwargs : 
            Additional arguments to pass to pandas.read_csv
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        df = pd.read_csv(file_path_or_buffer, **kwargs)
        
        # Set the dataframe name attribute based on file name if possible
        if isinstance(file_path_or_buffer, str):
            df.name = os.path.basename(file_path_or_buffer)
        else:
            df.name = "Uploaded Dataset"
            
        return df
    
    @staticmethod
    def load_excel(
        file_path_or_buffer: Union[str, io.BytesIO],
        sheet_name: Optional[Union[str, int]] = 0,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load data from an Excel file
        
        Parameters:
        -----------
        file_path_or_buffer : str or BytesIO
            Path to Excel file or BytesIO object
        sheet_name : str or int, optional
            Name or index of the sheet to load (default: 0)
        **kwargs : 
            Additional arguments to pass to pandas.read_excel
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        df = pd.read_excel(file_path_or_buffer, sheet_name=sheet_name, **kwargs)
        
        # Set the dataframe name attribute
        if isinstance(file_path_or_buffer, str):
            df.name = os.path.basename(file_path_or_buffer)
            if isinstance(sheet_name, str):
                df.name += f" - {sheet_name}"
        else:
            df.name = "Uploaded Dataset"
            
        return df
    
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """
        Detect the encoding of a file
        
        Parameters:
        -----------
        file_path : str
            Path to the file
            
        Returns:
        --------
        str
            Detected encoding
        """
        try:
            import chardet
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read())
            return result['encoding']
        except ImportError:
            print("chardet package not found. Using utf-8 as default encoding.")
            return 'utf-8'