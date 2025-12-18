"""
CSV Data Loader Module
Handles CSV file loading and column extraction using Polars.
"""

import polars as pl
from pathlib import Path
from typing import List, Optional, Tuple


class DataLoader:
    """Handles CSV file loading with Polars for high performance."""
    
    def __init__(self):
        self.df: Optional[pl.DataFrame] = None
        self.file_path: Optional[Path] = None
    
    def load_csv(self, file_path: str) -> pl.DataFrame:
        """
        Load a CSV file using Polars.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Polars DataFrame
        """
        self.file_path = Path(file_path)
        
        # Use scan_csv for lazy evaluation (memory efficient for large files)
        # Then collect to materialize
        self.df = pl.scan_csv(
            file_path,
            infer_schema_length=10000,
            ignore_errors=True
        ).collect()
        
        return self.df
    
    def get_columns(self) -> List[str]:
        """
        Get list of column names from loaded DataFrame.
        
        Returns:
            List of column names
        """
        if self.df is None:
            return []
        return self.df.columns
    
    def get_numeric_columns(self) -> List[str]:
        """
        Get list of numeric column names.
        
        Returns:
            List of numeric column names
        """
        if self.df is None:
            return []
        
        numeric_cols = []
        for col in self.df.columns:
            dtype = self.df[col].dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                numeric_cols.append(col)
        return numeric_cols
    
    def get_unique_values(self, column: str) -> List[str]:
        """
        Get unique values from a column.
        
        Args:
            column: Column name
            
        Returns:
            List of unique values as strings
        """
        if self.df is None or column not in self.df.columns:
            return []
        
        unique_vals = self.df[column].unique().to_list()
        return [str(v) for v in unique_vals if v is not None]
    
    def get_row_count(self) -> int:
        """Get the number of rows in the DataFrame."""
        if self.df is None:
            return 0
        return len(self.df)
    
    def get_data_preview(self, n_rows: int = 5) -> str:
        """
        Get a preview of the data.
        
        Args:
            n_rows: Number of rows to preview
            
        Returns:
            String representation of the preview
        """
        if self.df is None:
            return "No data loaded"
        return str(self.df.head(n_rows))
