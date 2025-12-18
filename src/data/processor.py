"""
Data Processor Module
Handles data cleaning and transformation (stack operation).
"""

import polars as pl
from typing import List, Optional, Tuple


class DataProcessor:
    """Handles data cleaning and transformation operations."""
    
    @staticmethod
    def clean_single_column(
        df: pl.DataFrame,
        group_col: str,
        data_type_col: str,
        value_col: str
    ) -> pl.DataFrame:
        """
        Clean data in single-column mode.
        Removes rows with null/empty values.
        
        Args:
            df: Input DataFrame
            group_col: Name of the group column
            data_type_col: Name of the data type column
            value_col: Name of the value column
            
        Returns:
            Cleaned DataFrame
        """
        # Filter out null values in the value column
        cleaned_df = df.filter(
            pl.col(value_col).is_not_null() & 
            pl.col(value_col).is_not_nan()
        )
        
        # Also filter out null group and data type values
        cleaned_df = cleaned_df.filter(
            pl.col(group_col).is_not_null() &
            pl.col(data_type_col).is_not_null()
        )
        
        return cleaned_df
    
    @staticmethod
    def stack_to_long(
        df: pl.DataFrame,
        group_col: str,
        data_cols: List[str]
    ) -> pl.DataFrame:
        """
        Transform multi-column data to long format (stack operation).
        
        Args:
            df: Input DataFrame with multiple data columns
            group_col: Name of the group column
            data_cols: List of data column names to stack
            
        Returns:
            Long-format DataFrame with columns: [group_col, 'data_type', 'value']
        """
        # Use unpivot (melt) to transform wide to long format
        long_df = df.unpivot(
            index=[group_col],
            on=data_cols,
            variable_name="data_type",
            value_name="value"
        )
        
        # Clean: remove null values
        long_df = long_df.filter(
            pl.col("value").is_not_null() &
            pl.col("value").is_not_nan() &
            pl.col(group_col).is_not_null()
        )
        
        # Ensure value column is float for calculations
        long_df = long_df.with_columns(
            pl.col("value").cast(pl.Float64)
        )
        
        return long_df
    
    @staticmethod
    def prepare_data(
        df: pl.DataFrame,
        mode: str,
        group_col: str,
        data_type_col: Optional[str] = None,
        value_col: Optional[str] = None,
        data_cols: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Prepare data based on mode (single or multi-column).
        
        Args:
            df: Input DataFrame
            mode: 'single' or 'multi'
            group_col: Name of the group column
            data_type_col: For single mode, name of the data type column
            value_col: For single mode, name of the value column
            data_cols: For multi mode, list of data columns to stack
            
        Returns:
            Processed DataFrame with consistent format:
            [group_col, 'data_type', 'value']
        """
        if mode == 'single':
            if data_type_col is None or value_col is None:
                raise ValueError("Single mode requires data_type_col and value_col")
            
            # Rename columns to standard names
            processed_df = df.select([
                pl.col(group_col).alias("group"),
                pl.col(data_type_col).alias("data_type"),
                pl.col(value_col).cast(pl.Float64).alias("value")
            ])
            
            # Clean
            processed_df = processed_df.filter(
                pl.col("value").is_not_null() &
                pl.col("value").is_not_nan() &
                pl.col("group").is_not_null() &
                pl.col("data_type").is_not_null()
            )
            
        elif mode == 'multi':
            if data_cols is None or len(data_cols) == 0:
                raise ValueError("Multi mode requires data_cols")
            
            # Stack columns
            processed_df = DataProcessor.stack_to_long(df, group_col, data_cols)
            
            # Rename group column to standard name
            processed_df = processed_df.rename({group_col: "group"})
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return processed_df
    
    @staticmethod
    def get_data_types(df: pl.DataFrame) -> List[str]:
        """
        Get unique data types from processed DataFrame.
        
        Args:
            df: Processed DataFrame with 'data_type' column
            
        Returns:
            List of unique data type names
        """
        if "data_type" not in df.columns:
            return []
        
        return df["data_type"].unique().sort().to_list()
    
    @staticmethod
    def get_groups(df: pl.DataFrame) -> List[str]:
        """
        Get unique groups from processed DataFrame.
        
        Args:
            df: Processed DataFrame with 'group' column
            
        Returns:
            List of unique group names
        """
        if "group" not in df.columns:
            return []
        
        return [str(g) for g in df["group"].unique().sort().to_list()]
    
    @staticmethod
    def filter_by_data_type(df: pl.DataFrame, data_type: str) -> pl.DataFrame:
        """
        Filter DataFrame for a specific data type.
        
        Args:
            df: Processed DataFrame
            data_type: Data type to filter for
            
        Returns:
            Filtered DataFrame
        """
        return df.filter(pl.col("data_type") == data_type)
