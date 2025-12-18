"""
Statistics Calculator Module
Handles statistical computations including descriptive stats and t-tests.
"""

import polars as pl
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings


@dataclass
class GroupStats:
    """Statistics for a single group."""
    group_name: str
    count: int
    mean: float
    median: float
    std: float
    variance: float
    p95: float
    p05: float
    std_diff_from_control: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: bool = False  # True if p <= 0.05


@dataclass
class DataTypeStats:
    """Statistics for a data type across all groups."""
    data_type: str
    control_group: str
    group_stats: Dict[str, GroupStats]
    
    def get_ordered_groups(self) -> List[str]:
        """Get groups ordered with control first."""
        groups = list(self.group_stats.keys())
        if self.control_group in groups:
            groups.remove(self.control_group)
            groups.insert(0, self.control_group)
        return groups
    
    def has_significant_results(self) -> bool:
        """Check if any group has significant p-value."""
        for name, gs in self.group_stats.items():
            if name != self.control_group and gs.is_significant:
                return True
        return False


class StatsCalculator:
    """Handles statistical calculations with multi-threading support."""
    
    SIGNIFICANCE_THRESHOLD = 0.05
    
    @staticmethod
    def compute_descriptive_stats(values: np.ndarray) -> Dict[str, float]:
        """
        Compute descriptive statistics for an array of values.
        
        Args:
            values: NumPy array of values
            
        Returns:
            Dictionary with count, mean, median, std, variance, p95, p05
        """
        if len(values) == 0:
            return {
                'count': 0,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'variance': np.nan,
                'p95': np.nan,
                'p05': np.nan
            }
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
            'variance': np.var(values, ddof=1) if len(values) > 1 else 0.0,
            'p95': np.percentile(values, 95),
            'p05': np.percentile(values, 5)
        }
    
    @staticmethod
    def perform_ttest(
        group_values: np.ndarray,
        control_values: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Perform independent samples t-test.
        
        Args:
            group_values: Values from the test group
            control_values: Values from the control group
            
        Returns:
            Tuple of (p_value, is_significant)
        """
        if len(group_values) < 2 or len(control_values) < 2:
            return (np.nan, False)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, p_value = stats.ttest_ind(group_values, control_values, equal_var=False)
                is_significant = p_value <= StatsCalculator.SIGNIFICANCE_THRESHOLD
                return (p_value, is_significant)
            except Exception:
                return (np.nan, False)
    
    @staticmethod
    def compute_group_stats(
        df: pl.DataFrame,
        group_name: str,
        control_values: Optional[np.ndarray] = None,
        control_std: Optional[float] = None,
        control_mean: Optional[float] = None
    ) -> GroupStats:
        """
        Compute statistics for a single group.
        
        Args:
            df: DataFrame filtered for this group
            group_name: Name of the group
            control_values: Values from control group for t-test
            control_std: Standard deviation of control group
            
        Returns:
            GroupStats object
        """
        values = df["value"].to_numpy()
        desc_stats = StatsCalculator.compute_descriptive_stats(values)
        
        # Compute comparison with control if available
        # Calculate standardized mean difference: (group_mean - control_mean) / control_std
        std_diff = None
        p_value = None
        is_significant = False
        
        if control_std is not None and control_mean is not None and control_std > 0:
            std_diff = (desc_stats['mean'] - control_mean) / control_std
        
        if control_values is not None and len(control_values) > 0:
            p_value, is_significant = StatsCalculator.perform_ttest(values, control_values)
        
        return GroupStats(
            group_name=group_name,
            count=int(desc_stats['count']),
            mean=desc_stats['mean'],
            median=desc_stats['median'],
            std=desc_stats['std'],
            variance=desc_stats['variance'],
            p95=desc_stats['p95'],
            p05=desc_stats['p05'],
            std_diff_from_control=std_diff,
            p_value=p_value,
            is_significant=is_significant
        )
    
    @staticmethod
    def compute_data_type_stats(
        df: pl.DataFrame,
        data_type: str,
        control_group: str
    ) -> DataTypeStats:
        """
        Compute statistics for all groups within a data type.
        
        Args:
            df: DataFrame filtered for this data type
            data_type: Name of the data type
            control_group: Name of the control group
            
        Returns:
            DataTypeStats object
        """
        groups = [str(g) for g in df["group"].unique().to_list()]
        group_stats: Dict[str, GroupStats] = {}
        
        # First compute control group stats
        control_df = df.filter(pl.col("group").cast(pl.Utf8) == control_group)
        control_values = control_df["value"].to_numpy() if len(control_df) > 0 else np.array([])
        
        control_stats = StatsCalculator.compute_group_stats(
            control_df, control_group, None, None
        )
        group_stats[control_group] = control_stats
        control_std = control_stats.std
        
        # Compute stats for other groups
        for group_name in groups:
            if group_name == control_group:
                continue
            
            group_df = df.filter(pl.col("group").cast(pl.Utf8) == group_name)
            gs = StatsCalculator.compute_group_stats(
                group_df, group_name, control_values, control_std, control_stats.mean
            )
            group_stats[group_name] = gs
        
        return DataTypeStats(
            data_type=data_type,
            control_group=control_group,
            group_stats=group_stats
        )
    
    @staticmethod
    def compute_all_stats(
        df: pl.DataFrame,
        control_group: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, DataTypeStats]:
        """
        Compute statistics for all data types.
        
        Args:
            df: Processed DataFrame with 'group', 'data_type', 'value' columns
            control_group: Name of the control group
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping data_type -> DataTypeStats
        """
        data_types = df["data_type"].unique().to_list()
        total = len(data_types)
        results: Dict[str, DataTypeStats] = {}
        
        for i, data_type in enumerate(data_types):
            # Filter for this data type
            type_df = df.filter(pl.col("data_type") == data_type)
            
            # Compute stats
            stats = StatsCalculator.compute_data_type_stats(
                type_df, str(data_type), control_group
            )
            results[str(data_type)] = stats
            
            # Update progress
            if progress_callback:
                progress_callback((i + 1) / total * 100)
        
        return results
    
    @staticmethod
    def get_values_for_group(
        df: pl.DataFrame,
        data_type: str,
        group: str
    ) -> np.ndarray:
        """
        Get values for a specific data type and group.
        
        Args:
            df: Processed DataFrame
            data_type: Data type to filter
            group: Group to filter
            
        Returns:
            NumPy array of values
        """
        filtered = df.filter(
            (pl.col("data_type") == data_type) &
            (pl.col("group").cast(pl.Utf8) == group)
        )
        return filtered["value"].to_numpy()
    
    @staticmethod
    def stats_to_table_data(stats: DataTypeStats) -> List[List[Any]]:
        """
        Convert DataTypeStats to table-ready data.
        
        Args:
            stats: DataTypeStats object
            
        Returns:
            List of rows for table display
        """
        headers = ['Group', 'Count', 'Mean', 'Median', 'Std', 'P95', 'P05', 'Std Diff', 'P-value']
        rows = [headers]
        
        for group_name in stats.get_ordered_groups():
            gs = stats.group_stats[group_name]
            row = [
                gs.group_name,
                gs.count,
                f"{gs.mean:.4f}" if not np.isnan(gs.mean) else "N/A",
                f"{gs.median:.4f}" if not np.isnan(gs.median) else "N/A",
                f"{gs.std:.4f}" if not np.isnan(gs.std) else "N/A",
                f"{gs.p95:.4f}" if not np.isnan(gs.p95) else "N/A",
                f"{gs.p05:.4f}" if not np.isnan(gs.p05) else "N/A",
                f"{gs.std_diff_from_control:.4f}" if gs.std_diff_from_control is not None and not np.isnan(gs.std_diff_from_control) else "-",
                f"{gs.p_value:.4f}" if gs.p_value is not None and not np.isnan(gs.p_value) else "-"
            ]
            rows.append(row)
        
        return rows
