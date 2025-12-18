"""
Chart Plotter Module
Creates combined visualizations: boxplot with swarm, Q-Q plot, and stats table.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from collections import defaultdict

from ..stats.calculator import DataTypeStats, GroupStats


# Color palette - control group is always blue
COLORS = {
    'control': '#3498db',  # Blue
    'palette': [
        '#e74c3c',  # Red
        '#2ecc71',  # Green
        '#9b59b6',  # Purple
        '#f39c12',  # Orange
        '#1abc9c',  # Teal
        '#e91e63',  # Pink
        '#00bcd4',  # Cyan
        '#ff5722',  # Deep Orange
        '#795548',  # Brown
        '#607d8b',  # Blue Grey
    ]
}


class ChartPlotter:
    """Creates scientific visualization charts."""
    
    # Font settings for scientific style
    FONT_FAMILY = 'sans-serif'
    TITLE_SIZE = 12
    LABEL_SIZE = 10
    TICK_SIZE = 9
    
    # Higher DPI for better resolution
    FIGURE_DPI = 150
    
    def __init__(self):
        """Initialize plotter with matplotlib settings."""
        # Set scientific style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.family'] = self.FONT_FAMILY
        plt.rcParams['axes.titlesize'] = self.TITLE_SIZE
        plt.rcParams['axes.labelsize'] = self.LABEL_SIZE
        plt.rcParams['xtick.labelsize'] = self.TICK_SIZE
        plt.rcParams['ytick.labelsize'] = self.TICK_SIZE
    
    def get_group_color(self, group: str, control_group: str, group_index: int) -> str:
        """
        Get color for a group.
        
        Args:
            group: Group name
            control_group: Name of control group
            group_index: Index of group (for non-control groups)
            
        Returns:
            Hex color string
        """
        if group == control_group:
            return COLORS['control']
        
        palette_idx = group_index % len(COLORS['palette'])
        return COLORS['palette'][palette_idx]
    
    def beeswarm_positions(
        self,
        y_values: np.ndarray,
        center: float,
        width: float = 0.3,
        point_size: float = 20
    ) -> np.ndarray:
        """
        Calculate x-positions for beeswarm plot.
        ONLY points with EXACTLY the same y value are spread symmetrically.
        All unique values stay centered on the group axis.
        
        Args:
            y_values: Y values for points
            center: Center x position
            width: Maximum spread width
            point_size: Size of points for overlap calculation
            
        Returns:
            Array of x positions
        """
        n = len(y_values)
        if n == 0:
            return np.array([])
        
        # Start with all points centered
        positions = np.full(n, center)
        
        # Find duplicate values and spread them
        # Round to handle floating point precision issues
        rounded_values = np.round(y_values, decimals=6)
        
        # Count occurrences of each value
        unique_values, inverse_indices, counts = np.unique(
            rounded_values, return_inverse=True, return_counts=True
        )
        
        # Only process values that appear more than once
        for val_idx, count in enumerate(counts):
            if count > 1:
                # Find all points with this value
                point_indices = np.where(inverse_indices == val_idx)[0]
                
                # Spread them symmetrically around center
                offsets = np.linspace(-width/2, width/2, count)
                for i, idx in enumerate(point_indices):
                    positions[idx] = center + offsets[i]
        
        return positions
    
    def plot_boxplot_swarm(
        self,
        ax: plt.Axes,
        data_by_group: Dict[str, np.ndarray],
        control_group: str,
        ordered_groups: List[str]
    ) -> None:
        """
        Plot boxplot with beeswarm scatter overlay.
        
        Args:
            ax: Matplotlib axes
            data_by_group: Dictionary mapping group name to values
            control_group: Name of control group
            ordered_groups: Groups in display order (control first)
        """
        positions = list(range(len(ordered_groups)))
        
        # Calculate means for connecting line
        means = []
        
        non_control_idx = 0
        for i, group in enumerate(ordered_groups):
            values = data_by_group.get(group, np.array([]))
            if len(values) == 0:
                means.append(np.nan)
                continue
            
            color = self.get_group_color(group, control_group, non_control_idx)
            if group != control_group:
                non_control_idx += 1
            
            # Draw boxplot
            bp = ax.boxplot(
                [values],
                positions=[i],
                widths=0.5,
                patch_artist=True,
                showfliers=False
            )
            
            # Style boxplot - semi-transparent
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.3)
                patch.set_edgecolor(color)
            
            for element in ['whiskers', 'caps', 'medians']:
                for line in bp[element]:
                    line.set_color(color)
                    line.set_linewidth(1.5)
            
            # Draw scatter points with beeswarm effect
            # Sample if too many points
            plot_values = values
            if len(values) > 500:
                sample_idx = np.random.choice(len(values), 500, replace=False)
                plot_values = values[sample_idx]
            
            x_positions = self.beeswarm_positions(plot_values, i, width=0.35)
            ax.scatter(
                x_positions,
                plot_values,
                c=color,
                s=12,
                alpha=0.6,
                edgecolors='white',
                linewidths=0.3,
                zorder=3
            )
            
            means.append(np.mean(values))
        
        # Connect means with line
        valid_means = [(i, m) for i, m in enumerate(means) if not np.isnan(m)]
        if len(valid_means) > 1:
            xs, ys = zip(*valid_means)
            ax.plot(xs, ys, 'k-', linewidth=1.5, alpha=0.7, zorder=4, marker='o', markersize=4)
        
        # Style axes
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered_groups, rotation=45, ha='right')
        ax.set_xlabel('Group')
        ax.set_ylabel('Value')
        ax.set_title('Distribution by Group')
    
    def plot_qq(
        self,
        ax: plt.Axes,
        data_by_group: Dict[str, np.ndarray],
        control_group: str,
        ordered_groups: List[str],
        y_limits: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Plot Q-Q plot (quantile plot) with scatter points.
        Full range from 0% to 100% to show all values including extremes.
        
        Args:
            ax: Matplotlib axes
            data_by_group: Dictionary mapping group name to values
            control_group: Name of control group
            ordered_groups: Groups in display order
            y_limits: Optional Y-axis limits to match boxplot
        """
        # Full quantile range from 0 to 1 (100 points)
        quantiles = np.linspace(0, 1, 101)
        
        non_control_idx = 0
        for group in ordered_groups:
            values = data_by_group.get(group, np.array([]))
            if len(values) == 0:
                continue
            
            color = self.get_group_color(group, control_group, non_control_idx)
            if group != control_group:
                non_control_idx += 1
            
            # Calculate quantile values (0% = min, 100% = max)
            q_values = np.percentile(values, quantiles * 100)
            
            # Plot line
            ax.plot(quantiles, q_values, color=color, linewidth=1.5, label=group, alpha=0.8)
            
            # Add scatter points including extremes
            # 0%, 5%, 10%, ..., 95%, 100%
            scatter_quantiles = np.linspace(0, 1, 21)
            scatter_values = np.percentile(values, scatter_quantiles * 100)
            ax.scatter(
                scatter_quantiles,
                scatter_values,
                c=color,
                s=20,
                alpha=0.8,
                edgecolors='white',
                linewidths=0.5,
                zorder=3
            )
        
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Value')
        ax.set_title('Quantile Plot')
        ax.set_xlim(0, 1)
        
        if y_limits:
            ax.set_ylim(y_limits)
    
    def plot_stats_table(
        self,
        ax: plt.Axes,
        stats: DataTypeStats
    ) -> None:
        """
        Plot statistics as a table.
        
        Args:
            ax: Matplotlib axes
            stats: DataTypeStats object
        """
        ax.axis('off')
        
        # Prepare table data
        headers = ['Group', 'Count', 'Mean', 'Median', 'Std', 'P95', 'P05', 'Std Diff', 'P-value']
        
        cell_data = []
        cell_colors = []
        
        non_control_idx = 0
        for group_name in stats.get_ordered_groups():
            gs = stats.group_stats[group_name]
            
            row = [
                gs.group_name,
                str(gs.count),
                f"{gs.mean:.3f}" if not np.isnan(gs.mean) else "N/A",
                f"{gs.median:.3f}" if not np.isnan(gs.median) else "N/A",
                f"{gs.std:.3f}" if not np.isnan(gs.std) else "N/A",
                f"{gs.p95:.3f}" if not np.isnan(gs.p95) else "N/A",
                f"{gs.p05:.3f}" if not np.isnan(gs.p05) else "N/A",
                f"{gs.std_diff_from_control:.3f}" if gs.std_diff_from_control is not None and not np.isnan(gs.std_diff_from_control) else "-",
                f"{gs.p_value:.4f}" if gs.p_value is not None and not np.isnan(gs.p_value) else "-"
            ]
            cell_data.append(row)
            
            # Color based on significance
            color = self.get_group_color(group_name, stats.control_group, non_control_idx)
            if group_name != stats.control_group:
                non_control_idx += 1
            
            row_colors = [(1, 1, 1, 0.3)] * len(headers)  # Light background
            # Highlight p-value cell if significant
            if gs.is_significant:
                row_colors[-1] = (0.9, 0.5, 0.5, 0.5)  # Red tint for significant
            cell_colors.append(row_colors)
        
        # Create table
        table = ax.table(
            cellText=cell_data,
            colLabels=headers,
            cellColours=cell_colors,
            colColours=[('#f0f0f0')] * len(headers),
            loc='center',
            cellLoc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    
    def create_combined_figure(
        self,
        data_type: str,
        data_by_group: Dict[str, np.ndarray],
        stats: DataTypeStats
    ) -> Figure:
        """
        Create combined figure with boxplot, Q-Q plot, and stats table.
        
        Args:
            data_type: Name of the data type
            data_by_group: Dictionary mapping group name to values
            stats: DataTypeStats object
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure with custom layout and higher DPI
        fig = plt.figure(figsize=(12, 9), dpi=self.FIGURE_DPI)
        
        # Grid: 2 rows, 2 columns
        # Top row: boxplot (left) and Q-Q plot (right)
        # Bottom row: table (spanning both columns)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        ax_box = fig.add_subplot(gs[0, 0])
        ax_qq = fig.add_subplot(gs[0, 1])
        ax_table = fig.add_subplot(gs[1, :])
        
        ordered_groups = stats.get_ordered_groups()
        
        # Plot boxplot with swarm
        self.plot_boxplot_swarm(ax_box, data_by_group, stats.control_group, ordered_groups)
        
        # Get y-limits from boxplot to share with Q-Q plot
        y_limits = ax_box.get_ylim()
        
        # Plot Q-Q with scatter points
        self.plot_qq(ax_qq, data_by_group, stats.control_group, ordered_groups, y_limits)
        
        # Plot table
        self.plot_stats_table(ax_table, stats)
        
        # Add shared legend
        handles = []
        non_control_idx = 0
        for group in ordered_groups:
            color = self.get_group_color(group, stats.control_group, non_control_idx)
            if group != stats.control_group:
                non_control_idx += 1
            handles.append(mpatches.Patch(color=color, label=group))
        
        fig.legend(
            handles=handles,
            loc='upper center',
            ncol=min(len(handles), 5),
            bbox_to_anchor=(0.5, 0.98),
            fontsize=9
        )
        
        # Main title
        fig.suptitle(f'Analysis: {data_type}', fontsize=14, fontweight='bold', y=1.02)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                plt.tight_layout()
            except Exception:
                pass
        
        return fig
    
    def figure_to_image(self, fig: Figure) -> bytes:
        """
        Convert figure to PNG bytes.
        
        Args:
            fig: Matplotlib Figure
            
        Returns:
            PNG image as bytes
        """
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=self.FIGURE_DPI, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        return buf.getvalue()
    
    def save_figure(self, fig: Figure, path: str) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib Figure
            path: Output file path
        """
        fig.savefig(path, dpi=self.FIGURE_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
