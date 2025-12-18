"""
Chartify Pro Main Application
Main window with control panel and chart viewer.
"""

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
import threading
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import polars as pl
import numpy as np

from .control_panel import ControlPanel
from .chart_viewer import ChartViewer
from ..data.loader import DataLoader
from ..data.processor import DataProcessor
from ..stats.calculator import StatsCalculator, DataTypeStats
from ..charts.plotter import ChartPlotter, create_figure_multiprocess
from ..report.ppt_generator import PPTGenerator


class ChartifyApp:
    """Main application window."""
    
    WINDOW_TITLE = "Chartify Pro"
    WINDOW_SIZE = "1400x800"
    MIN_WIDTH = 1200
    MIN_HEIGHT = 700
    
    def __init__(self):
        """Initialize the application."""
        # Create main window with ttkbootstrap theme
        self.root = ttk.Window(
            title=self.WINDOW_TITLE,
            themename="cosmo",  # Modern, clean theme
            size=(1400, 800),
            minsize=(self.MIN_WIDTH, self.MIN_HEIGHT)
        )
        
        # Data state
        self.loader = DataLoader()
        self.processor = DataProcessor()
        self.plotter = ChartPlotter()
        self.ppt_generator = PPTGenerator()
        
        self.df: Optional[pl.DataFrame] = None
        self.processed_df: Optional[pl.DataFrame] = None
        self.stats: Dict[str, DataTypeStats] = {}
        self.figures: Dict[str, Figure] = {}
        self.png_bytes: Dict[str, bytes] = {}  # Store PNG bytes for fast display
        self.ppt_path: Optional[str] = None
        
        self._create_layout()
    
    def _create_layout(self):
        """Create main layout with control panel and chart viewer."""
        # Configure root grid
        self.root.columnconfigure(0, weight=0, minsize=350)  # Control panel
        self.root.columnconfigure(1, weight=1)  # Chart viewer
        self.root.rowconfigure(0, weight=1)
        
        # Create control panel (left side)
        self.control_panel = ControlPanel(
            self.root,
            on_csv_loaded=self._on_csv_loaded,
            on_calculate=self._on_calculate,
            on_generate_ppt=self._on_generate_ppt,
            on_open_ppt=self._on_open_ppt
        )
        self.control_panel.grid(row=0, column=0, sticky=NSEW, padx=5, pady=5)
        
        # Create chart viewer (right side)
        self.chart_viewer = ChartViewer(self.root)
        self.chart_viewer.grid(row=0, column=1, sticky=NSEW, padx=5, pady=5)
    
    def _on_csv_loaded(self, path: str):
        """Handle CSV file loaded event."""
        try:
            self.df = self.loader.load_csv(path)
            columns = self.loader.get_columns()
            self.control_panel.update_columns(columns)
            
            # Show info
            row_count = self.loader.get_row_count()
            self.control_panel.set_progress(0, f"Loaded {row_count:,} rows, {len(columns)} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def _on_calculate(self):
        """Handle calculate button click."""
        settings = self.control_panel.get_settings()
        
        # Validate settings
        if not settings['csv_path']:
            messagebox.showwarning("Warning", "Please select a CSV file first.")
            return
        
        if not settings['group_col']:
            messagebox.showwarning("Warning", "Please select a group column.")
            return
        
        if not settings['control_group']:
            messagebox.showwarning("Warning", "Please select a control group.")
            return
        
        if settings['mode'] == 'single':
            if not settings['data_type_col'] or not settings['value_col']:
                messagebox.showwarning("Warning", "Please select data type and value columns.")
                return
        else:
            if not settings['data_cols']:
                messagebox.showwarning("Warning", "Please select at least one data column.")
                return
        
        # Run calculation in background thread
        thread = threading.Thread(target=self._run_calculation, args=(settings,))
        thread.daemon = True
        thread.start()
    
    def _run_calculation(self, settings: dict):
        """Run calculation in background thread."""
        try:
            self.control_panel.set_progress(5, "Processing data...")
            
            # Process data
            if settings['mode'] == 'single':
                self.processed_df = self.processor.prepare_data(
                    self.df,
                    mode='single',
                    group_col=settings['group_col'],
                    data_type_col=settings['data_type_col'],
                    value_col=settings['value_col']
                )
            else:
                self.processed_df = self.processor.prepare_data(
                    self.df,
                    mode='multi',
                    group_col=settings['group_col'],
                    data_cols=settings['data_cols']
                )
            
            self.control_panel.set_progress(20, "Calculating statistics...")
            
            # Calculate statistics
            def stats_progress(p):
                self.control_panel.set_progress(20 + p * 0.3, f"Calculating statistics... {p:.0f}%")
            
            self.stats = StatsCalculator.compute_all_stats(
                self.processed_df,
                settings['control_group'],
                progress_callback=stats_progress
            )
            
            self.control_panel.set_progress(50, "Generating charts (multiprocess)...")
            
            # Generate figures using multiprocessing
            self.figures = {}
            data_types = list(self.stats.keys())
            total = len(data_types)
            
            # Prepare data for multiprocessing
            tasks = []
            for data_type in data_types:
                # Get data for this type
                data_by_group = {}
                for group in self.stats[data_type].get_ordered_groups():
                    values = StatsCalculator.get_values_for_group(
                        self.processed_df, data_type, group
                    )
                    data_by_group[group] = values
                
                # Convert stats to serializable dict
                stats_dict = {}
                for name, gs in self.stats[data_type].group_stats.items():
                    stats_dict[name] = {
                        'group_name': gs.group_name,
                        'count': gs.count,
                        'mean': gs.mean,
                        'median': gs.median,
                        'std': gs.std,
                        'variance': gs.variance,
                        'p95': gs.p95,
                        'p05': gs.p05,
                        'std_diff_from_control': gs.std_diff_from_control,
                        'p_value': gs.p_value,
                        'is_significant': gs.is_significant
                    }
                
                tasks.append((
                    data_type,
                    data_by_group,
                    stats_dict,
                    self.stats[data_type].control_group,
                    self.stats[data_type].get_ordered_groups()
                ))
            
            # Execute in parallel using ProcessPoolExecutor
            completed = 0
            self.png_bytes = {}  # Store PNG bytes
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(create_figure_multiprocess, *task): task[0]
                    for task in tasks
                }
                
                for future in as_completed(futures):
                    data_type = futures[future]
                    try:
                        dt, png_bytes, has_significant = future.result()
                        
                        # Store PNG bytes directly - no Figure conversion needed!
                        self.png_bytes[dt] = png_bytes
                        
                        completed += 1
                        progress = 50 + completed / total * 40
                        self.control_panel.set_progress(progress, f"Generating charts... {completed}/{total}")
                        
                    except Exception as e:
                        print(f"Error processing {data_type}: {e}")
                        completed += 1
            
            self.control_panel.set_progress(90, "Displaying charts...")
            
            # Display in GUI (must be done in main thread)
            self.root.after(0, lambda: self._display_charts())
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Calculation failed: {str(e)}"))
            self.control_panel.set_progress(0, "Error occurred")
    
    def _display_charts(self):
        """Display charts in viewer."""
        def display_progress(p):
            self.control_panel.set_progress(90 + p * 0.1, f"Displaying... {p:.0f}%")
        
        # Use fast PNG bytes display
        self.chart_viewer.display_png_bytes(self.png_bytes, progress_callback=display_progress, stats=self.stats)
        
        self.control_panel.set_progress(100, f"Complete! {len(self.png_bytes)} charts generated.")
        self.control_panel.enable_ppt_buttons(True)
    
    def _on_generate_ppt(self):
        """Handle generate PPT button click."""
        if not self.png_bytes or not self.stats:
            messagebox.showwarning("Warning", "Please run calculation first.")
            return
        
        settings = self.control_panel.get_settings()
        
        # Run in background thread
        thread = threading.Thread(target=self._run_ppt_generation, args=(settings,))
        thread.daemon = True
        thread.start()
    
    def _run_ppt_generation(self, settings: dict):
        """Run PPT generation in background thread."""
        try:
            self.control_panel.set_progress(0, "Generating PPT report...")
            
            def ppt_progress(p):
                self.control_panel.set_progress(p, f"Generating PPT... {p:.0f}%")
            
            self.ppt_path = self.ppt_generator.generate_report_from_bytes(
                self.png_bytes,
                self.stats,
                settings['csv_path'],
                settings['ppt_template_path'],
                progress_callback=ppt_progress
            )
            
            self.root.after(0, lambda: self._on_ppt_complete())
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"PPT generation failed: {str(e)}"))
            self.control_panel.set_progress(0, "Error occurred")
    
    def _on_ppt_complete(self):
        """Handle PPT generation complete."""
        self.control_panel.set_progress(100, f"PPT saved: {Path(self.ppt_path).name}")
        self.control_panel.enable_open_ppt(True)
        messagebox.showinfo("Success", f"PPT report generated:\n{self.ppt_path}")
    
    def _on_open_ppt(self):
        """Handle open PPT button click."""
        if not self.ppt_path or not Path(self.ppt_path).exists():
            messagebox.showwarning("Warning", "No PPT file available.")
            return
        
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', self.ppt_path])
            elif sys.platform == 'win32':  # Windows
                os.startfile(self.ppt_path)
            else:  # Linux
                subprocess.run(['xdg-open', self.ppt_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {str(e)}")
    
    def run(self):
        """Run the application main loop."""
        # Bind group column change to update control groups
        def on_group_col_selected(event=None):
            group_col = self.control_panel.group_col_var.get()
            if group_col and self.df is not None:
                groups = self.loader.get_unique_values(group_col)
                self.control_panel.update_control_groups(groups)
        
        self.control_panel.group_col_combo.bind('<<ComboboxSelected>>', on_group_col_selected)
        
        self.root.mainloop()
