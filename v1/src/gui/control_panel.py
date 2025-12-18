"""
Control Panel Widget
Left side panel with all input controls and buttons.
"""

import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from typing import Callable, List, Optional
from pathlib import Path


class ControlPanel(ttk.Frame):
    """Left side control panel with file selection and processing controls."""
    
    def __init__(
        self,
        parent,
        on_csv_loaded: Callable,
        on_calculate: Callable,
        on_generate_ppt: Callable,
        on_open_ppt: Callable
    ):
        """
        Initialize control panel.
        
        Args:
            parent: Parent widget
            on_csv_loaded: Callback when CSV is loaded
            on_calculate: Callback for calculate button
            on_generate_ppt: Callback for generate PPT button
            on_open_ppt: Callback for open PPT button
        """
        super().__init__(parent, padding=10)
        
        self.on_csv_loaded = on_csv_loaded
        self.on_calculate = on_calculate
        self.on_generate_ppt = on_generate_ppt
        self.on_open_ppt = on_open_ppt
        
        # State
        self.csv_path: Optional[str] = None
        self.ppt_template_path: Optional[str] = None
        self.columns: List[str] = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all control widgets."""
        row = 0
        
        # === CSV File Selection ===
        ttk.Label(self, text="CSV File", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=W, pady=(0, 5)
        )
        row += 1
        
        self.csv_var = tk.StringVar(value="No file selected")
        ttk.Entry(self, textvariable=self.csv_var, state='readonly', width=30).grid(
            row=row, column=0, sticky=EW, padx=(0, 5)
        )
        ttk.Button(self, text="Browse", command=self._browse_csv, bootstyle="primary-outline").grid(
            row=row, column=1, sticky=E
        )
        row += 1
        
        # Separator
        ttk.Separator(self, orient=HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=EW, pady=10)
        row += 1
        
        # === PPT Template Selection ===
        ttk.Label(self, text="PPT Template (Optional)", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=W, pady=(0, 5)
        )
        row += 1
        
        self.ppt_var = tk.StringVar(value="No template (blank)")
        ttk.Entry(self, textvariable=self.ppt_var, state='readonly', width=30).grid(
            row=row, column=0, sticky=EW, padx=(0, 5)
        )
        ttk.Button(self, text="Browse", command=self._browse_ppt, bootstyle="secondary-outline").grid(
            row=row, column=1, sticky=E
        )
        row += 1
        
        # Separator
        ttk.Separator(self, orient=HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=EW, pady=10)
        row += 1
        
        # === Data Mode Selection ===
        ttk.Label(self, text="Data Mode", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=W, pady=(0, 5)
        )
        row += 1
        
        self.mode_var = tk.StringVar(value="single")
        mode_frame = ttk.Frame(self)
        mode_frame.grid(row=row, column=0, columnspan=2, sticky=W)
        
        ttk.Radiobutton(
            mode_frame, text="Single Column", variable=self.mode_var, 
            value="single", command=self._on_mode_change
        ).pack(side=LEFT, padx=(0, 10))
        ttk.Radiobutton(
            mode_frame, text="Multi Column", variable=self.mode_var, 
            value="multi", command=self._on_mode_change
        ).pack(side=LEFT)
        row += 1
        
        # Separator
        ttk.Separator(self, orient=HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=EW, pady=10)
        row += 1
        
        # === Column Selection ===
        ttk.Label(self, text="Column Configuration", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=W, pady=(0, 5)
        )
        row += 1
        
        # Group column
        ttk.Label(self, text="Group Column:").grid(row=row, column=0, sticky=W)
        self.group_col_var = tk.StringVar()
        self.group_col_combo = ttk.Combobox(self, textvariable=self.group_col_var, state='readonly', width=20)
        self.group_col_combo.grid(row=row, column=1, sticky=EW)
        self.group_col_combo.bind('<<ComboboxSelected>>', self._on_group_col_change)
        row += 1
        
        # Control group
        ttk.Label(self, text="Control Group:").grid(row=row, column=0, sticky=W, pady=(5, 0))
        self.control_group_var = tk.StringVar()
        self.control_group_combo = ttk.Combobox(self, textvariable=self.control_group_var, state='readonly', width=20)
        self.control_group_combo.grid(row=row, column=1, sticky=EW, pady=(5, 0))
        row += 1
        
        # Single mode: Data type column
        self.data_type_label = ttk.Label(self, text="Data Type Column:")
        self.data_type_label.grid(row=row, column=0, sticky=W, pady=(5, 0))
        self.data_type_col_var = tk.StringVar()
        self.data_type_col_combo = ttk.Combobox(self, textvariable=self.data_type_col_var, state='readonly', width=20)
        self.data_type_col_combo.grid(row=row, column=1, sticky=EW, pady=(5, 0))
        row += 1
        
        # Single mode: Value column
        self.value_label = ttk.Label(self, text="Value Column:")
        self.value_label.grid(row=row, column=0, sticky=W, pady=(5, 0))
        self.value_col_var = tk.StringVar()
        self.value_col_combo = ttk.Combobox(self, textvariable=self.value_col_var, state='readonly', width=20)
        self.value_col_combo.grid(row=row, column=1, sticky=EW, pady=(5, 0))
        row += 1
        
        # Multi mode: Data columns (Listbox with multiple selection)
        self.data_cols_label = ttk.Label(self, text="Data Columns:")
        self.data_cols_label.grid(row=row, column=0, sticky=NW, pady=(5, 0))
        
        self.data_cols_frame = ttk.Frame(self)
        self.data_cols_frame.grid(row=row, column=1, sticky=EW, pady=(5, 0))
        
        self.data_cols_listbox = tk.Listbox(
            self.data_cols_frame, 
            selectmode=tk.MULTIPLE, 
            height=5, 
            exportselection=False
        )
        self.data_cols_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.data_cols_frame, orient=VERTICAL, command=self.data_cols_listbox.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.data_cols_listbox.config(yscrollcommand=scrollbar.set)
        row += 1
        
        # Separator
        ttk.Separator(self, orient=HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=EW, pady=10)
        row += 1
        
        # === Action Buttons ===
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky=EW, pady=(5, 0))
        
        self.calculate_btn = ttk.Button(
            btn_frame, 
            text="Start Calculation", 
            command=self._on_calculate,
            bootstyle="success",
            state='disabled'
        )
        self.calculate_btn.pack(fill=X, pady=2)
        
        self.generate_ppt_btn = ttk.Button(
            btn_frame, 
            text="Generate PPT Report", 
            command=self._on_generate_ppt,
            bootstyle="info",
            state='disabled'
        )
        self.generate_ppt_btn.pack(fill=X, pady=2)
        
        self.open_ppt_btn = ttk.Button(
            btn_frame, 
            text="Open PPT", 
            command=self._on_open_ppt,
            bootstyle="secondary",
            state='disabled'
        )
        self.open_ppt_btn.pack(fill=X, pady=2)
        row += 1
        
        # === Progress Bar ===
        ttk.Separator(self, orient=HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=EW, pady=10)
        row += 1
        
        ttk.Label(self, text="Progress", font=('Helvetica', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=W, pady=(0, 5)
        )
        row += 1
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self, 
            variable=self.progress_var, 
            maximum=100,
            bootstyle="success-striped"
        )
        self.progress_bar.grid(row=row, column=0, columnspan=2, sticky=EW)
        row += 1
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, font=('Helvetica', 9)).grid(
            row=row, column=0, columnspan=2, sticky=W, pady=(5, 0)
        )
        
        # Configure column weights
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        
        # Initial mode display
        self._on_mode_change()
    
    def _browse_csv(self):
        """Open file dialog for CSV selection."""
        path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            self.csv_path = path
            self.csv_var.set(Path(path).name)
            self.on_csv_loaded(path)
    
    def _browse_ppt(self):
        """Open file dialog for PPT template selection."""
        path = filedialog.askopenfilename(
            title="Select PPT Template",
            filetypes=[("PowerPoint Files", "*.pptx"), ("All Files", "*.*")]
        )
        if path:
            self.ppt_template_path = path
            self.ppt_var.set(Path(path).name)
    
    def _on_mode_change(self):
        """Handle mode change between single and multi."""
        mode = self.mode_var.get()
        
        if mode == "single":
            # Show single mode widgets
            self.data_type_label.grid()
            self.data_type_col_combo.grid()
            self.value_label.grid()
            self.value_col_combo.grid()
            # Hide multi mode widgets
            self.data_cols_label.grid_remove()
            self.data_cols_frame.grid_remove()
        else:
            # Hide single mode widgets
            self.data_type_label.grid_remove()
            self.data_type_col_combo.grid_remove()
            self.value_label.grid_remove()
            self.value_col_combo.grid_remove()
            # Show multi mode widgets
            self.data_cols_label.grid()
            self.data_cols_frame.grid()
    
    def _on_group_col_change(self, event=None):
        """Handle group column selection change."""
        # This will be called from outside to update control group options
        pass
    
    def _on_calculate(self):
        """Handle calculate button click."""
        self.on_calculate()
    
    def _on_generate_ppt(self):
        """Handle generate PPT button click."""
        self.on_generate_ppt()
    
    def _on_open_ppt(self):
        """Handle open PPT button click."""
        self.on_open_ppt()
    
    def update_columns(self, columns: List[str]):
        """
        Update column dropdowns with new columns.
        
        Args:
            columns: List of column names
        """
        self.columns = columns
        
        # Update all column comboboxes
        self.group_col_combo['values'] = columns
        self.data_type_col_combo['values'] = columns
        self.value_col_combo['values'] = columns
        
        # Update multi-column listbox
        self.data_cols_listbox.delete(0, tk.END)
        for col in columns:
            self.data_cols_listbox.insert(tk.END, col)
        
        # Enable calculate button
        self.calculate_btn.config(state='normal')
    
    def update_control_groups(self, groups: List[str]):
        """
        Update control group dropdown.
        
        Args:
            groups: List of group names
        """
        self.control_group_combo['values'] = groups
        if groups:
            self.control_group_combo.set(groups[0])
    
    def get_settings(self) -> dict:
        """
        Get current settings.
        
        Returns:
            Dictionary with all settings
        """
        # Get selected data columns for multi mode
        selected_indices = self.data_cols_listbox.curselection()
        selected_data_cols = [self.data_cols_listbox.get(i) for i in selected_indices]
        
        return {
            'csv_path': self.csv_path,
            'ppt_template_path': self.ppt_template_path,
            'mode': self.mode_var.get(),
            'group_col': self.group_col_var.get(),
            'control_group': self.control_group_var.get(),
            'data_type_col': self.data_type_col_var.get(),
            'value_col': self.value_col_var.get(),
            'data_cols': selected_data_cols
        }
    
    def set_progress(self, value: float, status: str = None):
        """
        Update progress bar and status.
        
        Args:
            value: Progress value (0-100)
            status: Optional status text
        """
        self.progress_var.set(value)
        if status:
            self.status_var.set(status)
        self.update_idletasks()
    
    def enable_ppt_buttons(self, enable: bool = True):
        """Enable or disable PPT-related buttons."""
        state = 'normal' if enable else 'disabled'
        self.generate_ppt_btn.config(state=state)
    
    def enable_open_ppt(self, enable: bool = True):
        """Enable or disable open PPT button."""
        state = 'normal' if enable else 'disabled'
        self.open_ppt_btn.config(state=state)
