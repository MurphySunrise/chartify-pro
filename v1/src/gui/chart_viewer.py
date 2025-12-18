"""
Chart Viewer Widget
Right side scrollable panel for displaying charts in dynamic grid layout.
Supports click-to-zoom functionality with overlay (no layout recalculation).
"""

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from typing import List, Dict, Optional, Callable
from matplotlib.figure import Figure
from io import BytesIO
from PIL import Image, ImageTk
import warnings


class ChartViewer(ttk.Frame):
    """Scrollable chart display area with dynamic grid layout and zoom."""
    
    # Chart dimensions
    CHART_MIN_WIDTH = 350
    CHART_MIN_HEIGHT = 280
    CHART_PADDING = 10
    
    # Higher resolution for display
    DISPLAY_DPI = 150
    
    def __init__(self, parent):
        """
        Initialize chart viewer.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.figures: Dict[str, Figure] = {}
        self.photo_images: Dict[str, ImageTk.PhotoImage] = {}
        self.full_images: Dict[str, ImageTk.PhotoImage] = {}
        self.chart_widgets: Dict[str, ttk.Label] = {}
        
        # Zoom state - use overlay instead of hiding
        self.zoomed_chart: Optional[str] = None
        self.zoom_overlay: Optional[tk.Frame] = None
        
        self._create_widgets()
        
        # Bind resize event
        self.bind('<Configure>', self._on_resize)
        self._last_width = 0
    
    def _create_widgets(self):
        """Create scrollable canvas and frame."""
        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self, bg='#f8f9fa', highlightthickness=0)
        self.scrollbar_y = ttk.Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self, orient=HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        
        # Pack scrollbars and canvas
        self.scrollbar_y.pack(side=RIGHT, fill=Y)
        self.scrollbar_x.pack(side=BOTTOM, fill=X)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Create inner frame for charts
        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor=NW)
        
        # Bind events
        self.inner_frame.bind('<Configure>', self._on_frame_configure)
        
        # Bind mouse wheel
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind_all('<Button-4>', self._on_mousewheel)
        self.canvas.bind_all('<Button-5>', self._on_mousewheel)
    
    def _on_frame_configure(self, event=None):
        """Update scroll region when frame size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def _on_resize(self, event=None):
        """Handle window resize - recalculate grid columns."""
        has_figures = hasattr(self, 'figures') and self.figures
        has_png_data = hasattr(self, 'png_data') and self.png_data
        
        if (not has_figures and not has_png_data) or self.zoomed_chart:
            return
        
        new_width = self.winfo_width()
        if abs(new_width - self._last_width) > 50:
            self._last_width = new_width
            self._relayout_charts()
    
    def _calculate_columns(self) -> int:
        """Calculate number of columns based on available width."""
        available_width = self.winfo_width()
        if available_width < 100:
            available_width = 1000
        
        chart_total_width = self.CHART_MIN_WIDTH + self.CHART_PADDING * 2
        columns = max(1, int(available_width / chart_total_width))
        return columns
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        if self.zoomed_chart:
            return
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, 'units')
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, 'units')
    
    def clear(self):
        """Clear all displayed charts."""
        self.figures.clear()
        if hasattr(self, 'png_data'):
            self.png_data.clear()
        self.photo_images.clear()
        self.full_images.clear()
        self.chart_widgets.clear()
        self.zoomed_chart = None
        
        if self.zoom_overlay:
            self.zoom_overlay.destroy()
            self.zoom_overlay = None
        
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
    
    def _figure_to_photoimage(
        self, 
        fig: Figure, 
        width: int, 
        height: int
    ) -> ImageTk.PhotoImage:
        """
        Convert matplotlib figure to Tkinter PhotoImage.
        """
        buf = BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(buf, format='png', dpi=self.DISPLAY_DPI, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
        buf.seek(0)
        
        pil_image = Image.open(buf)
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        
        return ImageTk.PhotoImage(pil_image)
    
    def _bytes_to_photoimage(
        self,
        png_bytes: bytes,
        width: int,
        height: int
    ) -> ImageTk.PhotoImage:
        """
        Convert PNG bytes directly to Tkinter PhotoImage.
        Much faster than going through Figure object.
        """
        pil_image = Image.open(BytesIO(png_bytes))
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_image)
    
    def _on_chart_click(self, data_type: str):
        """Handle chart click - toggle zoom mode."""
        if self.zoomed_chart == data_type:
            self._restore_normal_view()
        else:
            self._zoom_to_chart(data_type)
    
    def _zoom_to_chart(self, data_type: str):
        """
        Zoom to display a single chart using overlay (no layout change).
        Supports both figures dict and png_data dict.
        """
        # Check if we have the data (either figures or png_data)
        has_figure = hasattr(self, 'figures') and data_type in self.figures
        has_png = hasattr(self, 'png_data') and data_type in self.png_data
        
        if not has_figure and not has_png:
            return
        
        # Close any existing overlay
        if self.zoom_overlay:
            self.zoom_overlay.destroy()
        
        self.zoomed_chart = data_type
        
        # Get dimensions
        width = self.winfo_width() - 20
        height = self.winfo_height() - 60
        
        if width < 100:
            width = 800
        if height < 100:
            height = 600
        
        # Get or create cached image
        cache_key = f"{data_type}_{width}_{height}"
        if cache_key not in self.full_images:
            if has_png:
                # Use PNG bytes directly (faster)
                photo = self._bytes_to_photoimage(self.png_data[data_type], width, height)
            else:
                # Fallback to figure
                fig = self.figures[data_type]
                photo = self._figure_to_photoimage(fig, width, height)
            self.full_images[cache_key] = photo
        
        photo = self.full_images[cache_key]
        
        # Create overlay frame using place() - sits on top without affecting layout
        self.zoom_overlay = tk.Frame(self, bg='#f8f9fa')
        self.zoom_overlay.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Title
        title_label = tk.Label(
            self.zoom_overlay,
            text=f"📊 {data_type} (click to return)",
            font=('Helvetica', 14, 'bold'),
            bg='#f8f9fa',
            cursor='hand2'
        )
        title_label.pack(fill=X, pady=10)
        title_label.bind('<Button-1>', lambda e: self._restore_normal_view())
        
        # Image
        img_label = tk.Label(self.zoom_overlay, image=photo, bg='#f8f9fa', cursor='hand2')
        img_label.pack(fill=BOTH, expand=True, padx=10, pady=10)
        img_label.bind('<Button-1>', lambda e: self._restore_normal_view())
        
        # Keep reference
        img_label.image = photo
    
    def _restore_normal_view(self):
        """Restore normal view by removing overlay - instant, no layout recalc."""
        self.zoomed_chart = None
        
        if self.zoom_overlay:
            self.zoom_overlay.destroy()
            self.zoom_overlay = None
    
    def _relayout_charts(self):
        """Relayout charts based on current window size."""
        # Check which data source we have
        has_figures = hasattr(self, 'figures') and self.figures
        has_png_data = hasattr(self, 'png_data') and self.png_data
        
        if (not has_figures and not has_png_data) or self.zoomed_chart:
            return
        
        # Determine which data source to use
        data_source = self.png_data if has_png_data else self.figures
        
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.chart_widgets.clear()
        
        columns = self._calculate_columns()
        
        # Order data types: mismatch (p < 0.05) first, then match (p >= 0.05)
        if hasattr(self, 'stats') and self.stats:
            mismatch = []
            match = []
            for data_type in data_source.keys():
                if data_type in self.stats and self.stats[data_type].has_significant_results():
                    mismatch.append(data_type)
                else:
                    match.append(data_type)
            mismatch.sort()
            match.sort()
            data_types = mismatch + match
        else:
            data_types = sorted(data_source.keys())
        
        for i, data_type in enumerate(data_types):
            row = i // columns
            col = i % columns
            
            chart_frame = ttk.Frame(self.inner_frame, padding=5)
            chart_frame.grid(row=row, column=col, padx=5, pady=5, sticky=NSEW)
            
            title_label = ttk.Label(
                chart_frame, 
                text=data_type, 
                font=('Helvetica', 10, 'bold'),
                anchor=CENTER,
                cursor='hand2'
            )
            title_label.pack(fill=X, pady=(0, 5))
            title_label.bind('<Button-1>', lambda e, dt=data_type: self._on_chart_click(dt))
            
            if data_type in self.photo_images:
                photo = self.photo_images[data_type]
            elif has_png_data:
                photo = self._bytes_to_photoimage(self.png_data[data_type], self.CHART_MIN_WIDTH, self.CHART_MIN_HEIGHT)
                self.photo_images[data_type] = photo
            else:
                fig = self.figures[data_type]
                photo = self._figure_to_photoimage(fig, self.CHART_MIN_WIDTH, self.CHART_MIN_HEIGHT)
                self.photo_images[data_type] = photo
            
            img_label = ttk.Label(chart_frame, image=photo, cursor='hand2')
            img_label.pack(fill=BOTH, expand=True)
            img_label.bind('<Button-1>', lambda e, dt=data_type: self._on_chart_click(dt))
            
            self.chart_widgets[data_type] = img_label
        
        for col in range(columns):
            self.inner_frame.columnconfigure(col, weight=1)
        
        self._on_frame_configure()
    
    def display_figures(
        self,
        figures: Dict[str, Figure],
        progress_callback: Optional[Callable[[float], None]] = None,
        stats: Optional[Dict] = None
    ):
        """Display figures in dynamic grid layout.
        
        Args:
            figures: Dictionary mapping data_type -> Figure
            progress_callback: Optional callback for progress updates
            stats: Optional stats dictionary for ordering (mismatch first)
        """
        self.clear()
        self.figures = figures
        self.stats = stats  # Store for relayout
        
        if not figures:
            return
        
        total = len(figures)
        
        # Order data types: mismatch (p < 0.05) first, then match (p >= 0.05)
        if stats:
            mismatch = []
            match = []
            for data_type in figures.keys():
                if data_type in stats and stats[data_type].has_significant_results():
                    mismatch.append(data_type)
                else:
                    match.append(data_type)
            mismatch.sort()
            match.sort()
            data_types = mismatch + match
        else:
            data_types = sorted(figures.keys())
        
        columns = self._calculate_columns()
        
        for i, data_type in enumerate(data_types):
            fig = figures[data_type]
            
            row = i // columns
            col = i % columns
            
            chart_frame = ttk.Frame(self.inner_frame, padding=5)
            chart_frame.grid(row=row, column=col, padx=5, pady=5, sticky=NSEW)
            
            title_label = ttk.Label(
                chart_frame, 
                text=data_type, 
                font=('Helvetica', 10, 'bold'),
                anchor=CENTER,
                cursor='hand2'
            )
            title_label.pack(fill=X, pady=(0, 5))
            title_label.bind('<Button-1>', lambda e, dt=data_type: self._on_chart_click(dt))
            
            photo = self._figure_to_photoimage(fig, self.CHART_MIN_WIDTH, self.CHART_MIN_HEIGHT)
            self.photo_images[data_type] = photo
            
            img_label = ttk.Label(chart_frame, image=photo, cursor='hand2')
            img_label.pack(fill=BOTH, expand=True)
            img_label.bind('<Button-1>', lambda e, dt=data_type: self._on_chart_click(dt))
            
            self.chart_widgets[data_type] = img_label
            
            if progress_callback:
                progress_callback((i + 1) / total * 100)
            
            self.update_idletasks()
        
        for col in range(columns):
            self.inner_frame.columnconfigure(col, weight=1)
        
        self._on_frame_configure()
        self._last_width = self.winfo_width()
    
    def get_figures(self) -> Dict[str, Figure]:
        """Get all displayed figures."""
        return self.figures
    
    def display_png_bytes(
        self,
        png_data: Dict[str, bytes],
        progress_callback: Optional[Callable[[float], None]] = None,
        stats: Optional[Dict] = None
    ):
        """Display charts from PNG bytes directly (faster for multiprocess).
        
        Args:
            png_data: Dictionary mapping data_type -> PNG bytes
            progress_callback: Optional callback for progress updates
            stats: Optional stats dictionary for ordering (mismatch first)
        """
        self.clear()
        self.png_data = png_data  # Store PNG data
        self.stats = stats
        
        if not png_data:
            return
        
        total = len(png_data)
        
        # Order data types: mismatch first
        if stats:
            mismatch = []
            match = []
            for data_type in png_data.keys():
                if data_type in stats and stats[data_type].has_significant_results():
                    mismatch.append(data_type)
                else:
                    match.append(data_type)
            mismatch.sort()
            match.sort()
            data_types = mismatch + match
        else:
            data_types = sorted(png_data.keys())
        
        columns = self._calculate_columns()
        
        for i, data_type in enumerate(data_types):
            png_bytes = png_data[data_type]
            
            row = i // columns
            col = i % columns
            
            chart_frame = ttk.Frame(self.inner_frame, padding=5)
            chart_frame.grid(row=row, column=col, padx=5, pady=5, sticky=NSEW)
            
            title_label = ttk.Label(
                chart_frame, 
                text=data_type, 
                font=('Helvetica', 10, 'bold'),
                anchor=CENTER,
                cursor='hand2'
            )
            title_label.pack(fill=X, pady=(0, 5))
            title_label.bind('<Button-1>', lambda e, dt=data_type: self._on_chart_click(dt))
            
            # Use bytes directly - much faster!
            photo = self._bytes_to_photoimage(png_bytes, self.CHART_MIN_WIDTH, self.CHART_MIN_HEIGHT)
            self.photo_images[data_type] = photo
            
            img_label = ttk.Label(chart_frame, image=photo, cursor='hand2')
            img_label.pack(fill=BOTH, expand=True)
            img_label.bind('<Button-1>', lambda e, dt=data_type: self._on_chart_click(dt))
            
            self.chart_widgets[data_type] = img_label
            
            if progress_callback:
                progress_callback((i + 1) / total * 100)
            
            # Update every 10 charts to avoid too many UI updates
            if i % 10 == 0:
                self.update_idletasks()
        
        self.update_idletasks()
        
        for col in range(columns):
            self.inner_frame.columnconfigure(col, weight=1)
        
        self._on_frame_configure()
        self._last_width = self.winfo_width()
