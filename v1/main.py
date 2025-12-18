#!/usr/bin/env python3
"""
Chartify Pro - CSV Data Analysis & PPT Report Generator
"""

import sys
import multiprocessing
from src.gui.app import ChartifyApp


def main():
    """Application entry point."""
    app = ChartifyApp()
    app.run()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows exe packaging
    main()
