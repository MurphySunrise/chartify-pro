#!/usr/bin/env python3
"""
Chartify Pro - CSV Data Analysis & PPT Report Generator
"""

import sys
from src.gui.app import ChartifyApp


def main():
    """Application entry point."""
    app = ChartifyApp()
    app.run()


if __name__ == "__main__":
    main()
