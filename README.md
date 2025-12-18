# Chartify Pro V1

A GUI application for CSV data analysis, plotting, and PowerPoint report generation.

## Features
- Load and analyze CSV data with Polars
- Generate combined visualizations (boxplot + Q-Q plot + stats table)
- Export to PowerPoint reports
- High-performance for large datasets

## Requirements
- Python 3.9+
- Dependencies: `pip install -r requirements.txt`

## Run from Source
```bash
cd v1
python3 main.py
```

## Build Executables

This project uses GitHub Actions to automatically build executables for Windows and macOS.

### Automatic Build (Recommended)

1. **Push to GitHub**
   ```bash
   cd /Users/murphy/Downloads/Chartify
   git init
   git add v1/
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/chartify-pro.git
   git push -u origin main
   ```

2. **Trigger Build**
   - Go to your GitHub repo → Actions → "Build Executables"
   - Click "Run workflow" to manually trigger
   - Or push a version tag: `git tag v1.0.0 && git push --tags`

3. **Download**
   - After build completes, download from Actions → Artifacts
   - Or from Releases page if you used a tag

### Build Locally (macOS only)

```bash
pip install pyinstaller
cd v1
pyinstaller --onefile --windowed --name "ChartifyPro" main.py
# Output: dist/ChartifyPro.app
```

## License
MIT
