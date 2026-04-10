# APM Corridor Visualizer Launch Script
# Quick launcher for the Streamlit visualizer

Write-Host ""
Write-Host "Launching APM Corridor Visualizer..." -ForegroundColor Green
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Use the virtual environment Python directly
$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"
$appFile = Join-Path $scriptDir "streamlit_app.py"

# Check if virtual environment exists
if (-not (Test-Path $pythonExe)) {
    Write-Host "Virtual environment not found at: $pythonExe" -ForegroundColor Red
    Write-Host "Please ensure the .venv folder exists." -ForegroundColor Yellow
    exit 1
}

# Launch Streamlit
Write-Host "Starting Streamlit server..." -ForegroundColor Cyan
Write-Host "Access URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

& $pythonExe -m streamlit run $appFile
