# Install RAG Agent Dependencies
# Run this script to install deepagents and other dependencies

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  SynapseTEA RAG Agent - Dependency Installer" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Virtual environment not activated!" -ForegroundColor Yellow
    Write-Host "Activating .venv..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
    Write-Host ""
}

Write-Host "Current Python environment:" -ForegroundColor Green
python --version
Write-Host ""

Write-Host "Installing/upgrading dependencies..." -ForegroundColor Green
Write-Host ""

# Install dependencies
uv pip install -r requirements.txt

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Test the RAG tools directly:"
Write-Host "   python data/rag_agent.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Start the interactive agent loop:"
Write-Host "   python data/test_agent_loop.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Run demo mode:"
Write-Host "   python data/test_agent_loop.py --demo" -ForegroundColor Yellow
Write-Host ""
