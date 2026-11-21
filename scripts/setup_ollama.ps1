<#
.SYNOPSIS
    Configure Ollama on Windows so the RAGForge app running in WSL2 can reach it.

.DESCRIPTION
    Ollama listens on 127.0.0.1 by default, which WSL2 cannot reach. This script:
      1. Sets OLLAMA_HOST=0.0.0.0:11434 for the current user.
      2. Restarts the Ollama app so the setting takes effect.
      3. Pulls the nomic-embed-text embedding model if it is missing.
      4. Prints the WSL gateway URL to use.

    Run this in a normal PowerShell window (no administrator required).

.NOTES
    Binding to 0.0.0.0 exposes Ollama to your local network. On untrusted
    networks, restrict access with a Windows Firewall rule or bind to the
    WSL-facing vEthernet address instead.
#>

$ErrorActionPreference = "Stop"

$ollamaDir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
$ollamaExe = Join-Path $ollamaDir "ollama.exe"
$appExe = Join-Path $ollamaDir "ollama app.exe"

if (-not (Test-Path $ollamaExe)) {
    throw "Could not find ollama.exe at $ollamaExe. Install Ollama first: https://ollama.com/download"
}

Write-Host "Setting OLLAMA_HOST=0.0.0.0:11434 for the current user..." -ForegroundColor Cyan
setx OLLAMA_HOST "0.0.0.0:11434" | Out-Null

Write-Host "Restarting Ollama..." -ForegroundColor Cyan
Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process -Name "ollama app" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

$env:OLLAMA_HOST = "0.0.0.0:11434"
Start-Process $appExe
Start-Sleep -Seconds 6

$version = (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/version -TimeoutSec 15).Content
Write-Host "Ollama is running: $version" -ForegroundColor Green

$models = (& $ollamaExe list) -join "`n"
if ($models -notmatch "nomic-embed-text") {
    Write-Host "Pulling nomic-embed-text..." -ForegroundColor Cyan
    & $ollamaExe pull nomic-embed-text
} else {
    Write-Host "nomic-embed-text is already installed." -ForegroundColor Green
}

Write-Host ""
Write-Host "Done. In WSL, discover the host gateway with:" -ForegroundColor Yellow
Write-Host '  source scripts/ollama_host.sh' -ForegroundColor Yellow
Write-Host "or simply run 'uv run ragforge check' and let RAGForge auto-detect it."
