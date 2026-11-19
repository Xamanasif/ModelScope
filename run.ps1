# ModelScope launcher (Windows PowerShell). Binds to 127.0.0.1 only.
#   .\run.ps1                 # serve API + built dashboard on http://127.0.0.1:8000
#   .\run.ps1 -Build          # rebuild the dashboard first (needs Node.js/npm)
#   .\run.ps1 -Port 8010
param([int]$Port = 8000, [switch]$Build)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Create the virtual environment and install requirements first; see README.md."
}
if ($Build -or -not (Test-Path (Join-Path $PSScriptRoot "frontend\dist\index.html"))) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Warning "npm not found: the API will run, but the dashboard is not built."
    } else {
        Push-Location (Join-Path $PSScriptRoot "frontend")
        try {
            if (-not (Test-Path "node_modules")) { npm ci }
            npm run build
            if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
        } finally { Pop-Location }
    }
}
Write-Host "ModelScope on http://127.0.0.1:$Port  (API docs: /docs). Opening it never trains a model. Ctrl+C to stop."
& $Python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
