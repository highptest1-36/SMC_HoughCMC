# setup_env.ps1 - one-shot creation of the GPU-ready conda env for journal_cmc.
#
# Run from THIS folder in an Anaconda PowerShell Prompt:
#     powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
# Optional custom env name:
#     powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -EnvName smc_cmc2
# CPU-only (skip CUDA torch):
#     powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -Cpu

param(
    [string]$EnvName = "smc_cmc",
    [switch]$Cpu
)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# conda must be available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: conda not found. Open an 'Anaconda PowerShell Prompt'"
    Write-Host "       (or run 'conda init powershell' and reopen the terminal)."
    exit 1
}

Write-Host "==> Creating conda env '$EnvName' (Python 3.10) ..."
conda create -n $EnvName python=3.10 -y

Write-Host "==> Upgrading pip ..."
conda run --no-capture-output -n $EnvName python -m pip install --upgrade pip

if ($Cpu) {
    Write-Host "==> Installing CPU torch 2.7.1 ..."
    conda run --no-capture-output -n $EnvName python -m pip install torch==2.7.1
} else {
    Write-Host "==> Installing CUDA torch 2.7.1 (cu126). About 2.5 GB, please wait ..."
    conda run --no-capture-output -n $EnvName python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
}

Write-Host "==> Installing remaining dependencies ..."
conda run --no-capture-output -n $EnvName python -m pip install -r requirements_journal.txt

Write-Host "==> Verifying environment ..."
conda run --no-capture-output -n $EnvName python run.py gpucheck

Write-Host ""
Write-Host "Done. To use it every session:"
Write-Host "    conda activate $EnvName"
Write-Host "    python run.py gpucheck"
