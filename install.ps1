# TensorLogic Installation Script for Windows (PowerShell)
# Usage: .\install.ps1 [-Prefix "C:\path"] [-Uninstall]

param(
    [string]$Prefix = "C:\Program Files\TensorLogic",
    [switch]$Uninstall,
    [switch]$Help
)

# Display help
if ($Help) {
    Write-Host "TensorLogic Installation Script for Windows"
    Write-Host ""
    Write-Host "Usage: .\install.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Prefix PATH     Install to PATH (default: C:\Program Files\TensorLogic)"
    Write-Host "  -Uninstall       Uninstall TensorLogic from Prefix"
    Write-Host "  -Help            Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\install.ps1"
    Write-Host "  .\install.ps1 -Prefix `"C:\Users\$env:USERNAME\TensorLogic`""
    Write-Host "  .\install.ps1 -Uninstall"
    exit 0
}

# Check if running as Administrator (for Program Files installation)
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($Prefix -like "C:\Program Files*" -and -not $isAdmin) {
    Write-Host "Error: Installing to '$Prefix' requires Administrator privileges." -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or install to a user directory:"
    Write-Host "  .\install.ps1 -Prefix `"C:\Users\$env:USERNAME\TensorLogic`""
    exit 1
}

# ====================================
# Uninstallation
# ====================================
if ($Uninstall) {
    Write-Host "Uninstalling TensorLogic from $Prefix..."

    if (-not (Test-Path $Prefix)) {
        Write-Host "TensorLogic is not installed at $Prefix" -ForegroundColor Yellow
        exit 0
    }

    # Remove binary
    if (Test-Path "$Prefix\bin\tl.exe") {
        Write-Host "  Removing $Prefix\bin\tl.exe"
        Remove-Item "$Prefix\bin\tl.exe" -Force
    }

    # Remove DLLs from bin directory
    if (Test-Path "$Prefix\bin") {
        Write-Host "  Removing DLL dependencies from bin directory..."
        Get-ChildItem "$Prefix\bin\*.dll" -ErrorAction SilentlyContinue | Where-Object {
            $_.Name -like "libtorch*" -or $_.Name -like "libc10*" -or $_.Name -like "libshm*"
        } | ForEach-Object {
            Remove-Item $_.FullName -Force
        }
    }

    # Remove libraries
    if (Test-Path "$Prefix\lib") {
        Write-Host "  Removing libraries from $Prefix\lib"
        Remove-Item "$Prefix\lib\*" -Recurse -Force
    }

    # Remove examples
    if (Test-Path "$Prefix\share\tensorlogic") {
        Write-Host "  Removing $Prefix\share\tensorlogic"
        Remove-Item "$Prefix\share\tensorlogic" -Recurse -Force
    }

    # Remove documentation
    if (Test-Path "$Prefix\share\doc\tensorlogic") {
        Write-Host "  Removing $Prefix\share\doc\tensorlogic"
        Remove-Item "$Prefix\share\doc\tensorlogic" -Recurse -Force
    }

    # Remove empty directories
    if (Test-Path "$Prefix\bin" -and (Get-ChildItem "$Prefix\bin" | Measure-Object).Count -eq 0) {
        Remove-Item "$Prefix\bin" -Force
    }
    if (Test-Path "$Prefix\lib" -and (Get-ChildItem "$Prefix\lib" | Measure-Object).Count -eq 0) {
        Remove-Item "$Prefix\lib" -Force
    }
    if (Test-Path "$Prefix\share" -and (Get-ChildItem "$Prefix\share" -Recurse | Measure-Object).Count -eq 0) {
        Remove-Item "$Prefix\share" -Recurse -Force
    }

    # Remove Prefix directory if empty
    if (Test-Path $Prefix -and (Get-ChildItem $Prefix -Recurse | Measure-Object).Count -eq 0) {
        Remove-Item $Prefix -Force
    }

    Write-Host "Uninstallation complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Note: You may need to manually remove $Prefix\bin from your PATH if it was added."
    exit 0
}

# ====================================
# Installation
# ====================================

Write-Host "Installing TensorLogic to $Prefix..."

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Verify required directories exist
if (-not (Test-Path "$ScriptDir\bin")) {
    Write-Host "Error: bin\ directory not found. Are you in the extracted archive directory?" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$ScriptDir\lib")) {
    Write-Host "Error: lib\ directory not found. Are you in the extracted archive directory?" -ForegroundColor Red
    exit 1
}

# Create installation directories
Write-Host "  Creating installation directories..."
New-Item -ItemType Directory -Force -Path "$Prefix\bin" | Out-Null
New-Item -ItemType Directory -Force -Path "$Prefix\lib" | Out-Null
New-Item -ItemType Directory -Force -Path "$Prefix\share\tensorlogic" | Out-Null
New-Item -ItemType Directory -Force -Path "$Prefix\share\doc\tensorlogic" | Out-Null

# Install binary
Write-Host "  Installing tl.exe to $Prefix\bin\tl.exe..."
Copy-Item "$ScriptDir\bin\tl.exe" "$Prefix\bin\tl.exe" -Force

# Install libraries
Write-Host "  Installing libtorch libraries to $Prefix\lib\..."
Copy-Item "$ScriptDir\lib\*" "$Prefix\lib\" -Recurse -Force

# Copy DLLs to bin directory so they're found automatically (Windows DLL search path)
Write-Host "  Copying DLL dependencies to bin directory..."
Get-ChildItem "$Prefix\lib\*.dll" | ForEach-Object {
    Copy-Item $_.FullName "$Prefix\bin\" -Force
}

# Install examples (if they exist)
if (Test-Path "$ScriptDir\examples") {
    Write-Host "  Installing example programs to $Prefix\share\tensorlogic\examples\..."
    Copy-Item "$ScriptDir\examples" "$Prefix\share\tensorlogic\" -Recurse -Force
} elseif (Test-Path "$ScriptDir\Programs") {
    Write-Host "  Installing example programs to $Prefix\share\tensorlogic\examples\..."
    New-Item -ItemType Directory -Force -Path "$Prefix\share\tensorlogic\examples" | Out-Null
    Copy-Item "$ScriptDir\Programs\*" "$Prefix\share\tensorlogic\examples\" -Recurse -Force
}

# Install documentation (if it exists)
if (Test-Path "$ScriptDir\doc") {
    Write-Host "  Installing documentation to $Prefix\share\doc\tensorlogic\..."
    Copy-Item "$ScriptDir\doc\*" "$Prefix\share\doc\tensorlogic\" -Recurse -Force
}

# ====================================
# Post-installation instructions
# ====================================

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "TensorLogic has been installed to: $Prefix"
Write-Host ""

# Check if Prefix\bin is in PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
$binPath = "$Prefix\bin"

if ($currentPath -like "*$binPath*") {
    Write-Host "You can now run TensorLogic with: tl" -ForegroundColor Green
} else {
    Write-Host "IMPORTANT: Add $binPath to your PATH to use TensorLogic:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run this command in PowerShell (as Administrator if installed to Program Files):" -ForegroundColor Cyan
    Write-Host ""
    if ($isAdmin) {
        Write-Host "  [Environment]::SetEnvironmentVariable(" -NoNewline
        Write-Host "`"Path`", `"" -NoNewline -ForegroundColor White
        Write-Host "`$env:Path;$binPath" -NoNewline -ForegroundColor Yellow
        Write-Host "`", `"Machine`")" -ForegroundColor White
    } else {
        Write-Host "  [Environment]::SetEnvironmentVariable(" -NoNewline
        Write-Host "`"Path`", `"" -NoNewline -ForegroundColor White
        Write-Host "`$env:Path;$binPath" -NoNewline -ForegroundColor Yellow
        Write-Host "`", `"User`")" -ForegroundColor White
    }
    Write-Host ""
    Write-Host "Then restart your terminal for changes to take effect."
    Write-Host ""
}

Write-Host "Test the installation with:"
Write-Host "  $Prefix\bin\tl.exe --version"
Write-Host ""
Write-Host "To uninstall, run:"
Write-Host "  .\install.ps1 -Uninstall -Prefix `"$Prefix`""
Write-Host ""
