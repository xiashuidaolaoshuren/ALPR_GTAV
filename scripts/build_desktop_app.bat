@echo off
REM Build script for GTA V ALPR Desktop Application
REM This script automates the Electron build process

echo ========================================
echo GTA V ALPR - Desktop Application Builder
echo ========================================
echo.

REM Check if Node.js is installed
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo [1/4] Checking Node.js installation...
node --version
npm --version
echo.

REM Navigate to electron directory
cd /d "%~dp0..\gui\electron"

REM Check if node_modules exists
if not exist "node_modules" (
    echo [2/4] Installing dependencies (first time)...
    call npm install
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [2/4] Dependencies already installed, skipping...
)
echo.

echo [3/4] Building Electron application...
echo This may take several minutes...
echo.
call npm run build:win
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed
    pause
    exit /b 1
)
echo.

echo [4/4] Build complete!
echo.
echo Installer location: gui\electron\dist\
dir dist\*.exe /b
echo.

echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo You can find the installer in: gui\electron\dist\
echo.
pause
