@echo off
REM ════════════════════════════════════════════════════════════════════════
REM 🚀 ULTIMATE TRADING SYSTEM - SETUP (Windows)
REM Automatic local environment setup
REM ════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo.
echo 🚀 Setting up Ultimate Trading System...
echo ════════════════════════════════════════════════════════════════════════
echo.

REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Step 1: Check Python
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo 📝 Checking Python installation...

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% found

REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Step 2: Create Virtual Environment
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo 📦 Creating virtual environment...

if exist venv (
    echo ⚠️  Virtual environment already exists
) else (
    python -m venv venv
    echo ✅ Virtual environment created
)

REM Activate venv
call venv\Scripts\activate.bat

echo ✅ Virtual environment activated

REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Step 3: Upgrade pip
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip setuptools wheel -q
echo ✅ Pip upgraded

REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Step 4: Install dependencies
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo 📥 Installing dependencies (this may take a few minutes)...

if exist requirements.txt (
    pip install -r requirements.txt
    echo ✅ Dependencies installed
) else (
    echo ❌ requirements.txt not found
    pause
    exit /b 1
)

REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Step 5: Create directories
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo 📁 Creating directories...

if not exist data mkdir data
if not exist logs mkdir logs
if not exist models mkdir models
if not exist reports mkdir reports

echo ✅ Directories created:
echo    • data     (watchlist, database)
echo    • logs     (system logs)
echo    • models   (trained ML models)
echo    • reports  (Excel output)

REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Step 6: Create .env file
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo 🔐 Setting up environment file...

if exist .env (
    echo ⚠️  .env already exists - skipping
) else (
    if exist .env.example (
        copy .env.example .env
        echo ✅ .env created from template
    ) else (
        echo ⚠️  .env.example not found
    )
)

REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Step 7: Test imports
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo ✅ Testing Python imports...

python << 'EOF'
try:
    import pandas; print("  ✅ pandas")
    import numpy; print("  ✅ numpy")
    import yfinance; print("  ✅ yfinance")
    from telegram.ext import Application; print("  ✅ python-telegram-bot")
    import streamlit; print("  ✅ streamlit")
    import xgboost; print("  ✅ xgboost")
    import sklearn; print("  ✅ scikit-learn")
    print("\n✅ All core imports successful!")
except ImportError as e:
    print(f"⚠️  Missing import: {e}")
EOF

REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REM Final Summary
REM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo ════════════════════════════════════════════════════════════════════════
echo ✅ SETUP COMPLETE!
echo ════════════════════════════════════════════════════════════════════════
echo.
echo 📋 NEXT STEPS:
echo.
echo 1️⃣  Edit .env with your API keys:
echo    notepad .env
echo.
echo 2️⃣  Start Telegram Bot:
echo    python main.py
echo.
echo 3️⃣  In another terminal, start Web Dashboard:
echo    streamlit run app.py
echo.
echo 4️⃣  Test in Telegram:
echo    /start
echo    /analyze NVDA
echo.
echo 📚 Full documentation:
echo    https://github.com/Superman7676/ultimate-trading-system
echo.
echo ════════════════════════════════════════════════════════════════════════
echo.

pause
