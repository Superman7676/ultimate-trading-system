#!/usr/bin/env python3
"""
Ultimate Trading System - Configuration Module
Centralized configuration management
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== TELEGRAM ====================
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# ==================== APIs ====================
FINHUB_API_KEY = os.getenv('FINHUUB_API_KEY')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

# ==================== DATABASE ====================
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'sqlite:///trading_system.db'
)

# ==================== TRADING PARAMETERS ====================
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 10000))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 2.0))
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 5.0))
RISK_REWARD_RATIO = float(os.getenv('RISK_REWARD_RATIO', 1.5))

# ==================== MARKET HOURS (EST) ====================
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
PRE_MARKET_START = "04:00"
AFTER_HOURS_END = "20:00"

# ==================== ML MODEL PARAMETERS ====================
LOOKBACK_PERIOD = 60  # days
PREDICTION_HORIZON = 5  # days
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# ==================== LOGGING ====================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = LOGS_DIR / 'trading_system.log'

# ==================== FEATURE FLAGS ====================
ENABLE_BACKTEST = os.getenv('ENABLE_BACKTEST', 'true').lower() == 'true'
ENABLE_LIVE_TRADING = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
ENABLE_ALERTS = os.getenv('ENABLE_ALERTS', 'true').lower() == 'true'
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# ==================== TECHNICAL INDICATORS ====================
TECHNICAL_INDICATORS = {
    'SMA': {'periods': [20, 50, 200]},
    'EMA': {'periods': [12, 26]},
    'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BOLLINGER': {'period': 20, 'std_dev': 2},
    'ATR': {'period': 14},
    'STOCHASTIC': {'period': 14, 'smooth_k': 3, 'smooth_d': 3},
}

# ==================== DATA SOURCES ====================
DATA_SOURCES = {
    'yfinance': {'enabled': True, 'timeout': 30},
    'alpha_vantage': {'enabled': True, 'timeout': 30},
    'finHub': {'enabled': False, 'timeout': 30},
}

# ==================== BACKTESTING ====================
BACKTEST_CONFIG = {
    'initial_capital': 100000,
    'commission': 0.001,  # 0.1% per trade
    'slippage': 0.0005,   # 0.05% slippage
    'max_positions': 5,
    'rebalance_frequency': 'weekly',
}

if __name__ == '__main__':
    # Print configuration for verification
    print("=" * 50)
    print("Ultimate Trading System - Configuration")
    print("=" * 50)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Logs Directory: {LOGS_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Reports Directory: {REPORTS_DIR}")
    print(f"Database: {DATABASE_URL}")
    print(f"Debug Mode: {DEBUG_MODE}")
    print(f"Enable Alerts: {ENABLE_ALERTS}")
    print(f"Enable Live Trading: {ENABLE_LIVE_TRADING}")
    print("=" * 50)
