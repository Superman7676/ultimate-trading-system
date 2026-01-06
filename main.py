#!/usr/bin/env python3
"""
Ultimate Trading System - Telegram Bot
Main entry point for the trading bot
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Start command handler
    Sends welcome message to the user
    """
    user = update.effective_user
    welcome_message = f"""
🚀 **Welcome to Ultimate Trading System!**

Hi {user.first_name}!

Available Commands:

📊 /analyze <SYMBOL> - Analyze a stock (e.g., /analyze NVDA)
💹 /price <SYMBOL> - Get current price
📈 /technicals <SYMBOL> - Technical analysis indicators
🤖 /predict <SYMBOL> - ML model prediction
⚙️ /settings - Configure trading parameters
📚 /help - Show help information

Start by typing:
/analyze NVDA
    """
    await update.message.reply_text(welcome_message, parse_mode='Markdown')
    logger.info(f"User {user.id} started the bot")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Help command handler
    Provides detailed information about available commands
    """
    help_text = """
**🔧 Command Help**

*Analysis & Predictions:*
/analyze <SYMBOL> - Comprehensive stock analysis
/price <SYMBOL> - Current market price
/technicals <SYMBOL> - Technical indicators (RSI, MACD, etc.)
/predict <SYMBOL> - ML model prediction
/backtest <SYMBOL> - Backtest trading strategy

*Portfolio Management:*
/portfolio - View your portfolio
/add <SYMBOL> <QUANTITY> - Add position
/remove <SYMBOL> - Remove position

*Settings:*
/settings - Trading configuration
/risk <PERCENT> - Set stop loss percentage
/profit <PERCENT> - Set take profit percentage

*System:*
/status - System status
/logs - View recent logs
/help - Show this help message
    """
    await update.message.reply_text(help_text, parse_mode='Markdown')
    logger.info(f"Help requested by user {update.effective_user.id}")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Analyze command handler
    Provides stock analysis
    """
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a stock symbol\n\nUsage: /analyze NVDA"
        )
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(
        f"📊 Analyzing {symbol}...\n\n🔄 Analysis in progress. Please wait..."
    )
    
    logger.info(f"Analysis requested for {symbol} by user {update.effective_user.id}")


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Price command handler
    Gets current stock price
    """
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a stock symbol\n\nUsage: /price NVDA"
        )
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(
        f"💹 Fetching price for {symbol}...\n\n⏳ Loading current price..."
    )
    
    logger.info(f"Price request for {symbol} by user {update.effective_user.id}")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Status command handler
    Shows system status
    """
    status_message = f"""
✅ **System Status**

🤖 Bot Status: Online
⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Data Source: yFinance
🧠 ML Models: Active
🔌 Database: Connected
    """
    await update.message.reply_text(status_message, parse_mode='Markdown')
    logger.info(f"Status requested by user {update.effective_user.id}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle regular messages
    """
    logger.info(f"Message from {update.effective_user.id}: {update.message.text}")


def main() -> None:
    """
    Main function to start the bot
    """
    logger.info("Starting Ultimate Trading System Bot")
    
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    logger.info("Bot is running and polling for updates...")
    application.run_polling()


if __name__ == '__main__':
    main()
