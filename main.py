from config import BotConfig
from bot.telegram_bot import TelegramBot
import logging
import sys

# Load configuration
config = BotConfig()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the Telegram bot."""
    # Create and run bot
    bot = TelegramBot(config)
    
    try:
        # Run the bot - it will manage its own event loop
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
