"""
Telegram bot implementation with Mistral and Ollama integration.
Handles user interactions, settings, model selection, conversation history, and file uploads.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from telegram.constants import ParseMode
import re

from config import BotConfig
from core.models.ollama.client import OllamaClient
from core.models.mistral.client import MistralClient
from .conversation_manager import get_conversation_manager
from .file_processor import get_file_processor

# Setup logging
logger = logging.getLogger(__name__)

class TelegramBot:
    """Main Telegram bot class."""
    
    def __init__(self, config: BotConfig):
        """
        Initialize the Telegram bot.
        
        Args:
            config: Bot configuration
        """
        self.config = config
        self.application = None
        self.user_settings: Dict[int, Dict[str, Any]] = {}  # user_id -> settings
        self.model_clients: Dict[str, Any] = {}  # Cache for model clients
        self.default_model = None  # Will be set during initialization
        self.conversation_manager = get_conversation_manager()  # Conversation history manager
        self.file_processor = get_file_processor()  # File upload processor
        
    def _is_user_authorized(self, user_id: int) -> bool:
        """
        Check if a user is authorized to use the bot.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if user is authorized (in admin list or valid user list), False otherwise
        """
        # Parse admin user IDs from config (comma-separated string)
        admin_ids = []
        if self.config.telegram_bot_admin:
            for id_str in self.config.telegram_bot_admin.split(','):
                id_str = id_str.strip()
                if not id_str:
                    continue
                try:
                    admin_ids.append(int(id_str))
                except ValueError:
                    logger.warning(f"Invalid admin ID format: '{id_str}', skipping")
        
        # Parse valid user IDs from config (comma-separated string)
        valid_user_ids = []
        if self.config.telegram_bot_user:
            for id_str in self.config.telegram_bot_user.split(','):
                id_str = id_str.strip()
                if not id_str:
                    continue
                try:
                    valid_user_ids.append(int(id_str))
                except ValueError:
                    logger.warning(f"Invalid valid user ID format: '{id_str}', skipping")
        
        # Check if user is in admin list or valid user list
        # If valid_user_ids is empty, only admin users are allowed
        # If valid_user_ids is not empty, users in either list are allowed
        if user_id in admin_ids:
            return True
        
        if valid_user_ids:
            return user_id in valid_user_ids
        
        # If no valid_user_ids specified and user is not admin, deny access
        return False
    
    def _is_user_admin(self, user_id: int) -> bool:
        """
        Check if a user is an admin.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            True if user is in admin list, False otherwise
        """
        # Parse admin user IDs from config (comma-separated string)
        admin_ids = []
        if self.config.telegram_bot_admin:
            for id_str in self.config.telegram_bot_admin.split(','):
                id_str = id_str.strip()
                if not id_str:
                    continue
                try:
                    admin_ids.append(int(id_str))
                except ValueError:
                    logger.warning(f"Invalid admin ID format: '{id_str}', skipping")
        
        return user_id in admin_ids
        
    async def initialize(self):
        """Initialize the bot and model clients."""
        # Initialize model clients
        if self.config.is_model_enabled("ollama"):
            ollama_config = self.config.get_model_config("ollama")
            self.model_clients["ollama"] = OllamaClient(ollama_config)
            await self.model_clients["ollama"].initialize()
            
        if self.config.is_model_enabled("mistral"):
            mistral_config = self.config.get_model_config("mistral")
            self.model_clients["mistral"] = MistralClient(mistral_config)
            await self.model_clients["mistral"].initialize()
        
        # Set default model
        self.default_model = "ollama" if "ollama" in self.model_clients else "mistral"
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        user_id = update.effective_user.id
        
        # Check if user is authorized
        if not self._is_user_authorized(user_id):
            error_message = (
                "⛔ **Access Denied**\n\n"
                "You are not authorized to use this chatbot.\n"
                "Please contact the administrator if you believe this is an error."
            )
            await update.message.reply_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return

        welcome_message = (
            "🌟 How can I assist you today? 🌟\n\n"
            "I'm your intelligent assistant powered by Mistral and Ollama models.\n\n"
            "Use /config to configure your preferred AI model."
        )
        
        if not self._is_user_admin(user_id):
            welcome_message = (
                "🌟 How can I assist you today? 🌟"
            )

        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.HTML
        )
        
        # Initialize user settings if not exists
        if user_id not in self.user_settings:
            # Initialize model clients if not done yet to get default model
            if not self.model_clients:
                await self.initialize()
            
            self.user_settings[user_id] = {
                "model": self.default_model or ("ollama" if "ollama" in self.model_clients else "mistral")
            }
    
    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle settings button click (legacy callback support)."""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        
        # Check if user is admin (only admins can change configuration)
        if not self._is_user_admin(user_id):
            error_message = (
                "⛔ **Admin Access Required**\n\n"
                "Only administrators can change bot configuration.\n"
                "Please contact the administrator if you need to modify settings."
            )
            await query.edit_message_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return
        
        current_settings = self.user_settings.get(user_id, {})
        
        # Initialize model clients if not done yet (lazy initialization)
        if not self.model_clients:
            await self.initialize()
        
        current_model = current_settings.get("model", self.default_model)
        
        # Create model selection buttons
        model_buttons = []
        available_models = list(self.model_clients.keys())
        
        for model in available_models:
            prefix = "✅ " if model == current_model else ""
            model_buttons.append(
                InlineKeyboardButton(
                    f"{prefix}{model.capitalize()}",
                    callback_data=f"model_{model}"
                )
            )
        
        # Create settings keyboard
        keyboard = [
            model_buttons,
            [InlineKeyboardButton("❌ Close", callback_data='close_settings')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        settings_message = (
            "⚙️ **Configuration** \n\n"
            f"🤖 Current Model: `{current_model.capitalize()}`\n\n"
            "Choose a model to switch:"
        )
        
        if query.message:
            await query.edit_message_text(
                settings_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
        else:
            await context.bot.send_message(
                chat_id=query.from_user.id,
                text=settings_message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
    
    async def handle_model_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle model selection from settings."""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        
        # Check if user is admin (only admins can change configuration)
        if not self._is_user_admin(user_id):
            error_message = (
                "⛔ **Admin Access Required**\n\n"
                "Only administrators can change bot configuration.\n"
                "Please contact the administrator if you need to modify settings."
            )
            await query.edit_message_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return
        
        model_name = query.data.replace("model_", "")
        
        # Update user settings
        if user_id not in self.user_settings:
            self.user_settings[user_id] = {}
        
        self.user_settings[user_id]["model"] = model_name
        
        # Show confirmation
        await query.edit_message_text(
            f"✅ Model changed to **{model_name.capitalize()}**",
            parse_mode=ParseMode.HTML
        )
        
        # Reopen settings after confirmation
        await asyncio.sleep(1)
        await self.settings(update, context)
    
    
    async def close_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close settings menu."""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        
        # Check if user is admin (only admins can access configuration)
        if not self._is_user_admin(user_id):
            error_message = (
                "⛔ **Admin Access Required**\n\n"
                "Only administrators can change bot configuration.\n"
                "Please contact the administrator if you need to modify settings."
            )
            await query.edit_message_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return
        
        await query.edit_message_text("❌ Settings closed.")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user messages and generate AI responses."""
        user_id = update.effective_user.id
        
        # Check if user is authorized
        if not self._is_user_authorized(user_id):
            error_message = (
                "⛔ **Access Denied**\n\n"
                "You are not authorized to use this chatbot.\n"
                "Please contact the administrator if you believe this is an error."
            )
            await update.message.reply_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return
        
        user_message = update.message.text
        
        # Store user message in conversation history
        self.conversation_manager.add_message(user_id, "user", user_message)
        
        # Get conversation history for API (limit to last 20 messages to avoid token overflow)
        conversation_history = self.conversation_manager.get_conversation_for_api(user_id, max_messages=20)
        
        # Initialize user settings if not exists
        if user_id not in self.user_settings:
            # Initialize model clients if not done yet to get default model
            if not self.model_clients:
                await self.initialize()
            
            self.user_settings[user_id] = {
                "model": self.default_model or ("ollama" if "ollama" in self.model_clients else "mistral")
            }

        # Get user settings
        settings = self.user_settings.get(user_id, {})
        
        # Get default model (fallback to first available if not set)
        default_model = self.default_model
        model_name = settings.get("model", default_model)
        
        # Send typing indicator with error handling
        try:
            await update.message.chat.send_action("typing")
        except Exception as e:
            logger.warning(f"Failed to send typing indicator: {e}")
            # Continue anyway - this is not critical
        
        # Send initial "Thinking..." message
        thinking_message = None
        try:
            thinking_message = await update.message.reply_text(
                "⏳ Thinking...",
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Failed to send thinking message: {e}")
            thinking_message = None
        
        try:
            # Initialize model clients if not done yet (lazy initialization)
            if not self.model_clients:
                await self.initialize()
            
            # Get the appropriate model client
            model_client = self.model_clients.get(model_name)
            if not model_client:
                error_text = "❌ Selected model is not available."
                if thinking_message:
                    try:
                        await context.bot.edit_message_text(
                            chat_id=update.message.chat_id,
                            message_id=thinking_message.message_id,
                            text=error_text,
                            parse_mode=ParseMode.HTML
                        )
                    except Exception:
                        await update.message.reply_text(error_text)
                else:
                    await update.message.reply_text(error_text)
                return
            
            # Generate response using conversation history
            response = await model_client.generate_with_history(
                conversation_history=conversation_history,
                use_agents_prompt=False  # Don't use system prompt for direct user messages
            )
            
            if response.success:
                # Store assistant response in conversation history
                self.conversation_manager.add_message(user_id, "assistant", response.content)
                
                # Format the response safely
                try:
                    # Escape the content first - use HTML mode for better Telegram compatibility
                    safe_content = self._safe_telegram_message(response.content, use_html=True)
                    
                    # Format the response with HTML tags
                    # response_text = f"🤖 <b>{model_name.capitalize()} Response:</b>\n\n{safe_content}"
                    response_text = safe_content

                    # Add metadata if needed
                    if hasattr(update.message, 'reply_to_message') and self._is_user_admin(user_id):
                        response_text += f"\n\n<i>📊 Tokens: {response.tokens_used} | Time: {response.response_time:.2f}s</i>"
                    
                    # Try to edit the thinking message
                    if thinking_message:
                        try:
                            await context.bot.edit_message_text(
                                chat_id=update.message.chat_id,
                                message_id=thinking_message.message_id,
                                text=response_text,
                                parse_mode=ParseMode.HTML
                            )
                        except Exception as edit_error:
                            logger.error(f"Failed to edit thinking message: {edit_error}")
                            # Fallback to sending new message
                            await update.message.reply_text(
                                response_text,
                                parse_mode=ParseMode.HTML
                            )
                    else:
                        await update.message.reply_text(
                            response_text,
                            parse_mode=ParseMode.HTML
                        )
                except Exception as format_error:
                    logger.error(f"Error formatting response: {format_error}")
                    # Fallback to plain text with Markdown escaping
                    safe_content = self._safe_telegram_message(response.content, use_html=False)
                    fallback_text = f"🤖 {model_name.capitalize()} Response:\n\n{safe_content}"
                    
                    if thinking_message:
                        try:
                            await context.bot.edit_message_text(
                                chat_id=update.message.chat_id,
                                message_id=thinking_message.message_id,
                                text=fallback_text,
                                parse_mode=ParseMode.MARKDOWN
                            )
                        except Exception:
                            await update.message.reply_text(
                                fallback_text,
                                parse_mode=ParseMode.MARKDOWN
                            )
                    else:
                        await update.message.reply_text(
                            fallback_text,
                            parse_mode=ParseMode.MARKDOWN
                        )
            else:
                # Escape error message too - use plain text escaping (no HTML)
                safe_error = self._safe_telegram_message(response.error_message or "Unknown error", use_html=False)
                error_text = f"❌ Error generating response: {safe_error}"
                
                if thinking_message:
                    try:
                        await context.bot.edit_message_text(
                            chat_id=update.message.chat_id,
                            message_id=thinking_message.message_id,
                            text=error_text,
                            parse_mode=ParseMode.HTML
                        )
                    except Exception:
                        await update.message.reply_text(error_text)
                else:
                    await update.message.reply_text(error_text)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            # Escape the error message - use plain text escaping (no HTML)
            safe_error = self._safe_telegram_message(str(e), use_html=False)
            error_text = f"❌ An error occurred: {safe_error}"
            
            if thinking_message:
                try:
                    await context.bot.edit_message_text(
                        chat_id=update.message.chat_id,
                        message_id=thinking_message.message_id,
                        text=error_text,
                        parse_mode=ParseMode.HTML
                    )
                except Exception:
                    await update.message.reply_text(error_text)
            else:
                await update.message.reply_text(error_text)
    
    async def info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /info command to display user information."""
        user = update.effective_user
        chat = update.effective_chat
        
        # Get user information
        user_id = user.id
        
        # Check if user is authorized
        if not self._is_user_authorized(user_id):
            error_message = (
                "⛔ **Access Denied**\n\n"
                "You are not authorized to use this chatbot.\n"
                "Please contact the administrator if you believe this is an error."
            )
            await update.message.reply_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return
        
        first_name = user.first_name or "N/A"
        last_name = user.last_name or "N/A"
        username = user.username or "N/A"
        language_code = user.language_code or "N/A"
        
        # Determine member status
        if chat.type in ["group", "supergroup"]:
            member = await context.bot.get_chat_member(chat.id, user.id)
            member_status = member.status
        else:
            member_status = "N/A (private chat)"
        
        # Create info message
        info_message = (
            "📋 **Your Information** \n\n"
            f"🆔 **User ID:** `{user_id}`\n"
            f"👤 **First Name:** `{first_name}`\n"
            f"👥 **Last Name:** `{last_name}`\n"
            f"🐙 **Username:** `@{username}`\n"
            f"🌐 **Language:** `{language_code}`\n"
            f"🔒 **Member Status:** `{member_status}`\n"
        )
        
        await update.message.reply_text(
            info_message,
            parse_mode=ParseMode.MARKDOWN
        )

    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /config command to configure bot settings."""
        user_id = update.effective_user.id
        
        # Check if user is admin (only admins can change configuration)
        if not self._is_user_admin(user_id):
            error_message = (
                "⛔ **Admin Access Required**\n\n"
                "Only administrators can change bot configuration.\n"
                "Please contact the administrator if you need to modify settings."
            )
            await update.message.reply_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return
        
        current_settings = self.user_settings.get(user_id, {})
        
        # Initialize model clients if not done yet (lazy initialization)
        if not self.model_clients:
            await self.initialize()
        
        current_model = current_settings.get("model", self.default_model)
        
        # Create model selection buttons
        model_buttons = []
        available_models = list(self.model_clients.keys())
        
        for model in available_models:
            prefix = "✅ " if model == current_model else ""
            model_buttons.append(
                InlineKeyboardButton(
                    f"{prefix}{model.capitalize()}",
                    callback_data=f"model_{model}"
                )
            )
        
        # Create settings keyboard
        keyboard = [
            model_buttons,
            [InlineKeyboardButton("❌ Close", callback_data='close_settings')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        config_message = (
            "⚙️ **Configuration** \n\n"
            f"🤖 Current Model: `{current_model.capitalize()}`\n\n"
            "Choose a model to switch:"
        )
        
        await update.message.reply_text(
            config_message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def clear_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /clear command to clear conversation history."""
        user_id = update.effective_user.id
        
        # Check if user is authorized
        if not self._is_user_authorized(user_id):
            error_message = (
                "⛔ **Access Denied**\n\n"
                "You are not authorized to use this chatbot.\n"
                "Please contact the administrator if you believe this is an error."
            )
            await update.message.reply_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return
        
        # Clear conversation history for this user
        self.conversation_manager.clear_conversation(user_id)
        
        await update.message.reply_text(
            "🗑️ Conversation history cleared! Starting fresh.",
            parse_mode=ParseMode.HTML
        )
    
    async def handle_file_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle file uploads (photos and documents)."""
        user_id = update.effective_user.id
        
        # Check if user is authorized
        if not self._is_user_authorized(user_id):
            error_message = (
                "⛔ **Access Denied**\n\n"
                "You are not authorized to use this chatbot.\n"
                "Please contact the administrator if you believe this is an error."
            )
            await update.message.reply_text(
                error_message,
                parse_mode=ParseMode.HTML
            )
            return
        
        # Process the file
        result = await self.file_processor.handle_file_with_followup(update, context)
        
        # If file was processed successfully and has content, generate AI response
        # For vision messages, extracted_text is empty but we still want to generate a response
        if result and (result.get("extracted_text", "").strip() or result.get("is_vision", False)):
            # Get user settings for model selection
            if user_id not in self.user_settings:
                if not self.model_clients:
                    await self.initialize()
                self.user_settings[user_id] = {
                    "model": self.default_model or ("ollama" if "ollama" in self.model_clients else "mistral")
                }
            
            settings = self.user_settings.get(user_id, {})
            default_model = self.default_model
            model_name = settings.get("model", default_model)
            
            # Send typing indicator with error handling
            try:
                await update.message.chat.send_action("typing")
            except Exception as e:
                logger.warning(f"Failed to send typing indicator: {e}")
                # Continue anyway - this is not critical
            
            # Send initial "Analyzing..." message
            analyzing_message = None
            try:
                analyzing_message = await update.message.reply_text(
                    "🤔 Analyzing file content...",
                    parse_mode=ParseMode.HTML
                )
            except Exception as e:
                logger.error(f"Failed to send analyzing message: {e}")
                analyzing_message = None
            
            try:
                # Initialize model clients if not done yet
                if not self.model_clients:
                    await self.initialize()
                
                # Get the appropriate model client
                model_client = self.model_clients.get(model_name)
                if not model_client:
                    error_text = "❌ Selected model is not available."
                    if analyzing_message:
                        try:
                            await context.bot.edit_message_text(
                                chat_id=update.message.chat_id,
                                message_id=analyzing_message.message_id,
                                text=error_text,
                                parse_mode=ParseMode.HTML
                            )
                        except Exception:
                            await update.message.reply_text(error_text)
                    else:
                        await update.message.reply_text(error_text)
                    return
                
                # Get conversation history for API
                conversation_history = self.conversation_manager.get_conversation_for_api(user_id, max_messages=20)
                
                # Generate response using conversation history (which now includes the file content)
                response = await model_client.generate_with_history(
                    conversation_history=conversation_history,
                    use_agents_prompt=False
                )
                
                if response.success:
                    # Store assistant response in conversation history
                    self.conversation_manager.add_message(user_id, "assistant", response.content)
                    
                    # Format the response safely
                    safe_content = self._safe_telegram_message(response.content, use_html=True)
                    response_text = safe_content
                    
                    # Add metadata if admin
                    if hasattr(update.message, 'reply_to_message') and self._is_user_admin(user_id):
                        response_text += f"\n\n<i>📊 Tokens: {response.tokens_used} | Time: {response.response_time:.2f}s</i>"
                    
                    # Update the analyzing message
                    if analyzing_message:
                        try:
                            await context.bot.edit_message_text(
                                chat_id=update.message.chat_id,
                                message_id=analyzing_message.message_id,
                                text=response_text,
                                parse_mode=ParseMode.HTML
                            )
                        except Exception as edit_error:
                            logger.error(f"Failed to edit analyzing message: {edit_error}")
                            await update.message.reply_text(
                                response_text,
                                parse_mode=ParseMode.HTML
                            )
                    else:
                        await update.message.reply_text(
                            response_text,
                            parse_mode=ParseMode.HTML
                        )
                else:
                    # Handle error
                    safe_error = self._safe_telegram_message(response.error_message or "Unknown error", use_html=False)
                    error_text = f"❌ Error generating response: {safe_error}"
                    
                    if analyzing_message:
                        try:
                            await context.bot.edit_message_text(
                                chat_id=update.message.chat_id,
                                message_id=analyzing_message.message_id,
                                text=error_text,
                                parse_mode=ParseMode.HTML
                            )
                        except Exception:
                            await update.message.reply_text(error_text)
                    else:
                        await update.message.reply_text(error_text)
                    
            except Exception as e:
                logger.error(f"Error handling file response: {e}")
                safe_error = self._safe_telegram_message(str(e), use_html=False)
                error_text = f"❌ An error occurred: {safe_error}"
                
                if analyzing_message:
                    try:
                        await context.bot.edit_message_text(
                            chat_id=update.message.chat_id,
                            message_id=analyzing_message.message_id,
                            text=error_text,
                            parse_mode=ParseMode.HTML
                        )
                    except Exception:
                        await update.message.reply_text(error_text)
                else:
                    await update.message.reply_text(error_text)

    def run(self):
        """Run the Telegram bot."""
        # Use the telegram library's recommended approach
        # Let the Application manage its own event loop
        try:
            # Create the Application
            self.application = Application.builder().token(self.config.telegram_bot_token).build()
            
            # Add error handler
            self.application.add_error_handler(self.error_handler)
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(CommandHandler("info", self.info))
            self.application.add_handler(CommandHandler("config", self.config_command))
            self.application.add_handler(CommandHandler("clear", self.clear_conversation))
            self.application.add_handler(CallbackQueryHandler(self.settings, pattern="^settings$"))
            self.application.add_handler(CallbackQueryHandler(self.handle_model_selection, pattern="^model_"))

            self.application.add_handler(CallbackQueryHandler(self.close_settings, pattern="^close_settings$"))
            
            # Text message handler
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            
            # File upload handlers
            self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_file_upload))
            self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_file_upload))
            
            # Start the bot - let Application manage the event loop
            logger.info("Starting Telegram bot...")
            self.application.run_polling()
            
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            raise
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors in the telegram bot."""
        logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
        
        # Check if it's a network timeout error
        if hasattr(context.error, '__class__') and context.error.__class__.__name__ == 'TimedOut':
            logger.warning("Network timeout occurred, retrying...")
            # Don't crash on network timeouts
            return
        
        # For other errors, try to notify the user if we have an update
        if update and hasattr(update, 'effective_chat'):
            try:
                error_msg = f"❌ An error occurred: {str(context.error)[:100]}..."
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=error_msg
                )
            except Exception:
                logger.error("Failed to send error message to user")
    

    
    async def stop(self):
        """Stop the Telegram bot."""
        if self.application:
            await self.application.stop()
            await self.application.shutdown()
        
        # Close model clients
        for client in self.model_clients.values():
            await client.close()
        
        logger.info("Telegram bot stopped.")
    
    
    def _markdown_to_html(self, text: str) -> str:
        """Convert Markdown formatting to HTML for Telegram."""
        # Convert headers
        text = re.sub(r'^###\s+(.*$)', r'<b>\1</b>', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.*$)', r'<b>\1</b>', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s+(.*$)', r'<b>\1</b>', text, flags=re.MULTILINE)
        
        # Convert bold **text** to <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Convert italic *text* to <i>text</i>
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        
        # Convert italic _text_ to <i>text</i>
        text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
        
        # Convert inline code `code` to <code>code</code>
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        
        # Convert strikethrough ~text~ to <s>text</s>
        text = re.sub(r'~(.*?)~', r'<s>\1</s>', text)
        
        # Convert spoilers ||text|| to <span class="tg-spoiler">text</span>
        text = re.sub(r'\|\|(.*?)\|\|', r'<span class="tg-spoiler">\1</span>', text)
        
        # Convert links [text](url) to <a href="url">text</a>
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
        
        # Simple HTML escaping - escape basic characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Unescape the tags we just created
        text = text.replace('&lt;b&gt;', '<b>')
        text = text.replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>')
        text = text.replace('&lt;/i&gt;', '</i>')
        text = text.replace('&lt;code&gt;', '<code>')
        text = text.replace('&lt;/code&gt;', '</code>')
        text = text.replace('&lt;s&gt;', '<s>')
        text = text.replace('&lt;/s&gt;', '</s>')
        text = text.replace('&lt;span class="tg-spoiler"&gt;', '<span class="tg-spoiler">')
        text = text.replace('&lt;/span&gt;', '</span>')
        text = text.replace('&lt;a href="', '<a href="')
        text = text.replace('"&gt;', '">')
        text = text.replace('&lt;/a&gt;', '</a>')
        
        return text
    
    def _safe_telegram_message(self, text: str, max_length: int = 4000, use_html: bool = True) -> str:
        """Prepare text for safe Telegram sending with HTML or Markdown formatting."""
        if use_html:
            # Convert Markdown to HTML
            html_text = self._markdown_to_html(text)
            
            # Limit message length
            if len(html_text) > max_length:
                # Try to find a good breaking point
                if len(text) > max_length * 0.8:  # If original was too long
                    html_text = html_text[:max_length-3] + '...'
                else:
                    # HTML expansion made it too long, try to truncate more intelligently
                    # Find the last closing tag and truncate there
                    last_tag = html_text.rfind('</')
                    if last_tag != -1 and last_tag < max_length-10:
                        html_text = html_text[:last_tag] + '</' + html_text[last_tag+2:html_text.find('>', last_tag)] + '>...'
                    else:
                        html_text = html_text[:max_length-3] + '...'
            
            return html_text
        else:
            # Original Markdown escaping (fallback)
            escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}']
            for char in escape_chars:
                text = text.replace(char, f'\\{char}')
            
            if len(text) > max_length:
                text = text[:max_length-3] + '...'
            
            return text
