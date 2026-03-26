"""Response streamer for Telegram with typing indicators and message chunking.

This module handles streaming LLM responses back to Telegram with proper
typing indicators, message chunking, and rate limiting.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


class ResponseStreamer:
    """Stream responses to Telegram with typing indicators and chunking."""

    def __init__(
        self,
        chunk_size: int = 4000,
        typing_delay: float = 0.1,
        message_delay: float = 0.05,
        edit_interval: float = 2.0,  # Minimum time between edits in seconds
        edit_buffer_size: int = 1000,  # Minimum buffer size before editing
    ) -> None:
        self.chunk_size = chunk_size
        self.typing_delay = typing_delay
        self.message_delay = message_delay
        self.edit_interval = edit_interval
        self.edit_buffer_size = edit_buffer_size

    async def stream_response(
        self,
        bot: Bot,
        chat_id: int,
        text_generator: AsyncGenerator[str, None],
        **kwargs: Any,
    ) -> None:
        """Stream message to Telegram chat with typing indicators."""
        logger.debug(
            f"Starting response streaming for chat_id={chat_id}, chunk_size={self.chunk_size}"
        )

        # Start typing indicator
        typing_task = asyncio.create_task(
            self._send_typing_indicator(bot, chat_id)
        )

        try:
            # Buffer for accumulating response
            buffer = ""
            message_count = 0

            # Process stream
            async for chunk in text_generator:
                buffer += chunk

                # Send chunks when buffer reaches size
                while len(buffer) >= self.chunk_size:
                    send_size = self.chunk_size
                    
                    # Try to break at sentence boundary
                    if "." in buffer[:send_size]:
                        last_period = buffer[:send_size].rfind(".")
                        if last_period > send_size // 2:  # Only if reasonable
                            send_size = last_period + 1

                    chunk_to_send = buffer[:send_size]
                    buffer = buffer[send_size:]

                    await self._send_chunk(bot, chat_id, chunk_to_send, message_count, **kwargs)
                    message_count += 1

                    # Small delay between messages
                    await asyncio.sleep(self.message_delay)

            # Send remaining buffer
            if buffer.strip():
                await self._send_chunk(bot, chat_id, buffer, message_count, **kwargs)

            logger.debug(
                f"Response streaming completed for chat_id={chat_id}, total_chunks={message_count + (1 if buffer.strip() else 0)}"
            )

        except Exception as exc:
            logger.error(
                f"Error during response streaming for chat_id={chat_id}: {exc}"
            )
            raise

        finally:
            # Stop typing indicator
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

    async def _send_typing_indicator(self, bot: Bot, chat_id: int) -> None:
        """Send typing indicator periodically."""
        while True:
            try:
                await bot.send_chat_action(
                    chat_id=chat_id,
                    action="typing",
                )
                await asyncio.sleep(self.typing_delay * 10)  # Telegram typing lasts ~5 seconds
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    f"Failed to send typing indicator for chat_id={chat_id}: {exc}"
                )
                await asyncio.sleep(1.0)

    async def _send_chunk(
        self,
        bot: Bot,
        chat_id: int,
        chunk: str,
        message_count: int,
        **kwargs: Any,
    ) -> None:
        """Send a chunk of the response as a Telegram message."""
        if not chunk.strip():
            return

        try:
            # For first message, send as reply if reply_to_message_id is provided
            if message_count == 0 and "reply_to_message_id" in kwargs:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                    reply_to_message_id=kwargs["reply_to_message_id"],
                    disable_web_page_preview=True,
                )
            else:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )

            logger.debug(
                f"Chunk sent for chat_id={chat_id}, length={len(chunk)}, message_number={message_count}"
            )

        except TelegramError as exc:
            logger.error(
                f"Failed to send message chunk for chat_id={chat_id}, length={len(chunk)}: {exc}"
            )
            raise

    async def stream_with_edit(
        self,
        bot: Bot,
        chat_id: int,
        message_id: int,
        text_generator: AsyncGenerator[str, None],
        **kwargs: Any,
    ) -> None:
        """Stream response by editing a single message (for inline mode)."""
        logger.debug(
            f"Starting response streaming with edit for chat_id={chat_id}, message_id={message_id}"
        )

        # Start typing indicator
        typing_task = asyncio.create_task(
            self._send_typing_indicator(bot, chat_id)
        )

        try:
            buffer = ""
            last_edit_time = asyncio.get_event_loop().time()

            async for chunk in text_generator:
                buffer += chunk

                # Edit message periodically or when buffer grows
                current_time = asyncio.get_event_loop().time()
                if current_time - last_edit_time > self.edit_interval or len(buffer) >= self.edit_buffer_size:
                    try:
                        logger.debug(
                            f"Editing message chat_id={chat_id}, message_id={message_id}, "
                            f"text_length={len(buffer) + 1}, preview={repr(buffer[:100])}..."
                        )
                        await bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=message_id,
                            text=buffer + "▌",  # Cursor indicator
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
                        last_edit_time = current_time
                    except TelegramError as exc:
                        error_msg = str(exc).lower()
                        # Ignore "message not modified" errors
                        if "message is not modified" not in error_msg:
                            logger.debug(
                                f"TelegramError details: {exc.__dict__}"
                            )
                            
                            # Check if it's a message not found error (400 Bad Request)
                            is_message_not_found = any(
                                phrase in error_msg
                                for phrase in [
                                    "message not found",
                                    "bad request",
                                    "message to edit not found",
                                    "400",
                                    "message_id invalid"
                                ]
                            )
                            
                            if is_message_not_found:
                                logger.warning(
                                    f"Message {message_id} not found for chat_id={chat_id}, stopping edit attempts. "
                                    f"Original message may have been deleted."
                                )
                                # Stop trying to edit this message
                                raise exc
                            
                            logger.warning(
                                f"Failed to edit message during streaming for chat_id={chat_id}: {exc}. "
                                f"Text length: {len(buffer)}, preview: {repr(buffer[:200])}"
                            )
                            # If it's a rate limit error, increase the edit interval and wait
                            if "flood control" in error_msg or "too many requests" in error_msg:
                                # Try to extract retry time from error message
                                retry_seconds = 5  # Default wait time
                                import re
                                match = re.search(r'retry in (\d+) seconds', error_msg, re.IGNORECASE)
                                if match:
                                    retry_seconds = int(match.group(1))
                                
                                # Increase edit interval significantly
                                self.edit_interval = min(self.edit_interval * 2.0, 30.0)  # Cap at 30 seconds
                                logger.info(f"Rate limited, increasing edit interval to {self.edit_interval}s, waiting {retry_seconds}s")
                                
                                # Wait for the retry time before trying again
                                await asyncio.sleep(retry_seconds)
                                # Update last_edit_time to prevent immediate retry
                                last_edit_time = asyncio.get_event_loop().time()

            # Final edit without cursor
            if buffer:
                try:
                    logger.debug(
                        f"Final edit message chat_id={chat_id}, message_id={message_id}, "
                        f"text_length={len(buffer)}, preview={repr(buffer[:100])}..."
                    )
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=buffer,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
                except TelegramError as exc:
                    error_msg = str(exc).lower()
                    if "message is not modified" not in error_msg:
                        logger.debug(
                            f"TelegramError details: {exc.__dict__}"
                        )
                        
                        # Check if it's a message not found error (400 Bad Request)
                        is_message_not_found = any(
                            phrase in error_msg
                            for phrase in [
                                "message not found",
                                "bad request",
                                "message to edit not found",
                                "400",
                                "message_id invalid"
                            ]
                        )
                        
                        if is_message_not_found:
                            logger.warning(
                                f"Message {message_id} not found for chat_id={chat_id} during final edit. "
                                f"Original message may have been deleted. Cannot send fallback from streamer."
                            )
                        else:
                            logger.warning(
                                f"Failed to finalize message edit for chat_id={chat_id}: {exc}. "
                                f"Text length: {len(buffer)}, preview: {repr(buffer[:200])}"
                            )

            logger.debug(
                f"Response streaming with edit completed for chat_id={chat_id}, final_length={len(buffer)}"
            )

        except Exception as exc:
            logger.error(
                f"Error during response streaming with edit for chat_id={chat_id}: {exc}"
            )
            raise

        finally:
            # Stop typing indicator
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

    async def create_progressive_response(
        self,
        bot: Bot,
        chat_id: int,
        initial_text: str = "Thinking...",
        **kwargs: Any,
    ) -> tuple[int, asyncio.Queue[str]]:
        """Create a progressive response message that gets updated.
        
        Returns:
            Tuple of (message_id, queue) where queue is used to send text updates
        """
        try:
            # Send initial message
            message = await bot.send_message(
                chat_id=chat_id,
                text=initial_text,
                parse_mode=ParseMode.HTML,
                **kwargs,
            )

            # Create queue for updates
            update_queue: asyncio.Queue[str] = asyncio.Queue()

            # Start update task
            asyncio.create_task(
                self._process_progressive_updates(
                    bot, chat_id, message.message_id, update_queue
                )
            )

            return message.message_id, update_queue

        except TelegramError as exc:
            logger.error(
                f"Failed to create progressive response for chat_id={chat_id}: {exc}"
            )
            raise

    async def _process_progressive_updates(
        self,
        bot: Bot,
        chat_id: int,
        message_id: int,
        update_queue: asyncio.Queue[str],
    ) -> None:
        """Process updates for progressive response."""
        buffer = ""
        last_edit_time = asyncio.get_event_loop().time()

        while True:
            try:
                # Wait for update with timeout
                try:
                    chunk = await asyncio.wait_for(update_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # No updates for 30 seconds, assume done
                    break

                if chunk is None:  # Sentinel for completion
                    break

                buffer += chunk

                # Edit message periodically
                current_time = asyncio.get_event_loop().time()
                if current_time - last_edit_time > 0.5 or len(buffer) >= 1000:
                    try:
                        await bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=message_id,
                            text=buffer,
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
                        last_edit_time = current_time
                    except TelegramError as exc:
                        if "message is not modified" not in str(exc).lower():
                            logger.warning(
                                f"Failed to update progressive response for chat_id={chat_id}: {exc}"
                            )

            except Exception as exc:
                logger.error(
                    f"Error in progressive update processing for chat_id={chat_id}: {exc}"
                )
                break

        # Final edit
        if buffer:
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=buffer,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            except TelegramError as exc:
                if "message is not modified" not in str(exc).lower():
                    logger.warning(
                        f"Failed to finalize progressive response for chat_id={chat_id}: {exc}"
                    )

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on response streamer."""
        return {
            "healthy": True,
            "component": "response_streamer",
            "config": {
                "chunk_size": self.chunk_size,
                "typing_delay": self.typing_delay,
                "message_delay": self.message_delay,
                "edit_interval": self.edit_interval,
                "edit_buffer_size": self.edit_buffer_size,
            },
        }