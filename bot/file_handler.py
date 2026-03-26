"""
File handler for processing uploaded files (images, PDFs, DOCX, etc.)
Downloads files from Telegram, extracts text content, and integrates with conversation.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, BinaryIO
from datetime import datetime
import asyncio

from telegram import Update, Message
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file uploads from Telegram users."""
    
    def __init__(self, upload_dir: str = "data/uploads"):
        """
        Initialize the file handler.
        
        Args:
            upload_dir: Directory to store uploaded files
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file extensions and their handlers
        self.supported_extensions = {
            '.jpg': 'image', '.jpeg': 'image', '.png': 'image', 
            '.gif': 'image', '.bmp': 'image', '.webp': 'image',
            '.pdf': 'document', '.docx': 'document', '.doc': 'document',
            '.txt': 'document', '.rtf': 'document', '.md': 'document'
        }
    
    async def download_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[Tuple[Path, str, str]]:
        """
        Download a file from Telegram message.
        
        Args:
            update: Telegram update object
            context: Telegram context object
            
        Returns:
            Tuple of (file_path, file_name, file_type) or None if download fails
        """
        message = update.message
        
        # Determine file type and get file object
        file_obj = None
        file_name = "unknown"
        file_type = "unknown"
        
        if message.photo:
            # Get the largest photo
            file_obj = await message.photo[-1].get_file()
            file_name = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            file_type = "image"
        elif message.document:
            document = message.document
            file_obj = await document.get_file()
            file_name = document.file_name or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            file_type = "document"
        else:
            logger.warning(f"Unsupported message type for file download: {message}")
            return None
        
        if not file_obj:
            logger.error("Failed to get file object from message")
            return None
        
        # Create user-specific directory
        user_id = update.effective_user.id
        user_dir = self.upload_dir / str(user_id)
        user_dir.mkdir(exist_ok=True)
        
        # Generate safe filename
        safe_filename = self._sanitize_filename(file_name)
        file_path = user_dir / safe_filename
        
        try:
            # Download the file
            await file_obj.download_to_drive(file_path)
            logger.info(f"Downloaded file: {file_path} ({file_type})")
            
            return (file_path, safe_filename, file_type)
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove unsafe characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Keep only alphanumeric, dots, hyphens, and underscores
        import re
        filename = re.sub(r'[^\w\.\-]', '_', filename)
        # Limit length
        if len(filename) > 100:
            name, ext = os.path.splitext(filename)
            filename = name[:95] + ext
        return filename
    
    def get_file_extension(self, filename: str) -> str:
        """
        Get file extension in lowercase.
        
        Args:
            filename: Filename
            
        Returns:
            File extension with dot (e.g., '.pdf')
        """
        return os.path.splitext(filename)[1].lower()
    
    def is_supported_file(self, filename: str) -> bool:
        """
        Check if file type is supported.
        
        Args:
            filename: Filename
            
        Returns:
            True if file type is supported
        """
        ext = self.get_file_extension(filename)
        return ext in self.supported_extensions
    
    def get_file_category(self, filename: str) -> Optional[str]:
        """
        Get file category (image/document) based on extension.
        
        Args:
            filename: Filename
            
        Returns:
            File category or None if not supported
        """
        ext = self.get_file_extension(filename)
        return self.supported_extensions.get(ext)
    
    async def cleanup_user_files(self, user_id: int, keep_recent: int = 10) -> None:
        """
        Clean up old files for a user, keeping only the most recent ones.
        
        Args:
            user_id: Telegram user ID
            keep_recent: Number of recent files to keep
        """
        user_dir = self.upload_dir / str(user_id)
        if not user_dir.exists():
            return
        
        # Get all files sorted by modification time (newest first)
        files = list(user_dir.glob("*"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Remove old files
        for file in files[keep_recent:]:
            try:
                file.unlink()
                logger.debug(f"Cleaned up old file: {file}")
            except Exception as e:
                logger.warning(f"Failed to clean up file {file}: {e}")
    
    def get_user_file_count(self, user_id: int) -> int:
        """
        Get number of files uploaded by a user.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Number of files
        """
        user_dir = self.upload_dir / str(user_id)
        if not user_dir.exists():
            return 0
        return len(list(user_dir.glob("*")))
    
    async def process_uploaded_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[Dict[str, Any]]:
        """
        Process an uploaded file: download and prepare for content extraction.
        
        Args:
            update: Telegram update object
            context: Telegram context object
            
        Returns:
            Dictionary with file info or None if processing fails
        """
        # Download the file
        result = await self.download_file(update, context)
        if not result:
            return None
        
        file_path, filename, file_type = result
        
        # Check if file is supported
        if not self.is_supported_file(filename):
            logger.warning(f"Unsupported file type: {filename}")
            # Clean up unsupported file
            try:
                file_path.unlink()
            except:
                pass
            return None
        
        # Get file category
        file_category = self.get_file_category(filename)
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Prepare file info dictionary
        file_info = {
            "path": str(file_path),
            "filename": filename,
            "type": file_type,
            "category": file_category,
            "size": file_size,
            "user_id": update.effective_user.id,
            "timestamp": datetime.now().isoformat(),
            "message_id": update.message.message_id if update.message else None
        }
        
        logger.info(f"Processed file: {filename} ({file_category}, {file_size} bytes)")
        
        return file_info


# Singleton instance
_file_handler: Optional[FileHandler] = None

def get_file_handler(upload_dir: str = "data/uploads") -> FileHandler:
    """
    Get or create the file handler singleton.
    
    Args:
        upload_dir: Directory to store uploaded files
        
    Returns:
        FileHandler instance
    """
    global _file_handler
    if _file_handler is None:
        _file_handler = FileHandler(upload_dir)
    return _file_handler