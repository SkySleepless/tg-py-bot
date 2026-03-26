"""
File processor orchestrator that handles the complete file upload pipeline.
Coordinates file handling, content extraction, and conversation integration.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from .file_handler import get_file_handler
from .content_extractor import get_content_extractor
from .conversation_manager import get_conversation_manager

logger = logging.getLogger(__name__)


class FileProcessor:
    """Orchestrates the complete file upload and processing pipeline."""
    
    def __init__(self):
        """Initialize the file processor."""
        self.file_handler = get_file_handler()
        self.content_extractor = get_content_extractor()
        self.conversation_manager = get_conversation_manager()
        
        # Image extensions that should use vision API
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    async def process_file_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
        """
        Process a file upload message from Telegram.
        
        Args:
            update: Telegram update object
            context: Telegram context object
            
        Returns:
            Dictionary with processing results
        """
        user_id = update.effective_user.id
        message = update.message
        
        # Send initial processing message
        processing_msg = await message.reply_text(
            "📎 Processing your file...",
            parse_mode="HTML"
        )
        
        try:
            # Step 1: Download the file
            file_info = await self.file_handler.process_uploaded_file(update, context)
            if not file_info:
                await processing_msg.edit_text("❌ Failed to download the file.")
                return {"success": False, "error": "File download failed"}
            
            # Step 2: Check if file is an image
            if self._is_image_file(file_info["filename"]):
                # Handle image with vision API format
                return await self._process_image_with_vision(
                    user_id, file_info, processing_msg
                )
            else:
                # Handle non-image files with traditional extraction
                return await self._process_document_file(
                    user_id, file_info, processing_msg
                )
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            await processing_msg.edit_text(f"❌ An error occurred while processing the file: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _format_file_content(self, filename: str, extracted_text: str, max_length: int = 4000) -> str:
        """
        Format extracted file content for conversation history.
        
        Args:
            filename: Original filename
            extracted_text: Extracted text content
            max_length: Maximum length of formatted content
            
        Returns:
            Formatted content string
        """
        # Truncate if too long
        if len(extracted_text) > max_length:
            extracted_text = extracted_text[:max_length] + "...\n[Content truncated due to length]"
        
        # Format with file context
        formatted = f"[File: {filename}]\n\n{extracted_text}"
        
        return formatted
    
    def _create_vision_message(self, file_info: Dict[str, Any], prompt: str = "Describe this image in detail.") -> Dict[str, Any]:
        """
        Create a vision message in OpenAI-compatible format.
        
        Args:
            file_info: File information dictionary
            prompt: Text prompt to accompany the image
            
        Returns:
            Vision message dictionary
        """
        import base64
        from pathlib import Path
        
        image_path = file_info["path"]
        filename = file_info["filename"]
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine image MIME type from file extension
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        # Create vision message
        vision_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}\n\n[Image: {filename}]"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}"
                    }
                }
            ]
        }
        
        return vision_message
    
    def _is_image_file(self, filename: str) -> bool:
        """
        Check if a file is an image based on its extension.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if file is an image, False otherwise
        """
        from pathlib import Path
        ext = Path(filename).suffix.lower()
        return ext in self.image_extensions
    
    async def handle_file_with_followup(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[Dict[str, Any]]:
        """
        Process a file and prepare for AI response generation.
        
        Args:
            update: Telegram update object
            context: Telegram context object
            
        Returns:
            Processing results or None if failed
        """
        # Process the file
        result = await self.process_file_message(update, context)
        
        if not result["success"]:
            return None
        
        # If content was extracted, we can now generate AI response
        if result.get("extracted_text", "").strip():
            # The content is already added to conversation history
            # Return the result so the bot can generate a response
            return result
        
        return result
    
    async def _process_image_with_vision(self, user_id: int, file_info: Dict[str, Any],
                                        processing_msg: Any) -> Dict[str, Any]:
        """
        Process an image file using vision API format.
        
        Args:
            user_id: User ID
            file_info: File information dictionary
            processing_msg: Processing message to update
            
        Returns:
            Processing results
        """
        try:
            # Create vision message
            vision_message = self._create_vision_message(file_info)
            
            # Add vision message to conversation history
            self.conversation_manager.add_message(user_id, "user", vision_message["content"])
            
            # Don't clean up file - keep it saved for future reference
            # self.content_extractor.cleanup_file(file_info["path"])
            
            # Update processing message
            file_size_kb = file_info["size"] / 1024
            success_msg = (
                f"✅ Image processed with vision API!\n"
                f"🖼️ **{file_info['filename']}** ({file_size_kb:.1f} KB)\n"
                f"👁️ Ready for vision analysis"
            )
            
            await processing_msg.edit_text(success_msg, parse_mode="HTML")
            
            # Prepare result
            result = {
                "success": True,
                "content": vision_message,  # Return the full vision message
                "extracted_text": "",  # No extracted text for vision
                "file_info": file_info,
                "extraction_metadata": {"processing": "vision_api"},
                "is_vision": True  # Flag to indicate this is a vision message
            }
            
            logger.info(f"Successfully processed image with vision API for user {user_id}: {file_info['filename']}")
            return result
            
        except Exception as e:
            logger.error(f"Vision processing failed for {file_info['filename']}: {e}")
            # Fall back to OCR extraction
            logger.info(f"Falling back to OCR for {file_info['filename']}")
            return await self._process_document_file(user_id, file_info, processing_msg)
    
    async def _process_document_file(self, user_id: int, file_info: Dict[str, Any],
                                    processing_msg: Any) -> Dict[str, Any]:
        """
        Process a document file with traditional text extraction.
        
        Args:
            user_id: User ID
            file_info: File information dictionary
            processing_msg: Processing message to update
            
        Returns:
            Processing results
        """
        try:
            # Extract content
            extraction_result = self.content_extractor.extract_text(
                file_info["path"],
                file_info["category"]
            )
            
            if not extraction_result["success"]:
                # Don't clean up file even on failure - keep it for debugging
                # self.content_extractor.cleanup_file(file_info["path"])
                
                error_msg = f"❌ Failed to extract text: {extraction_result['error']}"
                if "OCR not available" in error_msg:
                    error_msg += "\n\nInstall OCR dependencies: `pip install pytesseract pillow`"
                elif "PDF library" in error_msg:
                    error_msg += "\n\nInstall PDF library: `pip install pdfplumber`"
                elif "python-docx" in error_msg:
                    error_msg += "\n\nInstall DOCX library: `pip install python-docx`"
                
                await processing_msg.edit_text(error_msg)
                return {"success": False, "error": extraction_result["error"]}
            
            # Format content for conversation
            extracted_text = extraction_result["content"]
            if not extracted_text.strip():
                # Don't clean up file even if no text found - keep it for debugging
                # self.content_extractor.cleanup_file(file_info["path"])
                await processing_msg.edit_text("📄 File processed, but no text content was found.")
                return {"success": True, "content": "", "file_info": file_info}
            
            # Format the message with file context
            formatted_content = self._format_file_content(file_info["filename"], extracted_text)
            
            # Add to conversation history
            self.conversation_manager.add_message(user_id, "user", formatted_content)
            
            # Don't clean up file - keep it saved for future reference
            # self.content_extractor.cleanup_file(file_info["path"])
            
            # Update processing message
            file_size_kb = file_info["size"] / 1024
            success_msg = (
                f"✅ File processed successfully!\n"
                f"📄 **{file_info['filename']}** ({file_size_kb:.1f} KB)\n"
                f"📝 Extracted {len(extracted_text.split())} words"
            )
            
            await processing_msg.edit_text(success_msg, parse_mode="HTML")
            
            # Prepare result
            result = {
                "success": True,
                "content": formatted_content,
                "extracted_text": extracted_text,
                "file_info": file_info,
                "extraction_metadata": extraction_result["metadata"],
                "is_vision": False
            }
            
            logger.info(f"Successfully processed document for user {user_id}: {file_info['filename']}")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            await processing_msg.edit_text(f"❌ An error occurred while processing the file: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_file_summary(self, file_info: Dict[str, Any], extraction_metadata: Dict[str, Any]) -> str:
        """
        Generate a summary of file processing results.
        
        Args:
            file_info: File information dictionary
            extraction_metadata: Extraction metadata
            
        Returns:
            Summary string
        """
        filename = file_info.get("filename", "Unknown")
        file_type = file_info.get("type", "Unknown")
        file_size_kb = file_info.get("size", 0) / 1024
        
        summary = f"**File:** {filename}\n"
        summary += f"**Type:** {file_type}\n"
        summary += f"**Size:** {file_size_kb:.1f} KB\n"
        
        if extraction_metadata:
            if "pages" in extraction_metadata:
                summary += f"**Pages:** {extraction_metadata['pages']}\n"
            if "library" in extraction_metadata:
                summary += f"**Extracted with:** {extraction_metadata['library']}\n"
            if "ocr_engine" in extraction_metadata:
                summary += f"**OCR Engine:** {extraction_metadata['ocr_engine']}\n"
        
        return summary


# Singleton instance
_file_processor: Optional[FileProcessor] = None

def get_file_processor() -> FileProcessor:
    """
    Get or create the file processor singleton.
    
    Returns:
        FileProcessor instance
    """
    global _file_processor
    if _file_processor is None:
        _file_processor = FileProcessor()
    return _file_processor