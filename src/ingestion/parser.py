from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import json
from src.core.syslog2 import *

@dataclass
class ChatMessage:
    id: str
    timestamp: datetime
    sender: str
    content: str
    
class ChatParser:
    """Parses chat dump files into structured ChatMessage objects."""
    
    def parse_file(self, file_path: str) -> List[ChatMessage]:
        """
        Parses a chat dump file. Supports JSON format from TelegramFetcher.
        
        Args:
            file_path: Path to the chat dump file.
            
        Returns:
            List of ChatMessage objects.
        """
        messages = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Check if file is JSON
                if file_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            # Handle Telegram dump format
                            msg = ChatMessage(
                                id=str(item.get('id')),
                                timestamp=datetime.fromisoformat(item.get('date')),
                                sender=item.get('sender', 'Unknown'),
                                content=item.get('content', '')
                            )
                            messages.append(msg)
                    else:
                        # Fallback or error for unexpected JSON structure
                        syslog2(LOG_WARNING, "unexpected json structure", file_path=file_path)
                else:
                    # Fallback to line-based parsing for text files
                    for i, line in enumerate(f):
                        if line.strip():
                            msg = ChatMessage(
                                id=str(i),
                                timestamp=datetime.now(), # Placeholder
                                sender="Unknown", 
                                content=line.strip()
                            )
                            messages.append(msg)
                            
        except Exception as e:
            syslog2(LOG_ERR, "file read failed", error=str(e))
            raise
            
        return messages
