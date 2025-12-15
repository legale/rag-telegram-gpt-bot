"""Use case for ingesting messages from a file."""

from __future__ import annotations

import os
import re
from typing import List
from pathlib import Path

from src.core.domain import Message
from src.core.interfaces import MessageStore
from src.ingestion.parser import ChatParser, ChatMessage
from src.lib.syslog2 import *


class IngestMessages:
    """
    Use case for parsing and storing messages from a file.
    
    This use case depends only on MessageStore interface, making it testable.
    """

    def __init__(self, message_store: MessageStore, parser: ChatParser = None):
        """
        Initialize use case.

        Args:
            message_store: MessageStore interface implementation
            parser: Optional ChatParser instance (creates default if None)
        """
        self.message_store = message_store
        self.parser = parser or ChatParser()

    def execute(self, file_path: str) -> int:
        """
        Parse file and store messages.

        Args:
            file_path: Path to chat dump file

        Returns:
            Number of messages saved
        """
        if not file_path:
            raise ValueError("file_path must be provided")

        syslog2(LOG_NOTICE, "starting parse and store messages", file_path=file_path)

        # Parse file
        syslog2(LOG_NOTICE, "parsing file", file_path=file_path)
        chat_messages = self.parser.parse_file(file_path)
        syslog2(LOG_NOTICE, "file parsed", messages_count=len(chat_messages))

        # Determine chat_id from filename or default
        filename = os.path.basename(file_path)
        chat_id_match = re.search(r"telegram_dump_(-?\d+)", filename)
        chat_id = chat_id_match.group(1) if chat_id_match else "unknown_chat"
        syslog2(LOG_NOTICE, "identified chat_id", chat_id=chat_id)

        # Convert ChatMessage to domain Message objects
        syslog2(LOG_NOTICE, "preparing messages for storage")
        domain_messages: List[Message] = []
        for i, msg in enumerate(chat_messages, 1):
            print(f"\rProcessing messages: {i}/{len(chat_messages)}", flush=True, end="")
            # Composite ID: {chat_id}_{msg_id} to ensure global uniqueness
            composite_id = f"{chat_id}_{msg.id}"
            
            domain_message = Message(
                id=composite_id,
                chat_id=chat_id,
                from_id=msg.sender,
                text=msg.content,
                timestamp=msg.timestamp,
                meta={}  # Can be extended with additional metadata
            )
            domain_messages.append(domain_message)
        print()  # Newline after progress
        syslog2(LOG_NOTICE, "messages prepared for storage", count=len(domain_messages))

        # Save messages using MessageStore
        syslog2(LOG_NOTICE, "saving messages to store")
        try:
            saved_count = self.message_store.save_batch(domain_messages)
            skipped_count = len(domain_messages) - saved_count
            if skipped_count > 0:
                syslog2(LOG_NOTICE, "messages saved", inserted=saved_count, skipped=skipped_count, total=len(domain_messages))
            else:
                syslog2(LOG_NOTICE, "messages saved", inserted=saved_count, total=len(domain_messages))
        except Exception as e:
            syslog2(LOG_ERR, "error saving messages", error=str(e))
            raise

        syslog2(LOG_NOTICE, "ingest messages complete", messages_saved=saved_count)
        return saved_count

