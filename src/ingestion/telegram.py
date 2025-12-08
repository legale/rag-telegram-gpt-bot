#!/usr/bin/env python3
import sys
import os

try:
    from telethon.sync import TelegramClient
    from telethon.tl.types import Channel, Chat
except ImportError:
    # If dependencies are missing, try to re-run with poetry
    poetry_cmd = "poetry"
    # Check if poetry is in PATH, if not try default pipx location
    if os.system("which poetry > /dev/null 2>&1") != 0:
        if os.path.exists("/home/ru/.local/bin/poetry"):
            poetry_cmd = "/home/ru/.local/bin/poetry"
            
    try:
        os.execvp(poetry_cmd, [poetry_cmd, "run", "python"] + sys.argv)
    except FileNotFoundError:
        print(f"Error: '{poetry_cmd}' not found. Please ensure poetry is installed and in PATH.", file=sys.stderr)
        sys.exit(1)

from typing import List, Optional
import json
from datetime import datetime
import os
import argparse
import sys
import logging
from src.core.syslog2 import *

# Suppress Telethon debug logging messages
logging.getLogger('telethon').setLevel(logging.WARNING)

# Helper to serialize datetime
def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

class TelegramFetcher:
    def __init__(self, api_id: int, api_hash: str, session_name: str = "legale_bot_session"):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(session_name, api_id, api_hash)

    def _find_chat(self, id_or_name):
        """Finds a chat by ID (int/str) or Title (str)."""
        # Try to convert to int if it looks like one
        target_id = None
        try:
            target_id = int(id_or_name)
        except ValueError:
            pass

        syslog2(LOG_INFO, "searching for chat", name=id_or_name)
        
        # Iterate dialogs to find match
        # Note: This might be slow if there are many dialogs, but it's reliable for Titles.
        for dialog in self.client.iter_dialogs():
            # Check ID
            if target_id is not None and dialog.id == target_id:
                return dialog
            # Check Title
            if dialog.name == id_or_name:
                return dialog
                
        return None

    def list_channels(self):
        """Lists all available dialogs (channels/chats)."""
        with self.client:
            print(f"{'ID':<15} | {'Name'}")
            print("-" * 50)
            for dialog in self.client.iter_dialogs():
                print(f"{dialog.id:<15} | {dialog.name}")

    def list_members(self, id_or_name: str):
        """Lists members of a specific chat."""
        with self.client:
            chat = self._find_chat(id_or_name)
            if not chat:
                syslog2(LOG_ERR, "chat not found", name=id_or_name)
                return

            print(f"Members of '{chat.name}' ({chat.id}):")
            print(f"{'ID':<15} | {'Name'} | {'Username'}")
            print("-" * 60)
            
            try:
                for user in self.client.iter_participants(chat):
                    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
                    username = f"@{user.username}" if user.username else "N/A"
                    print(f"{user.id:<15} | {name:<20} | {username}")
            except Exception as e:
                syslog2(LOG_ERR, "fetch members failed", error=str(e))

    def dump_chat(self, id_or_name: str, limit: int = 1000, output_file: Optional[str] = None):
        """Dumps messages from a chat."""
        with self.client:
            chat = self._find_chat(id_or_name)
            if not chat:
                syslog2(LOG_ERR, "chat not found", name=id_or_name)
                return

            syslog2(LOG_INFO, "found chat", name=chat.name, id=chat.id)
            syslog2(LOG_INFO, "fetching messages", limit=limit)
            
            messages_data = []
            count = 0
            last_date = None
            
            # Use iter_messages with limit
            for message in self.client.iter_messages(chat, limit=limit):
                count += 1
                if message.date:
                    last_date = message.date
                
                # Progress update on every message (overwrites same line)
                date_str = last_date.strftime('%Y-%m-%d %H:%M') if last_date else "Unknown"
                sys.stdout.write(f"\rFetched {count} messages... (Last: {date_str})")
                sys.stdout.flush()

                if message.text:
                    sender_name = "Unknown"
                    if message.sender:
                        if hasattr(message.sender, 'first_name') and message.sender.first_name:
                             sender_name = message.sender.first_name
                             if hasattr(message.sender, 'last_name') and message.sender.last_name:
                                 sender_name += f" {message.sender.last_name}"
                        elif hasattr(message.sender, 'title'):
                            sender_name = message.sender.title
                            
                    msg_data = {
                        "id": message.id,
                        "date": message.date,
                        "sender": sender_name,
                        "content": message.text
                    }
                    messages_data.append(msg_data)
            
            print() # Newline after progress bar
            
            messages_data.sort(key=lambda x: x['date'])
            
            # Form output filename based on chat ID
            chat_id = abs(chat.id)  # Use absolute value for negative IDs
            
            if not output_file:
                # Default: save in current directory
                output_file = f"telegram_dump_{chat_id}.json"
            elif os.path.isdir(output_file):
                # If output_file is a directory, create filename with chat ID in that directory
                output_file = os.path.join(output_file, f"telegram_dump_{chat_id}.json")
            # If output_file is already a full path to a file, use it as is
                
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(messages_data, f, default=json_serial, ensure_ascii=False, indent=2)
            print(f"Saved {len(messages_data)} messages to {output_file}")
            syslog2(LOG_INFO, "saved messages", count=len(messages_data), path=output_file)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Configuration
    API_ID = os.getenv("TELEGRAM_API_ID")
    API_HASH = os.getenv("TELEGRAM_API_HASH")
    
    if not API_ID or not API_HASH:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env file")
        sys.exit(1)
        
    # Convert API_ID to int
    try:
        API_ID = int(API_ID)
    except ValueError:
        print("Error: TELEGRAM_API_ID must be an integer")
        sys.exit(1)

    
    parser = argparse.ArgumentParser(description="Telegram Utility Script")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # list_chan
    subparsers.add_parser("list_chan", help="List all available channels/chats")
    
    # list_chan_members
    members_parser = subparsers.add_parser("list_chan_members", help="List members of a channel")
    members_parser.add_argument("target", help="Chat ID or Name")
    
    # dump_chan
    dump_parser = subparsers.add_parser("dump_chan", help="Dump chat history")
    dump_parser.add_argument("target", help="Chat ID or Name")
    dump_parser.add_argument("--limit", type=int, default=1000, help="Number of messages to fetch")
    dump_parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()
    
    fetcher = TelegramFetcher(API_ID, API_HASH)
    
    if args.command == "list_chan":
        fetcher.list_channels()
    elif args.command == "list_chan_members":
        fetcher.list_members(args.target)
    elif args.command == "dump_chan":
        fetcher.dump_chat(args.target, limit=args.limit, output_file=args.output)
    else:
        parser.print_help()
