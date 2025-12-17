
import json
import logging
from typing import List, Optional
from datetime import datetime

from src.storage.db import Database, MessageModel

# Setup logger
logger = logging.getLogger(__name__)

class AliasDiscoveryService:
    """Service for discovering user aliases using LLM."""
    
    SYSTEM_PROMPT = (
        "ты ассистент поиска по хистори чата. Пользователи обычно скрываются за никами, "
        "поэтому их username обычно редко имеет что-то общее с реальным именем. "
        "Но в разговорах другие пользователи часто обращаются по настоящему имени. "
        "Проанализируй сообщения пользователя и соседние с ним сообщения, чтобы составить список alias для его обычного имени. "
        "Верни в формате: {\"user\": \"<username>\", \"alias\": \"'name1' 'name2 super man' 'name3'\"}"
    )

    def __init__(self, db: Database, bot_instance):
        """
        Initialize the service.
        
        Args:
            db: Database instance
            bot_instance: LegaleBot instance (or compatible interface with .complete method)
        """
        self.db = db
        self.bot = bot_instance

    async def discover_aliases(self, username: str, max_messages: int = 15) -> List[str]:
        """
        Discover aliases for a user.
        
        Args:
            username: Username to analyze
            max_messages: Number of recent user messages to use for context
            
        Returns:
            List of discovered aliases
        """
        # 1. Gather context
        # Get recent messages from the user
        user_msgs = self.db.get_messages_by_user(username, limit=max_messages)
        
        if not user_msgs:
            logger.info(f"No messages found for user {username}")
            return []
            
        context_lines = []
        seen_ids = set()
        
        # For each message, get context
        for msg in user_msgs:
            # Get neighbors (small window, e.g. 3 before, 3 after)
            neighbors = self.db.get_neighbor_messages(msg, window_count=3, max_tokens=1000)
            
            for m in neighbors:
                if m.msg_id not in seen_ids:
                    # Format: [YYYY-MM-DD HH:MM] [user: User] Text
                    line = f"[{m.ts.strftime('%Y-%m-%d %H:%M')}] [user: {m.from_id}] {m.text}"
                    context_lines.append(line)
                    seen_ids.add(m.msg_id)
            
            context_lines.append("---")
            
        full_context = "\n".join(context_lines)
        
        # 2. Call LLM
        prompt = f"Target User: {username}\n\nChat Log:\n{full_context}"
        
        try:
            response_text = await self.bot.complete(
                prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1 # Low temp for extraction
            )
            
            # 3. Parse Response
            # Expected JSON: {"user": "...", "alias": "'name1' 'name2'"}
            # Clean possible markdown code blocks
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            
            if "alias" in data:
                # Parse space-separated quoted strings: 'name1' 'name2'
                # Simple parsing: split by "'" and filter
                raw_aliases = data["alias"]
                # This format "'name1' 'name2'" suggests splitting by ' ' might break multi-word names.
                # Regex would be better, or simple split if quotes are consistent.
                # Let's use simple split by " " is risky if contains spaces inside quotes.
                # But example: 'name2 super man' -> spaces inside.
                
                parts = raw_aliases.split("'")
                # 'name1' 'name2 super man'
                # Split gives: ["", "name1", " ", "name2 super man", ""]
                # Filter out empty and spaces
                aliases = [p.strip() for p in parts if p.strip()]
                return aliases
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse alias JSON for {username}: {response_text}")
        except Exception as e:
            logger.error(f"Error discovering aliases for {username}: {e}")
            
        return []

    def save_aliases(self, username: str, aliases: List[str]) -> None:
        """
        Save discovered aliases to database.
        
        Args:
            username: Username to update
            aliases: List of aliases to add
        """
        if not aliases:
            return
            
        # Get existing aliases to merge? Or overwrite? 
        # Typically discovery adds new ones. 
        # But for now, let's just update (overwrite or merge logic in DB method?)
        # db.update_user_aliases overwrites. 
        # Let's merge manually.
        user = self.db.get_user(username)
        current_aliases = []
        if user and user.aliases:
            try:
                current_aliases = json.loads(user.aliases)
            except:
                pass
        
        # Merge set
        updated = list(set(current_aliases + aliases))
        self.db.update_user_aliases(username, updated)
        logger.info(f"Updated aliases for {username}: {updated}")

