"""
Access control logic for Legale Bot.
Handles admin authentication and access checks.
"""

from typing import Callable, Optional, Protocol


class ConfigProvider(Protocol):
    """Protocol for configuration providers that supply admin password."""
    
    @property
    def admin_password(self) -> str:
        """Get admin password from configuration."""
        ...


class AdminAccessControl:
    """Handles admin access control logic."""
    
    def __init__(self, admin_store, config_provider: ConfigProvider):
        """
        Initialize AdminAccessControl.
        
        Args:
            admin_store: AdminStore instance for admin data
            config_provider: ConfigProvider instance for password access
        """
        self.admin_store = admin_store
        self.config_provider = config_provider
    
    def is_admin(self, user_id: int) -> bool:
        """
        Check if user is admin.
        
        Args:
            user_id: Telegram user ID
        
        Returns:
            True if user is admin
        """
        if not self.admin_store.admin_exists():
            return False
        admin = self.admin_store.get_admin()
        return admin is not None and admin.get('user_id') == user_id
    
    def verify_password(self, password: str) -> bool:
        """
        Verify admin password.
        
        Args:
            password: Password to verify
        
        Returns:
            True if password is correct
        """
        return password == self.config_provider.admin_password
    
    def get_admin(self) -> Optional[dict]:
        """
        Get current admin info.
        
        Returns:
            Dict with admin info or None if no admin set
        """
        return self.admin_store.get_admin()


def check_access(
    *,
    user_id: int,
    chat_id: int,
    is_private: bool,
    is_command: bool,
    allowed_chats: list[int],
    is_admin: Callable[[int], bool],
    command_text: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Pure access control policy (no transport concerns).

    Returns (allowed, reason_code).
    """
    if is_command and command_text and (command_text.startswith("/admin_set") or command_text.startswith("/set_admin")):
        return True, None

    if is_admin(user_id):
        return True, None

    if is_private:
        return False, "private_non_admin"

    if is_command:
        return True, None

    if chat_id in allowed_chats:
        return True, None
    return False, "chat_not_whitelisted"
