
import os
import sys
import asyncio

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app.bootstrap import create_app
from src.core.command_service import CommandService

async def main():
    print("Initializing app...")
    os.environ["OPENROUTER_API_KEY"] = "sk-dummy"
    # Mock DB URL and Vector DB path
    db_url = "sqlite:///test_verify_profile.sqlite"
    vector_db_path = "test_verify_profile_chroma"
    
    app = create_app(
        db_url=db_url,
        vector_db_path=vector_db_path,
        log_level=6  # LOG_INFO
    )
    
    command_service = app.command_service
    print("App initialized.")
    
    # Check registration
    handler_sync = command_service.dispatcher.handlers.get("profile")
    handler_async = command_service.dispatcher.async_handlers.get("profile")
    
    print(f"Sync handler for /profile: {handler_sync}")
    print(f"Async handler for /profile: {handler_async}")
    
    if handler_async:
        print("SUCCESS: /profile command registered as ASYNC.")
        sys.exit(0)
    else:
        print("FAILURE: /profile command NOT registered as ASYNC.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
