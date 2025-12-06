#!/bin/bash
# Fix permissions for legale-bot user

echo "Fixing permissions for legale-bot user..."

# Set ownership
sudo chown -R legale-bot:legale-bot /home/legale-bot/legale-bot/

# Set directory permissions (rwx for owner only)
sudo find /home/legale-bot/legale-bot/ -type d -exec chmod 700 {} \;

# Set file permissions (rw for owner only)
sudo find /home/legale-bot/legale-bot/ -type f -exec chmod 600 {} \;

# Make scripts executable
sudo chmod 700 /home/legale-bot/legale-bot/src/ingestion/*.py
sudo chmod 700 /home/legale-bot/legale-bot/src/bot/*.py

# Special permissions for databases
sudo chmod 600 /home/legale-bot/legale-bot/legale_bot.db
sudo chmod 600 /home/legale-bot/legale-bot/legale_bot_session.session

# ChromaDB directory needs write access
sudo chmod -R 700 /home/legale-bot/legale-bot/chroma_db/

echo "✓ Permissions fixed!"
echo ""
echo "Verify with:"
echo "  ls -la /home/legale-bot/legale-bot/"
