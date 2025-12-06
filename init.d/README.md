# Installation Instructions for Init Script

## For SysV Init Systems (Debian/Ubuntu without systemd)

1. **Copy script to init.d**:
   ```bash
   sudo cp init.d/legale-bot /etc/init.d/
   sudo chmod +x /etc/init.d/legale-bot
   ```

2. **Edit configuration** (if needed):
   ```bash
   sudo nano /etc/init.d/legale-bot
   # Update USER and WORKDIR variables
   ```

3. **Enable service**:
   ```bash
   sudo update-rc.d legale-bot defaults
   ```

4. **Start service**:
   ```bash
   sudo service legale-bot start
   ```

5. **Check status**:
   ```bash
   sudo service legale-bot status
   ```

6. **View logs**:
   ```bash
   tail -f /var/log/syslog | grep legale-bot
   ```

## Management Commands

```bash
# Start
sudo service legale-bot start

# Stop
sudo service legale-bot stop

# Restart
sudo service legale-bot restart

# Status
sudo service legale-bot status
```

## Notes

- This script is for legacy systems without systemd
- For modern systems, use the systemd service file instead: `systemd/legale-bot.service`
- The script runs the bot in foreground mode with output redirected to syslog
- PID file is stored at `/var/run/legale-bot.pid`
