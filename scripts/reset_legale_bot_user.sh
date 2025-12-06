#!/bin/bash
# Скрипт для полной очистки и пересоздания окружения legale-bot

echo "=== Очистка окружения legale-bot ==="

# 1. Остановить все процессы legale-bot
echo "1. Останавливаем процессы..."
sudo pkill -u legale-bot -f python
sudo systemctl stop legale-bot 2>/dev/null || true

# 2. Удалить пользователя legale-bot (если существует)
echo "2. Удаляем пользователя legale-bot..."
sudo userdel -r legale-bot 2>/dev/null || true

# 3. Удалить домашний каталог (если остался)
echo "3. Удаляем домашний каталог..."
sudo rm -rf /home/legale-bot

# 4. Создать пользователя заново
echo "4. Создаем пользователя legale-bot..."
sudo useradd -m -s /bin/bash legale-bot

# 5. Скопировать проект с правильными правами
echo "5. Копируем проект..."
sudo rsync -avh --exclude='.git' --exclude='__pycache__' --exclude='.pytest_cache' \
    /home/ru/legale-bot/ /home/legale-bot/legale-bot/

# 6. Установить владельца
echo "6. Устанавливаем владельца..."
sudo chown -R legale-bot:legale-bot /home/legale-bot/legale-bot

# 7. Установить правильные права
echo "7. Устанавливаем права..."
sudo chmod 755 /home/legale-bot/legale-bot
sudo chmod 664 /home/legale-bot/legale-bot/*.db 2>/dev/null || true
sudo chmod 664 /home/legale-bot/legale-bot/*.session 2>/dev/null || true
sudo chmod 755 /home/legale-bot/legale-bot/chroma_db 2>/dev/null || true

# 8. Скопировать .env файл
echo "8. Копируем .env..."
sudo cp /home/ru/legale-bot/.env /home/legale-bot/legale-bot/.env
sudo chown legale-bot:legale-bot /home/legale-bot/legale-bot/.env
sudo chmod 600 /home/legale-bot/legale-bot/.env

echo ""
echo "=== Готово! ==="
echo ""
echo "Теперь выполните:"
echo "  sudo su - legale-bot"
echo "  cd legale-bot"
echo "  poetry install"
echo "  poetry run python src/ingestion/telegram.py dump_chan \"Vpn-friends\" --limit 1000 --output telegram_dump.json"
