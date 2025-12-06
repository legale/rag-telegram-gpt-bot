# Решение проблемы с правами доступа

## Проблема
```
sqlite3.OperationalError: attempt to write a readonly database
```

Эта ошибка возникает, когда пользователь `legale-bot` не имеет прав на запись в файлы базы данных.

## Причина
При копировании файлов через `rsync` права доступа были установлены как `rw-r--r--` (644), что означает только чтение для владельца файла. SQLite требует права на запись.

## Решение

### Быстрое исправление (уже выполнено)
```bash
# Установить владельца всех файлов
sudo chown -R legale-bot:legale-bot /home/legale-bot/legale-bot/

# Исправить права на файлы баз данных
sudo chmod 600 /home/legale-bot/legale-bot/legale_bot.db
sudo chmod 600 /home/legale-bot/legale-bot/legale_bot_session.session

# Исправить права на ChromaDB
sudo chmod -R 700 /home/legale-bot/legale-bot/chroma_db/
```

### Автоматический скрипт
Создан скрипт для автоматического исправления прав:

```bash
# Запустить скрипт
sudo /home/ru/legale-bot/scripts/fix_permissions.sh
```

### Проверка
```bash
# Проверить права
ls -la /home/legale-bot/legale-bot/legale_bot*.{db,session}

# Должно быть:
# -rw------- 1 legale-bot legale-bot ... legale_bot.db
# -rw------- 1 legale-bot legale-bot ... legale_bot_session.session
```

## Правильный способ копирования в будущем

При копировании файлов используйте:

```bash
# Вариант 1: rsync с сохранением прав и владельца
sudo rsync -avh --chown=legale-bot:legale-bot /home/ru/legale-bot/ /home/legale-bot/legale-bot/

# Вариант 2: rsync + исправление прав после
rsync -avh /home/ru/legale-bot/ /home/legale-bot/legale-bot/
sudo chown -R legale-bot:legale-bot /home/legale-bot/legale-bot/
sudo chmod 600 /home/legale-bot/legale-bot/*.db
sudo chmod 600 /home/legale-bot/legale-bot/*.session
```

## Теперь можно запускать команды

```bash
# Переключиться на пользователя legale-bot
sudo su - legale-bot

# Перейти в каталог проекта
cd /home/legale-bot/legale-bot

# Запустить команду
poetry run python src/ingestion/telegram.py dump_chan "Vpn-friends" --limit 1000000 --output telegram_dump.json
```

## Дополнительно: Права для systemd

Если используете systemd service, убедитесь что в файле `/etc/systemd/system/legale-bot.service` указан правильный пользователь:

```ini
[Service]
User=legale-bot
Group=legale-bot
WorkingDirectory=/home/legale-bot/legale-bot
```
