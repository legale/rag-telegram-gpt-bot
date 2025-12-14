# Рекомендации по улучшению покрытия тестами после рефакторинга

## Общая статистика покрытия
- **Общее покрытие**: 64% (6167 statements, 2244 missing)
- **507 тестов пройдено**, 1 пропущен

## Список тестов для добавления

### 1. src/bot/core.py (64% покрытие, -36% от 100%)

#### Новые helper-методы требуют прямого тестирования:

**test_build_history_for_prompt** (`tests/test_bot_core_chat.py`)
- Проверить построение истории с разным количеством сообщений
- Проверить ограничение max_messages
- Проверить формат возвращаемых данных (role, content)

**test_calculate_token_usage** (`tests/test_bot_core_chat.py`)
- Проверить расчет токенов для system_prompt
- Проверить расчет токенов с user_content
- Проверить возвращаемые поля (prompt_tokens, completion_tokens, total_tokens, estimated_cost)

**test_ensure_context_limit_resets** (`tests/test_bot_core_chat.py`)
- Проверить сброс истории при превышении max_context_tokens
- Проверить возврат предупреждающего сообщения
- Проверить что история действительно сброшена

**test_is_token_limit_error** (`tests/test_bot_core_chat.py`)
- Проверить распознавание различных форматов ошибок token limit
- Проверить false positives (ошибки не связанные с token limit)

**test_retry_after_reset** (`tests/test_bot_core_chat.py`)
- Проверить повторный вызов LLM после сброса контекста
- Проверить что используется пустая история
- Проверить обработку ошибок при повторном вызове

### 2. src/core/retrieval.py (47% покрытие, -53% от 100%)

#### Новые helper-методы для поиска требуют тестирования:

**test_search_with_rephrased_query** (`tests/test_retrieval.py`)
- Проверить поиск с raw_query и rephrased_query
- Проверить возврат двух списков результатов
- Проверить корректность использования vector_store

**test_merge_search_results_by_distance** (`tests/test_retrieval.py`)
- Проверить объединение результатов из нескольких источников
- Проверить удаление дубликатов по chunk_id
- Проверить сохранение минимального distance для дубликатов
- Проверить обработку пустых списков

**test_convert_to_distance_format** (`tests/test_retrieval.py`)
- Проверить конвертацию distance (float)
- Проверить конвертацию score (1.0 - score)
- Проверить обработку отсутствующих полей (infinity)

**test_apply_distance_threshold** (`tests/test_retrieval.py`)
- Проверить фильтрацию по threshold
- Проверить threshold=None (без фильтрации)
- Проверить корректность сравнения distance <= threshold

**test_apply_rag_ntop_limit** (`tests/test_retrieval.py`)
- Проверить ограничение количества результатов rag_ntop
- Проверить rag_ntop=0 (без ограничения)
- Проверить сортировку по distance перед ограничением

**test_rephrase_query_for_rag** (`tests/test_retrieval.py`)
- Проверить вызов LLM для перефразирования
- Проверить парсинг JSON ответа
- Проверить обработку ошибок парсинга
- Проверить fallback на исходный query

### 3. src/core/message_search.py (68% покрытие, -32% от 100%)

#### Новые helper-методы требуют тестирования:

**test_parse_msg_id** (`tests/test_message_search.py`)
- Проверить парсинг обычного числового ID
- Проверить парсинг составного ID (chat_id_msg_id)
- Проверить обработку невалидных форматов
- Проверить обработку ошибок ValueError

**test_format_message_parts** (`tests/test_message_search.py`)
- Проверить форматирование HTML частей сообщения
- Проверить включение debug_rag информации
- Проверить обработку отсутствующих полей
- Проверить экранирование HTML символов

### 4. src/bot/tgbot.py (41% покрытие, -59% от 100%) ⚠️ ПРИОРИТЕТ

#### Низкое покрытие требует значительного расширения тестов:

**test_ensure_required_components** (`tests/test_tgbot.py`)
- Проверить успешную инициализацию MessageHandler
- Проверить отсутствие admin_manager
- Проверить отсутствие bot_instance
- Проверить отсутствие bot.retrieval

**test_send_message_parts** (`tests/test_tgbot.py`)
- Проверить отправку нескольких частей сообщений
- Проверить обработку пустого списка
- Проверить использование empty_message
- Проверить логирование (log_context)
- Проверить обработку ошибок отправки
- Проверить возврат количества отправленных частей

**test_process_document_update** (`tests/test_tgbot.py`)
- Проверить обработку загрузки документов
- Проверить валидацию файлов
- Проверить вызов handle_file_upload
- Проверить обработку ошибок

**test_process_text_update** (`tests/test_tgbot.py`)
- Проверить обработку текстовых сообщений
- Проверить вызов handle_message
- Проверить обработку ошибок

**test_handle_public_commands** (`tests/test_tgbot.py`)
- Проверить обработку /help
- Проверить обработку /admin_set
- Проверить инициализацию компонентов
- Проверить отсутствие admin_manager

**test_check_access** (`tests/test_tgbot.py`)
- Проверить разрешенные чаты
- Проверить заблокированные чаты
- Проверить приватные сообщения (admin)
- Проверить приватные сообщения (non-admin)

**test_determine_response_decision** (`tests/test_tgbot.py`)
- Проверить решение о ответе на основе frequency
- Проверить упоминание бота в группе
- Проверить приватные сообщения
- Проверить команды (всегда отвечать)

**test_handle_search_mention** (`tests/test_tgbot.py`)
- Проверить обработку упоминания бота
- Проверить извлечение query из текста
- Проверить вызов handle_message
- Проверить обработку ошибок

**test_init_runtime_for_current_profile** (`tests/test_tgbot.py`)
- Проверить успешную инициализацию
- Проверить получение профиля
- Проверить создание LegaleBot
- Проверить регистрацию команд
- Проверить обработку ошибок

### 5. src/bot/admin_commands.py (42% покрытие, -58% от 100%)

#### Новые helper-методы требуют тестирования:

**test_get_chat_id_from_args** (`tests/test_admin_commands.py`)
- Проверить извлечение chat_id из args
- Проверить использование текущего chat_id
- Проверить парсинг числовых значений
- Проверить обработку невалидных значений

**test_handle_chats_list** (`tests/test_admin_commands.py`)
- Проверить пустой список
- Проверить список с чатами
- Проверить форматирование вывода

**test_handle_chats_add** (`tests/test_admin_commands.py`)
- Проверить добавление текущего чата
- Проверить добавление указанного чата
- Проверить обработку дубликатов
- Проверить обработку ошибок

**test_handle_chats_remove** (`tests/test_admin_commands.py`)
- Проверить удаление чата
- Проверить обработку несуществующего чата
- Проверить обработку ошибок

**test_validate_upload** (`tests/test_admin_commands.py`)
- Проверить валидацию JSON файлов
- Проверить валидацию размера файла
- Проверить обработку невалидных форматов
- Проверить обработку слишком больших файлов

**test_download_file** (`tests/test_admin_commands.py`)
- Проверить загрузку файла
- Проверить создание временного файла
- Проверить обработку ошибок загрузки
- Проверить форматирование размера файла

**test_is_waiting_for_file** (`tests/test_admin_commands.py`)
- Проверить ожидание файла
- Проверить отсутствие ожидания
- Проверить пользователя не в словаре

**test_start_ingestion_task** (`tests/test_admin_commands.py`)
- Проверить создание задачи
- Проверить отправку статусного сообщения
- Проверить запуск задачи в фоне
- Проверить обработку ошибок

### 6. src/bot/admin_tasks.py (85% покрытие, -15% от 100%)

#### Уже хорошо покрыто, но можно добавить:

**test_update_progress_simple** (`tests/test_admin_tasks.py`)
- Проверить обновление сообщения
- Проверить форматирование прогресса
- Проверить обработку ошибок обновления

### 7. src/ingestion/pipeline.py (42% покрытие, -58% от 100%)

#### Новые helper-методы и исключения требуют тестирования:

**test_clear_vector_collection** (`tests/test_pipeline.py`)
- Проверить очистку коллекции
- Проверить подсчет before/after
- Проверить удаление всех элементов
- Проверить обработку пустой коллекции

**test_clear_topic_names** (`tests/test_pipeline.py`)
- Проверить сброс имен топиков на 'unknown'
- Проверить для L1 и L2 топиков
- Проверить подсчет обновленных записей

**test_init_raises_configuration_error** (`tests/test_pipeline.py`)
- Проверить ConfigurationError при отсутствии profile_dir
- Проверить ConfigurationError при невалидной конфигурации
- Проверить IngestionRuntimeError при ошибках выполнения

## Список тестов для удаления/обновления

### 1. Обновленные тесты (уже исправлены)

✅ **test_run_stage1_missing_profile_dir** - обновлен для использования ConfigurationError вместо SystemExit
✅ **test_parse_and_store_chunks_no_messages** - обновлен для использования IngestionPipelineError
✅ **test_clear_stage3** - обновлен для работы с новым _clear_vector_collection
✅ **test_clear_all** - обновлен для работы с новым _clear_vector_collection
✅ **test_handle_file_upload_not_waiting** - обновлен для новой логики проверки ожидания
✅ **test_retrieval_service_with_topics** - обновлен для работы с llm_client
✅ **test_handle_find_command_send_error** - обновлен для нового поведения _send_message_parts

### 2. Тесты которые могут быть удалены (дублирование)

**Нет тестов для удаления** - все существующие тесты проверяют публичное API и не дублируют функциональность.

## Приоритеты

### Высокий приоритет (критично для качества кода):
1. **src/bot/tgbot.py** (41% покрытие) - основной модуль бота
2. **src/bot/admin_commands.py** (42% покрытие) - админские команды
3. **src/ingestion/pipeline.py** (42% покрытие) - критичный пайплайн инжестии

### Средний приоритет (улучшение качества):
4. **src/core/retrieval.py** (47% покрытие) - RAG поиск
5. **src/bot/core.py** (64% покрытие) - ядро бота

### Низкий приоритет (уже достаточно покрыто):
6. **src/core/message_search.py** (68% покрытие)
7. **src/bot/admin_tasks.py** (85% покрытие)

## Метрики покрытия по модулям

| Модуль | Покрытие | Statements | Missing | Новые методы |
|--------|----------|------------|---------|--------------|
| src/bot/core.py | 64% | 159 | 57 | _build_history_for_prompt, _calculate_token_usage, _ensure_context_limit, _is_token_limit_error, _retry_after_reset |
| src/core/retrieval.py | 47% | 595 | 315 | _search_with_rephrased_query, _merge_search_results_by_distance, _convert_to_distance_format, _apply_distance_threshold, _apply_rag_ntop_limit |
| src/core/message_search.py | 68% | 132 | 42 | _parse_msg_id, _format_message_parts |
| src/bot/tgbot.py | 41% | 670 | 398 | _ensure_required_components, _send_message_parts, process_document_update, process_text_update |
| src/bot/admin_commands.py | 42% | 519 | 300 | _get_chat_id_from_args, _handle_chats_*, _validate_upload, _download_file, _start_ingestion_task, _is_waiting_for_file |
| src/bot/admin_tasks.py | 85% | 121 | 18 | _update_progress_simple |

## Итоговые рекомендации

1. **Немедленно добавить**: Тесты для src/bot/tgbot.py (критично низкое покрытие 41%)
2. **Добавить в ближайшее время**: Тесты для новых helper-методов в src/bot/core.py и src/core/retrieval.py
3. **Улучшить покрытие**: src/bot/admin_commands.py и src/ingestion/pipeline.py до 70%+
4. **Поддерживать**: Высокое покрытие src/bot/admin_tasks.py (85%)

## Статус после рефакторинга

✅ Все существующие тесты проходят (507 passed, 1 skipped)
✅ Все сломанные тесты исправлены
✅ Новые методы покрываются косвенно через существующие тесты
⚠️ Прямое тестирование новых helper-методов отсутствует и рекомендуется для улучшения качества


