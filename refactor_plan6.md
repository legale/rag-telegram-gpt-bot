# План рефакторинга Legale Bot

## Краткий обзор проблем

Основные проблемы в коде связаны с высокой цикломатической сложностью крупных функций, использованием глобальных переменных, дублированием логики и недостаточной тестируемостью. Наиболее проблемные области: модуль `tgbot.py` с множеством глобальных переменных и сложной логикой обработки webhook-запросов, модуль `core.py` с большими методами класса `LegaleBot`, модуль `pipeline.py` с длинными функциями ингеста данных, и модуль `hybrid_retrieval.py` с разветвленной логикой поиска. Также обнаружено значительное дублирование кода при создании embedding-клиентов, обработке ошибок и парсинге команд. Широкое использование `except Exception` без специфичной обработки снижает надежность кода. Отсутствие таймаутов для HTTP-запросов создает риск зависания при работе с внешними API. Много легаси-кода с комментариями о deprecated функциях и неиспользуемых методах усложняет поддержку.

## План задач рефакторинга

- [w] file=src/bot/tgbot.py убрать глобальные переменные bot_instance, admin_manager, admin_router, task_manager, ingest_commands, command_dispatcher: создать класс RuntimeContext для хранения состояния и передавать его через dependency injection (частично выполнено: RuntimeContext создан, глобальные переменные убраны, но dependency injection не реализован - везде используется get_runtime_context() как глобальный синглтон вместо передачи через параметры функций)

- [x] file=src/bot/tgbot.py func=init_runtime_for_current_profile уменьшить цикломатическую сложность: разбить на отдельные функции _create_bot_instance(), _create_admin_components(), _create_command_dispatcher()

- [x] file=src/bot/tgbot.py func=_process_webhook_update уменьшить цикломатическую сложность: вынести обработку команд и обработку обычных сообщений в отдельные функции _handle_command() и _handle_user_message()

- [x] file=src/bot/core.py func=chat уменьшить цикломатическую сложность: вынести логику получения контекста в _get_context_for_query(), логику построения промпта в _build_llm_messages(), логику вызова LLM в _call_llm_with_context()

- [x] file=src/bot/core.py func=__init__ уменьшить цикломатическую сложность: вынести создание embedding_client в _create_embedding_client(), создание retrieval_service в _create_retrieval_service()

- [x] file=src/core/hybrid_retrieval.py func=search уменьшить цикломатическую сложность: вынести FTS-only режим в _search_fts_only(), векторный reranking в _rerank_with_vectors(), обработку ошибок embedding в _handle_embedding_error()

- [x] file=src/core/hybrid_retrieval.py func=_pack_context уменьшить цикломатическую сложность: вынести дедупликацию по msg_id в _deduplicate_by_msg_id(), проверку token budget в _check_token_budget(), обогащение сообщениями в _enrich_with_messages()

- [x] file=src/ingestion/pipeline.py func=parse_and_store_chunks уменьшить цикломатическую сложность: вынести получение сообщений из БД в _load_messages_from_db(), конвертацию в ChatMessage в _convert_to_chat_messages(), сохранение chunks в _save_chunks_to_db()

- [x] file=src/ingestion/pipeline.py func=generate_embeddings уменьшить цикломатическую сложность: вынести получение chunks без embeddings в _get_chunks_without_embeddings(), генерацию embeddings батчами в _generate_embeddings_batch(), сохранение embeddings в _save_embeddings_to_db()

- [x] file=src/app/bootstrap.py убрать дублирование логики создания embedding_client: вынести в отдельную функцию _create_embedding_client_from_config() и использовать её в create_hybrid_search() и create_hybrid_retrieval()

- [x] file=src/bot/core.py file=src/app/bootstrap.py убрать дублирование логики создания embedding_client из BotConfig: использовать единую функцию из bootstrap.py

- [x] file=src/bot/tgbot.py file=src/bot/admin_router.py убрать дублирование логики обработки ошибок: использовать единый метод handle_error() из BaseAdminCommand или создать ErrorHandler utility

- [ ] file=src/bot/tgbot.py file=src/bot/cli.py убрать дублирование логики парсинга команд: использовать единый CommandDispatcher для всех точек входа (частично выполнено: используется dispatcher в main(), но есть неиспользуемая дублирующая функция handle_cli_command в cli.py)

- [x] file=src/core/llm.py func=complete добавить явные таймауты для HTTP-запросов: использовать timeout параметр в OpenAI client (timeout=30.0)

- [x] file=src/core/embedding.py func=get_embeddings добавить явные таймауты для HTTP-запросов: использовать timeout параметр в OpenAI client (timeout=60.0 для batch)

- [x] file=src/ingestion/telegram.py func=dump_chat добавить явные таймауты для Telethon API вызовов: использовать timeout параметр в client методах

- [x] file=src/bot/tgbot.py func=_process_webhook_update заменить широкий except Exception на специфичные исключения: обрабатывать ValueError, KeyError, AttributeError отдельно с соответствующими сообщениями

- [x] file=src/ingestion/pipeline.py func=parse_and_store_messages заменить широкий except Exception на специфичные исключения: обрабатывать FileNotFoundError, json.JSONDecodeError, ValueError отдельно

- [x] file=src/core/hybrid_retrieval.py func=search заменить широкий except Exception на специфичные исключения: обрабатывать EmbeddingError, VectorIndexError отдельно с fallback на FTS-only

- [ ] file=src/bot/core.py func=_call_llm_with_retry заменить широкий except Exception на специфичные исключения: обрабатывать APIError, TimeoutError, RateLimitError отдельно (частично выполнено: TimeoutError обрабатывается отдельно, но APIError и RateLimitError обрабатываются внутри общего except Exception с проверкой типа по строке)

- [x] file=src/bot/tgbot.py удалить неиспользуемую функцию _send_find_results_simple (помечена как deprecated, используется _send_message_parts_unified)

- [x] file=src/storage/db.py удалить закомментированные методы clear_chunk_topic_l1_assignments, clear_chunk_topic_l2_assignments, update_chunk_topics (clustering deprecated)

- [x] file=src/storage/db.py удалить закомментированные методы get_by_topic_l1, get_by_topic_l2 из ChunkStore (clustering deprecated)

- [x] file=src/storage/vector_store.py удалить неиспользуемые методы get_topics_l1_collection, get_topics_l2_collection (clustering deprecated)

- [x] file=src/ingestion/pipeline.py удалить закомментированные методы run_stage4, run_stage5, run_stage6, run_stage7 (clustering deprecated)

- [x] file=src/bot/admin_commands.py удалить дублирующийся импорт CommandValidator (импортируется дважды в строках 16-17)

- [x] file=src/bot/tgbot.py func=_register_admin_commands уменьшить цикломатическую сложность: вынести регистрацию каждой группы команд в отдельные функции _register_profile_commands(), _register_ingest_commands(), _register_stats_commands()

- [x] file=src/bot/tgbot.py func=lifespan уменьшить цикломатическую сложность: вынести инициализацию profile_manager в _init_profile_manager(), инициализацию runtime в _init_runtime(), инициализацию telegram_app в _init_telegram_app()

- [x] file=src/bot/core.py func=_get_or_build_context уменьшить цикломатическую сложность: вынести проверку необходимости обновления контекста в _should_refresh_context(), построение нового контекста в _build_new_context(), оценку качества контекста в _evaluate_context_quality()

- [x] file=src/bot/core.py func=_build_prompt_and_history упростить: убрать параметр max_messages из сигнатуры, использовать значение по умолчанию из конфига

- [x] file=src/core/message_search.py func=_prepare_message_parts уменьшить цикломатическую сложность: вынести получение сообщений для chunk в _get_messages_for_chunk(), форматирование сообщений в _format_message_parts_for_chunk()

- [x] file=src/core/message_search.py func=_format_message_parts уменьшить цикломатическую сложность: вынести парсинг msg_id в _parse_message_id(), создание message_data в _create_message_data()

- [x] file=src/ingestion/chunker.py func=chunk_messages уменьшить цикломатическую сложность: вынести предварительный подсчет токенов в _precompute_token_counts(), создание chunks в _create_chunks_from_messages(), обработку overlap в _apply_overlap()

- [x] file=src/adapters/persistence/sqlite_chunk_store.py func=save_batch уменьшить цикломатическую сложность: вынести подготовку данных chunk в _prepare_chunk_data(), проверку существования chunk в _chunk_exists(), создание/обновление chunk в _create_or_update_chunk()

- [x] file=src/adapters/persistence/sqlite_message_store.py func=save_batch уменьшить цикломатическую сложность: вынести конвертацию domain messages в dict в _convert_messages_to_dicts(), сохранение metadata в _save_message_metadata()

- [x] file=src/storage/db.py func=_ensure_schema уменьшить цикломатическую сложность: вынести создание FTS5 таблиц в _create_fts5_tables(), создание triggers в _create_fts5_triggers(), проверку колонок в _check_and_add_columns()

- [x] file=src/storage/db.py func=add_messages_batch уменьшить цикломатическую сложность: вынести проверку существующих msg_ids в _get_existing_msg_ids(), фильтрацию новых сообщений в _filter_new_messages()

- [x] file=src/bot/config.py func=_load упростить логику: вынести создание файла с defaults в _create_default_config(), загрузку существующего config в _load_existing_config(), добавление missing defaults в _add_missing_defaults()

- [x] file=src/bot/admin_router.py func=_route_with_subcommand уменьшить цикломатическую сложность: вынести проверку доступа в _check_admin_access(), выполнение handler в _execute_handler_with_error_handling()

- [x] file=src/bot/admin_tasks.py func=_persist_sql уменьшить цикломатическую сложность: вынести подготовку chunk models в _prepare_chunk_models(), сохранение в БД в _save_chunks_to_database()

- [x] file=src/bot/admin_tasks.py func=_persist_vectors уменьшить цикломатическую сложность: вынести подготовку данных для vector store в _prepare_vector_data(), синхронизацию в _sync_to_vector_store()

- [x] file=src/core/query_rewriter.py func=rephrase_for_embedding добавить обработку ошибок LLM: обрабатывать APIError, TimeoutError с fallback на возврат оригинального query

- [x] file=src/core/llm.py func=complete добавить retry логику с exponential backoff для RateLimitError и временных ошибок сети

- [x] file=src/core/embedding.py func=get_embeddings_batched добавить retry логику с exponential backoff для временных ошибок API

- [x] file=src/bot/core.py func=_retry_after_reset упростить: убрать дублирование построения промпта, использовать _build_prompt_and_history()

- [x] file=src/bot/core.py func=_is_token_limit_error упростить: использовать регулярное выражение или список ключевых слов вместо множественных проверок строк

- [x] file=src/core/hybrid_retrieval.py func=_cosine_similarity вынести в отдельный модуль src/core/distance_utils.py для переиспользования (уже есть similarity_to_distance, добавить обратную функцию)

- [x] file=src/ingestion/pipeline.py func=_prepare_chunk_data_for_vector_store уменьшить цикломатическую сложность: вынести парсинг embedding_json в _parse_chunk_embedding(), подготовку metadata в _prepare_chunk_metadata()

- [x] file=src/ingestion/pipeline.py func=_check_and_fix_collection_dimension упростить: вынести проверку dimension в _check_collection_dimension(), пересоздание collection в _recreate_collection_if_needed()

- [x] file=src/bot/tgbot.py func=_parse_webhook_update уменьшить цикломатическую сложность: вынести валидацию request body в _validate_webhook_request(), парсинг Update в _parse_update_from_json()

- [x] file=src/bot/tgbot.py func=_setup_webhook_endpoint упростить: вынести создание endpoint handler в _create_webhook_handler()

- [x] file=src/bot/tgbot.py func=main уменьшить цикломатическую сложность: вынести парсинг аргументов в _parse_cli_arguments(), выполнение команд в _execute_bot_command()

- [x] file=src/bot/tgbot.py func=run_server уменьшить цикломатическую сложность: вынести создание app в _create_fastapi_app(), настройку logging в _setup_server_logging(), запуск uvicorn в _start_uvicorn_server()

- [x] file=src/bot/tgbot.py func=run_daemon уменьшить цикломатическую сложность: вынести создание daemon context в _create_daemon_context(), настройку signal handlers в _setup_signal_handlers()

- [x] file=src/core/prompt.py func=construct_prompt уменьшить цикломатическую сложность: вынести построение context части в _build_context_section(), построение history части в _build_history_section(), построение task части в _build_task_section()

- [x] file=src/core/prompt.py func=get_system_prompt упростить: использовать единый метод для получения system prompt из BotConfig без дублирования логики

- [x] file=src/bot/utils/response_formatter.py создать единый класс ResponseFormatter для форматирования всех типов ответов (уже существует, проверить использование во всех местах)

- [x] file=src/bot/utils/command_validator.py создать единый класс CommandValidator для валидации всех команд (уже существует, проверить использование во всех местах)

- [x] file=src/bot/admin.py func=is_admin упростить: вынести проверку пароля в _verify_password(), проверку существования admin в _admin_exists()

- [x] file=src/bot/admin.py func=set_admin упростить: вынести валидацию пароля в _validate_admin_password(), сохранение admin в _save_admin_info()

- [x] file=src/core/commands.py func=FindCommandHandler.handle уменьшить цикломатическую сложность: вынести парсинг аргументов в _parse_find_args(), создание retrieval service в _create_retrieval_service(), выполнение поиска в _execute_search()

- [x] file=src/core/commands.py func=ModelCommandHandler.handle упростить: вынести переключение модели в _switch_model(), сохранение в config в _save_model_to_config()

- [x] file=src/core/dispatcher.py func=dispatch упростить: вынести нормализацию command name в _normalize_command_name(), поиск handler в _find_handler(), выполнение handler в _execute_handler()

- [x] file=src/core/dispatcher.py func=dispatch_async упростить: использовать общие методы нормализации и поиска handler из dispatch()

- [x] file=src/storage/vector_store.py func=add_documents_with_embeddings уменьшить цикломатическую сложность: вынести валидацию входных данных в _validate_batch_inputs(), обработку батчей в _process_batch()

- [x] file=src/storage/vector_store.py func=query упростить: вынести вычисление query embeddings в _compute_query_embeddings(), выполнение запроса в _execute_vector_query()

- [x] file=src/adapters/vector/chroma_vector_index.py func=query уменьшить цикломатическую сложность: вынести конвертацию filter в where clause в _convert_filter_to_where(), конвертацию результатов в ScoredDoc в _convert_results_to_scored_docs()

- [x] file=src/adapters/persistence/sqlite_fts_index.py func=search уменьшить цикломатическую сложность: вынести построение WHERE clause в _build_where_clause(), выполнение FTS запроса в _execute_fts_query(), нормализацию текста в _normalize_query_text()

- [x] file=src/core/message_search.py func=search_message_contents упростить: вынести фильтрацию по threshold в _filter_results_by_threshold(), подготовку message parts в _prepare_message_parts_from_results()

- [x] file=src/core/message_search.py func=search_message_links упростить: вынести получение link info из chunk в _get_link_info_from_chunk(), построение link в _build_message_link()

- [x] file=src/ingestion/parser.py func=parse_file упростить: вынести парсинг JSON в _parse_json_file(), парсинг текстового файла в _parse_text_file()

- [x] file=src/ingestion/chunker.py func=_format_message упростить: использовать f-string template вместо конкатенации строк

- [x] file=src/ingestion/chunker.py func=_create_prefix упростить: использовать f-string template вместо конкатенации строк

- [x] file=src/bot/tgbot.py func=_send_message_parts_unified упростить: вынести отправку одной части в _send_single_message_part(), обработку ошибок отправки в _handle_send_error()

- [x] file=src/bot/tgbot.py func=is_bot_mentioned упростить: вынести проверку mention entity в _check_mention_entity(), проверку text_mention entity в _check_text_mention_entity()

- [x] file=src/bot/tgbot.py func=_ensure_required_components упростить: вынести проверку admin_manager в _check_admin_manager(), проверку bot_instance в _check_bot_instance()

- [x] file=src/bot/tgbot.py func=_ensure_handler_available упростить: вынести выполнение handler в _execute_handler(), отправку response в _send_handler_response()

- [x] file=src/bot/tgbot.py func=_handle_public_commands уменьшить цикломатическую сложность: вынести обработку каждой команды в отдельные функции _handle_id_command(), _handle_help_command(), _handle_admin_set_command()

- [x] file=src/bot/tgbot.py func=_map_log_level_to_constant упростить: использовать словарь для маппинга вместо множественных if-elif

- [x] file=src/bot/tgbot.py func=_map_syslog2_to_logging_level упростить: использовать словарь для маппинга вместо множественных if-elif

- [x] file=src/bot/tgbot.py func=setup_logging уменьшить цикломатическую сложность: вынести создание handler в _create_log_handler(), настройку formatter в _setup_formatter(), настройку loggers в _configure_loggers()

- [x] file=src/bot/core.py func=_calculate_token_usage упростить: вынести построение messages для подсчета в _build_messages_for_token_count()

- [x] file=src/bot/core.py func=get_token_usage упростить: использовать _build_prompt_and_history() вместо дублирования логики построения промпта

- [x] file=src/bot/core.py func=_ensure_context_limit упростить: вынести проверку лимита токенов в _check_token_limit_exceeded(), сброс контекста в _reset_context_if_needed()

- [x] file=src/core/hybrid_retrieval.py func=retrieve упростить: вынести конвертацию SearchResult в dict в _convert_search_results_to_dicts(), фильтрацию по threshold в _filter_by_threshold()

- [x] file=src/core/hybrid_retrieval.py func=search_chunks_basic упростить: вынести конвертацию результатов в _convert_to_basic_format(), сортировку по distance в _sort_by_distance()

- [x] file=src/storage/db.py func=get_messages_by_chunk уменьшить цикломатическую сложность: вынести получение start message в _get_start_message(), получение end message в _get_end_message(), получение messages в диапазоне в _get_messages_in_range()

- [x] file=src/storage/db.py func=get_chunk_link_info упростить: вынести извлечение chat_id в _extract_chat_id(), извлечение msg_id в _extract_msg_id(), извлечение chat_username в _extract_chat_username()

- [x] file=src/storage/db.py func=fts_search уменьшить цикломатическую сложность: вынести построение WHERE clause в _build_fts_where_clause(), выполнение FTS запроса в _execute_fts_query(), конвертацию результатов в _convert_fts_results()

- [x] file=src/bot/config.py упростить: вынести валидацию значений в отдельные методы _validate_chunk_token_min(), _validate_chunk_token_max(), _validate_chunk_overlap_ratio() и т.д.

- [x] file=src/bot/config.py func=save упростить: вынести создание директории в _ensure_profile_dir(), сохранение файла в _write_config_file(), установку permissions в _set_file_permissions()

- [x] file=src/core/chunk_utils.py func=build_chunk_dict_from_domain_chunk упростить: вынести извлечение metadata в _extract_chunk_metadata(), создание dict в _create_chunk_dict()

- [x] file=src/core/distance_utils.py func=similarity_to_distance добавить обратную функцию distance_to_similarity для консистентности

- [x] file=src/bot/utils/telegram_common.py func=split_message_if_needed упростить: вынести проверку необходимости split в _needs_splitting(), разбиение сообщения в _split_message()

- [x] file=src/bot/utils/telegram_links.py func=build_message_link упростить: вынести построение link для username в _build_link_with_username(), построение link без username в _build_link_without_username()

- [x] file=src/bot/utils/access_control.py func=check_access упростить: вынести проверку admin доступа в _check_admin_access(), проверку allowed chats в _check_allowed_chats()

- [ ] file=src/bot/utils/frequency_controller.py func=should_respond упростить: вынести проверку frequency limit в _check_frequency_limit(), обновление счетчика в _update_counter()

- [ ] file=src/bot/utils/database_stats.py func=get_stats упростить: вынести получение stats из БД в _get_database_stats(), форматирование stats в _format_stats()

- [ ] file=src/bot/utils/health_checker.py func=check_health упростить: вынести проверку БД в _check_database_health(), проверку vector store в _check_vector_store_health()

- [ ] file=src/bot/utils/response_formatter.py func=format_error_message упростить: вынести форматирование error message в _format_error(), добавление context в _add_context()

- [ ] file=src/bot/utils/command_validator.py func=validate упростить: вынести валидацию command name в _validate_command_name(), валидацию args в _validate_args()

- [ ] file=src/core/query_rewriter.py func=rephrase_for_embedding упростить: вынести построение prompt для rephrasing в _build_rephrase_prompt(), вызов LLM в _call_llm_for_rephrasing()

- [ ] file=src/core/llm.py func=count_tokens упростить: вынести подсчет токенов для одного message в _count_message_tokens()

- [ ] file=src/core/embedding.py func=get_embeddings упростить: вынести очистку текста в _clean_texts(), вызов API в _call_embedding_api()

- [ ] file=src/core/embedding.py func=get_embeddings_batched упростить: вынести обработку одного батча в _process_batch(), обновление прогресса в _update_progress()

- [ ] file=src/core/embedding.py func=create_embedding_client упростить: вынести создание LocalEmbeddingClient в _create_local_client(), создание EmbeddingClient в _create_api_client()

- [ ] file=src/core/prompt.py func=construct_prompt упростить: вынести проверку custom_template в _should_use_custom_template(), построение промпта из template в _build_from_template()

- [ ] file=src/storage/vector_store.py func=_recreate_collection_with_dimension вынести в публичный метод recreate_collection() для переиспользования

- [ ] file=src/ingestion/pipeline.py func=_get_llm_client упростить: вынести получение model из config в _get_model_from_config(), создание LLMClient в _create_llm_client()

- [ ] file=src/ingestion/pipeline.py func=run_all упростить: использовать список stage функций и вызывать их в цикле вместо явных вызовов

- [x] file=src/bot/tgbot.py func=_register_command_group упростить: использовать словарь для маппинга method_name -> subcommand вместо явных вызовов register()

- [ ] file=src/bot/admin_commands.py func=BaseAdminCommand.get_profile_paths упростить: вынести валидацию profile_name в _validate_profile_name(), получение paths в _get_paths_for_profile()

- [ ] file=src/bot/admin_commands.py func=BaseAdminCommand.handle_error упростить: вынести логирование ошибки в _log_error(), форматирование сообщения в _format_error_message()

- [ ] file=src/core/admin_commands.py func=AdminSetCommandHandler.handle упростить: вынести валидацию пароля в _validate_password(), установку admin в _set_admin_user()

- [ ] file=src/core/admin_commands.py func=AdminGetCommandHandler.handle упростить: вынести получение admin info в _get_admin_info(), форматирование ответа в _format_admin_info()

- [ ] file=src/core/admin_commands.py func=AdminCommandHandler.handle упростить: вынести парсинг команды в _parse_admin_command(), выполнение команды в _execute_admin_command()

- [ ] file=src/app/main_cli.py func=create_dispatcher упростить: вынести регистрацию sync handlers в _register_sync_handlers(), регистрацию async handlers в _register_async_handlers()

- [ ] file=src/app/main_cli.py func=handle_command упростить: вынести парсинг command name и args в _parse_command(), создание context в _create_command_context()

- [ ] file=src/bot/cli.py func=handle_cli_command упростить: использовать CommandDispatcher вместо дублирования логики обработки команд

- [ ] file=src/bot/cli.py func=main упростить: вынести инициализацию bot в _init_bot(), обработку команд в _handle_user_input(), обработку ошибок в _handle_error()

- [ ] file=src/bot/command_parser.py func=parse_find_command_args упростить: вынести парсинг rag_method в _parse_rag_method(), парсинг query в _parse_query(), валидацию args в _validate_args()

- [ ] file=src/core/ingest_use_cases/ingest_messages.py func=execute упростить: вынести парсинг файла в _parse_file(), сохранение messages в _save_messages()

- [ ] file=src/core/ingest_use_cases/process_chunks.py func=execute упростить: вынести получение messages в _load_messages(), создание chunks в _create_chunks(), сохранение chunks в _save_chunks()

- [ ] file=src/core/ingest_use_cases/generate_embeddings.py func=execute упростить: вынести получение chunks без embeddings в _get_chunks_without_embeddings(), генерацию embeddings в _generate_embeddings(), сохранение embeddings в _save_embeddings()

- [ ] file=src/core/ingest_use_cases/sync_to_vector_store.py func=execute упростить: вынести подготовку данных в _prepare_vector_data(), синхронизацию в _sync_to_vector_store()

- [ ] file=src/core/ingest_use_cases/pipeline_orchestrator.py func=run упростить: вынести выполнение stage в _execute_stage(), обработку ошибок stage в _handle_stage_error()

- [ ] file=src/adapters/persistence/sqlite_chunk_store.py func=get_by_ids упростить: вынести получение chunks из БД в _fetch_chunks_from_db(), конвертацию в domain objects в _convert_to_domain_chunks()

- [ ] file=src/adapters/persistence/sqlite_chunk_store.py func=update_topics упростить: вынести обновление topic для одного chunk в _update_chunk_topics(), batch update в _batch_update_topics()

- [ ] file=src/adapters/persistence/sqlite_message_store.py func=get_by_chat упростить: вынести получение messages из БД в _fetch_messages_from_db(), конвертацию в domain objects в _convert_to_domain_messages()

- [ ] file=src/adapters/persistence/sqlite_message_store.py func=get_context упростить: вынести вычисление time window в _calculate_time_window(), получение messages в window в _get_messages_in_window()

- [ ] file=src/adapters/persistence/sqlite_fts_index.py func=normalize_text упростить: вынести lowercase в _to_lowercase(), замену ё->е в _replace_yo(), удаление пунктуации в _remove_punctuation()

- [ ] file=src/adapters/vector/chroma_vector_index.py func=upsert упростить: вынести извлечение данных из VectorDoc в _extract_vector_doc_data(), подготовку documents в _prepare_documents()

- [ ] file=src/adapters/vector/chroma_vector_index.py func=get_embeddings_by_ids упростить: вынести получение embeddings из collection в _fetch_embeddings(), конвертацию результатов в _convert_to_dict()

- [ ] file=src/adapters/llm/llm_adapter.py func=complete упростить: вынести построение messages в _build_messages(), извлечение kwargs в _extract_kwargs(), вызов LLM в _call_llm()

- [ ] file=src/adapters/embedding/embedder_adapter.py упростить: методы embed_documents и embed_query просто делегируют к client, можно оставить как есть или добавить валидацию входных данных

- [ ] file=src/core/search.py func=search упростить: вынести вычисление query embedding в _compute_query_embedding(), векторный поиск в _perform_vector_search(), обогащение результатами в _enrich_with_messages()

- [ ] file=src/core/search.py func=search упростить: вынести фильтрацию по threshold в _filter_by_threshold(), получение chunks по IDs в _get_chunks_by_ids(), построение SearchResult в _build_search_results()

- [ ] file=src/core/domain.py упростить: добавить методы валидации для domain objects (Message.validate(), Chunk.validate())

- [ ] file=src/core/interfaces.py упростить: добавить документацию с примерами использования для каждого Protocol

- [ ] file=src/lib/syslog2.py func=syslog2 упростить: вынести форматирование сообщения в _format_message(), логирование в _log_message()

- [ ] file=src/lib/argparse2.py func=parse упростить: вынести парсинг опций в _parse_options(), парсинг команд в _parse_commands(), валидацию в _validate_parsed()

- [ ] file=src/lib/argparse2.py func=cmd_parse упростить: вынести парсинг опций в _parse_options(), парсинг команды в _parse_command(), парсинг аргументов в _parse_args()

- [ ] file=src/lib/argparse2.py func=gen_help упростить: вынести генерацию help для опций в _gen_options_help(), генерацию help для команд в _gen_commands_help()

- [ ] file=src/bot/tgbot.py func=_process_webhook_update упростить: оставить только parse_update()-_to_app_request()-call app.handle_request()-send_response(), весь access control-rate limit-command routing вынести из transport

- [ ] file=src/app/bootstrap.py func=create_app добавить: собрать все зависимости и вернуть объект App с методами handle_request(), handle_command(), ingest(), никаких импортов src/bot внутри core

- [ ] file=src/app/app.py func=handle_request создать: единая точка обработки входа (telegram или cli), вход AppRequest (user_id-chat_id-text-transport-meta), выход AppResponse (text-actions)

- [ ] file=src/app/types.py создать: dataclass AppRequest-AppResponse, CommandRequest-QueryRequest, чтобы transport не знал про core детали

- [ ] file=src/core/command_service.py func=dispatch создать: единый сервис команд, принимает CommandRequest, возвращает CommandResult, регистрации команд только здесь

- [ ] file=src/core/chat_service.py func=chat создать: единый сервис чата (retrieval+prompt+llm), без доступа к transport и файловой системе

- [ ] file=src/core/access_control.py func=check_access создать: логика allowed_chats-is_admin сюда, transport только вызывает и мапит отказ в текст

- [ ] file=src/core/rate_limit.py func=allow создать: логика frequency controller сюда, transport только вызывает

- [ ] file=src/core/commands.py удалить: перенести все user команды в src/core/commands/user.py, оставить только thin registration layer или удалить целиком

- [ ] file=src/core/admin_commands.py удалить: перенести admin команды в src/core/commands/admin.py, убрать дублирование с src/bot/admin_commands.py

- [ ] file=src/bot/admin_commands.py удалить: транспортный слой не содержит бизнес команд, только адаптация входа-выхода

- [ ] file=src/core/dispatcher.py func=dispatch убрать из runtime: заменить на CommandService с простым registry dict[name]=handler, без async ветвления если не критично

- [ ] file=src/bot/command_parser.py func=parse_command упростить: парсит только текст в CommandRequest (name-args-raw), никакой логики разрешений-частоты-списка команд

- [ ] file=src/bot/core.py class=LegaleBot расщепить: вынести conversation_state в src/core/conversation_state.py, context_provider в src/core/context_provider.py, prompt_builder в src/core/prompt_builder.py, llm_gateway в src/core/llm_gateway.py, LegaleBot оставить как thin facade или удалить

- [ ] file=src/core/hybrid_retrieval.py class=HybridRetrievalService разгрузить: вынести packing (dedup-neighbors-token budget) в src/core/context_packer.py func=pack(), вынести rephrase в отдельный этап ContextProvider

- [ ] file=src/core/message_search.py объединить: слить в src/core/hybrid_retrieval.py или src/core/context_provider.py, чтобы не было второго параллельного слоя "high-level search"

- [ ] file=src/ingestion/pipeline.py class=IngestionPipeline убрать оркестрацию: перенести orchestration в src/core/use_cases/ingest.py func=run_ingest(), а src/ingestion оставить как адаптеры parser-chunker-telegram_fetcher

- [ ] dir=src/core/ingest_use_cases удалить: слить ingest_messages-process_chunks-generate_embeddings-sync_to_vector_store-pipeline_orchestrator в один use case src/core/use_cases/ingest.py с шагами, зависимости только через interfaces

- [ ] file=src/storage/db.py разделить: вынести ORM модели в src/storage/models.py, вынести schema-migrations в src/storage/migrations/, db.py оставить только engine-session factory

- [ ] file=src/adapters/persistence/sqlite_message_store.py func=* запретить импорт src/storage/db.py как "бог-объект": использовать session factory и models, реализовать только MessageStore контракт

- [ ] file=src/adapters/persistence/sqlite_chunk_store.py func=* убрать embedding_json из ответственности стора: ChunkStore хранит текст-мета-связи, embeddings источник правды только в VectorIndex или только в SQLite (выбрать один)

- [ ] file=src/adapters/vector/chroma_vector_index.py func=query расширить: возвращать (chunk_id-score-metadata) без необходимости читать chunks из SQLite для каждого кандидата, минимизировать roundtrips

- [ ] file=src/core/interfaces.py пересмотреть: оставить "несущие стены" только для границ (MessageStore-ChunkStore-FTSIndex-VectorIndex-Embedder-LLM-ConfigProvider), удалить все что не используется напрямую use cases

- [ ] file=src/bot/config.py class=BotConfig перенести: в src/app/config_store.py (профиль и файлы), core получает ConfigProvider интерфейс, AdminManager не читает json напрямую

- [ ] file=src/bot/admin.py class=AdminManager заменить: на src/core/access_control.py + src/app/config_store.py, чтобы админ логика не жила в transport

- [ ] file=legale.py func=main упростить: CLI только собирает App через bootstrap и вызывает app.handle_command(), никакой ручной сборки Database-VectorStore внутри команд

- [ ] file=src/bot/cli.py func=main упростить: CLI как transport, делает AppRequest и печатает AppResponse, без прямого вызова LegaleBot и без регистрации команд

- [ ] file=src/app/main_cli.py func=create_dispatcher удалить: заменить на create_app() + CommandService registry, чтобы не было второго центра регистрации

- [ ] file=src/lib/syslog2.py scope=api унифицировать: сделать LoggingPort интерфейс в core (минимальный), syslog2 оставить реализацией, core не зависит от конкретного логгера

- [ ] file=src/core/llm.py class=LLMClient разгрузить: вынести retry-timeouts-backoff в src/core/llm_gateway.py, LLMClient оставить как тонкий HTTP клиент

- [ ] file=src/core/embedding.py class=EmbeddingClient разгрузить: вынести выбор local-vs-api в src/app/bootstrap.py, core использует только Embedder интерфейс, никаких if generator=="local" внутри core

- [ ] file=tests/* добавить: тесты на границы слоев - core use cases тестируются с in-memory фейками интерфейсов, без sqlite-chroma-telethon-fastapi*
