# 🐭 Мишка-шкряботушка — Лісовий RAG-помічник

Telegram-бот для лісових звірів на базі **Python 3.11+**, **aiogram 3.x**, **Groq API**, **FastEmbed**, **ChromaDB** та **SQLite (SQLAlchemy 2.0 async)**.

## Швидкий старт (в один рядок)

1. Створіть файл конфігурації `.env` на основі шаблону [`.env.example`](.env.example) та вкажіть ваші ключі `BOT_TOKEN` і `GROQ_API_KEY`:

```bash
cp .env.example .env
```

2. Запустіть бота у фоновому режимі через Docker Compose:

```bash
docker compose up -d --build
```

Дані реляційної бази (`SQLite`) та векторного сховища (`ChromaDB`) автоматично зберігаються на хості у директорії `./data`.
