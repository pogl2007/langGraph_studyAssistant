# Ассистент по учебным материалам

Студент загружает лекции и методички, ассистент отвечает на вопросы **только по ним** — со ссылкой
на файл и страницу — и честно говорит «в материалах этого нет», если ответа там нет.

Стек: Python 3.12, Qdrant, BAAI/bge-m3, OpenAI-совместимый LLM (Ollama / vLLM / внешний API),
Streamlit, Docker Compose.

> Статус: **этап 1 — MVP.** Загрузка PDF, простой чанкинг, dense-поиск, ответ с цитатами,
> минимальный интерфейс. Гибридный поиск, реранкер, LangGraph, API, MCP — следующие этапы.
> Почему сделано именно так — в [DECISIONS.md](DECISIONS.md).

## Как устроено сейчас

```
PDF ──► pymupdf ──► чанки по 500 токенов ──► bge-m3 ──► Qdrant (dense)
                                                           │
вопрос ──► bge-m3 ──► top-5 по косинусу ◄──────────────────┘
                          │
                          ▼
           промпт: фрагменты в <document>, ссылки номерами
                          │
                          ▼
                LLM ──► [2] → [лекция.pdf, стр. 7]
```

## Запуск в Docker

```bash
git clone https://github.com/pogl2007/langGraph_studyAssistant.git && cd langGraph_studyAssistant
cp .env.example .env        # вписать LLM_API_KEY или указать Ollama
docker compose up -d --build
```

Интерфейс — http://localhost:8501, Qdrant — http://localhost:6333/dashboard.
При первом запуске скачивается модель эмбеддингов (~2,2 ГБ) в volume `hf_models`.

## Локальная разработка

```bash
docker compose up -d qdrant
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/streamlit run ui/streamlit_app.py
```

## Тесты

```bash
.venv/Scripts/python -m pytest
```

Тесты не тянут модели и не ходят в сеть: Qdrant поднимается в памяти, эмбеддинги — детерминированный
«мешок слов», LLM замокан.

## Структура

```
app/
  ingest/      парсинг PDF, чанкинг, индексация
  retrieval/   эмбеддинги, Qdrant, поиск
  answer/      промпты, цитаты, линейный RAG-пайплайн
  llm/         клиент OpenAI-совместимого API
  config.py    настройки из .env
  services.py  сборка компонентов
ui/            Streamlit
tests/
```

## Метрики

Появятся на этапе 2 (golden set) — только измеренные.
