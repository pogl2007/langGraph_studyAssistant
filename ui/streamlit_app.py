"""Минимальный интерфейс MVP: загрузка PDF и чат с цитатами.

Пока ядро вызывается прямо из процесса Streamlit. На этапе 7 интерфейс перейдёт на HTTP API.
"""

import asyncio
import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.answer.pipeline import Answer  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.services import build_pipeline, get_indexer, get_store  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()

st.set_page_config(page_title="Ассистент по материалам", page_icon="📚")
st.title("Ассистент по учебным материалам")


@st.cache_resource(show_spinner="Загружаю модель эмбеддингов…")
def indexer():
    return get_indexer()


def list_files() -> list[dict]:
    store = get_store()
    if not store.client.collection_exists(store.collection):
        return []
    return store.list_files()


async def ask(question: str, course: str) -> Answer:
    indexer()  # модель эмбеддингов уже в кэше, пайплайн возьмёт её же
    pipeline = build_pipeline()
    try:
        return await pipeline.answer(question, course=course or None)
    finally:
        await pipeline.llm.aclose()


with st.sidebar:
    course = st.text_input("Курс", value="default", help="Поиск идёт только по материалам этого курса")
    uploads = st.file_uploader("Лекции (PDF)", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Проиндексировать", type="primary"):
        for up in uploads:
            data = up.getvalue()
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                st.error(f"{up.name}: больше {settings.max_upload_mb} МБ")
                continue
            with st.spinner(f"Индексирую {up.name}…"):
                r = indexer().index_pdf(up.name, data, course=course or "default")
            st.success(f"{r.filename}: {r.pages} стр., {r.chunks} чанков за {r.seconds:.0f} с")

    st.subheader("Проиндексировано")
    files = list_files()
    if not files:
        st.caption("Пока ничего")
    for f in files:
        st.caption(f"**{f['filename']}** · курс {f['course']} · {f['pages_total']} стр. · {f['chunks_total']} чанков")


if "messages" not in st.session_state:
    st.session_state.messages = []


def show_answer(answer: Answer) -> None:
    st.markdown(answer.text)
    if answer.citations:
        with st.expander(f"Источники ({len(answer.citations)})"):
            for c in answer.citations:
                st.markdown(f"**{c.label}**")
                st.caption(c.snippet + "…")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            show_answer(msg["answer"])
        else:
            st.markdown(msg["content"])

if question := st.chat_input("Вопрос по материалам"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Ищу в материалах…"):
                answer = asyncio.run(ask(question, course))
        except Exception as e:  # сеть, ключ, таймаут провайдера
            st.session_state.messages.pop()  # вопрос без ответа в истории не оставляем
            st.error(f"Не удалось получить ответ: {e}")
            st.stop()
        show_answer(answer)
    st.session_state.messages.append({"role": "assistant", "answer": answer})
