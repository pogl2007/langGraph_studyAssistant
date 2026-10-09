"""Интерфейс ассистента. Тонкий клиент: вся логика в API, здесь только отображение.

Запуск: streamlit run ui/streamlit_app.py  (API должен быть поднят, адрес — STUDY_API_URL)
"""

import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client import ApiClient  # noqa: E402

API_URL = os.getenv("STUDY_API_URL", "http://localhost:8000")
NODE_NAMES = {
    "summarize": "сжатие истории",
    "route": "разбор запроса",
    "retrieve": "поиск",
    "grade": "оценка найденного",
    "rewrite": "переформулировка",
    "generate": "ответ",
    "no_answer": "нет в материалах",
    "make_quiz": "составление теста",
    "ask_quiz": "ожидание ответов",
    "check_quiz": "проверка теста",
    "chitchat": "разговор",
    "out_of_scope": "вне темы",
    "progress": "слабые темы",
}

st.set_page_config(page_title="Ассистент по материалам", page_icon="📚", layout="wide")
api = ApiClient(API_URL)
ss = st.session_state
ss.setdefault("thread_id", uuid.uuid4().hex)
ss.setdefault("messages", [])  # {"role", "text", "citations", "trace"}
ss.setdefault("quiz", None)
ss.setdefault("pending", None)  # сообщение, которое надо отправить в этом прогоне


def citations_md(citations: list[dict]) -> str:
    seen, parts = set(), []
    for c in citations:
        label = c.get("label") or f"[{c['filename']}, стр. {c['page']}]"
        if label in seen:
            continue
        seen.add(label)
        parts.append(f"[{label.strip('[]')}]({c['url']})" if c.get("url") else label)
    return " · ".join(parts)


def run_stream(events) -> dict | None:
    """Показывает ход графа и токены, возвращает итоговое сообщение ассистента."""
    status = st.status("Думаю…", expanded=False)
    text_box = st.empty()
    streamed, trace, final = "", [], None
    for name, data in events:
        if name == "status":
            status.update(label=data["text"])
        elif name == "node":
            trace.append(data["node"])
            status.write(f"✓ {NODE_NAMES.get(data['node'], data['node'])}")
        elif name == "route":
            status.write(f"маршрут: **{data['route']}** ({'правила' if data['by'] == 'rules' else 'LLM'})")
        elif name == "grade":
            best = f", лучший скор {data['best']}" if data.get("best") is not None else ""
            status.write(f"найденное: **{'подходит' if data['grade'] == 'relevant' else 'не подходит'}**{best}")
        elif name == "rewrite":
            status.write(f"новый запрос: «{data['query']}»")
        elif name == "token":
            streamed += data["text"]
            text_box.markdown(streamed + "▌")
        elif name == "quiz":
            ss.quiz = data
            text = f"Тест по теме «{data['topic']}»: {len(data['questions'])} вопросов — ответьте ниже."
            final = {"role": "assistant", "text": text, "citations": [], "trace": trace}
        elif name == "answer":
            final = {"role": "assistant", "text": data["text"], "citations": data["citations"], "trace": trace}
        elif name == "error":
            final = {"role": "assistant", "text": f"⚠️ {data['message']}", "citations": [], "trace": trace}
    status.update(label=" → ".join(NODE_NAMES.get(n, n) for n in trace) or "готово", state="complete")
    text_box.empty()
    return final


def show_message(i: int, m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m.get("citations"):
            st.caption("Источники: " + citations_md(m["citations"]))
        if m["role"] == "assistant" and i > 0:
            if m.get("trace"):
                st.caption(" → ".join(NODE_NAMES.get(n, n) for n in m["trace"]))
            rating = st.feedback("thumbs", key=f"fb_{ss.thread_id}_{i}")
            if rating is not None and not m.get("rated"):
                m["rated"] = True
                question = ss.messages[i - 1]["text"] if ss.messages[i - 1]["role"] == "user" else ""
                api.feedback(thread_id=ss.thread_id, rating=1 if rating == 1 else -1, question=question, answer=m["text"])
                st.toast("Спасибо за оценку")


# ---------- боковая панель ----------

with st.sidebar:
    st.header("📚 Материалы")
    user_id = st.text_input("Имя", value="student", help="Слабые темы запоминаются для этого имени")
    course = st.text_input("Курс", value="physics", help="Поиск идёт только по материалам курса") or None

    uploads = st.file_uploader("Лекции (PDF)", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Проиндексировать", type="primary"):
        tasks = []
        for up in uploads:
            try:
                tasks.append(api.upload(up.name, up.getvalue(), course or "default"))
            except Exception as e:
                st.error(f"{up.name}: {e}")
        bar = st.progress(0.0, text="Индексация…")
        pending = {t["task_id"]: t["filename"] for t in tasks}
        done = 0
        while pending:
            for task_id in list(pending):
                info = api.task(task_id)
                if info["status"] in ("done", "failed"):
                    done += 1
                    name = pending.pop(task_id)
                    if info["status"] == "done":
                        r = info["result"]
                        verb = "без изменений" if r["status"] == "unchanged" else f"{r['chunks']} чанков за {r['seconds']:.0f} с"
                        st.success(f"{name}: {r['pages']} стр., {verb}")
                    else:
                        st.error(f"{name}: {info['error']}")
            bar.progress(done / max(len(tasks), 1), text=f"Индексация… {done}/{len(tasks)}")
            time.sleep(1)
        bar.empty()

    try:
        docs = api.documents(course)
    except Exception as e:
        st.error(f"API недоступен: {e}")
        docs = []
    st.subheader(f"Проиндексировано ({len(docs)})")
    for d in docs:
        col1, col2 = st.columns([5, 1])
        col1.caption(f"**{d['filename']}** · {d['pages_total']} стр. · {d['chunks_total']} чанков")
        if col2.button("✕", key=f"del_{d['file_id']}", help="Удалить"):
            api.delete(d["file_id"])
            st.rerun()

    st.divider()
    topic = st.text_input("Тест по теме", placeholder="например, теорема Гаусса")
    c1, c2 = st.columns(2)
    if c1.button("Составить тест", disabled=not topic):
        ss.pending = f"Составь тест по теме {topic}"
    if c2.button("Что повторить?"):
        ss.pending = "Что мне повторить?"
    if st.button("Новый диалог"):
        ss.thread_id, ss.messages, ss.quiz = uuid.uuid4().hex, [], None
        st.rerun()


# ---------- чат ----------

st.title("Ассистент по учебным материалам")
st.caption("Отвечаю только по загруженным лекциям, со ссылками на страницы. Если ответа нет — так и скажу.")

for i, m in enumerate(ss.messages):
    show_message(i, m)

if ss.quiz:
    quiz = ss.quiz
    with st.chat_message("assistant"), st.form("quiz_form"):
        st.markdown(f"**Тест: {quiz['topic']}**")
        answers = []
        for qi, q in enumerate(quiz["questions"]):
            choice = st.radio(f"{qi + 1}. {q['question']}", q["options"], index=None, key=f"q_{ss.thread_id}_{qi}")
            answers.append(q["options"].index(choice) if choice is not None else None)
        if st.form_submit_button("Проверить", type="primary"):
            ss.quiz = None
            with st.chat_message("assistant"):
                reply = run_stream(api.resume(answers, ss.thread_id, user_id, course))
            if reply:
                ss.messages.append(reply)
            st.rerun()

question = st.chat_input("Вопрос по материалам") or ss.pop("pending", None)
if question:
    ss.pending = None
    ss.messages.append({"role": "user", "text": question})
    show_message(len(ss.messages) - 1, ss.messages[-1])
    with st.chat_message("assistant"):
        reply = run_stream(api.chat(question, ss.thread_id, user_id, course))
    if reply:
        ss.messages.append(reply)
    st.rerun()
