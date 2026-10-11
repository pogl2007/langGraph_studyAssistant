"""Дешёвые правила маршрутизации. LLM зовём, только если ни одно правило не сработало.

Правила дают ответ за микросекунды и без токенов, а у бесплатного провайдера лимит 5 запросов
в минуту — каждый сэкономленный вызов заметен.
"""

import re

from app.graph.state import Route

_QUIZ = re.compile(
    r"(\bтест\w*|\bквиз\w*|проверь меня|проверочн\w*|вопрос\w* для самопроверки|погоняй меня|\bэкзаменуй)",
    re.IGNORECASE,
)
_QUIZ_TOPIC = re.compile(r"\b(?:по теме|на тему|по)\s+[«\"]?(.+?)[»\"]?\s*[.!?]*$", re.IGNORECASE)
_DRAW = re.compile(
    r"^(нарисуй|изобрази|сгенерируй (картинку|рисунок|изображение|схему)|покажи (картинкой|на рисунке)|"
    r"сделай (рисунок|иллюстрацию|картинку))",
    re.IGNORECASE,
)
_DRAW_PREFIX = re.compile(
    r"^(нарисуй|изобрази|сгенерируй|покажи|сделай)\s+(мне\s+)?(картинку|рисунок|изображение|схему|иллюстрацию)?\s*"
    r"(к теме|по теме|на тему|про|для)?\s*",
    re.IGNORECASE,
)
_PROGRESS = re.compile(
    r"(что (мне )?(стоит |надо |нужно )?повторить|мои ошибки|слаб\w+ тем|в чём я ошиба|в чем я ошиба|мой прогресс)",
    re.IGNORECASE,
)
_SMALL_TALK = re.compile(
    r"^(привет\w*|здравствуй\w*|добр\w+ (день|вечер|утро)|доброе утро|спасибо\w*|благодар\w*|пока|"
    r"до свидания|ок|окей|понятно|ясно|круто|отлично|hi|hello|thanks?)[\s!.)]*$",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(
    r"^(что|как|почему|зачем|чем|какой|какая|какое|какие|каков\w*|сколько|когда|где|кто|"
    r"объясни|расскажи|сформулируй|дай определение|определи|опиши|выведи|докажи|сравни|в чём|в чем)\b",
    re.IGNORECASE,
)
_FOLLOW_START = re.compile(r"^(а|и|ещё|еще|тогда|то есть|подробнее|поясни|почему так)\b", re.IGNORECASE)
_REFERENCE = re.compile(
    r"\b(он|она|оно|они|его|её|ее|их|этот|эта|это|этого|этой|тот|та|второ\w+|перв\w+|предыдущ\w+|последн\w+)\b",
    re.IGNORECASE,
)


def classify(text: str) -> tuple[Route, str | None] | None:
    """(маршрут, тема теста) или None, если правило не уверено."""
    t = text.strip()
    if _PROGRESS.search(t):
        return "progress", None
    if _DRAW.match(t):
        return "draw", draw_topic(t)
    if _QUIZ.search(t):
        m = _QUIZ_TOPIC.search(t)
        return "quiz", (m.group(1).strip() if m else None)
    if _SMALL_TALK.match(t):
        return "chitchat", None
    if t.endswith("?") or _QUESTION_START.match(t):
        return "question", None
    return None


def draw_topic(text: str) -> str:
    """«Нарисуй схему силовых линий точечного заряда» → «силовых линий точечного заряда»."""
    topic = _DRAW_PREFIX.sub("", text.strip()).strip(" .!?«»\"")
    return topic or text.strip()


def small_talk_reply(text: str) -> str | None:
    """Шаблонный ответ на «привет» и «спасибо» — LLM для этого не нужна."""
    t = text.strip().lower()
    if not _SMALL_TALK.match(t):
        return None
    if re.match(r"^(спасибо|благодар|thank)", t):
        return "Пожалуйста! Если в лекциях что-то осталось непонятным — спрашивайте или попросите тест по теме."
    if re.match(r"^(пока|до свидания)", t):
        return "До встречи! Удачи с учёбой."
    if re.match(r"^(ок|окей|понятно|ясно|круто|отлично)", t):
        return "Отлично. Задайте следующий вопрос или попросите тест по этой теме."
    return (
        "Привет! Я отвечаю на вопросы по загруженным лекциям со ссылками на страницы и составляю тесты. "
        "Например: «Что такое напряжённость электрического поля?» или «Составь тест по теореме Гаусса»."
    )


def is_follow_up(text: str, has_history: bool) -> bool:
    """Вопрос опирается на предыдущий: «а что насчёт второго?», «подробнее про это»."""
    if not has_history:
        return False
    words = text.split()
    # «первый закон Ньютона» — самостоятельный вопрос, а «а второй?» — нет: местоимение
    # или порядковое считаем ссылкой на прошлое только в коротком вопросе
    return bool(_FOLLOW_START.match(text.strip())) or len(words) <= 2 or (
        bool(_REFERENCE.search(text)) and len(words) <= 5
    )
