import pytest

from app.graph.rules import classify, is_follow_up


@pytest.mark.parametrize(
    ("text", "route", "topic"),
    [
        ("Нарисуй схему силовых линий точечного заряда", "draw", "силовых линий точечного заряда"),
        ("Сгенерируй картинку к теме циклотрон", "draw", "циклотрон"),
        ("Изобрази конденсатор", "draw", "конденсатор"),
        ("Составь тест по теме Закон Кулона", "quiz", "Закон Кулона"),
        ("Проверь меня", "quiz", None),
        ("Что мне повторить?", "progress", None),
        ("Привет!", "chitchat", None),
        ("Что такое циклотрон?", "question", None),
        ("Сформулируй первое правило Кирхгофа", "question", None),
    ],
)
def test_rules_route_without_llm(text, route, topic):
    assert classify(text) == (route, topic)


def test_ambiguous_text_goes_to_llm():
    assert classify("Запиши систему уравнений Максвелла") is None


@pytest.mark.parametrize(
    ("text", "history", "expected"),
    [
        ("а закон Кулона?", True, True),
        ("а второй?", True, True),
        ("Что утверждает первый закон Ньютона и где он применяется в электростатике?", True, False),
        ("а закон Кулона?", False, False),
    ],
)
def test_follow_up_detection(text, history, expected):
    assert is_follow_up(text, history) is expected
