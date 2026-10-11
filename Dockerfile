FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models

WORKDIR /srv

# torch только CPU-сборки: обычный torch с PyPI под Linux тянет CUDA на несколько гигабайт.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Зависимости берём из pyproject.toml — один источник правды, без отдельного requirements.txt.
# Слой с зависимостями кэшируется и не пересобирается при правке кода.
COPY pyproject.toml .
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" \
        > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY app ./app
COPY certs ./certs

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models /srv/data \
    && chown -R app /models /srv/data
USER app

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
