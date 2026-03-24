# официальный slim-образ Python 3.12 для меньшего размера
FROM python:3.12-slim

WORKDIR /usr/src/app

COPY ./app ./
COPY ./requirements.txt ./
COPY ./model.joblib ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Для безопасности создаётся пользователь с правами на управление директорей
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /usr/src/app
USER appuser

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
