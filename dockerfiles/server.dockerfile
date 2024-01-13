FROM python:3.10-slim
EXPOSE $PORT

WORKDIR /
RUN apt update
RUN apt-get update && \
    apt-get upgrade -y
RUN pip install --upgrade pip

COPY src/requirements.txt requirements.txt
COPY src/prediction_server.py prediction_server.py
COPY pyproject.toml pyproject.toml
COPY src/ src/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install python-multipart

CMD exec uvicorn prediction_server:app --port $PORT --host 0.0.0.0 --workers 1
