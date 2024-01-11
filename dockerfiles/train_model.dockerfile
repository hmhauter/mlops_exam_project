# Base image
FROM python:3.10-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
RUN apt update
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install --upgrade pip
RUN git clone https://github.com/hmhauter/mlops_exam_project.git
RUN pip install -r mlops_exam_project/requirements.txt --no-cache-dir
RUN pip install mlops_exam_project/ --no-deps --no-cache-dir
RUN pip install dvc
RUN pip install "dvc[gdrive]"

ENTRYPOINT ["python", "-u", "mlops_exam_project/src/train_model.py"]