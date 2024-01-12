FROM python:3.10-slim-buster

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
RUN pip install mlops_exam_project/ --no-deps --no-cache-dir
RUN pip install dvc
RUN pip install "dvc[gs]"
RUN apt-get -y update
RUN apt-get -y install curl

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-459.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-459.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh