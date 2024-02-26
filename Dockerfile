FROM python:3.11

WORKDIR /fastapi_docker

COPY . /fastapi_docker

RUN pip install -r requirements.txt
