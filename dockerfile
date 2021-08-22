# Using python container
FROM python:3.8-slim-buster

# Setting environment that ensures that the python output is sent straight to terminal
ENV PYTHONUNBUFFERED 1

WORKDIR /app
ADD . /app
COPY . /requirements.txt /run/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . /app

