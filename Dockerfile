FROM python:3.7.0

RUN mkdir /dis

RUN apt-get update && \
  apt-get upgrade -y && \
  pip install --upgrade pip && \
  pip install numpy==1.17.0 && \
  pip install torch==1.4.0 

COPY ./ /dis

WORKDIR /dis

CMD ['python', 'main.py']
