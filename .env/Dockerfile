FROM ubuntu:22.04

RUN apt update && \
    apt install git python3.10 python3-pip -y && \
    cd /usr/bin && \
    ln -s python3.10 python

WORKDIR /root

COPY .env/requirements.txt /tmp

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt


# visdom ray-dashboard tensorboard
EXPOSE 8097 8265 6006
