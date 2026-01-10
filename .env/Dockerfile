FROM ubuntu:24.04

RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y git python3.13 python3-pip && \
    cd /usr/bin && \
    ln -sf python3.13 python && \
    ln -sf python3.13 python3

COPY ./ /root/FL-bench

RUN pip install --upgrade --ignore-installed pip setuptools wheel && \
    pip install -r /root/FL-bench/.env/requirements.txt

WORKDIR /root

# visdom ray-dashboard tensorboard
EXPOSE 8097 8265 6006