FROM ubuntu:22.04

RUN apt update && \
    apt install build-essential git python3.10 python3-pip -y && \
    cd /usr/bin && \
    ln -s python3.10 python

WORKDIR /root

RUN git clone https://github.com/KarhouTam/FL-bench.git

WORKDIR /root/FL-bench

RUN pip install -r requirements.txt

# port for visdom
EXPOSE 8097
