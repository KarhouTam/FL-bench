FROM ubuntu:22.04

RUN apt update && \
    apt install git python3.10 python3-pip -y && \
    cd /usr/bin && \
    ln -s python3.10 python

WORKDIR /root

RUN git clone https://github.com/KarhouTam/FL-bench.git

WORKDIR /root/FL-bench

RUN pip install -r .environment/requirements.txt

# port for visdom
EXPOSE 8097
