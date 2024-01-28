ARG IMAGE_SOURCE=registry.cn-hangzhou.aliyuncs.com/karhou/linux:ubuntu-basic

FROM ${IMAGE_SOURCE}

ARG CHINA_MAINLAND=true

WORKDIR /etc/apt

RUN if [ "${CHINA_MAINLAND}" = "false" ]; then \
    rm sources.list && \
    mv sources.list.bak sources.list ; \
    fi
    
RUN apt update && \
    apt install python3-pip -y && \
    cd /usr/bin && \
    ln -s python3.10 python 

RUN if [ "${CHINA_MAINLAND}" = "true" ]; then \
    pip config set global.index-url https://mirrors.sustech.edu.cn/pypi/simple ; \
    fi 

WORKDIR /root

RUN git clone https://github.com/KarhouTam/FL-bench.git

WORKDIR /root/FL-bench

RUN pip install -r requirements.txt

# port for visdom
EXPOSE 8097
