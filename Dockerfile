ARG IMAGE_SOURCE=registry.cn-hangzhou.aliyuncs.com/karhou/linux:ubuntu-basic

FROM ${IMAGE_SOURCE}

ARG CHINA_MAINLAND=true
ARG FL_BENCH_ROOT=../

WORKDIR /etc/apt

RUN if [ "${CHINA_MAINLAND}" = "false" ]; then \
    rm sources.list && \
    mv sources.list.bak sources.list ; \
    fi

RUN apt update && \
    apt install -y \
    python3.10 \
    python3-pip

RUN if [ "${CHINA_MAINLAND}" = "true" ]; then \
    pip config set global.index-url https://mirrors.sustech.edu.cn/pypi/simple ; \
    fi 

RUN pip install --upgrade pip && \
    pip install poetry

COPY ${FL_BENCH_ROOT} /root/FL-bench

WORKDIR /root/FL-bench

RUN if [ ${CHINA_MAINLAND} = "false" ]; then \
    sed -i "26,30d" pyproject.toml && \
    poetry lock --no-update ; \
    fi

RUN poetry install

CMD poetry shell

# port for visdom
EXPOSE 8097
