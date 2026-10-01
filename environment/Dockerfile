FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
WORKDIR /task
COPY data/ /task/data/
COPY public_examples/ /task/examples/
COPY data_tools.py verify_data.py /tmp/
RUN python3 /tmp/verify_data.py && rm /tmp/verify_data.py /tmp/data_tools.py
