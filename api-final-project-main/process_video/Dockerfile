FROM python:3.8.13 as base

# Install some packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    vim \
    wget \
    curl

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/src/

ADD ./ /src/

WORKDIR /src

FROM base as build
ENTRYPOINT ["python3", "/src/process_video.py"]

FROM base as test
RUN ["pytest", "-v", "/src/tests"]
