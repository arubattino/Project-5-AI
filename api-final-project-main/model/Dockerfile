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

# Tensorflow (and Keras) with GPU support
RUN pip3 install tensorflow==2.8.0

ENV PYTHONPATH=$PYTHONPATH:/src/

ADD ./ /src/

WORKDIR /src

FROM base as build
ENTRYPOINT ["python3", "/src/ml_service.py"]

FROM base as test
RUN ["pytest", "-v", "/src/tests"]


