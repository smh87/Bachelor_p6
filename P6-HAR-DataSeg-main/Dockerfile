
FROM nvcr.io/nvidia/tensorflow:21.04-tf2-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
WORKDIR /src    
COPY ./locked-requirements.txt /locked-requirements.txt
COPY ./ ./

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r /locked-requirements.txt
ENTRYPOINT ["python", "test.py"]


