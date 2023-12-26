FROM nvcr.io/nvidia/tritonserver:23.10-py3

COPY ./requirements.txt /opt/tritonserver/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /opt/tritonserver/requirements.txt

CMD ["tritonserver", "--model-repository=/models", "--log-warning=True", "--log-error=True"]
