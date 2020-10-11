FROM tensorflow/tensorflow:latest-gpu
RUN mkdir keras_models
WORKDIR keras_models
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]