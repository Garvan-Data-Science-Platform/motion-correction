FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY setup.py README.rst HISTORY.rst ./

RUN pip install .

COPY . .

CMD python working.py