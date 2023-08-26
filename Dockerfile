FROM python:3.10.5-slim-buster

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app.py app.py
COPY data data
RUN python -m nltk.downloader all

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0" ]