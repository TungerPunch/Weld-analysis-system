FROM python:3.9-slim

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY . .

CMD [ "python", "./main.py"]