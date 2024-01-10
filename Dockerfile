FROM python:3.11-slim

WORKDIR /opt/app

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

CMD [ "python", "-m" , "src" ]
