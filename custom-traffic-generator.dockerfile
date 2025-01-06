FROM python:3.8-slim
RUN apt-get update && apt-get install -y net-tools iputils-ping
COPY traffic_generator.py /app/traffic_generator.py
WORKDIR /app
RUN pip install scapy
CMD ["python3", "/app/traffic_generator.py"]
