FROM debian:latest
RUN apt-get update && apt-get install -y tcpdump net-tools iputils-ping iproute2
WORKDIR /app
CMD ["tcpdump"]
