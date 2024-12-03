from scapy.all import IP, TCP, send
import time

def generate_periodic_traffic(src_ip, dst_ip, dst_port, interval, duration):
    start_time = time.time()
    print(f"Generating traffic from {src_ip} to {dst_ip}:{dst_port} for {duration} seconds.")
    while time.time() - start_time < duration:
        packet = IP(src=src_ip, dst=dst_ip) / TCP(dport=dst_port)
        print(f"Sending packet: {packet.summary()}")  # Log packet details
        send(packet, verbose=True)  # Ensure Scapy sends the packet
        time.sleep(interval)

if __name__ == "__main__":
    src_ip = "10.0.0.1"
    dst_ip = "10.0.0.2"
    dst_port = 80
    interval = 0.2    # Traffic interval in seconds
    duration = 3600     # Traffic generation duration in seconds
    generate_periodic_traffic(src_ip, dst_ip, dst_port, interval, duration)
