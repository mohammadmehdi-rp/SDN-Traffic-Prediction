from scapy.all import IP, TCP, send
import random
import time

def generate_variable_traffic(src_ip, dst_ip, dst_port, interval, duration):
    start_time = time.time()
    print(f"Generating traffic from {src_ip} to {dst_ip}:{dst_port} for {duration} seconds.")

    # Open a file to log traffic details for machine learning
    with open("traffic_log.csv", "w") as log_file:
        log_file.write("timestamp,src_ip,dst_ip,dst_port,packet_size\n")

        while time.time() - start_time < duration:
            # Generate a random packet size (e.g., between 50 and 1500 bytes)
            packet_size = random.randint(50, 1500)
            
            # Create a packet with a dummy payload of the specified size
            payload = "X" * packet_size
            packet = IP(src=src_ip, dst=dst_ip) / TCP(dport=dst_port) / payload
            
            # Log the packet details
            timestamp = time.time()
            log_file.write(f"{timestamp},{src_ip},{dst_ip},{dst_port},{packet_size}\n")
            
            print(f"Sending packet: {packet.summary()} (Size: {packet_size} bytes)")
            
            # Send the packet
            send(packet, verbose=False)
            
            # Introduce variability in the interval
            time.sleep(interval + random.uniform(-0.1, 0.1))

if __name__ == "__main__":
    src_ip = "10.0.0.1"
    dst_ip = "10.0.0.2"
    dst_port = 80
    interval = 0.2    # Base traffic interval in seconds
    duration = 60     # Traffic generation duration in seconds
    generate_variable_traffic(src_ip, dst_ip, dst_port, interval, duration)
