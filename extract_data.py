import pandas as pd
from scapy.all import rdpcap

def extract_data(pcap_file, output_csv):
    print(f"Reading from {pcap_file}...")
    packets = rdpcap(pcap_file)
    print(f"Number of packets: {len(packets)}")

    data = []
    for packet in packets:
        if packet.haslayer("IP"):
            data.append({
                "timestamp": packet.time,
                "src_ip": packet["IP"].src,
                "dst_ip": packet["IP"].dst,
                "src_port": packet.sport,
                "dst_port": packet.dport,
                "protocol": packet["IP"].proto,
                "length": len(packet)
            })
        elif packet.haslayer("IPv6"):
            data.append({
                "timestamp": packet.time,
                "src_ip": packet["IPv6"].src,
                "dst_ip": packet["IPv6"].dst,
                "src_port": packet.sport if packet.haslayer("TCP") or packet.haslayer("UDP") else None,
                "dst_port": packet.dport if packet.haslayer("TCP") or packet.haslayer("UDP") else None,
                "protocol": packet["IPv6"].nh,
                "length": len(packet)
            })

    df = pd.DataFrame(data)
    print(f"Number of rows in DataFrame: {len(df)}")
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    extract_data("capture.pcap", "traffic_data.csv")
