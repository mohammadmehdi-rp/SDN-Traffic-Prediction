import pandas as pd

def aggregate_traffic(input_csv, output_csv, interval=1):
    print(f"Reading traffic log from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    # Resample data into intervals and aggregate
    aggregated = df.resample(f"{interval}S").agg({
        'packet_size': ['sum', 'count', 'mean']
    })
    
    # Flatten multi-level columns
    aggregated.columns = ['total_bytes', 'total_packets', 'avg_packet_size']
    aggregated.reset_index(inplace=True)
    
    print(f"Aggregated data:\n{aggregated.head()}")
    aggregated.to_csv(output_csv, index=False)
    print(f"Aggregated data saved to {output_csv}")

if __name__ == "__main__":
    aggregate_traffic("traffic_log.csv", "traffic_aggregated.csv", interval=1)
