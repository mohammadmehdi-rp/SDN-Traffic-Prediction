import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    print("Initial DataFrame:")
    print(df.head())

    # Replace IPv6 placeholders with '0.0.0.0'
    df['src_ip'] = df['src_ip'].replace('::', '0.0.0.0')
    df['dst_ip'] = df['dst_ip'].replace('::', '0.0.0.0')
    print("DataFrame after replacing IPv6 placeholders:")
    print(df.head())

    # Handle missing values
    df.fillna(0, inplace=True)
    print("DataFrame after handling missing values:")
    print(df.head())

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')

    # Extract temporal features (hour, minute, second)
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["second"] = df["timestamp"].dt.second

    # Drop timestamp column
    df.drop(columns=["timestamp"], inplace=True)
    print("DataFrame after processing timestamp:")
    print(df.head())

    # Normalize numerical data
    scaler = StandardScaler()
    df[["src_port", "dst_port", "length", "hour", "minute", "second"]] = scaler.fit_transform(
        df[["src_port", "dst_port", "length", "hour", "minute", "second"]]
    )
    print("DataFrame after normalization:")
    print(df.head())

    # Drop unnecessary columns (keep features that can indicate patterns)
    df.drop(columns=['src_ip', 'dst_ip', 'protocol'], inplace=True)

    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_data("traffic_data.csv", "preprocessed_traffic_data.csv")
