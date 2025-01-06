import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_aggregated_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    print("Initial DataFrame:")
    print(df.head())

    # Convert timestamp to datetime and extract temporal features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["second"] = df["timestamp"].dt.second

    # Drop timestamp after extracting features
    df.drop(columns=["timestamp"], inplace=True)
    print("DataFrame after extracting temporal features:")
    print(df.head())

    # Handle missing values
    # Option 1: Impute missing values with the median
    df.fillna(df.median(), inplace=True)

    # Option 2: Alternatively, you can drop rows with missing values
    # df.dropna(inplace=True)

    print("DataFrame after handling missing values:")
    print(df.head())

    # Normalize numerical columns
    scaler = StandardScaler()
    df[["total_bytes", "total_packets", "avg_packet_size", "hour", "minute", "second"]] = scaler.fit_transform(
        df[["total_bytes", "total_packets", "avg_packet_size", "hour", "minute", "second"]]
    )
    print("DataFrame after normalization:")
    print(df.head())

    # Save preprocessed data
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

if __name__ == "__main__":
    preprocess_aggregated_data("traffic_aggregated.csv", "preprocessed_traffic_data.csv")
