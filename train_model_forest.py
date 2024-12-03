import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = "preprocessed_traffic_data.csv"  # Update with the correct file path if needed
data = pd.read_csv(file_path)

# Feature engineering: Combine time columns into a single feature
data['time_in_seconds'] = data['hour'] * 3600 + data['minute'] * 60 + data['second']

# Define features and target
features = ['src_port', 'dst_port', 'time_in_seconds']
target = 'length'

# Split the dataset
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_rf = rf_model.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error (MSE): {mse_rf}")
print(f"Random Forest - R^2 Score: {r2_rf}")

# Prediction Example: Predict traffic for new data
# Assume you have new data in the same format as the original data
new_data = pd.DataFrame({
    'src_port': [8080],  # Example src_port
    'dst_port': [80],    # Example dst_port
    'time_in_seconds': [10000]  # Example time_in_seconds
})

# Scale the new data
new_data_scaled = scaler.transform(new_data)

# Make predictions on the new data
predicted_length = rf_model.predict(new_data_scaled)
print(f"Predicted Packet Length: {predicted_length[0]}")
