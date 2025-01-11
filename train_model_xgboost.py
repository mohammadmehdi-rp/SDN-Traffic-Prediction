import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_and_evaluate_model(input_csv):
    # Load preprocessed data
    data = pd.read_csv(input_csv)

    # Define features and target
    features = ['total_packets', 'avg_packet_size', 'hour', 'minute', 'second']
    target = 'total_bytes'

    # Split the dataset
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost Regressor
    xgb_model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"XGBoost Regressor - Mean Squared Error (MSE): {mse}")
    print(f"XGBoost Regressor - R^2 Score: {r2}")

    # Define the range for the "ideal" prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    
    # Scatter plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, label="Predictions")
    plt.plot([min_val, max_val], [min_val, max_val], 
              color="blue", linestyle="--", linewidth=2,
              label="Ideal Prediction")
    plt.xlabel("Actual Traffic Volume (Total Bytes)")
    plt.ylabel("Predicted Traffic Volume (Total Bytes)")
    plt.title("Actual vs Predicted Traffic Volume (XGBoost)")
    plt.legend()
    plt.grid(True)
    plt.savefig("actual_vs_predicted_xgboost.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Time-domain plot: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label="Actual Traffic Volume", color="green", alpha=0.7)
    plt.plot(range(len(y_test)), y_pred, label="Predicted Traffic Volume", color="orange", alpha=0.7)
    plt.xlabel("Time Index")
    plt.ylabel("Traffic Volume (Total Bytes)")
    plt.title("Time-Domain Analysis of Actual vs Predicted Traffic Volume (XGBoost)")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_domain_actual_vs_predicted_xgboost.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Return the model for prediction
    return xgb_model

if __name__ == "__main__":
    # Train and evaluate the model
    xgb_model = train_and_evaluate_model("preprocessed_traffic_data.csv")

    # Predict traffic volume for new data
    new_data = pd.DataFrame({
        'total_packets': [200],       
        'avg_packet_size': [512],     
        'hour': [14],                 
        'minute': [30],             
        'second': [15]             
    })

    predicted_volume = xgb_model.predict(new_data)
    print(f"Predicted Traffic Volume (Total Bytes): {predicted_volume[0]}")
