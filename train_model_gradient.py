import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
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

    # Train the Gradient Boosting Regressor
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_gb = gb_model.predict(X_test)
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    print(f"Gradient Boosting - Mean Squared Error (MSE): {mse_gb}")
    print(f"Gradient Boosting - R^2 Score: {r2_gb}")

    # Plot actual vs predicted values
    plt.scatter(y_test, y_pred_gb)
    plt.xlabel("Actual Traffic Volume (Total Bytes)")
    plt.ylabel("Predicted Traffic Volume (Total Bytes)")
    plt.title("Actual vs Predicted Traffic Volume (Gradient Boosting)")
    # Save the figure before showing it
    plt.savefig("actual_vs_predicted_gradient.png", dpi=300, bbox_inches='tight') 
    plt.show()

    # Return the model for prediction
    return gb_model, X_test, y_test

def predict_traffic(model, new_data):
    # Make a prediction for the new data
    predicted_traffic = model.predict(new_data)
    print(f"Predicted Traffic Volume (Total Bytes): {predicted_traffic[0]}")

if __name__ == "__main__":
    # Train and evaluate the model
    gb_model, X_test, y_test = train_and_evaluate_model("preprocessed_traffic_data.csv")

    # Example prediction with new data (adjust as necessary based on feature values)
    new_data = pd.DataFrame({
        'total_packets': [1000],      # Example data
        'avg_packet_size': [200],     # Example data
        'hour': [10],                 # Example data
        'minute': [30],               # Example data
        'second': [0]                 # Example data
    })

    # Make a prediction for the new data
    predict_traffic(gb_model, new_data)
