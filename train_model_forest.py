import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

    # Train the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"Random Forest - Mean Squared Error (MSE): {mse_rf}")
    print(f"Random Forest - R^2 Score: {r2_rf}")

    # Plot actual vs predicted values
    plt.scatter(y_test, y_pred_rf)
    plt.xlabel("Actual Traffic Volume (Total Bytes)")
    plt.ylabel("Predicted Traffic Volume (Total Bytes)")
    plt.title("Actual vs Predicted Traffic Volume (Forest)")
    # Save the figure before showing it
    plt.savefig("actual_vs_predicted_forest.png", dpi=300, bbox_inches='tight') 
    plt.show()

    return rf_model, X_test, y_test

def predict_traffic(rf_model, new_data):
    # Predict traffic volume for new input
    prediction = rf_model.predict(new_data)
    print(f"Predicted Traffic Volume (Total Bytes): {prediction[0]}")

if __name__ == "__main__":
    # Train and evaluate the model
    rf_model, X_test, y_test = train_and_evaluate_model("preprocessed_traffic_data.csv")

    # Example prediction with new data (adjust as necessary based on feature values)
    new_data = pd.DataFrame({
        'total_packets': [1000],      # Example data
        'avg_packet_size': [200],     # Example data
        'hour': [10],                 # Example data
        'minute': [30],               # Example data
        'second': [0]                 # Example data
    })

    # Make a prediction for the new data
    predict_traffic(rf_model, new_data)
