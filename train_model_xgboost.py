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

    # Plot actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Traffic Volume (Total Bytes)")
    plt.ylabel("Predicted Traffic Volume (Total Bytes)")
    plt.title("Actual vs Predicted Traffic Volume XGBoost")
    # Save the figure before showing it
    plt.savefig("actual_vs_predicted_xgboost.png", dpi=300, bbox_inches='tight') 
    plt.show()

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

if __name__ == "__main__":
    train_and_evaluate_model("preprocessed_traffic_data.csv")
