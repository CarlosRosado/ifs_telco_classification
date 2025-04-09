import os
import pytest
from unittest.mock import patch, MagicMock
from src.model.train_mlflow import ModelTrainer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch

@pytest.fixture
def mock_env_vars():
    """
    Mock environment variables for the test.
    """
    with patch.dict(os.environ, {
        "LOCAL_DATA_PATH": "/app/data_files/ML_telco_customer_churn_data.csv",
        "MODEL_OUTPUT_PATH": "./models/churn_model.pkl",
        "MLFLOW_TRACKING_URI": "http://localhost:5001"
    }):
        yield


@patch("os.makedirs")  
@patch("os.path.exists", return_value=False) 
@patch("src.model.train_mlflow.mlflow")
@patch("src.model.train_mlflow.DataLoader.load_data")  
def test_train_model(mock_load_data, mock_mlflow, mock_path_exists, mock_makedirs, mock_env_vars):
    """
    Test the train_model method of the ModelTrainer class
    """

    mock_df = pd.DataFrame({
        "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "MonthlyCharges": [70.35, 89.10, 29.85, 56.75, 99.99, 45.50, 80.20, 60.00, 70.00, 50.00, 90.00, 40.00],
        "TotalCharges": [70.35, 178.20, 29.85, 113.50, 499.95, 45.50, 560.40, 120.00, 140.00, 100.00, 180.00, 80.00],
        "Churn": ["No", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No"],
        "PhoneService": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
        "StreamingTV": ["Yes", "No", "No", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes", "No"],
        "StreamingMovies": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "InternetService": ["Fiber optic", "DSL", "No", "Fiber optic", "DSL", "DSL", "Fiber optic", "DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Electronic check", "Mailed check", "Electronic check", "Mailed check", "Electronic check", "Mailed check", "Electronic check", "Bank transfer (automatic)", "Mailed check"]
    })
    mock_load_data.return_value = mock_df

    # Mock mlflow methods
    mock_mlflow.set_experiment.return_value = None
    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    # Create an instance of ModelTrainer
    trainer = ModelTrainer(
        data_path=os.environ["LOCAL_DATA_PATH"],
        model_output_path=os.environ["MODEL_OUTPUT_PATH"],
        mlflow_tracking_uri=os.environ["MLFLOW_TRACKING_URI"]
    )

    # Call the train_model method
    trainer.train_model()

    # Assertions to verify the mocks were called
    mock_load_data.assert_called_once_with("/app/data_files/ML_telco_customer_churn_data.csv")
    mock_mlflow.set_experiment.assert_called_once_with("IFS - Telco Churn Prediction")
    mock_mlflow.start_run.assert_called_once()