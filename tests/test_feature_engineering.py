import pandas as pd
import pytest
from src.preprocessing.feature_engineering import FeatureEngineer

@pytest.fixture
def mock_raw_data():
    """
    Creates a mock raw dataset for testing.
    """
    data = {
        "customerID": ["001", "002", "003"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No"],
        "tenure": [12, 24, 36],
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 42.30],
        "TotalCharges": ["29.85", "1889.50", "1840.75"],
        "Churn": ["No", "Yes", "No"]
    }
    return pd.DataFrame(data)

def test_preprocess_target(mock_raw_data):
    """
    Test the preprocess_target method.
    """
    processed_data = FeatureEngineer.preprocess_target(mock_raw_data)
    assert "Churn" in processed_data.columns, "The 'Churn' column is missing."
    assert processed_data["Churn"].isin([0, 1]).all(), "The 'Churn' column was not converted to binary values."

def test_generate_new_features(mock_raw_data):
    """
    Test the generate_new_features method.
    """
    processed_data = FeatureEngineer.generate_new_features(mock_raw_data)
    assert "tenure_years" in processed_data.columns, "The 'tenure_years' feature was not created."
    assert "avg_monthly_charge" in processed_data.columns, "The 'avg_monthly_charge' feature was not created."
    assert "is_high_value_customer" in processed_data.columns, "The 'is_high_value_customer' feature was not created."
    assert "has_multiple_services" in processed_data.columns, "The 'has_multiple_services' feature was not created."

def test_encode_categorical_variables(mock_raw_data):
    """
    Test the encode_categorical_variables method.
    """
    processed_data = FeatureEngineer.encode_categorical_variables(mock_raw_data)
    assert "gender" in processed_data.columns, "The 'gender' column is missing after encoding."
    assert "InternetService_Fiber optic" in processed_data.columns, "One-hot encoding for 'InternetService' failed."
    assert processed_data["gender"].isin([0, 1]).all(), "The 'gender' column was not encoded correctly."

def test_scale_numerical_features(mock_raw_data):
    """
    Test the scale_numerical_features method.
    """
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    processed_data = FeatureEngineer.generate_new_features(mock_raw_data)
    scaled_data = FeatureEngineer.scale_numerical_features(processed_data, numerical_cols)
    for col in numerical_cols:
        assert scaled_data[col].between(0, 1).all(), f"The column '{col}' was not scaled correctly."

def test_engineer_features(mock_raw_data):
    """
    Test the engineer_features method.
    """
    processed_data = FeatureEngineer.engineer_features(mock_raw_data)
    expected_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines_No", "MultipleLines_Yes", "InternetService_DSL", "InternetService_Fiber optic",
        "InternetService_No", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract_Month-to-month", "Contract_One year",
        "Contract_Two year", "PaperlessBilling", "PaymentMethod_Bank transfer (automatic)",
        "PaymentMethod_Electronic check", "PaymentMethod_Mailed check", "MonthlyCharges",
        "TotalCharges", "Churn", "tenure_years", "avg_monthly_charge", "is_high_value_customer",
        "has_multiple_services"
    ]
    for col in expected_columns:
        assert col in processed_data.columns, f"The column '{col}' is missing after feature engineering."