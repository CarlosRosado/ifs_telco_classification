import pandas as pd
import pytest
from src.preprocessing.clean_data import DataCleaner

@pytest.fixture
def mock_raw_data():
    """
    Creates a mock raw dataset for testing.
    """
    data = {
        "tenure": [1, 2, None],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": ["29.85", "188.95", None],
        "Churn": ["No", "Yes", "No"],
        "UnnecessaryColumn": ["A", "B", "C"]
    }
    return pd.DataFrame(data)

def test_clean_data(mock_raw_data):
    """
    Test that the clean_data method successfully cleans the dataset.
    """

    cleaned_data = DataCleaner.clean_data(mock_raw_data)

    # Assertions
    assert isinstance(cleaned_data, pd.DataFrame), "The returned object is not a DataFrame"
    
    # Check if TotalCharges is converted to numeric
    if "TotalCharges" in cleaned_data.columns:
        assert pd.api.types.is_numeric_dtype(cleaned_data["TotalCharges"]), "TotalCharges is not numeric"

    # Check that the cleaned DataFrame contains the expected columns
    expected_columns = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    assert all(col in cleaned_data.columns for col in expected_columns), "The cleaned DataFrame is missing expected columns"

    # Check that the DataFrame has the correct number of rows
    expected_rows = len(mock_raw_data.dropna(subset=["tenure", "TotalCharges"]))
    assert len(cleaned_data) == expected_rows, (
        f"The number of rows in the cleaned DataFrame is incorrect. Expected {expected_rows}, got {len(cleaned_data)}"
    )