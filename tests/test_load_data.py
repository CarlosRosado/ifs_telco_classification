import os
import pytest
import pandas as pd
from src.data.load_data import DataLoader

@pytest.fixture
def mock_valid_csv(tmp_path):
    """
    Creates a temporary valid CSV file for testing.
    """
    data = {
        "tenure": [1, 2, 3],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": [29.85, 188.95, 108.15],
        "Churn": ["No", "Yes", "No"]
    }
    file_path = tmp_path / "valid_data.csv"
    pd.DataFrame(data).to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def mock_empty_csv(tmp_path):
    """
    Creates a temporary empty CSV file for testing.
    """
    file_path = tmp_path / "empty_data.csv"
    file_path.write_text("")
    return file_path

@pytest.fixture
def mock_invalid_csv(tmp_path):
    """
    Creates a temporary invalid CSV file for testing.
    """
    file_path = tmp_path / "invalid_data.csv"
    file_path.write_text("This is not a valid CSV file.")
    return file_path

def test_load_data_success(mock_valid_csv):
    """
    Test that the load_data method successfully loads a valid CSV file.
    """
    df = DataLoader.load_data(mock_valid_csv)
    assert isinstance(df, pd.DataFrame), "The returned object is not a DataFrame"
    assert not df.empty, "The DataFrame is empty"
    assert list(df.columns) == ["tenure", "MonthlyCharges", "TotalCharges", "Churn"], "Column names do not match"

def test_load_data_file_not_found():
    """
    Test that the load_data method raises a FileNotFoundError for a missing file.
    """
    with pytest.raises(FileNotFoundError, match="Dataset not found at path"):
        DataLoader.load_data("non_existent_file.csv")

