import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.model.evaluation import ModelEvaluator

@pytest.fixture
def mock_binary_classification_data():
    """
    Creates mock data for binary classification testing.
    """
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])  
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 0]) 
    y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3, 0.7, 0.6, 0.9, 0.1]) 
    return y_true, y_pred, y_proba

def test_evaluate_model_without_proba(mock_binary_classification_data):
    """
    Test the evaluate_model method without predicted probabilities (y_proba).
    """
    y_true, y_pred, _ = mock_binary_classification_data

    metrics = ModelEvaluator.evaluate_model(y_true, y_pred, dataset_name="Test")

    # Assertions for metrics
    assert metrics["Accuracy"] == accuracy_score(y_true, y_pred), "Accuracy does not match"
    assert metrics["Precision"] == precision_score(y_true, y_pred), "Precision does not match"
    assert metrics["Recall"] == recall_score(y_true, y_pred), "Recall does not match"
    assert metrics["F1-Score"] == f1_score(y_true, y_pred), "F1-Score does not match"
    assert "AUC" not in metrics, "AUC should not be calculated when y_proba is not provided"

def test_evaluate_model_with_proba(mock_binary_classification_data):
    """
    Test the evaluate_model method with predicted probabilities (y_proba).
    """
    y_true, y_pred, y_proba = mock_binary_classification_data

    metrics = ModelEvaluator.evaluate_model(y_true, y_pred, y_proba=y_proba, dataset_name="Test")

    # Assertions for metrics
    assert metrics["Accuracy"] == accuracy_score(y_true, y_pred), "Accuracy does not match"
    assert metrics["Precision"] == precision_score(y_true, y_pred), "Precision does not match"
    assert metrics["Recall"] == recall_score(y_true, y_pred), "Recall does not match"
    assert metrics["F1-Score"] == f1_score(y_true, y_pred), "F1-Score does not match"
    assert metrics["AUC"] == roc_auc_score(y_true, y_proba), "AUC does not match"

def test_evaluate_model_invalid_input():
    """
    Test the evaluate_model method with invalid input.
    """
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1]  

    with pytest.raises(ValueError, match="Found input variables with inconsistent numbers of samples"):
        ModelEvaluator.evaluate_model(y_true, y_pred)