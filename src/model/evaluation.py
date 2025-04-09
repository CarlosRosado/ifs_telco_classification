import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelEvaluator:
    """
    Class evaluation models.

    Methods:
        evaluate_model(y_true, y_pred, y_proba=None, dataset_name="Test"):
            Evaluates the model and logs metrics like accuracy, precision, recall, F1-score, AUC, and confusion matrix.
    """

    @staticmethod
    def evaluate_model(y_true, y_pred, y_proba=None, dataset_name="Test"):
        """
        Evaluates the model and logs metrics

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            y_proba (array-like, optional): Predicted probabilities
            dataset_name (str): Name of the dataset

        Returns:
            dict: A dictionary containing all the evaluation metrics.
        """
        try:
            logging.info(f"--- {dataset_name} Dataset Evaluation ---")

            # Calculate metrics
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1-Score": f1_score(y_true, y_pred),
            }

            # Add AUC 
            if y_proba is not None:
                metrics["AUC"] = roc_auc_score(y_true, y_proba)

            # Log metrics
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}")

            # Log classification report
            logging.info("\nClassification Report:")
            logging.info("\n" + classification_report(y_true, y_pred))

            # Log confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"],
            )
            logging.info("\nConfusion Matrix:\n" + cm_df.to_string())

            return metrics
        except Exception as e:
            logging.error(f"An error occurred during model evaluation: {e}", exc_info=True)
            raise