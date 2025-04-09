import os
import sys
import logging
import pickle
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from src.data.load_data import DataLoader
from src.preprocessing.clean_data import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer
from src.model.evaluation import ModelEvaluator

# Dynamically add the project root directory to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()

# Load configuration from .env
DATA_URL = os.getenv("DATA_URL")
LOCAL_DATA_PATH = os.getenv("LOCAL_DATA_PATH")
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


class ModelTrainer:
    """
    A class to train the model in mlflow

    Attributes:
        data_path (str): Path to the dataset file.
        model_output_path (str): Path to save the trained model.
        mlflow_tracking_uri (str): URI for the MLflow tracking server.
    """

    def __init__(self, data_path, model_output_path, mlflow_tracking_uri):
        """
        Initialize the ModelTrainer class with configuration.

        Args:
            data_path (str): Path to the dataset file.
            model_output_path (str): Path to save the trained model.
            mlflow_tracking_uri (str): URI for the MLflow tracking server.
        """
        self.data_path = data_path
        self.model_output_path = model_output_path
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset.

        Returns:
            tuple: X and y
        """
        try:
            logging.info("Loading data...")
            df = DataLoader.load_data(self.data_path)
            logging.info("Cleaning data...")
            df = DataCleaner.clean_data(df)
            logging.info("Engineering features...")
            df = FeatureEngineer.engineer_features(df)

            # Split data into features and target
            logging.info("Splitting data into features and target...")
            X = df.drop('Churn', axis=1) 
            y = df['Churn']
            return X, y
        except Exception as e:
            logging.error(f"Error during data loading and preprocessing: {e}", exc_info=True)
            raise

    def train_model(self):
        """
        Train a Random Forest model with hyperparameter tuning, log metrics to MLflow, and save the model locally.
        """
        try:
            logging.info("Starting the training process.")

            # Set up MLflow tracking
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("IFS - Telco Churn Prediction")

            # Start an MLflow run
            with mlflow.start_run(run_name="IFS - Random Forest Training"):
                # Load and preprocess data
                X, y = self.load_and_preprocess_data()

                # Split data into training and test sets
                logging.info("Splitting data into training and test sets...")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Class imbalance using SMOTE
                logging.info("Applying SMOTE to handle class imbalance...")
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                # grid for RandomizedSearchCV
                logging.info("Defining hyperparameter grid...")
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [5, 10],
                    'class_weight': ['balanced'],
                    'max_features': ['sqrt']
                }

                # Initialize the Random Forest model
                logging.info("Initializing Random Forest model...")
                rf = RandomForestClassifier(random_state=42)

                # Use RandomizedSearchCV for hyperparameter tuning
                logging.info("Starting RandomizedSearchCV for hyperparameter tuning...")
                random_search = RandomizedSearchCV(
                    estimator=rf,
                    param_distributions=param_grid,
                    n_iter=10,
                    cv=3,
                    scoring='f1',
                    verbose=1,
                    n_jobs=-1,
                    random_state=42
                )
                random_search.fit(X_train_resampled, y_train_resampled)
                logging.info("RandomizedSearchCV completed.")

                # Get the best model from RandomizedSearchCV
                best_model = random_search.best_estimator_
                logging.info(f"Best Hyperparameters: {random_search.best_params_}")

                # Log parameters to MLflow
                mlflow.log_params(random_search.best_params_)

                # Evaluate the model on the training set
                logging.info("Evaluating the model on the training set...")
                y_train_pred = best_model.predict(X_train)
                y_train_proba = best_model.predict_proba(X_train)[:, 1]
                train_metrics = ModelEvaluator.evaluate_model(y_train, y_train_pred, y_train_proba, dataset_name="Train")
                mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

                # Evaluate the model on the test set
                logging.info("Evaluating the model on the test set...")
                y_test_pred = best_model.predict(X_test)
                y_test_proba = best_model.predict_proba(X_test)[:, 1]
                test_metrics = ModelEvaluator.evaluate_model(y_test, y_test_pred, y_test_proba, dataset_name="Test")
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                # Precision-Recall Curve and Threshold Tuning
                logging.info("Calculating Precision-Recall Curve and tuning threshold...")
                precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
                best_threshold = thresholds[(precision[:-1] + recall[:-1]).argmax()]
                logging.info(f"Best Threshold: {best_threshold}")
                mlflow.log_metric("best_threshold", best_threshold)

                # Log model to MLflow
                logging.info("Logging the model to MLflow...")
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="ifs_random_forest_model",
                    registered_model_name="ifs_Random_Forest",
                    input_example=X_test[:5]
                )

                # Save the model locally
                logging.info("Saving the model locally...")
                model_dir = os.path.dirname(self.model_output_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                with open(self.model_output_path, 'wb') as f:
                    pickle.dump(best_model, f)

                logging.info("Training process completed successfully.")

        except Exception as e:
            logging.error(f"An error occurred during training: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    # Load configuration from .env
    DATA_URL = os.getenv("DATA_URL")
    LOCAL_DATA_PATH = os.getenv("LOCAL_DATA_PATH")
    MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    trainer = ModelTrainer(
        data_path=LOCAL_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI
    )
    trainer.train_model()