import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureEngineer:
    """
    A class to handle feature engineering tasks, including preprocessing the target variable,
    generating new features, encoding categorical variables, and scaling numerical features.

    """

    @staticmethod
    def preprocess_target(df):
        """
        Converts the target variable 'Churn' into binary values (1 for 'Yes', 0 for 'No').

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with the target variable preprocessed.
        """
        try:
            logging.info("Preprocessing target variable 'Churn'...")
            if 'Churn' not in df.columns:
                raise ValueError("The target column 'Churn' is missing from the dataset.")
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            if df['Churn'].isnull().any():
                raise ValueError("The target column 'Churn' contains invalid values.")
            logging.info("Target variable 'Churn' preprocessed successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while preprocessing the target variable: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_new_features(df):
        """
        Generates new features

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with new features added.
        """
        try:
            logging.info("Generating new features...")
            if 'tenure' not in df.columns or 'MonthlyCharges' not in df.columns or 'TotalCharges' not in df.columns:
                raise ValueError("Required columns for feature generation are missing.")
            
            # Convert TotalCharges to numeric
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Fill NaN values in TotalCharges with 0
            df['TotalCharges'] = df['TotalCharges'].fillna(0)
            
            df['tenure_years'] = df['tenure'] / 12
            df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)  # Avoid division by zero
            df['is_high_value_customer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
            df['has_multiple_services'] = (
                (df['PhoneService'] == 'Yes') & 
                (df['InternetService'] != 'No') & 
                (df['StreamingTV'] == 'Yes') & 
                (df['StreamingMovies'] == 'Yes')
            ).astype(int)
            logging.info("New features generated successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while generating new features: {e}", exc_info=True)
            raise

    @staticmethod
    def encode_categorical_variables(df):
        """
        Encodes categorical variables:
        - Converts binary variables (e.g., Yes/No, True/False) into 1/0.
        - special cases like 'No internet service' and 'No phone service'.
        - Uses one-hot encoding for multi-class categorical variables.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with categorical variables encoded.
        """
        try:
            logging.info("Encoding categorical variables...")
            # Convert binary variables to 1/0
            binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            for col in binary_cols:
                if col in df.columns:
                    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, True: 1, False: 0})
            
            # special cases like 'No internet service' and 'No phone service'
            special_case_cols = ['StreamingTV', 'StreamingMovies', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
            for col in special_case_cols:
                if col in df.columns:
                    df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
            
            # Create separate categories for 'MultipleLines'
            if 'MultipleLines' in df.columns:
                df = pd.get_dummies(df, columns=['MultipleLines'], drop_first=False)
            
            # One-hot encode multi-class categorical variables
            multi_class_cols = ['InternetService', 'Contract', 'PaymentMethod']
            df = pd.get_dummies(df, columns=multi_class_cols, drop_first=False)
            
            for col in df.columns:
                if df[col].dtype == 'bool':  
                    df[col] = df[col].astype(int)
            
            logging.info("Categorical variables encoded successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while encoding categorical variables: {e}", exc_info=True)
            raise

    @staticmethod
    def scale_numerical_features(df, columns):
        """
        Scales numerical features using MinMaxScaler.

        Args:
            df (pd.DataFrame): The input dataset.
            columns (list): List of numerical columns to scale.

        Returns:
            pd.DataFrame: The dataset with scaled numerical features.
        """
        try:
            logging.info(f"Scaling numerical features: {columns}...")
            scaler = MinMaxScaler()
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"The following numerical columns are missing from the dataset: {missing_cols}")
            
            df[columns] = scaler.fit_transform(df[columns])
            logging.info("Numerical features scaled successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while scaling numerical features: {e}", exc_info=True)
            raise

    @staticmethod
    def engineer_features(df):
        """
        Performs feature engineering by:
        - Dropping unnecessary columns (e.g., customerID)
        - Preprocessing the target variable
        - Generating new features
        - Encoding categorical variables
        - Scaling numerical features

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with engineered features.
        """
        try:
            logging.info("Starting feature engineering process...")
            
            # Drop unnecessary columns
            if 'customerID' in df.columns:
                df = df.drop(columns=['customerID'])
            
            # Preprocess the target variable
            df = FeatureEngineer.preprocess_target(df)
            
            # Generate new features
            df = FeatureEngineer.generate_new_features(df)
            
            # Encode categorical variables
            df = FeatureEngineer.encode_categorical_variables(df)
            
            # Scale numerical features
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'tenure_years', 'avg_monthly_charge']
            df = FeatureEngineer.scale_numerical_features(df, numerical_cols)
            
            logging.info("Feature engineering process completed successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred during the feature engineering process: {e}", exc_info=True)
            raise