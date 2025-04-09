import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataCleaner:
    """
    A class for cleaning of datasets, missing values,
    removing outliers and dropping duplicate rows.

    Methods:
        handle_missing_values(df):
            Handles missing values in the dataset.
        remove_outliers(df, columns):
            Removes outliers from specified numerical columns using the IQR method.
        drop_duplicates(df):
            Removes duplicate rows from the dataset.
        clean_data(df):
            Cleans the dataset by handling missing values, removing outliers, and dropping duplicates.
    """

    @staticmethod
    def handle_missing_values(df):
        """
        Missing values in the dataset.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with missing values handled.
        """
        try:
            logging.info("Handling missing values...")
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)
            df = df[df['tenure'] > 0]
            logging.info("Missing values handled successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while handling missing values: {e}", exc_info=True)
            raise

    @staticmethod
    def remove_outliers(df, columns):
        """
        Removes outliers from the specified numerical columns using the IQR method.

        Args:
            df (pd.DataFrame): The input dataset.
            columns (list): List of numerical columns to check for outliers.

        Returns:
            pd.DataFrame: The dataset with outliers removed.
        """
        try:
            logging.info(f"Removing outliers for columns: {columns}...")
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            logging.info("Outliers removed successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while removing outliers: {e}", exc_info=True)
            raise

    @staticmethod
    def drop_duplicates(df):
        """
        Removes duplicate rows from the dataset.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The dataset with duplicate rows removed.
        """
        try:
            logging.info("Dropping duplicate rows...")
            df = df.drop_duplicates()
            logging.info("Duplicate rows dropped successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred while dropping duplicates: {e}", exc_info=True)
            raise

    @staticmethod
    def clean_data(df):
        """
        Cleans the dataset by:
        - Handling missing values
        - Removing outliers
        - Dropping duplicate rows

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        try:
            logging.info("Starting data cleaning process...")
            df = DataCleaner.handle_missing_values(df)
            df = DataCleaner.remove_outliers(df, ['tenure', 'MonthlyCharges', 'TotalCharges'])
            df = DataCleaner.drop_duplicates(df)
            logging.info("Data cleaning process completed successfully.")
            return df
        except Exception as e:
            logging.error(f"An error occurred during the data cleaning process: {e}", exc_info=True)
            raise