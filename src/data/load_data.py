import pandas as pd
import os
import logging


class DataLoader:
    """
    A class to loading datasets from a specified path.

    Methods:
        load_data(data_path):
            Loads the dataset from the specified path.
            Includes error handling for file not found, empty file, and invalid file format.
    """

    @staticmethod
    def load_data(data_path):
        """
        Loads the dataset from the specified path.

        Args:
            data_path (str): The path to the dataset file.

        Returns:
            pd.DataFrame: The loaded dataset as a pandas DataFrame.
        """
        try:
            # Check if the file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at path: {data_path}")

            # Load the dataset
            df = pd.read_csv(data_path)

            # Check if the dataset is empty
            if df.empty:
                raise ValueError("The dataset is empty.")

            return df

        except FileNotFoundError as fnf_error:
            logging.error(f"FileNotFoundError: {fnf_error}")
            raise

        except pd.errors.EmptyDataError:
            logging.error("Error: The file is empty or contains no data.")
            raise

        except pd.errors.ParserError:
            logging.error("Error: The file format is invalid.")
            raise

        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise