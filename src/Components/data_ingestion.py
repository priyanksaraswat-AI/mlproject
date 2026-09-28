
import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "raw.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info(
            "Entered the data ingestion method or component"
        )

        try:

            # Read dataset
            df = pd.read_csv(
                "notebook/data/stud.csv"
            )

            logging.info(
                "Read the dataset as dataframe"
            )

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # Save raw data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Raw data saved successfully"
            )

            # Train-test split
            logging.info(
                "Train Test split initiated"
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save training data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save testing data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Ingestion of data is completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            raise CustomException(e, sys)


if __name__ == "__main__":

    print("Data ingestion started")

    # Create DataIngestion object
    obj = DataIngestion()

    # Run data ingestion
    train_data, test_data = obj.initiate_data_ingestion()

    print("Data ingestion completed")

    # Start data transformation
    print("Data transformation started")

    data_transformation = DataTransformation()

    data_transformation.initiate_data_transformation(
        train_data,
        test_data
    )

    print("Data transformation completed")

    print("Complete pipeline executed successfully")
