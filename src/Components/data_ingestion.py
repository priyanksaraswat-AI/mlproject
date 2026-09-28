
# ============================================================
# DATA INGESTION COMPONENT
# ============================================================

# This message confirms that Python has started executing
# this file.
print("DATA INGESTION FILE STARTED")


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

# os is used for working with files, folders and file paths.
import os

# sys is used to access system-specific information.
# We use it when passing exception details to CustomException.
import sys

# dataclass is used to create configuration classes
# without writing a lot of boilerplate code.
from dataclasses import dataclass

# pandas is used for reading and manipulating datasets.
import pandas as pd

# train_test_split is used to divide the dataset into
# training data and testing data.
from sklearn.model_selection import train_test_split


# ============================================================
# 2. IMPORT CUSTOM PROJECT COMPONENTS
# ============================================================

# Import our custom exception class.
#
# This provides detailed information about where an
# error occurred.
from src.exception import CustomException

# Import our project logging utility.
#
# It records information about what the program is doing.
from src.logger import logging

# Import DataTransformation.
#
# After data ingestion is completed, the train and test
# datasets will be sent to the transformation stage.
from src.components.data_transformation import DataTransformation


# ============================================================
# 3. DATA INGESTION CONFIGURATION
# ============================================================

@dataclass
class DataIngestionConfig:

    # --------------------------------------------------------
    # Training data path
    # --------------------------------------------------------

    # Define the location where the training dataset
    # will be saved.
    #
    # Result:
    # artifacts/train.csv
    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )


    # --------------------------------------------------------
    # Testing data path
    # --------------------------------------------------------

    # Define the location where the testing dataset
    # will be saved.
    #
    # Result:
    # artifacts/test.csv
    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )


    # --------------------------------------------------------
    # Raw data path
    # --------------------------------------------------------

    # Define the location where the original dataset
    # will be saved.
    #
    # Result:
    # artifacts/raw.csv
    raw_data_path: str = os.path.join(
        "artifacts",
        "raw.csv"
    )


# ============================================================
# 4. DATA INGESTION CLASS
# ============================================================

class DataIngestion:


    # ========================================================
    # 4.1 CONSTRUCTOR
    # ========================================================

    def __init__(self):

        # Create an object of DataIngestionConfig.
        #
        # This allows us to access:
        #
        # self.ingestion_config.train_data_path
        # self.ingestion_config.test_data_path
        # self.ingestion_config.raw_data_path
        #
        # throughout this class.
        self.ingestion_config = DataIngestionConfig()


    # ========================================================
    # 4.2 DATA INGESTION METHOD
    # ========================================================

    def initiate_data_ingestion(self):

        # Write information into the log file.
        logging.info(
            "Entered the data ingestion method or component"
        )


        # ====================================================
        # ERROR HANDLING START
        # ====================================================

        # try contains the main data ingestion logic.
        #
        # If something goes wrong, the except block below
        # will handle the error.
        try:


            # =================================================
            # STEP 1: READ ORIGINAL DATASET
            # =================================================

            # Read the original student dataset.
            #
            # File:
            # notebook/data/stud.csv
            df = pd.read_csv(
                "notebook/data/stud.csv"
            )

            # Write information into the log file.
            logging.info(
                "Read the dataset as dataframe"
            )


            # =================================================
            # STEP 2: CREATE ARTIFACTS DIRECTORY
            # =================================================

            # Create the artifacts folder if it does not
            # already exist.
            #
            # os.path.dirname() extracts:
            #
            # artifacts
            #
            # from:
            #
            # artifacts/train.csv
            #
            # exist_ok=True means:
            #
            # If the folder already exists, don't throw an error.
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )


            # =================================================
            # STEP 3: SAVE RAW DATA
            # =================================================

            # Save the complete original dataset.
            #
            # File:
            # artifacts/raw.csv
            #
            # index=False prevents pandas from adding
            # an extra index column.
            #
            # header=True keeps the column names.
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            # Write confirmation into the log file.
            logging.info(
                "Raw data saved successfully"
            )


            # =================================================
            # STEP 4: TRAIN / TEST SPLIT
            # =================================================

            # Write information into the log file.
            logging.info(
                "Train Test split initiated"
            )


            # Split the original dataset into:
            #
            # 80% -> Training data
            # 20% -> Testing data
            #
            # random_state=42 ensures that the same
            # split is generated every time.
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )


            # =================================================
            # STEP 5: SAVE TRAINING DATA
            # =================================================

            # Save the training dataset.
            #
            # File:
            # artifacts/train.csv
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )


            # =================================================
            # STEP 6: SAVE TESTING DATA
            # =================================================

            # Save the testing dataset.
            #
            # File:
            # artifacts/test.csv
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )


            # =================================================
            # STEP 7: LOG COMPLETION
            # =================================================

            # Write a completion message to the log.
            logging.info(
                "Ingestion of data is completed"
            )


            # =================================================
            # STEP 8: RETURN TRAIN AND TEST PATHS
            # =================================================

            # Return the paths of the training and testing
            # datasets.
            #
            # These values will be received by:
            #
            # train_data, test_data
            #
            # in the main program.
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        # ====================================================
        # ERROR HANDLING
        # ====================================================

        # If any error occurs inside the try block,
        # execution comes here.
        except Exception as e:

            # Convert the original error into our
            # CustomException.
            #
            # e   = original error
            # sys = system/traceback information
            raise CustomException(
                e,
                sys
            )


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

# This condition checks whether this Python file is being
# executed directly.
#
# When we run:
#
# python -m src.components.data_ingestion
#
# Python sets __name__ to "__main__".
if __name__ == "__main__":


    # ========================================================
    # STEP 1: START DATA INGESTION
    # ========================================================

    print(
        "Data ingestion started"
    )


    # Create an object of the DataIngestion class.
    obj = DataIngestion()


    # Call the data ingestion method.
    #
    # It will:
    #
    # 1. Read stud.csv
    # 2. Save raw.csv
    # 3. Split the dataset
    # 4. Save train.csv
    # 5. Save test.csv
    #
    # The method returns:
    #
    # train_data = "artifacts/train.csv"
    # test_data  = "artifacts/test.csv"
    train_data, test_data = obj.initiate_data_ingestion()


    # Print completion message.
    print(
        "Data ingestion completed"
    )


    # ========================================================
    # STEP 2: START DATA TRANSFORMATION
    # ========================================================

    print(
        "Data transformation started"
    )


    # Create an object of DataTransformation.
    data_transformation = DataTransformation()


    # Run the data transformation process.
    #
    # train_data -> path of train.csv
    # test_data  -> path of test.csv
    #
    # The transformation process will:
    #
    # 1. Read train.csv and test.csv
    # 2. Separate features and target
    # 3. Handle missing values
    # 4. Encode categorical variables
    # 5. Scale features
    # 6. Save preprocessor.pkl
    data_transformation.initiate_data_transformation(
        train_data,
        test_data
    )


    # Print completion message.
    print(
        "Data transformation completed"
    )


    # ========================================================
    # STEP 3: COMPLETE PIPELINE
    # ========================================================

    print(
        "Complete pipeline executed successfully"
    )

