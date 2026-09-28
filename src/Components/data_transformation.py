
# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

# sys is used to access Python system-related information.
# We use it here when raising our CustomException.
import sys

# os provides functions for working with files and directories.
# We use it to create the path for preprocessor.pkl.
import os

# dataclass helps us create configuration classes
# without writing a lot of boilerplate code.
from dataclasses import dataclass


# NumPy is used for numerical operations.
# Here, we use np.c_ to combine processed features with the target column.
import numpy as np

# Pandas is used to read and manipulate CSV/dataframe data.
import pandas as pd


# ColumnTransformer allows us to apply different preprocessing
# pipelines to different groups of columns.
from sklearn.compose import ColumnTransformer

# SimpleImputer is used to handle missing/null values.
from sklearn.impute import SimpleImputer

# Pipeline allows us to execute multiple preprocessing steps
# sequentially.
from sklearn.pipeline import Pipeline

# OneHotEncoder converts categorical values into numerical values.
# StandardScaler standardizes numerical values.
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Import our custom exception class.
# It gives us detailed error information when something fails.
from src.exception import CustomException

# Import our custom logging configuration.
# This allows us to write information/errors into log files.
from src.logger import logging

# Import the function used to save the preprocessing object
# as a pickle file.
from src.utils import save_object


# ============================================================
# 2. DATA TRANSFORMATION CONFIGURATION
# ============================================================

@dataclass
class DataTransformationConfig:

    # Define the location where the trained preprocessing object
    # will be stored.
    #
    # os.path.join() creates:
    # artifacts/preprocessor.pkl
    #
    # This file will contain the fitted preprocessing pipeline.
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


# ============================================================
# 3. DATA TRANSFORMATION CLASS
# ============================================================

class DataTransformation:

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

    def __init__(self):

        # Create an object of DataTransformationConfig.
        #
        # This gives us access to:
        # self.data_transformation_config.preprocessor_obj_file_path
        #
        # which contains:
        # artifacts/preprocessor.pkl
        self.data_transformation_config = DataTransformationConfig()


    # ========================================================
    # 4. CREATE PREPROCESSING PIPELINE
    # ========================================================

    def get_data_transformer_object(self):

        """
        Create the preprocessing pipeline.

        Numerical columns:
            - Missing values are replaced with median.
            - Values are standardized using StandardScaler.

        Categorical columns:
            - Missing values are replaced with most frequent value.
            - Categories are converted into numerical values
              using OneHotEncoder.
            - Values are standardized using StandardScaler.

        Finally, both pipelines are combined using ColumnTransformer.
        """

        # Start try block so that any error can be captured
        # and converted into our CustomException.
        try:

            # ------------------------------------------------
            # Define numerical columns
            # ------------------------------------------------

            # These columns contain numerical values.
            #
            # Example:
            # reading_score = 72
            # writing_score = 74
            numerical_columns = [
                "reading_score",
                "writing_score"
            ]


            # ------------------------------------------------
            # Define categorical columns
            # ------------------------------------------------

            # These columns contain categorical/text values.
            #
            # Example:
            # gender = "male"
            # lunch = "standard"
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]


            # =================================================
            # 5. NUMERICAL PIPELINE
            # =================================================

            # Pipeline allows us to execute multiple
            # preprocessing steps one after another.
            #
            # For numerical data we will perform:
            #
            # Step 1 -> Handle missing values
            # Step 2 -> Standardize the values

            num_pipeline = Pipeline(

                # steps contains the list of preprocessing operations.
                steps=[

                    # -----------------------------------------
                    # Step 1: Handle missing numerical values
                    # -----------------------------------------

                    (
                        # Name of this step.
                        "imputer",

                        # SimpleImputer replaces missing values.
                        SimpleImputer(

                            # Replace missing numerical values
                            # with the median of that column.
                            strategy="median"
                        )
                    ),

                    # -----------------------------------------
                    # Step 2: Standardize numerical values
                    # -----------------------------------------

                    (
                        # Name of this step.
                        "scaler",

                        # StandardScaler converts values approximately
                        # to mean = 0 and standard deviation = 1.
                        StandardScaler()
                    ),
                ]
            )


            # =================================================
            # 6. CATEGORICAL PIPELINE
            # =================================================

            # For categorical columns we perform:
            #
            # Step 1 -> Handle missing values
            # Step 2 -> Convert categories to numerical values
            # Step 3 -> Standardize the encoded values

            cat_pipeline = Pipeline(

                # List of categorical preprocessing steps.
                steps=[

                    # -----------------------------------------
                    # Step 1: Handle missing categorical values
                    # -----------------------------------------

                    (
                        # Name of the step.
                        "imputer",

                        # Replace missing categorical values
                        # with the most frequently occurring value.
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),


                    # -----------------------------------------
                    # Step 2: One Hot Encoding
                    # -----------------------------------------

                    (
                        # Name of the step.
                        "one_hot_encoder",

                        # Convert categorical values into
                        # numerical/binary columns.
                        OneHotEncoder(

                            # If a new category appears in test/new data
                            # that was not present during training,
                            # do not throw an error.
                            #
                            # Instead, encode it appropriately.
                            handle_unknown="ignore",

                            # Return a normal dense NumPy array
                            # instead of a sparse matrix.
                            #
                            # Note:
                            # sparse_output=False requires
                            # scikit-learn 1.2 or newer.
                            sparse_output=False
                        )
                    ),


                    # -----------------------------------------
                    # Step 3: Standardize encoded values
                    # -----------------------------------------

                    (
                        # Name of the step.
                        "scaler",

                        # Standardize the encoded numerical values.
                        StandardScaler()
                    ),
                ]
            )


            # =================================================
            # 7. LOG INFORMATION
            # =================================================

            # Write the categorical column names into the log file.
            logging.info(
                f"Categorical columns: {categorical_columns}"
            )

            # Write the numerical column names into the log file.
            logging.info(
                f"Numerical columns: {numerical_columns}"
            )


            # =================================================
            # 8. COMBINE BOTH PIPELINES
            # =================================================

            # ColumnTransformer allows us to apply:
            #
            # num_pipeline -> numerical columns
            #
            # cat_pipeline -> categorical columns
            #
            # This creates one complete preprocessing object.

            preprocessor = ColumnTransformer(

                # transformers contains the pipelines
                # and the columns on which they should operate.
                transformers=[

                    # -----------------------------------------
                    # Numerical columns
                    # -----------------------------------------

                    (
                        # Name given to this transformer.
                        "num_pipeline",

                        # Pipeline that handles numerical columns.
                        num_pipeline,

                        # Columns on which num_pipeline will operate.
                        numerical_columns
                    ),


                    # -----------------------------------------
                    # Categorical columns
                    # -----------------------------------------

                    (
                        # Name given to this transformer.
                        "cat_pipeline",

                        # Pipeline that handles categorical columns.
                        cat_pipeline,

                        # Columns on which cat_pipeline will operate.
                        categorical_columns
                    ),
                ]
            )


            # Return the complete preprocessing object.
            #
            # This object has not been fitted yet.
            return preprocessor


        # If any error occurs inside this method...
        except Exception as e:

            # Raise our custom exception with detailed
            # information about the error.
            raise CustomException(e, sys)


    # ========================================================
    # 9. INITIATE DATA TRANSFORMATION
    # ========================================================

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        # Start try block so that transformation errors
        # can be captured by CustomException.
        try:

            # ------------------------------------------------
            # Read training and testing datasets
            # ------------------------------------------------

            # Read train.csv into a Pandas DataFrame.
            train_df = pd.read_csv(train_path)

            # Read test.csv into a Pandas DataFrame.
            test_df = pd.read_csv(test_path)


            # Write information into the log file.
            logging.info(
                "Reading of train and test data completed"
            )


            # ------------------------------------------------
            # Create preprocessing object
            # ------------------------------------------------

            # Log that we are creating the preprocessing object.
            logging.info(
                "Obtaining preprocessing object"
            )

            # Call our previously created method.
            #
            # This returns the ColumnTransformer containing:
            # - numerical pipeline
            # - categorical pipeline
            preprocessing_obj = self.get_data_transformer_object()


            # =================================================
            # 10. DEFINE TARGET COLUMN
            # =================================================

            # This is the column that we want to predict.
            #
            # In this project:
            #
            # Input features:
            # gender
            # race_ethnicity
            # parental_level_of_education
            # lunch
            # test_preparation_course
            # reading_score
            # writing_score
            #
            # Target:
            # math_score

            target_column_name = "math_score"


            # =================================================
            # 11. SEPARATE FEATURES AND TARGET
            # =================================================

            # Remove math_score from training data.
            #
            # What remains becomes the input/features (X).
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )


            # Extract math_score from training data.
            #
            # This becomes the target/output (y).
            target_feature_train_df = train_df[
                target_column_name
            ]


            # Remove math_score from testing data.
            #
            # What remains becomes test input/features (X_test).
            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )


            # Extract math_score from testing data.
            #
            # This becomes the test target/output (y_test).
            target_feature_test_df = test_df[
                target_column_name
            ]


            # =================================================
            # 12. APPLY PREPROCESSING
            # =================================================

            # Write information into the log file.
            logging.info(
                "Applying preprocessing object on training "
                "and testing dataframe"
            )


            # ------------------------------------------------
            # FIT + TRANSFORM TRAINING DATA
            # ------------------------------------------------

            # fit_transform() performs TWO operations:
            #
            # 1. FIT
            #    Learn preprocessing information from training data.
            #
            #    Example:
            #    - Median values
            #    - Mean
            #    - Standard deviation
            #    - Categories for OneHotEncoder
            #
            # 2. TRANSFORM
            #    Apply the learned preprocessing to training data.
            #
            # IMPORTANT:
            # We fit the preprocessing object ONLY on training data.

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )


            # ------------------------------------------------
            # TRANSFORM TEST DATA
            # ------------------------------------------------

            # transform() applies the preprocessing information
            # learned from the training data.
            #
            # We DO NOT use fit_transform() on test data.
            #
            # This prevents data leakage from test data.

            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )


            # =================================================
            # 13. ADD TARGET COLUMN BACK
            # =================================================

            # Combine:
            #
            # Processed training features
            # +
            # Original math_score target
            #
            # np.c_ combines arrays column-wise.

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]


            # Do the same thing for test data.
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]


            # =================================================
            # 14. SAVE PREPROCESSING OBJECT
            # =================================================

            # Write information into the log file.
            logging.info(
                "Saving preprocessing object"
            )


            # Save the fitted preprocessing object as a pickle file.
            #
            # file_path:
            # artifacts/preprocessor.pkl
            #
            # obj:
            # preprocessing_obj
            #
            # The save_object() function from utils.py
            # handles the actual file creation.

            save_object(

                # Location where the pickle file will be saved.
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),

                # Object that we want to save.
                obj=preprocessing_obj
            )


            # Write confirmation into the log file.
            logging.info(
                "Preprocessing object saved successfully"
            )


            # =================================================
            # 15. RETURN TRANSFORMED DATA
            # =================================================

            # Return three things:
            #
            # 1. Transformed training data
            # 2. Transformed testing data
            # 3. Location of saved preprocessing object

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        # If any error occurs during transformation...
        except Exception as e:

            # Convert the original error into our
            # CustomException with detailed information.
            raise CustomException(e, sys)
