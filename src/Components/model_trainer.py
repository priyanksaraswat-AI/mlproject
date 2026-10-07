
# ---------------------------------------------------------
# Import required Python libraries
# ---------------------------------------------------------
print("MODEL TRAINER STARTED")
import os
import sys

# dataclass is used to create configuration classes
from dataclasses import dataclass

# Machine Learning regression models
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor


from xgboost import XGBRegressor

# Custom exception and logging from our project
from src.exception import CustomException
from src.logger import logging

# Utility functions from our project
from src.utils import save_object, evaluate_models


# ---------------------------------------------------------
# Model Trainer Configuration
# ---------------------------------------------------------

@dataclass
class ModelTrainerConfig:
    """
    Stores the path where the best trained model
    will be saved.
    """

    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


# ---------------------------------------------------------
# Model Trainer Class
# ---------------------------------------------------------
print("MODEL TRAINER CLASS LOADING")
class ModelTrainer:

    def __init__(self):
        """
        Initialize ModelTrainerConfig.
        """

        self.model_trainer_config = ModelTrainerConfig()


    # -----------------------------------------------------
    # Model Training Method
    # -----------------------------------------------------

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        preprocessor_path
    ):
        """
        Train multiple regression models,
        evaluate them, select the best model,
        save the best model and return its R2 score.
        """

        try:

            # -------------------------------------------------
            # STEP 1: Separate features and target
            # -------------------------------------------------

            logging.info(
                "Splitting training and testing input data"
            )

            # All columns except the last column = features
            x_train = train_array[:, :-1]

            # Last column = target
            y_train = train_array[:, -1]

            # All columns except the last column = features
            x_test = test_array[:, :-1]

            # Last column = target
            y_test = test_array[:, -1]


            # -------------------------------------------------
            # STEP 2: Define ML models
            # -------------------------------------------------

            models = {

                "Random Forest":
                    RandomForestRegressor(),

                "Decision Tree":
                    DecisionTreeRegressor(),

                "Gradient Boosting":
                    GradientBoostingRegressor(),

                "Linear Regression":
                    LinearRegression(),

                "K-Neighbors":
                    KNeighborsRegressor(),

                "XGBoost":
                    XGBRegressor(),

                "CatBoost":
                    CatBoostRegressor(
                        verbose=False
                    ),

                "AdaBoost":
                    AdaBoostRegressor()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                  "K-Neighbors": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },


                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0]
                },

                "CatBoost": {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "iterations": [100, 200]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            # -------------------------------------------------
            # STEP 3: Train and evaluate all models
            # -------------------------------------------------

            logging.info(
                "Training and evaluating all models"
            )

            model_report = evaluate_models(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params
            )


            # Print the scores in PowerShell
            print("\nMODEL REPORT:")
            for model_name, score in model_report.items():
                print(
                    f"{model_name}: "
                    f"{score:.4f}"
                )


            # -------------------------------------------------
            # STEP 4: Find best model score
            # -------------------------------------------------

            best_model_score = max(
                model_report.values()
            )


            # -------------------------------------------------
            # STEP 5: Find best model name
            # -------------------------------------------------

            best_model_name = max(
                model_report,
                key=model_report.get
            )


            # -------------------------------------------------
            # STEP 6: Get best model object
            # -------------------------------------------------

            best_model = models[
                best_model_name
            ]


            # -------------------------------------------------
            # STEP 7: Validate best model
            # -------------------------------------------------

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found with acceptable "
                    "R2 score"
                )


            logging.info(
                f"Best model found: {best_model_name}"
            )

            logging.info(
                f"Best model R2 score: "
                f"{best_model_score}"
            )


            print(
                f"\nBest Model: {best_model_name}"
            )

            print(
                f"Best Model R2 Score: "
                f"{best_model_score:.4f}"
            )


            # -------------------------------------------------
            # STEP 8: Save best model
            # -------------------------------------------------

            save_object(
                file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),
                obj=best_model
            )


            logging.info(
                "Best model saved successfully"
            )


            print(
                "\nBest model saved to:"
            )

            print(
                self.model_trainer_config
                .trained_model_file_path
            )


            # -------------------------------------------------
            # STEP 9: Final prediction using best model
            # -------------------------------------------------

            predicted = best_model.predict(
                x_test
            )


            # -------------------------------------------------
            # STEP 10: Calculate final R2 score
            # -------------------------------------------------

            test_r2_score = r2_score(
                y_test,
                predicted
            )


            logging.info(
                f"Final test R2 score: "
                f"{test_r2_score}"
            )


            # Return final score
            return test_r2_score


        except Exception as e:

            raise CustomException(
                e,
                sys
            )

