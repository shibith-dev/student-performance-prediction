from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifacts, DataValidationArtifact
from src.exception.exception import CustomException
from src.logging.logging import logging

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import joblib
import os

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e)
        
    def read_df(self, file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e)
    
    def get_preprocessor(self):
        try:
            # Column groups :
            nominal_cols = ["Department", "Hometown", "Gender", "Job", "Extra"]
            ordinal_cols = ["Income", "Preparation", "Gaming", "Attendance"]
            numeric_cols = ["HSC", "SSC", "Computer", "English", "Semester", "Last"]

            # Ordinal columns category order :
            income_order = ["Low (Below 15,000)", "Lower middle (15,000-30,000)", "Upper middle (30,000-50,000)", "High (Above 50,000)"]
            preparation_order = ["0-1 Hour", "2-3 Hours", "More than 3 Hours"]
            gaming_order = ["0-1 Hour", "2-3 Hours", "More than 3 Hours"]
            attendance_order = ["Below 40%", "40%-59%", "60%-79%", "80%-100%"]

            # Pipelines : 
            nominal_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            ordinal_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(categories=[income_order, preparation_order, gaming_order, attendance_order], handle_unknown = "use_encoded_value", unknown_value = -1))
                ]
            )

            numeric_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            )

            # column transformer :
            preprocessor = ColumnTransformer(
                transformers=[
                    ("nominal", nominal_pipeline, nominal_cols),
                    ("ordinal", ordinal_pipeline, ordinal_cols),
                    ("numeric", numeric_pipeline, numeric_cols)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e)     

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Initializing Data Transformation.")
            train_df = self.read_df(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_df(self.data_validation_artifact.valid_test_file_path)
            
            target_column = self.data_transformation_config.target_column

            x_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]
            
            x_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]

            preprocessor = self.get_preprocessor()

            logging.info("Fitting preprocessor on training data.")

            x_train_transformed = preprocessor.fit_transform(x_train)
            x_test_transformed = preprocessor.transform(x_test)

            # combine transformed feature and target:
            train_arr = np.c_[x_train_transformed, np.array(y_train)]
            test_arr = np.c_[x_test_transformed, np.array(y_test)]

            
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            joblib.dump(preprocessor, self.data_transformation_config.transformed_object_file_path)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            np.save(arr = train_arr, file = self.data_transformation_config.transformed_train_file_path)
            np.save(arr = test_arr, file = self.data_transformation_config.transformed_test_file_path)

            logging.info("Data Transformation Completed")

            return DataTransformationArtifacts(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path 
            )
            
        except Exception as e:
            logging.error("Data Transformation Failed.")
            raise CustomException(e)


