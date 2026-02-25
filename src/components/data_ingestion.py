import pandas as pd
import numpy as np
from src.entity.config_entity import DataIngestionConfig
from src.exception.exception import CustomException
from src.logging.logging import logging
from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient
import os
from sklearn.model_selection import train_test_split

load_dotenv()

MONGO_DB_URI = os.getenv("MONGO_DB_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e)
    
    def get_data_from_db(self) -> pd.DataFrame:
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logging.info("Connecting to mongoDB")
            mongo_client = MongoClient(MONGO_DB_URI)
            collection = mongo_client[database_name][collection_name]
            logging.info("Connection Successfull")

            logging.info("Loading the data from mongodb")
            data = collection.find()
            df = pd.DataFrame(data=data)
            logging.info(f"{len(df)} records were Successfully loaded")

            if "_id" in df.columns:
                df = df.drop("_id", axis=1) # removing unnecessory mongodb "_id"

            # In mongodb missing values often get stored as string "na", not actual null values. so we convert into np.nan                
            df.replace({"na":np.nan}, inplace=True) 
            logging.info("Removed _id column and replaced 'na' into actuall null values")

            logging.info(f"Returned {len(df)} records")
            return df
        except Exception as e:
            logging.error("Error occured while importing data from MongoDB")
            raise CustomException(e)
        
    def save_data_to_feature_store(self, dataframe:pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_name = os.path.dirname(feature_store_file_path)

            logging.info("Creating the feature_Store directory.")
            os.makedirs(dir_name, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info("Successfully saved the CSV file.")
            return dataframe
        except Exception as e:
            logging.error("Error Occured while saving the raw data to feature store")
            raise CustomException(e)
        
    def apply_train_test_split(self, dataframe:pd.DataFrame):
        try:
            logging.info("Train Test Split started.")
            train, test = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info(f"Train Test Split Completed. Train shape: {train.shape}, Test: {test.shape}")

            logging.info("Creating artifact directory.")
            dir_name = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_name, exist_ok=True)

            train.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info("Train Test sets saved Successfully.")
        except Exception as e:
            logging.error("Error occured while train-test split")
            raise CustomException(e)
        
    def initialize_data_ingestion(self):
        try:
            logging.info("Data Ingestion Initialized.")
            df = self.get_data_from_db()
            df = self.save_data_to_feature_store(dataframe=df)
            self.apply_train_test_split(dataframe=df)
            logging.info("Data Ingestion Completed Successfully.")
        except Exception as e:
            logging.error("Data Ingestion Failed.")
            raise CustomException(e)
    
