import pandas as pd
import numpy as np
import os
from src.logging.logging import logging
from src.exception.exception import CustomException
import pymongo
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URI = os.getenv("MONGO_DB_URI")

'''
    Our main goal is to load the local data, convert into json format and load into mongodb
'''

class StudentDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e)
    
    def csv_to_json(self, file_path):
        try:
            logging.info(f"Reading csv from the file path: {file_path}")

            df = pd.read_csv(file_path)
            df.reset_index(drop=True, inplace=True) # remove extra index

            logging.info(f"Loaded successfully, shape:{df.shape}")

            records = df.to_dict(orient="records")
            return records
        except Exception as e:
            logging.error("Error occured while converting CSV into JSON")
            raise CustomException(e)
    
    def insert_data_to_mongodb(self, records, database, collection):
        try:
            logging.info("Connecting to DB")

            mongo_client = pymongo.MongoClient(MONGO_DB_URI)
            
            db = mongo_client[database]
            collection = db[collection]

            logging.info(f"Inserting {len(records)} records into collection: {collection.name}")
            collection.insert_many(records)
            logging.info("Data Insertion Successful")
            return len(records)
        except Exception as e:
            logging.error("Error Occured while inserting the data into MongoDB")
            raise CustomException(e)


if __name__ == "__main__":
    try:
        FILE_PATH = "E:/Career/Professional_ML/02_Machine_Learning/01_Projects/student_performance_prediction/data/ResearchInformation3.csv"
        DATABASE = "ShibithAI"
        COLLECTION = "StudentData"

        studentObj = StudentDataExtract()
        records = studentObj.csv_to_json(file_path=FILE_PATH)
        no_of_records = studentObj.insert_data_to_mongodb(records=records, database=DATABASE, collection=COLLECTION)
        
        logging.info(f"Pipeline Completed Successfully. Total inserted: {no_of_records}")
    except Exception as e:
        logging.error(f"Pipeline Failed")
        raise CustomException(e)