from src.exception.exception import CustomException
from src.logging.logging import logging
import yaml
import os

def read_yaml(file_path: str)->dict:
    try:
        logging.info(f"Reading yaml file from {file_path}")
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        logging.info(f"yaml file successfully loaded")
    except Exception as e:
        logging.error("Error Occured while loading yaml file")
        raise CustomException(e)
    

    
def write_yaml(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e)
