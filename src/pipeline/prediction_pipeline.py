from src import constants
import joblib
import os

class PredictionPipeline:
    def __init__(self):
        model_file_path = os.path.join(constants.FINAL_MODEL_DIR,constants.MODEL_FILE_NAME)
        self.model = joblib.load(model_file_path)

    def predict(self, data):
        y_pred = self.model.predict(data)
        return y_pred
    