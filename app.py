from src.logging.logging import logging

from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline

application = Flask(__name__)

app = application

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/train", methods=["GET", "POST"])
def train_route():
    if request.method == "GET":
        return render_template("train.html")
    else:
        try:
            logging.info("Training Pipeline Started.")
            train_pipeline_obj = TrainingPipeline()
            train_pipeline_obj.run_training_pipeline()
            return render_template("train.html", status="successful")
        except Exception as e:
            logging.error(f"Pipeline Failed : {str(e)}")
            return render_template("train.html", status="failed")
    

@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    try:
        if request.method == "GET":
            return render_template("predict.html")
        else:
            logging.info("Prediction pipeline started.")
            form_data = request.form.to_dict()
            data = pd.DataFrame([form_data])
            predict_pipeline_obj = PredictionPipeline()
            result = predict_pipeline_obj.predict(data=data)
            return render_template("predict.html", result = result[0])
    except Exception as e:
        logging.error(f"Prediction pipeline Failed : {str(e)}")
        return render_template("predict.html", result = "Prediction Failed")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)





