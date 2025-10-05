from typing import List, Dict

from fastapi import FastAPI
from ray import serve
from pydantic import BaseModel

app = FastAPI()


@serve.deployment()
@serve.ingress(app)
class CarbonMonoxidePredictor:
    class CMPredictionRequest(BaseModel):
        data: List[List[float]]

    DEFAULT_VERSION = 1

    def __init__(self):
        self.version = self.DEFAULT_VERSION
        self.model = None

    def reconfigure(self, config: Dict):
        self.version = config.get("version", self.DEFAULT_VERSION)

        import mlflow
        self.model = mlflow.sklearn.load_model(f"models:/carbon-monoxide-predictor/{self.version}")

    @app.get("/")
    def root(self):
        return f"Carbon Monoxide Predictor. Version: {self.version}"

    @app.post("/predict")
    def predict(self, input: CMPredictionRequest):
        import numpy as np

        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_

        X = np.array(input.data)
        predictions = self.model.predict(X).tolist()
        return {"predictions": predictions}


carbon_monoxide_predictor_app = CarbonMonoxidePredictor.bind()
