from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from ray import serve

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class ActiveFiresPredictor:
    DEFAULT_VERSION = 1

    def __init__(self):
        self.version = self.DEFAULT_VERSION
        self.model = None

    def reconfigure(self, config: Dict):
        self.version = config.get("version", self.DEFAULT_VERSION)

        import mlflow
        self.model = mlflow.keras.load.load_model(f"models:/active-fires-predictor/{self.version}")

    @app.get("/")
    def root(self):
        return f"Active Fires Predictor. Version: {self.version}"

    @app.post("/predict")
    async def predict(self, request: Request):
        import numpy as np
        import io

        contents = await request.body()
        tensor = np.load(io.BytesIO(contents))

        prediction = self.model.predict(tensor)

        buffer = io.BytesIO()
        np.save(buffer, prediction)
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="application/octet-stream")


active_fires_predictor_app = ActiveFiresPredictor.bind()
