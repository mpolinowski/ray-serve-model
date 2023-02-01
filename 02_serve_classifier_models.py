import numpy as np
import os
import pandas as pd
import pickle
import ray
from ray import serve
import requests
from sklearn.preprocessing import StandardScaler

# normalize prediction example
df_input = pd.read_csv("data/prediction_example.csv")
df_norm = StandardScaler().fit_transform(df_input)

# model locations
RANDOM_FOREST_WHITE_MODEL_PATH = os.path.join("models/wine-white-quality_random_forest.pkl")
RANDOM_FOREST_RED_MODEL_PATH = os.path.join("models/wine-red-quality_random_forest.pkl")
XGBOOST_WHITE_MODEL_PATH = os.path.join("models/wine-white-quality_xgboost.pkl")
XGBOOST_RED_MODEL_PATH = os.path.join("models/wine-red-quality_xgboost.pkl")

# tart Ray
ray.init()

# tart Serve
serve.start()


# RANDOM FOREST - WHITE WINE

## define deployment
@serve.deployment(route_prefix="/random_forest/white_wines")
class RandomForestModelWhite:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
    async def __call__(self, request):
        payload = await request.json()
        return self.serve(payload)

    def serve(self, request):
        model_features = [
            request["fixed acidity"],
            request["volatile acidity"],
            request["citric acid"],
            request["residual sugar"],
            request["chlorides"],
            request["free sulfur dioxide"],
            request["total sulfur dioxide"],
            request["density"],
            request["pH"],
            request["sulphates"],
            request["alcohol"],
        ]
        prediction = self.model.predict([model_features])[0]
        return {"result": str(prediction)}


# RANDOM FOREST - RED WINE

## define deployment
@serve.deployment(route_prefix="/random_forest/red_wines")
class RandomForestModelRed:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
    async def __call__(self, request):
        payload = await request.json()
        return self.serve(payload)

    def serve(self, request):
        model_features = [
            request["fixed acidity"],
            request["volatile acidity"],
            request["citric acid"],
            request["residual sugar"],
            request["chlorides"],
            request["free sulfur dioxide"],
            request["total sulfur dioxide"],
            request["density"],
            request["pH"],
            request["sulphates"],
            request["alcohol"],
        ]
        prediction = self.model.predict([model_features])[0]
        return {"result": str(prediction)}



# XGBOOST - WHITE WINE

#define deployment
@serve.deployment(route_prefix="/xgboost/white_wines")
class XGBoostModelWhite:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    async def __call__(self, request):
        payload = await request.json()
        return self.serve(payload)

    def serve(self, request):
        model_features = np.array([
            request["fixed acidity"],
            request["volatile acidity"],
            request["citric acid"],
            request["residual sugar"],
            request["chlorides"],
            request["free sulfur dioxide"],
            request["total sulfur dioxide"],
            request["density"],
            request["pH"],
            request["sulphates"],
            request["alcohol"],
        ])
        prediction = self.model.predict(model_features.reshape(1,11))[0]
        return {"result": str(prediction)}



# XGBOOST - RED WINE

#define deployment
@serve.deployment(route_prefix="/xgboost/red_wines")
class XGBoostModelRed:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    async def __call__(self, request):
        payload = await request.json()
        return self.serve(payload)

    def serve(self, request):
        model_features = np.array([
            request["fixed acidity"],
            request["volatile acidity"],
            request["citric acid"],
            request["residual sugar"],
            request["chlorides"],
            request["free sulfur dioxide"],
            request["total sulfur dioxide"],
            request["density"],
            request["pH"],
            request["sulphates"],
            request["alcohol"],
        ])
        prediction = self.model.predict(model_features.reshape(1,11))[0]
        return {"result": str(prediction)}


# DEPLOY MODELS

XGBoostModelRed.deploy(XGBOOST_RED_MODEL_PATH)
XGBoostModelWhite.deploy(XGBOOST_WHITE_MODEL_PATH)
RandomForestModelRed.deploy(RANDOM_FOREST_RED_MODEL_PATH)
RandomForestModelWhite.deploy(RANDOM_FOREST_WHITE_MODEL_PATH)


# list current deployment
print(serve.list_deployments())


# SAMPLE API REQUEST

## take the first and second row of the
## example file and predict how good those
## wines should be based on the features presented
## the first should score high for reds and the second for whites

sample_request_input_red = {
    "fixed acidity": df_norm[0][0],
    "volatile acidity": df_norm[0][1],
    "citric acid": df_norm[0][2],
    "residual sugar": df_norm[0][3],
    "chlorides": df_norm[0][4],
    "free sulfur dioxide":  df_norm[0][5],
    "total sulfur dioxide": df_norm[0][6],
    "density": df_norm[0][7],
    "pH": df_norm[0][8],
    "sulphates": df_norm[0][9],
    "alcohol":  df_norm[0][10],
}

sample_request_input_white = {
    "fixed acidity": df_norm[1][0],
    "volatile acidity": df_norm[1][1],
    "citric acid": df_norm[1][2],
    "residual sugar": df_norm[1][3],
    "chlorides": df_norm[1][4],
    "free sulfur dioxide":  df_norm[1][5],
    "total sulfur dioxide": df_norm[1][6],
    "density": df_norm[1][7],
    "pH": df_norm[1][8],
    "sulphates": df_norm[1][9],
    "alcohol":  df_norm[1][10],
}

# GET PREDICTIONS

## http api requests
print(":: Random Forrest Classifier - White Wines ::")
print("")
print(requests.get("http://localhost:8000/random_forest/white_wines", json=sample_request_input_white).text)
print("")
print(":: Random Forrest Classifier - Red Wines ::")
print("")
print(requests.get("http://localhost:8000/random_forest/red_wines", json=sample_request_input_red).text)
print("")
print(":: XGBoost Classifier - White Wines ::")
print("")
print(requests.get("http://localhost:8000/xgboost/white_wines", json=sample_request_input_white).text)
print("")
print(":: XGBoost Classifier - Red Wines ::")
print("")
print(requests.get("http://localhost:8000/xgboost/red_wines", json=sample_request_input_red).text)