import asyncio
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


# models locations
RANDOM_FOREST_MODEL_PATH = os.path.join("models/wine-red-quality_random_forest.pkl")
XGBOOST_MODEL_PATH = os.path.join("models/wine-red-quality_xgboost.pkl")

# start Ray
ray.init()

# start Serve
serve.start()

#define deployments
@serve.deployment(route_prefix="/random_forest/red_wines")
class RandomForestModel:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    async def __call__(self, request):
        payload = await request.json()
        return self.serve(payload)

    def serve(self, request):
        input_vector = [
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
        prediction = self.model.predict([input_vector])[0]
        return {"result": str(prediction)}

@serve.deployment(route_prefix="/xgboost/red_wines")
class XGBoostModel:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    async def __call__(self, request):
        payload = await request.json()
        return self.serve(payload)

    def serve(self, request):
        input_vector = np.array([
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
        prediction = self.model.predict(input_vector.reshape(1,11))[0]
        return {"result": str(prediction)}

RandomForestModel.deploy(RANDOM_FOREST_MODEL_PATH)
XGBoostModel.deploy(XGBOOST_MODEL_PATH)


@serve.deployment(route_prefix="/consensus")
class Speculative:
    def __init__(self):
        self.rfhandle = RandomForestModel.get_handle(sync=False)
        self.xgboosthandle = XGBoostModel.get_handle(sync=False)
    async def __call__(self, request):
        payload = await request.json()
        f1, f2 = await asyncio.gather(self.rfhandle.serve.remote(payload),
                self.xgboosthandle.serve.remote(payload))

        rfresurlt = ray.get(f1)['result']
        xgresurlt = ray.get(f2)['result']
        ones = []
        zeros = []
        if rfresurlt == "1":
            ones.append("Random forest")
        else:
            zeros.append("Random forest")
        if xgresurlt == "1":
            ones.append("XGBoost")
        else:
            zeros.append("XGBoost")
        if len(ones) >= 2:
            return {"result": "1", "methods": ones}
        else:
            return {"result": "0", "methods": zeros}
            

Speculative.deploy()

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

print(":: Random Forrest Classifier - Red Wines ::")
print("")
print(requests.get("http://localhost:8000/random_forest/red_wines", json=sample_request_input_red).text)
print("")
print(":: XGBoost Classifier - Red Wines ::")
print("")
print(requests.get("http://localhost:8000/xgboost/red_wines", json=sample_request_input_red).text)
print("")
print(":: Consensus Results ::")
print("")
print(requests.get("http://localhost:8000/consensus", json=sample_request_input_red).text)