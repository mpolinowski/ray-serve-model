# DEPENDENCIES

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px

# normalizing features
from sklearn.preprocessing import StandardScaler
# training/validation split
from sklearn.model_selection import train_test_split
# get performance metrics after model fitting
from sklearn.metrics import classification_report
# random forest classifier
from sklearn.ensemble import RandomForestClassifier

# xgboost classifier
import xgboost as xgb


# DATASET

df = pd.read_csv("data/winequality-red_cleaned.csv")
# df = pd.read_csv("data/winequality-white_cleaned.csv")


# DATA PREPROCESSING

## make binary quality classification
df['good'] = [1 if x >= 7 else 0 for x in df['quality']]
## separate feature and target variables
X = df.drop(['quality', 'good'], axis = 1)
y = df['good']

## normalize feature
X_features = X
X = StandardScaler().fit_transform(X)

## 25/75 val/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)


# MODEL FITTING

## random forrest classifier
forrest_model = RandomForestClassifier(random_state=42)
## use training dataset for fitting
forrest_model.fit(X_train, y_train)
## run prediction based of the validation dataset
y_pred2 = forrest_model.predict(X_test)
## get performance metrics
print("")
print(":: RandomForestClassifier ::")
print("")
print(classification_report(y_test, y_pred2))

## xgboost classifier
xgboost_model = xgb.XGBClassifier(random_state=42)
## use training dataset for fitting
xgboost_model.fit(X_train, y_train)
## run prediction based of the validation dataset
y_pred5 = xgboost_model.predict(X_test)
## get performance metrics
print("")
print(":: XGBClassifier ::")
print("")
print(classification_report(y_test, y_pred5))


#save models - random forest and XGBoost
RANDOM_FOREST_MODEL_PATH = os.path.join("models/wine-red-quality_random_forest.pkl")
with open(RANDOM_FOREST_MODEL_PATH, "wb") as f:
    pickle.dump(forrest_model, f)
    
XGBOOST_MODEL_PATH = os.path.join("models/wine-red-quality_xgboost.pkl")
with open(XGBOOST_MODEL_PATH, "wb") as f:
    pickle.dump(xgboost_model, f)