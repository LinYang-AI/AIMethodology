import shap
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import time

from source.constants import MLFLOW_TRACKING_URI
from source.helpers import load_data, split_features_target
from source.preprocess import preprocess
import experiment

mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
shap.initjs()

df_train, df_test = load_data()

X, y = split_features_target(df_train)
X.drop("id", axis=1, inplace=True)
X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

new_features = ['building_class_Residential', 'energy_star_rating', 'february_avg_temp',
                'march_min_temp', 'july_min_temp', 'july_avg_temp', 'october_max_temp',
                'november_min_temp', 'december_avg_temp', 'avg_temp']

X_tr, X_tst = preprocess(X_train, X_test)
X_train_num, X_test_num = X_tr[new_features], X_tst[new_features]


estimator = RandomForestRegressor(n_jobs=-1)

mlflow.set_experiment('random_forest_estimator')
with mlflow.start_run():
    experiment.train_estimator(
        estimator, X_train_num, y_train, X_test_num, y_test)
