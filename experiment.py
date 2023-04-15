#!/usr/bin/env python
# coding: utf-8

# Energy consumption prediction

import shap
import time
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from source.constants import MLFLOW_TRACKING_URI
from source.helpers import load_data, split_features_target
from source.preprocess import preprocess
from source.feature_selection import select_features


mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
shap.initjs()

df_train, df_test = load_data()

df_train.head()

X, y = split_features_target(df_train)
X.drop("id", axis=1, inplace=True)
X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

X_tr, X_tst = preprocess(X_train, X_test)
X_tr.shape, X_tst.shape

new_features = select_features(X_tr, y_train)
print(new_features)

new_features = ['building_class_Residential', 'energy_star_rating', 'february_avg_temp',
                'march_min_temp', 'july_min_temp', 'july_avg_temp', 'october_max_temp',
                'november_min_temp', 'december_avg_temp', 'avg_temp']

exp_id = 895323239743716977

estimator = RandomForestRegressor(n_jobs=-1)


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def train_estimator(estimator, _X_train, _y_train, _X_test, _y_test):
    with mlflow.start_run(experiment_id=exp_id):
        estimator.fit(_X_train, _y_train)

        y_pred = estimator.predict(_X_test)
        mse = mean_squared_error(_y_test, y_pred)
        rmsle = compute_rmsle(_y_test, y_pred)

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(_X_test, _y_test, approximate=True)

        plt.figure()
        shap.summary_plot(shap_values, _X_test, plot_type='bar')
        summary_plot1 = plt.gcf()
        time.sleep(1)
        summary_plot1.savefig('shap_summary_plot1.png')
        plt.show()

        plt.figure()
        shap.summary_plot(shap_values, _X_test)
        summary_plot2 = plt.gcf()
        time.sleep(1)
        summary_plot2.savefig('shap_summary_plot2.png')
        plt.show()

        mlflow.sklearn.log_model(estimator, 'estimator')
        mlflow.log_metrics({"testing_mse": mse, "testing_rmsle": rmsle})
        mlflow.log_artifact('shap_summary_plot1.png')
        mlflow.log_artifact('shap_summary_plot2.png')


# In[80]:
X_train_num, X_test_num = X_tr[new_features], X_tst[new_features]
train_estimator(estimator, X_train_num, y_train, X_test_num, y_test)

# ## SHAP Model interpretation
# In[81]:
explainer = shap.TreeExplainer(estimator)
shap_values = explainer.shap_values(X_test_num, y_test, approximate=True)

# In[84]:
observation_idx = 0
shap.force_plot(explainer.expected_value,
                shap_values[observation_idx, :], X_test_num.iloc[observation_idx, :])
