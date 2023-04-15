## ./main.py

Machine Learning Model Training with Random Forest Regressor

Purpose: This python script aims to train a machine learning model using Random Forest Regressor to predict the target variable based on a set of features. The model is trained using a training dataset and evaluated using a testing dataset. The purpose of this code is to document the steps taken to train the model, preprocess the data, and evaluate the model's performance.

Dependencies:

- shap
- mlflow
- sklearn
- numpy
- matplotlib
- source.constants
- source.helpers
- source.preprocess
- experiment

Inputs:

- Training dataset (df_train): pandas DataFrame
- Testing dataset (df_test): pandas DataFrame
  Outputs:
- None

Procedure:

1. Import necessary libraries and dependencies
2. Load the training and testing datasets
3. Preprocess the data by splitting the features and target variable, dropping irrelevant features, and filling missing values
4. Split the preprocessed data into training and testing datasets
5. Define the new set of features to be used in training the model
6. Preprocess the training and testing datasets by selecting only the new set of features
7. Define the Random Forest Regressor model and set the number of jobs to -1
8. Set the mlflow experiment for tracking model metrics and parameters
9. Train the model using the training dataset and evaluate using the testing dataset
10. Record the model's performance metrics using mlflow

Source code:

```python
import shap
import time
import mlflow
import experiment
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from source.constants import MLFLOW_TRACKING_URI
from source.helpers import load_data, split_features_target
from source.preprocess import preprocess


mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
shap.initjs()

df_train, df_test = load_data()
X, y = split_features_target(df_train)
X.drop("id", axis=1, inplace=True)
X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

new_features = ['building_class_Residential', 'energy_star_rating', 'february_avg_temp','march_min_temp', 'july_min_temp', 'july_avg_temp', 'october_max_temp', 'november_min_temp', 'december_avg_temp', 'avg_temp']

X_tr, X_tst = preprocess(X_train, X_test)
X_train_num, X_test_num = X_tr[new_features], X_tst[new_features]

estimator = RandomForestRegressor(n_jobs=-1)

mlflow.set_experiment('random_forest_estimator')

with mlflow.start_run():
	experiment.train_estimator(
		estimator, X_train_num, y_train, X_test_num, y_test)
```

## ./source/

### ./source/constants.py

Defining File Paths for Data and MLFlow

Purpose: This code aims to define the file paths for the data and MLFlow tracking.

Dependencies:

- pathlib
  Inputs:
- None
  Outputs:
- None
  Procedure:

1. Import necessary libraries and dependencies
2. Define the root directory path using pathlib
3. Define the path for the data directory by concatenating the root directory path with "data"
4. Define the path for the MLFlow tracking directory by concatenating the root directory path with "mlruns"

Source code:

```python
from pathlib import Path
# ==== Paths ====

ROOT_DIR = Path(__file__).parents[1]

PATH_DATA = ROOT_DIR / "data"

MLFLOW_TRACKING_URI = ROOT_DIR / "mlruns"
```

### ./source/feature_selection.py

This module `feature_selection.py` aims to perform feature selection using Recursive Feature Elimination (RFE) with Linear Regression.

Dependencies:

- sklearn.feature_selection.RFE
- sklearn.linear_model.LinearRegression
  Inputs:
- X: pandas DataFrame or array-like of shape (n_samples, n_features) The training input samples.
- y: pandas Series or array-like of shape (n_samples,) The target values (class labels in classification, real numbers in regression).
- n_features_to_select: int or None (default=None) The number of features to select. If None, half of the features are selected.

Outputs:

- mask: numpy array of shape (n_features,) The mask of selected features.
- ranking: numpy array of shape (n_features,) The ranking of features, with ranking[i] being the rank of the i-th feature. The smaller the rank, the more important the feature.

Procedure:

1. Import necessary libraries and dependencies: RFE and Linear Regression from scikit-learn.
2. Define a function, "rfe_regression", that takes in training data (X and y) and the number of features to select.
   - Instantiate a Linear Regression estimator.
   - Instantiate an RFE object with the estimator and the number of features to select.
   - Fit the RFE object to the training data.
   - Return the mask of selected features and the ranking of all features.
3. Define a function, "select_features", that takes in training data (X_tr and y_tr) and the number of features to select (default is 10).
   - Call the "rfe_regression" function with the training data and number of features to select.
   - Retrieve the mask of selected features.
   - Subset the columns of the training data using the selected features.
   - Return the list of selected feature names.

Source code:

```python
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

def rfe_regression(X, y, n_features_to_select):
	estimator = LinearRegression()
	rfe = RFE(estimator, n_features_to_select=n_features_to_select)
	rfe.fit(X, y)
	return rfe.support_, rfe.ranking_

def select_features(X_tr, y_tr, num_features=10):
	mask, ranking = rfe_regression(X_tr, y_tr, num_features)
	new_features = X_tr.columns[mask]
	return new_features
```

The methods inside this module:
`ref_regression(X, y, n_features_to_select)`

- Input:
  - X (array-like of shape (n_samples, n_features)): The input data.
  - y (array-like of shape (n_samples,)): The target values.
  - n_features_to_select (int): The number of features to select.
- Output:
  - support (array of shape (n_features,)): The mask of selected features.
  - ranking (array of shape (n_features,)): The feature ranking with the selected features having rank 1.
    This method performs Recursive Feature Elimination (RFE) using a Linear Regression estimator. It selects the top `n_features_to_select` features that best contribute to predicting the target variable `y`. It returns the boolean mask of selected features and the ranking of all features based on their importance in predicting the target variable.

`select_features(X_tr, y_tr, num_features=10)`

- Input:
  - X_tr (array-like of shape (n_samples, n_features)): The training data.
  - y_tr (array-like of shape (n_samples,)): The target values of the training data.
  - num_features (int): The number of features to select.
- Output:
  - new_features (array of shape (num_features,)): The selected features.
    This method performs feature selection on the training data `X_tr` and target values `y_tr`. It uses `rfe_regression()` to select the top `num_features` features that best contribute to predicting the target variable. It returns an array of the selected feature names.

### ./source/helpers.py

This module `helpers.py` provides functionality for loading and splitting data.
`load_data() -> Tuple[pd.DataFrame, pd.DataFrame]`

- Output:
  - df_train (pandas DataFrame): The training data.
  - df_test (pandas DataFrame): The test data.
    This function loads the train and test data from CSV files located in the `PATH_DATA` directory. It returns a tuple of the training data and test data as pandas DataFrames.

`split_features_target(df: pd.DataFrame, target_col: str = "site_eui") -> Tuple[pd.DataFrame, pd.DataFrame]`

- Input:
  - df (pandas DataFrame): The input data.
  - target_col (str): The name of the target column in the input data. Default is "site_eui".
- Output:
  - features (pandas DataFrame): The input data without the target column.
  - target (pandas DataFrame): The target column.
    This function takes an input pandas DataFrame `df` and splits it into two DataFrames: `features` and `target`. The `features` DataFrame contains all columns from `df` except for the `target_col` column. The `target` DataFrame contains only the `target_col` column from `df`. It returns a tuple of the `features` DataFrame and the `target` DataFrame.

### ./source/preprocess.py

This module `preprocess.py` contains a set of methods used for pre-processing the dataset for machine learning modelling.

**Functions**

1. `state_encoder(df) -> pandas.DataFrame`:
   - This function takes a pandas DataFrame as input and applies a lambda function to the input dataframe to return the state number instead of the state abbreviation.
   - The input is a pandas DataFrame containing state abbreviations, and the output is a pandas DataFrame containing the state numbers as integers.
2. `preprocess(X_train, X_test) -> Tuple[pandas.DataFrame, pandas.DataFrame]`:
   - This function takes two pandas DataFrames as input representing the training and test data and preprocesses them.
   - The input includes a DataFrame containing features and a DataFrame containing the target column.
   - This function does the following steps:
     - Splits the data into numerical and categorical columns.
     - Applies OneHotEncoder to the categorical columns.
     - Scales the numerical columns using StandardScaler.
     - Applies CountEncoder to the "facility_type" column.
     - Applies a custom transformer to encode the state abbreviations to state numbers.
     - Concatenates the preprocessed columns into a single DataFrame.
   - The output of this function is a tuple containing the preprocessed training data and preprocessed test data as pandas DataFrames.
