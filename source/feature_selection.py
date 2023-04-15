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