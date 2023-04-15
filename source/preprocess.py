from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

def state_encoder(df):
    return df.apply(lambda x: x.str.slice(start=-1).astype(int))

def preprocess(X_train, X_test):
    other_cols = ['facility_type', 'State_Factor']
    categorical_cols = list(set(X_train.select_dtypes('object').columns) - set(other_cols))
    numerical_cols = list(X_train.select_dtypes(['int64', 'float64']).columns)

    transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), categorical_cols),
                                              ('num', StandardScaler(), numerical_cols),
                                              ('count', CountEncoder(handle_unknown='ignore'), ['facility_type']),
                                              ('custom', FunctionTransformer(state_encoder), ['State_Factor'])])
    
    X_tr = transformer.fit_transform(X_train)
    X_tst = transformer.transform(X_test)
    categorical_feature_names = transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = list(categorical_feature_names) + numerical_cols + other_cols
    X_tr = pd.DataFrame(X_tr, columns = feature_names)
    X_tst = pd.DataFrame(X_tst, columns = feature_names)
    return X_tr, X_tst