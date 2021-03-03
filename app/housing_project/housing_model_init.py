from .housing_pipeline import PreProcess_Data
from .housing_variables import num_list, cat_list, interesting_features, skewed, df_cols, numerical
import pandas as pd
import numpy as np
import pickle

def process_data(data):
    #data = pd.read_csv('data.csv')
    path = 'C:/Users/Alfred/flask_programming_projects/machine_learning_webpage/housing_project/'
    median_imputer = pickle.load(open(path + 'housing_median_imputer', 'rb'))
    mode_imputer = pickle.load(open(path + 'housing_mode_imputer', 'rb'))
    encoder = pickle.load(open(path + 'housing_encoder', 'rb'))
    data = PreProcess_Data(data)
    data.drop_columns(['Id', 'Alley', 'Utilities', 'PoolQC'])
    data.impute(num_list, cat_list, median_imputer, mode_imputer)
    data.create_ordinals()
    data.create_bins()
    data.create_new_features(interesting_features)
    data.drop_columns(['PoolArea', 'MoSold'])
    data.log_transform(skewed)
    data.create_polynomials(numerical)
    data.one_hot_encode(encoder)
    data = data.get_data()
    data = data[df_cols]
    return data

def predict(data):
    path = 'C:/Users/Alfred/flask_programming_projects/machine_learning_webpage/housing_project/'
    model = pickle.load(open(path + 'ridge', 'rb'))
    predictions = pd.DataFrame(np.exp(model.predict(data)), columns=['SalePrice'])
    predictions = predictions.round(0)
    predictions = predictions.astype('int64')
    return predictions

def housing_run(data):
    processed_data = process_data(data)
    #predictions = pd.DataFrame(np.exp(predict(df_processed)), columns=['SalePrice'])
    return predict(processed_data)
