from .preprocess_housing import PreProcess_Housing
from .variables_housing import impute_numerical_features, impute_categorical_features, interesting_features, skewed_features, polynomial_features, final_feature_order
import pandas as pd
import numpy as np
import pickle

# load dataframe and create instance of PreProcess_Data
# process dataframe object
def process_data(raw_data, path):
    #data = pd.read_csv('data.csv')
    median_imputer = pickle.load(open(path + 'median_imputer_housing', 'rb'))
    mode_imputer = pickle.load(open(path + 'mode_imputer_housing', 'rb'))
    encoder = pickle.load(open(path + 'encoder_housing', 'rb'))

    data = PreProcess_Housing(raw_data)
    data.drop_features(['Id', 'Alley', 'Utilities', 'PoolQC'])
    data.impute(impute_numerical_features, impute_categorical_features, median_imputer, mode_imputer)
    data.create_ordinals()
    data.create_bins()
    data.create_new_features(interesting_features)
    data.drop_features(['PoolArea', 'MoSold'])
    data.log_transform(skewed_features)
    data.create_polynomials(polynomial_features)
    data.one_hot_encode(encoder)
    data = data.get_data()
    data = data[final_feature_order]
    return data

# load model and predict on processed data
def predict(processed_data, path):
    model = pickle.load(open(path + 'stack_housing', 'rb'))
    predictions = pd.DataFrame(np.exp(model.predict(processed_data)), columns=['SalePrice'])
    predictions = predictions.round(0)
    predictions = predictions.astype('int64')
    return predictions

# execute process_data and predict functions
def run_housing(raw_data):
    path = '/code/housing_project/'
    processed_data = process_data(raw_data, path)
    #predictions = pd.DataFrame(np.exp(predict(df_processed)), columns=['SalePrice'])
    return predict(processed_data, path)
