from .preprocess_titanic import PreProcess_Titanic
from .variables_titanic import final_feature_order
import pandas as pd
import pickle
import xgboost
import lightgbm
import sklearn

# process csv file
def process_data(raw_data, path):
    median_imputer = pickle.load(open(path + 'median_imputer_titanic', 'rb'))
    mode_imputer = pickle.load(open(path + 'mode_imputer_titanic', 'rb'))
    encoder = pickle.load(open(path + 'encoder_titanic', 'rb'))

    data = PreProcess_Titanic(raw_data)
    data.drop_columns(['PassengerId', 'Name'])
    data.create_cabin_bins()
    data.create_sibsp_bins()
    data.create_parch_bins()
    data.format_ticket()
    data.drop_columns(['TicketPrefix'])
    data.drop_columns(['Ticket'])
    data.create_fare_bins()
    data.impute(median_imputer, mode_imputer)
    data.one_hot_encode(encoder)
    data.create_age_bins()
    df = data.return_data()
    df = df[final_feature_order]
    return df

# make predictions from processed csv file
def predict(processed_data, path):
    model = pickle.load(open(path + 'voter_titanic', 'rb'))
    predictions = pd.DataFrame(model.predict(processed_data), columns=['Survived'])
    predictions = predictions.replace({'Survived': {0: 'No', 1: 'Yes'}})
    return predictions

# function gets called with raw csv data as parameter and returns predictions
def run_titanic(raw_data):
    path = '/code/titanic_project/'
    processed_data = process_data(raw_data, path)
    return predict(processed_data, path)