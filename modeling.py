import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from math import sqrt 

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.api import Holt, ExponentialSmoothing
np.random.seed(0)

import yfinance as yf
import mplfinance as mpf





def evaluate(target_var, validate, yhat_df):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse


def plot_and_eval(target_var, train, validate, yhat_df):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1, color='#377eb8')
    plt.plot(validate[target_var], label='Validate', linewidth=1, color='#ff7f00')
    plt.plot(yhat_df[target_var], label='yhat', linewidth=2, color='#a65628')
    plt.legend(['Train','Validate','BTC Predictions'])
    plt.title('BTC Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    rmse = evaluate(target_var, validate, yhat_df)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()
    
    
def append_eval_df(model_type, target_var):
    '''
    This function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)    


def make_baseline_predictions(validate, btc_predictions=None, ):
    '''
    This function is to make a dataframe for the baseline predictions
    '''
    yhat_df = pd.DataFrame({'btc_price': btc_predictions},
                          index=validate.index)
    return yhat_df


def resplit_btc(btc_df):
    '''
    This function is made to clean the btc dataframe and split it into train, validate, and test
    '''
    # preparing dataframe
    btc_df = btc_df['Adj Close']
    btc_df = pd.DataFrame(btc_df)
    btc_df.index.name = 'date'
    btc_df.rename(columns={'Adj Close': 'btc_price'}, inplace= True)
    
    #splitting
    train_size = int(round(btc_df.shape[0] * 0.6))
    validate_size = int(round(btc_df.shape[0] * 0.2))
    test_size = int(round(btc_df.shape[0] * 0.2))
    
    validate_end_index = train_size + validate_size
    
    train = btc_df[:train_size]
    validate = btc_df[train_size:validate_end_index]
    test = btc_df[validate_end_index:]
    
    return train, validate, test


def baseline_model(train, validate):
    '''
    This function takes in the train and validate datasets and makes a baseline model of 50 moving average
    '''
    period = 50
    train.rolling(period).mean()
    
    rolling_btc = round(train['btc_price'].rolling(period).mean()[-1], 2)
    
    yhat_df = make_baseline_predictions(validate, rolling_btc)
    
    return yhat_df



