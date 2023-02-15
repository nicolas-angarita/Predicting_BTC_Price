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

    
def plot_eval(target_var, train, validate, yhat_df):
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
    plt.ylim(0,20000)
    plt.show()    


def append_eval_df(model_type, target_var, validate, yhat_df):
    '''
    This function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var,validate, yhat_df)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
       
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])

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
    
    return btc_df, train, validate, test


def baseline_model(train, validate):
    '''
    This function takes in the train and validate datasets and makes a baseline model of 50 moving average
    '''
    period = 50
    train.rolling(period).mean()
    
    rolling_btc = round(train['btc_price'].rolling(period).mean()[-1], 2)
    
    yhat_df = make_baseline_predictions(validate, rolling_btc)
    
    return yhat_df


def holts_linear(col, train, validate):
    '''
    This function makes a holts linear model
    '''
    # create our Holt Object
    model = Holt(train[col], exponential=False, damped=True)
    model = model.fit(optimized=True)   
    
    yhat_df = model.predict(start = validate.index[0], end = validate.index[-1])
    yhat_df = pd.DataFrame({'btc_price': yhat_df},
                          index=validate.index)
    return yhat_df


def holts_seasonal(train, validate):
    '''
    This function makes a holts seasonal model
    '''
    hst_price_fit4 = ExponentialSmoothing(train.btc_price, seasonal_periods=365, trend='add',
                                          seasonal='mul', damped=True).fit()
    
    yhat_list = hst_price_fit4.forecast(validate.shape[0]).tolist()
    yhat_df = pd.DataFrame({'btc_price': yhat_list}, index=validate.index)
    
    return yhat_df


def previous_cycle(btc_df):
    '''
    This function makes a previous cycle model
    '''
    train = btc_df[:'2021-04-15']
    validate = btc_df['2021-04-16':'2022-03-15']
    test = btc_df['2022-03-16':]
    
    train.diff(365)
    
    yhat_df = train['2020-05-17':'2021-04-15'] + train.diff(365).mean()

    yhat_df.index = validate.index
    
    return yhat_df, train, validate


def comparing_rmse(train, validate):
    '''
    This function makes a data frame to compare the rmse of the different models
    '''
    periods = [50,100]
    eval_dfs = []

    for p in periods: 
        rolling_btc = round(train['btc_price'].rolling(p).mean()[-1], 2)
        yhat_df = make_baseline_predictions(validate, rolling_btc)
        model_type = str(p) + '_day_moving_avg'
        eval_df = append_eval_df(model_type, 'btc_price', validate, yhat_df)
        eval_dfs.append(eval_df)
        
    #Holts winter linear
    yhat_df = holts_linear('btc_price', train, validate)
    eval_df = append_eval_df('holts_optimized','btc_price', validate, yhat_df)
    eval_dfs.append(eval_df)
    
    #Holts Seasonal
    yhat_df = holts_seasonal(train, validate)
    eval_df = append_eval_df('holts_add_mul', 'btc_price', validate, yhat_df)
    eval_dfs.append(eval_df)
    
    rmse_compare = pd.concat(eval_dfs, axis = 0)
    
    return rmse_compare.sort_values('rmse').reset_index().drop(columns= 'index')


def test_model_split(btc_df):
    '''
    
    '''
    #splitting
    train_size = int(round(btc_df.shape[0] * 0.6))
    validate_size = int(round(btc_df.shape[0] * 0.2))
    test_size = int(round(btc_df.shape[0] * 0.2))
    
    validate_end_index = train_size + validate_size
    
    train = btc_df[:train_size]
    validate = btc_df[train_size:validate_end_index]
    test = btc_df[validate_end_index:]
    
    return train, validate, test


def make_test_predictions(train, validate, test):
    
    rolling_btc = round(train['btc_price'].rolling(100).mean()[-1], 2)
    
    # Create a DataFrame with the predictions and the corresponding index range
    index_range = pd.date_range(start=validate.index[0], end=test.index[-1], freq='D')
    yhat_df = pd.DataFrame({'btc_price': rolling_btc}, index=index_range)
    
    yhat_df = yhat_df['2021-06-09':]
    
    return yhat_df 


def final_plot(train, validate, test, yhat_df, target_var):
    '''
    
    '''
    plt.figure(figsize=(12,4))
    plt.plot(train[target_var], color='#377eb8', label='train')
    plt.plot(validate[target_var], color='#ff7f00', label='validate')
    plt.plot(test[target_var], color='#4daf4a',label='test')
    plt.plot(yhat_df[target_var], color='#a65628', label='yhat')
    plt.legend()
    plt.title(target_var)
    plt.show()