import pandas as pd
import acquire as a

from scipy.stats import pearsonr
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt


def plot(df):
    '''
    This function is to graph all columns within the df
    '''
    df.plot()
    plt.title('Comparing Price Between Assets')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(['BTC','GOLD', 'DJI'])
    plt.show()


def line_plot(train, validate, test):
    '''
    A function that takes in split data and plots the columns in the train dataset
    '''
    for col in train.columns:
        plt.figure(figsize=(14,8))
        plt.plot(train[col], color='#377eb8', label = 'Train')
        plt.plot(validate[col], color='#ff7f00', label = 'Validate')
        plt.plot(test[col], color='#4daf4a', label = 'Test')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.title(f'{col[:3]}')
        plt.show()   
        
        
def seasonal_charts(train):
    '''
    This function takes in the train dataset and takes a look at the trends and seasons of the
    variable
    '''
   
    result = sm.tsa.seasonal_decompose(train['btc_price'].resample('M').mean())
    result.plot()
    plt.show()    
        
        
def comparing_assets(df, y1, y2):
    '''
    This function takes in a dataframe and plots 2 columns 
    '''
    sns.lineplot(data=df, x='date', y= y1, color='blue', label='BTC')
    
    y2_label = y2.split('_')[0].upper()
    
    sns.lineplot(data=df, x='date', y= y2, color='green', label= y2_label)
    plt.ylim(0,50000)
    plt.title(f"BTC Compared To {y2_label} Price")
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.show()
    

def zoomed_in(df):
    sns.lineplot(data=df, x='date', y='btc_price', color='blue', label='BTC')
    sns.lineplot(data=df, x='date', y='gold_price', color='gold' ,label='Gold')
    plt.ylim(0,3000)
    plt.title('BTC vs. Gold')
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.show()
    
    
def pearsons_test(df, col1, col2):
    '''
    This function takes in a df and 2 col names to run a pearson's stats test
    '''
    α = 0.05
    # Calculate the Pearson's correlation coefficient
    r, p = pearsonr(df[col1], df[col2])

    print("Pearson's correlation coefficient:", r)
    print("p-value:", p)

    if p < α:
        print('We reject the null hypothesis') 
    else:
        print('We fail to reject the null hypothesis')    
        
        
def volatility(year):
    '''
    This function is calling the asset_price function from acquire.py, then calculating the
    percent change of the close column. After, we calculate the standard deviation for the 
    volatility. Printing out the result and the plotting it.
    '''
    btc_df, gold_df, dji_df = a.asset_price(year)
    
    # Calculate daily returns for each
    btc_returns = btc_df['Close'].pct_change()
    dji_returns = dji_df['Close'].pct_change()
    gold_returns = gold_df['Close'].pct_change()
    
    # Calculate the standard deviation of daily returns
    btc_volatility = btc_returns.std()
    dji_volatility = dji_returns.std()
    gold_volatility = gold_returns.std()
    
     # Compare the volatilities
    print("BTC Volatility:", btc_volatility)
    print("DJI Volatility:", dji_volatility)
    print("Gold Volatility:", gold_volatility)

    #plot the results
    btc_returns.plot(c = 'black')
    dji_returns.plot(c = 'orange')
    gold_returns.plot(c = 'dodgerblue')
    plt.xlim('2014-09-17', '2023-01-01')
    plt.ylabel('Daily Percent')
    plt.title('Volatility')
    plt.legend(['BTC','DJI', 'GOLD'], loc = 'lower left')
    plt.show()