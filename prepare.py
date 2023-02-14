import pandas as pd
from datetime import datetime



def clean_df(df):
    '''
    This function takes in a data frame and will rename the columns and index. Make a max timestamp
    and resample the data into yearly, quarterly, and monthly. 
    '''
    df.rename(columns= {'Adj Close_x': 'btc_price',
                        'Adj Close_y': 'gold_price',
                        'Adj Close': 'dji_price'}, inplace = True )
    
    df.index.name = 'date'

    max_date = pd.Timestamp('2022-12-31')
    df = df[df.index <= max_date]
    
    df = df.round(2)
    
    df_year = df.resample('Y').mean()
    df_month = df.resample('M').mean()
    df_quarter = df.resample('3M').mean()
    
    df_year = df_year.round(2)
    df_month = df_month.round(2)
    df_quarter = df_quarter.round(2)
    
    return df, df_year, df_month, df_quarter

def split_data_explore(df):
    '''
    This function takes in a clean df and splits it to train, validate, and test to explore upon
    '''
    train_size = int(round(df.shape[0] * 0.5))
    validate_size = int(round(df.shape[0] * 0.3))
    test_size = int(round(df.shape[0] * 0.2))
    
    
    validate_end_index = train_size + validate_size
    
    train = df[:train_size]
    validate = df[train_size:validate_end_index]
    test = df[validate_end_index:]

    
    return train, validate, test