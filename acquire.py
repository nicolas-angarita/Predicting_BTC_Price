import pandas as pd
import yfinance as yf




def asset_price(year):
    '''
    This function will take in a specific year and will get the data from yahoo finance library for 
    Bitcoin, dow Jones Industrial Average, and Gold; making sure it does all the way back to the
    year that was put in the function. 
    '''
    btc = "BTC-USD"
    btc_df = yf.download(btc, start=f"{year}-01-01", end="2023-2-13")

    gold = "GC=F"
    gold_df = yf.download(gold, start=f"{year}-01-01", end="2023-2-13")

    dji = "^DJI"
    dji_df = yf.download(dji, start=f"{year}-01-01", end="2023-2-13")
    
    return btc_df, gold_df, dji_df


def merged_assets(btc_df, gold_df, dji_df):
    '''
    This function takes in the data frames for BTC, DJI, and Gold to merge all 3 of them into one dataframe 
    on the adjusted close column
    '''
    
    df_btc_gold = pd.merge(btc_df['Adj Close'], gold_df['Adj Close'], on="Date")

    df = pd.merge(df_btc_gold, dji_df['Adj Close'], on="Date")
    
    return df