# Predicting Bitcon's Price (BTC)

# Project Goals

 - Identify if there is a correlation between BTC(Bitcoin), DJI(Dow Jones Industrial Average), and Gold 
 - Build a model to best predict BTC's price in the next 6 months

# Project Description

We are looking to build a supervised machine learning model using time series analysis algorithm to best predict prices for BTC for the next 6 months.
We plan to explore and find if there is any correlations between BTC, DJI, and Gold prices to help us predict price. The time frame we are looking at is 
from 9/17/2014 to the end of 2022. After we have explored and made our models we will recommend to use DJI and Gold data or not to help predict price. 
We also will provide any usesful insights on interesting findings to predict BTC's price for the next 6 months.

# Initial Questions

 1. Does BTC's price have a correlation with the DJI?
 2. Does BTC's price have a relationship with Gold?
 3. How does the volatility of BTC compare with that of the DJI and Gold?


# The Plan

 - Create README with project goals, project description, initial hypotheses, planning of project, data dictionary, and come up with recommedations/takeaways

### Acquire Data
 - Acquire data from Yahoo Finance Library and dowload the historical data from BTC, DJI, & Gold. Create a function to later import the data into a juptyer notebook. (acquire.py)

### Prepare Data
 - Clean and prepare the data creating a function that will give me data that is ready to be explored upon. Within this step we will also write a function to split our data into train, validate, and test. (prepare.py) 
 
### Explore Data
- Create visuals on our data 

- Create at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, document any findings and takeaways that are observed.

### Model Data 
 - Create a baseline model of BTC (Moving Average model)
 
 - Create, Fit, Predict on train subset on 3 time series models.
 
 - Evaluate models on train and validate datasets.
 
 - Evaluate which model performs the best and on that model use the test data subset.
 
### Delivery  
 - Create a Final Report Notebook to document conclusions, takeaways, and next steps in recommadations for predicitng house values. Also, inlcude visualizations to help explain why the model that was selected is the best to better help the viewer understand. 


## Data Dictionary


| Target Variable |     Definition     |
| --------------- | ------------------ |
|      btc_price    | price of Bitcoin |

| Feature  | Definition |
| ------------- | ------------- |
| open | Opening price of the asset |
| high | The high price point of the asset  |
| low | The low price point of the asset |
| close | The closing price of the asset |
| adj close | The closing price after adjustments  |
| volume | The amount of an asset that changes hands | 
| dji_price | The price of the Dow Jones Industrial Average |
| gold | The price of gold per ounce|



## Steps to Reproduce

 - You will need to have the Yahoo Finance library downloaded to get the historical data of BTC, DJI, and Gold

- Clone my repo including the acquire.py, prepare.py, explore.py, and modeling.py 

- Put the data in a file containing the cloned repo.

- Run notebook.

## Conclusions

**Bitcoin predictions were used by minimizing RMSE within our models. Both DJI and Gold prove to have mild to strong correlations with BTC's price.**


 
**Best Model's performance:<br>**
**- My best model reduced the root mean squared error by 3,974 dollars compared to the baseline results.**

**- RMSE 9280.0 dollars on in-sample (train), RMSE 9280.0 dollars on out-of-sample data (validate) and RMSE of 27436.26 dollars on the test data.**

## Recommendations
- I would recommend gathering more data from BTC's start date of trading.

- I would also recommend collecting data or finding more correlations to BTC's price, as predicitng price on just BTC historical data is not optimal. This is a good base to better understand the historical price, but a multivariate time series model would have to be made to get better predictions.
## Next Steps

- Find more correlations to better predict BTC price (DXY to BTC price)

- Consider adding different hyperparameters to models for better results
    
- Take this data and build multivariate models
