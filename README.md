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
 4. How does the behavior of BTC, the DJI, and Gold change during periods of market stress or economic uncertainty?
 



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
|      BTC price    | price of Bitcoin |

| Feature  | Definition |
| ------------- | ------------- |
| year_built | The year the house was built  |
| lot_sqft | The square feet of the lot  |
| long | The longitude coordinates of the house |
| lat | The latitude coordinates of the house |
| transaction_month | The month of the transaction date |
| bathrooms | Number of bathrooms in the house | 
| bedrooms | Number of bedrooms in the house |
| fips | Code identifier for county |
| sqft | Square feet of the property|
| county | Name of the county the house is located |


## Steps to Reproduce

 - You will need to have the Yahoo Finance library downloaded to get the historical data of BTC, DJI, and Gold

- Clone my repo including the acquire.py, prepare.py, explore.py, and modeling.py 

- Put the data in a file containing the cloned repo.

- Run notebook.

## Conclusions

**TBD Home value predictions were used by minimizing RMSE within our models. County and transaction month have proven to be the most valuable, but there is still room for improvement.**


 
**Best Model's performance:<br>
My best model reduced the root mean squared error by TBD compared to the baseline results.**

**RMSE reduced by $82,493.41 on in-sample data (train), RMSE $84,100.76 on out-of-sample data (validate), and RMSE of $80,897.87 on the test data when compared to the baseline RMSE.**

## Recommendations
- TBD

## Next Steps

- I would add more columns/features from the database to see if there are better relationships to help predict price.
- Consider adding different hyperparameters to models for better results.
