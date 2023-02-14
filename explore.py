import pandas as pd

import matplotlib.pyplot as plt



def line_plot(train):
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