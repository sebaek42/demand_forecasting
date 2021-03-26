import numpy as np
import pandas as pd
import statsmodels.api as sm
from SARIMAX import Sarimax
from CNN import Cnn
from LSTM import Lstm
from arch.bootstrap import MovingBlockBootstrap, CircularBlockBootstrap, StationaryBootstrap, optimal_block_length


# Split data
# Default train data ratio = 0.8

def split_train_test(data, train_ratio):
    # train:test = 8:2
    length = len(data)
    tr_start, tr_end = 0, int(length * train_ratio)+1 
    te_start, te_end = tr_end, length 
    
    train = data[tr_start:tr_end]
    test = data[te_start:te_end]

    return train, test
    
def fit_model_and_predict(passenger, exog_norm, exog_org, ratio=0.8, n_periods=12):
    train_passenger, test_passenger = split_train_test(passenger, ratio)
    train_exog_norm, test_exog_norm = split_train_test(exog_norm, ratio)
    train_exog_org, test_exog_org = split_train_test(exog_org, ratio)

    models = (
        Sarimax(train_passenger, test_passenger, train_exog_norm, test_exog_norm),
        Cnn(train_passenger, test_passenger, train_exog_norm, test_exog_norm),
        Lstm(train_passenger, test_passenger, train_exog_norm, test_exog_norm)
    )

    pred = []
    for model in models:
        model.fit()
        pred.append(model.predict(n_periods=n_periods))

    return np.array(list(map(int , sum(pred) / 3)))