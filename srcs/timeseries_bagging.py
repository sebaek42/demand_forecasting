import numpy as np
import pandas as pd
import statsmodels.api as sm
from SARIMAX import Sarimax
from arch.bootstrap import MovingBlockBootstrap, CircularBlockBootstrap, StationaryBootstrap, optimal_block_length

class TS_Bagging:
    def __init__(self, passenger, exog):
        self.passenger = passenger
        self.exog = exog
    
    def split_train_test(self):
        # train:test = 8:2
        length = len(self.passenger)
        tr_start, tr_end = 0, int(length * 0.8)+1 
        te_start, te_end = tr_end, length

        self.train_passenger = self.passenger[tr_start:tr_end]
        self.test_passenger = self.passenger[te_start:te_end]

        self.train_exog = self.exog[tr_start:tr_end]
        self.test_exog = self.exog[te_start:te_end]

    def seasonal_decompose(self):
        # Decompose Data
        train_decomposed = sm.tsa.seasonal_decompose(self.train_passenger, freq=12, extrapolate_trend='freq')
        test_decomposed = sm.tsa.seasonal_decompose(self.test_passenger, freq=12, extrapolate_trend='freq')
        
        self.train_resid = train_decomposed.resid
        self.train_other = train_decomposed.trend + train_decomposed.seasonal
    
        self.test_resid = test_decomposed.resid
        self.test_other = test_decomposed.trend + test_decomposed.seasonal
        
    def forecast(self):
        self.split_train_test()
        self.seasonal_decompose()

        #Boot Strap
        block_size = int(optimal_block_length(self.train_passenger)['stationary'] + 0.5)
        bs = MovingBlockBootstrap(block_size, self.train_resid, y=self.train_exog)
        bs_data = [data for data in bs.bootstrap(3)]

        # Forecast
        """
        cnn = CNN(bs_data[0][0][0], bs_data[0][1]['y'])
        lstm = LSTM(bs_data[1][0][0], bs_data[1][1]['y'])
        """
        sarimax = Sarimax(bs_data[2][0][0], self.test_resid, self.train_other, 
                    self.test_other, bs_data[2][1]['y'], self.test_exog)
        
    
        sarimax_pred = sarimax.sarimax_predict()
        return sarimax_pred