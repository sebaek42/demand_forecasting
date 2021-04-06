# multivariate lstm example + walk_forward_validation
import os
import datetime
import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack
from numpy import asarray
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

class Lstm:
    def __init__(self, train_data, test_data, train_exog, test_exog):
        self.train_data = train_data
        self.test_data = test_data
        self.train_exog = train_exog
        self.test_exog = test_exog

    
    def prepare_data(self):
        
        self.train_exog = [self.train_exog[col].values.reshape(len(self.train_exog[col]), 1) for col in self.train_exog.columns]
        self.test_exog = [self.test_exog[col].values.reshape(len(self.test_exog[col]), 1) for col in self.test_exog.columns]
 
        self.train_data = self.train_data.values.reshape(len(self.train_data), 1)
        self.test_data = self.test_data.values.reshape(len(self.test_data), 1)
 
        train_data_shifted_1 = self.shift(self.train_data, 1)
        self.train_exog.append(train_data_shifted_1)
        test_data_shifted_1 = self.join_from_to(self.last_data(self.train_data, 1), self.shift(self.test_data, 1), 1)
        self.test_exog.append(test_data_shifted_1)

        train_dataset = hstack(self.train_exog + [self.train_data])
        test_dataset = hstack(self.test_exog + [self.test_data])
        return train_dataset, test_dataset

    def join_from_to(self, data1, data2, size):
        e = data2.copy()
        e[:size] = data1[:size]
        return e


    def split_sequences(self, sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)

        return array(X), array(y)

    def train_test_split(self, X, y):
        length = len(y)
        tr_start, tr_end = 0, int(length * 0.8)+1 
        te_start, te_end = tr_end, length

        train_X = X[tr_start:tr_end]
        test_X = X[te_start:te_end]

        train_y = y[tr_start:tr_end]
        test_y = y[te_start:te_end]

        return train_X, test_X, train_y, test_y

    def LSTM_forecast(self, trainX, trainy, testX, n_steps, n_features):
        # fit model
        trainX = asarray(trainX)
        trainy = asarray(trainy)
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(trainX, trainy, epochs=200, verbose=0)
        # make a one-step prediction
        yhat = model.predict(testX, verbose=0)
        yhat = np.transpose(yhat)
        yhat = yhat[0]
        return yhat, model

    def split_data(self, dataset, n_steps=1):
        X, y = [], []
        remain_X = []
        remain_y = []
        for i in range(0, len(dataset), n_steps):
            end_idx = i + n_steps
            if end_idx > len(dataset):
                end_idx = len(dataset)
                remain_X, remain_y = dataset[i:end_idx, :-1], dataset[i:end_idx, -1] 
                break

            seq_x, seq_y = dataset[i:end_idx, :-1], dataset[end_idx-1, -1] 
            X.append(seq_x)
            y.append(seq_y)
        return list(map(array, [X, y, remain_X, remain_y]))

    def walk_forward_validation(self, trainX, trainy, testX, testy, n_steps, n_features):
        history_X = [x for x in trainX]
        history_y = [y for y in trainy]
        predictions = list()
        # step over each time-step in the test set
        for test_X, test_y in zip(testX, testy):
            test_X = array([test_X])
            # fit model on history and make a prediction
            yhat, model = self.LSTM_forecast(trainX, trainy, test_X, n_steps, n_features)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history_X.append(test_X)
            history_y.append(test_y)
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (test_y, yhat))
            print('오차율=%.1f ' % (((test_y - yhat) / test_y) * 100))
        # estimate prediction error
        error = mean_absolute_error(testy, predictions)
        return error, testy, predictions, model

    def join_remain_to_test(self, test_X, test_y, remain_X, remain_y):
        remain_X = array([remain_X]).reshape(len(remain_X), 1, len(remain_X[0]))
        test_X = np.concatenate([remain_X, test_X])
        test_y = np.concatenate([test_y, remain_y])

        return test_X, test_y

    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = np.zeros(shape=(abs(n), 1))
            e[n:] = xs[:-n]
        else:
            e[n:] = np.zeros(shape=(abs(n), 1))
            e[:n] = xs[-n:]
        return array(e)

    def last_data(self, xs, n):
        e = np.empty_like(xs)
        e[n:] = np.zeros(shape=(len(xs)-n, 1))
        e[:n] = xs[-n:]      
        return array(e)

    def fit(self):
        n_steps = 3
        train_dataset, test_dataset = self.prepare_data()

        train_X, train_y= map(np.array, self.split_sequences(train_dataset, n_steps))
        test_X, test_y = map(np.array, self.split_sequences(test_dataset, n_steps))

        # model의 학습 데이터 차원을 맞추기 위해
        # block size 보다 작은 데이터는 test 데이터 앞쪽에 추가한다.
        # test_X, test_y = self.join_remain_to_test(test_X, test_y, remain_X, remain_y)
        mae, y, yhat, model = self.walk_forward_validation(train_X, train_y, test_X, test_y, train_X.shape[1], train_X.shape[2])
        
        self.model = model
        self.pred_exog = test_X

        print('MAE: %.3f' % mae)

    # 최대 1년
    def predict(self, n_periods=6):
        exog = self.pred_exog[-n_periods:]
        pred = []
        for e in exog:
            e = array([e])
            yhat = self.model.predict(e, verbose=0)
            yhat = np.transpose(yhat)
            yhat = yhat[0][0]
            pred.append(yhat)

        print('LSTM Done!')
        # return pred.values
        return np.array(pred)