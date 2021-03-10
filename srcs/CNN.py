import numpy as np
from numpy import array
from numpy import asarray
from numpy import hstack
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
import pandas as pd
import matplotlib.pyplot as plt

class Cnn:
    def __init__(self, train_resid, test_resid, train_other, test_other, train_exog, test_exog):
        self.train_resid = train_resid
        self.test_resid = test_resid
        self.train_other = train_other
        self.test_other = test_other
        self.train_exog = train_exog
        self.test_exog = test_exog
    
    def prepare_data(self):
        exog = self.train_exog.append(self.test_exog)
        resid = self.train_resid.append(self.test_resid)
        other = self.train_other.append(self.test_other)
        exog_reshape = [exog[col].values.reshape(len(exog[col]), 1) for col in self.train_exog.columns]
        resid = resid.values.reshape(len(resid), 1)
        other = other.values.reshape(len(other), 1)
        
        dataset = hstack(exog_reshape + [resid])
        return dataset


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

    def walk_forward_validation(self, trainX, trainy, testX, testy, n_steps, n_features):
        predictions = list()
        history_x = [x for x in trainX]
        history_y = [y for y in trainy]
        # step over each time-step in the test set
        for i in range(len(testX)):
            # split test row into input and output columns
            test_X, test_y = testX[i:i+1], testy[i:i+1]
            # fit model on history and make a prediction
            yhat = self.CNN_forecast(history_x, history_y, test_X, n_steps, n_features)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history_x.append(test_X[0])
            history_y.append(test_y[0])
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (test_y, yhat))
            print('오차율=%.1f ' % (((test_y - yhat) / test_y) * 100))
        # estimate prediction error
        error = mean_absolute_error(testy, predictions)
        return error, testy, predictions

    def CNN_forecast(self, trainX, trainy, testX, n_steps, n_features):
        trainX = asarray(trainX)
        trainy = asarray(trainy)
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(trainX, trainy, epochs=1000, verbose=0)
        yhat = model.predict(testX, verbose=0)
        yhat = np.transpose(yhat)
        yhat = yhat[0]
        return yhat
    def cnn_predict(self):
        n_steps = 3
        dataset = self.prepare_data()
        X, y = self.split_sequences(dataset, n_steps)
        n_features = X.shape[2]

        train_X, test_X, train_y, test_y = self.train_test_split(X, y)
        mae, y, yhat = self.walk_forward_validation(train_X, train_y, test_X, test_y, n_steps, n_features)

        print('MAE: %.3f' % mae)
        plt.plot(test_y, label='Expected')
        plt.plot(yhat, label='Predicted')
        plt.legend()
        plt.show()