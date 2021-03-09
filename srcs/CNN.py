import os
import datetime
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import hstack
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot

m1 = '/Users/baegseungho/tmp_x/월_나스닥지수.xlsx'
m2 = '/Users/baegseungho/tmp_x/월_민간소비증감율.xlsx'
m3 = '/Users/baegseungho/tmp_x/월_소비자물가지수.xlsx'
m4 = '/Users/baegseungho/tmp_x/월_수출액.xlsx'
m5 = '/Users/baegseungho/tmp_x/월_실업률.xlsx'
m6 = '/Users/baegseungho/tmp_x/월_싱가포르항공유.xlsx'
m7 = '/Users/baegseungho/tmp_x/월_일본연휴수.xlsx'
m8 = '/Users/baegseungho/tmp_x/월_중국연휴수.xlsx'
m9 = '/Users/baegseungho/tmp_x/월_코스피.xlsx'
m10 = '/Users/baegseungho/tmp_x/월_한국연휴수.xlsx'
m11 = '/Users/baegseungho/tmp_x/월_환율.xlsx'
m12 = '/Users/baegseungho/tmp_x/월_전세계코로나신규확진자.xlsx'

# m1 = '/Users/baegseungho/normalized/월_나스닥지수.xlsx'
# m2 = '/Users/baegseungho/normalized/월_민간소비증감율.xlsx'
# m3 = '/Users/baegseungho/normalized/월_소비자물가지수.xlsx'
# m4 = '/Users/baegseungho/normalized/월_수출액.xlsx'
# m5 = '/Users/baegseungho/normalized/월_실업률.xlsx'
# m6 = '/Users/baegseungho/normalized/월_싱가포르항공유.xlsx'
# m7 = '/Users/baegseungho/normalized/월_일본연휴수.xlsx'
# m8 = '/Users/baegseungho/normalized/월_중국연휴수.xlsx'
# m9 = '/Users/baegseungho/normalized/월_코스피.xlsx'
# m10 = '/Users/baegseungho/normalized/월_한국연휴수.xlsx'
# m11 = '/Users/baegseungho/normalized/월_환율.xlsx'
# m12 = '/Users/baegseungho/normalized/월_전세계코로나신규확진자.xlsx'
y1 = '/Users/baegseungho/tmp_y/월_전체승객.xlsx'



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
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

def prepare_data():
    df = pd.read_excel(m1, engine='openpyxl')
    data = np.array(df)
    in_seq1 = data[:,1]
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    df = pd.read_excel(m2, engine='openpyxl')
    data = np.array(df)
    in_seq2 = data[:,1]
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    df = pd.read_excel(m3, engine='openpyxl')
    data = np.array(df)
    in_seq3 = data[:,1]
    in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    df = pd.read_excel(m4, engine='openpyxl')
    data = np.array(df)
    in_seq4 = data[:,1]
    in_seq4 = in_seq4.reshape((len(in_seq4), 1))
    df = pd.read_excel(m5, engine='openpyxl')
    data = np.array(df)
    in_seq5 = data[:,1]
    in_seq5 = in_seq5.reshape((len(in_seq5), 1))
    df = pd.read_excel(m6, engine='openpyxl')
    data = np.array(df)
    in_seq6 = data[:,1]
    in_seq6 = in_seq6.reshape((len(in_seq6), 1))
    df = pd.read_excel(m7, engine='openpyxl')
    data = np.array(df)
    in_seq7 = data[:,1]
    in_seq7 = in_seq7.reshape((len(in_seq7), 1))
    df = pd.read_excel(m8, engine='openpyxl')
    data = np.array(df)
    in_seq8 = data[:,1]
    in_seq8 = in_seq8.reshape((len(in_seq8), 1))
    df = pd.read_excel(m9, engine='openpyxl')
    data = np.array(df)
    in_seq9 = data[:,1]
    in_seq9 = in_seq9.reshape((len(in_seq9), 1))
    df = pd.read_excel(m10, engine='openpyxl')
    data = np.array(df)
    in_seq10 = data[:,1]
    in_seq10 = in_seq10.reshape((len(in_seq10), 1))
    df = pd.read_excel(m11, engine='openpyxl')
    data = np.array(df)
    in_seq11 = data[:,1]
    in_seq11 = in_seq11.reshape((len(in_seq11), 1))
    df = pd.read_excel(m12, engine='openpyxl')
    data = np.array(df)
    in_seq12 = data[:,1]
    in_seq12 = in_seq12.reshape((len(in_seq12), 1))
    df = pd.read_excel(y1, engine='openpyxl')
    data = np.array(df)
    out_seq = data[:,1]
    out_seq = out_seq.reshape((len(out_seq), 1))
    dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, in_seq6, in_seq7, in_seq8, in_seq9, in_seq10, in_seq11, in_seq12, out_seq))
    return dataset

def prepare_data_normalized():
    df = pd.read_excel(m1, engine='openpyxl')
    data = np.array(df)
    in_seq1 = data[:,1]
    norm = np.linalg.norm(in_seq1)
    in_seq1 = in_seq1/norm
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))

    df = pd.read_excel(m2, engine='openpyxl')
    data = np.array(df)
    in_seq2 = data[:,1]
    norm = np.linalg.norm(in_seq2)
    in_seq2 = in_seq2/norm
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))

    df = pd.read_excel(m3, engine='openpyxl')
    data = np.array(df)
    in_seq3 = data[:,1]
    norm = np.linalg.norm(in_seq3)
    in_seq3 = in_seq3/norm
    in_seq3 = in_seq3.reshape((len(in_seq3), 1))

    df = pd.read_excel(m4, engine='openpyxl')
    data = np.array(df)
    in_seq4 = data[:,1]
    norm = np.linalg.norm(in_seq4)
    in_seq4 = in_seq4/norm
    in_seq4 = in_seq4.reshape((len(in_seq4), 1))

    df = pd.read_excel(m5, engine='openpyxl')
    data = np.array(df)
    in_seq5 = data[:,1]
    norm = np.linalg.norm(in_seq5)
    in_seq5 = in_seq5/norm
    in_seq5 = in_seq5.reshape((len(in_seq5), 1))

    df = pd.read_excel(m6, engine='openpyxl')
    data = np.array(df)
    in_seq6 = data[:,1]
    norm = np.linalg.norm(in_seq6)
    in_seq6 = in_seq6/norm
    in_seq6 = in_seq6.reshape((len(in_seq6), 1))

    df = pd.read_excel(m7, engine='openpyxl')
    data = np.array(df)
    in_seq7 = data[:,1]
    norm = np.linalg.norm(in_seq7)
    in_seq7 = in_seq7/norm
    in_seq7 = in_seq7.reshape((len(in_seq7), 1))

    df = pd.read_excel(m8, engine='openpyxl')
    data = np.array(df)
    in_seq8 = data[:,1]
    norm = np.linalg.norm(in_seq8)
    in_seq8 = in_seq1/norm
    in_seq8 = in_seq8.reshape((len(in_seq8), 1))

    df = pd.read_excel(m9, engine='openpyxl')
    data = np.array(df)
    in_seq9 = data[:,1]
    norm = np.linalg.norm(in_seq9)
    in_seq9 = in_seq9/norm
    in_seq9 = in_seq9.reshape((len(in_seq9), 1))

    df = pd.read_excel(m10, engine='openpyxl')
    data = np.array(df)
    in_seq10 = data[:,1]
    norm = np.linalg.norm(in_seq10)
    in_seq10 = in_seq10/norm
    in_seq10 = in_seq10.reshape((len(in_seq10), 1))

    df = pd.read_excel(m11, engine='openpyxl')
    data = np.array(df)
    in_seq11 = data[:,1]
    norm = np.linalg.norm(in_seq11)
    in_seq11 = in_seq11/norm
    in_seq11 = in_seq11.reshape((len(in_seq11), 1))

    df = pd.read_excel(m12, engine='openpyxl')
    data = np.array(df)
    in_seq12 = data[:,1]
    norm = np.linalg.norm(in_seq12)
    in_seq12 = in_seq12/norm
    in_seq12 = in_seq12.reshape((len(in_seq12), 1))

    df = pd.read_excel(y1, engine='openpyxl')
    data = np.array(df)
    out_seq = data[:,1]
    out_seq = out_seq.reshape((len(out_seq), 1))
    dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, in_seq6, in_seq7, in_seq8, in_seq9, in_seq10, in_seq11, in_seq12, out_seq))
    return dataset

def CNN_forecast(trainX, trainy, testX, n_steps, n_features):
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

def walk_forward_validation(trainX, trainy, testX, testy, n_steps, n_features):
    predictions = list()
    history_x = [x for x in trainX]
    history_y = [y for y in trainy]
    # step over each time-step in the test set
    for i in range(len(testX)):
        # split test row into input and output columns
        test_X, test_y = testX[i:i+1], testy[i:i+1]
        # fit model on history and make a prediction
        yhat = CNN_forecast(history_x, history_y, test_X, n_steps, n_features)
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

#############################################################
dataset = prepare_data()
# dataset = prepare_data_normalized()
#############################################################
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)

n_features = X.shape[2]

# train, test split 맨뒤 n개 셋을 테스트로
n = 18
trainX = X[0:-n].astype(float)
trainy = y[0:-n].astype(float)
testX = X[-n:].astype(float)
testy = y[-n:].astype(float)
# walk_forward_validation으로 매번 학습마다 학습데이터 업데이트
mae, y, yhat = walk_forward_validation(trainX, trainy, testX, testy, n_steps, n_features)
print('MAE: %.3f' % mae)

pyplot.plot(testy, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()
