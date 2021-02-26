# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Data
#
# ### 미주, 중국, 일본, 동북아시아, 동남아시아, 중동, 유럽, 오세아니아, 기타 입국, 출국 승객 데이터
#
# ### Temporory Effects
#
# 1. 국내외 BTS 콘서트
# 2. 국내안전요인: 북한 대남도발, 자연재해
# 3. 해외안전요인: 주요 국(중국, 일본, 미국, 대만 등) 자연재해 및 테러, 시위 등
# 4. 국제팬데믹: 사스, 메르스, 코로나
# 5. 주요국 연휴 수: 중국, 일본, 한국 등 성수기(1월, 2월, 7월, 8월) 제외
#                    중국 - 7일, 일본 - 영향 없음, 한국 - 4일 
# 6. 경제적요소: 민간소비증감율 - 9% 이상 증가, 소비자물가지수 - (-0.5) 이상 하락, 환율 (+)9 이상 급등, (-)6 이상 급락 
# 7. 현재 미정: 유가, 싱가포르 항공유 >> 공항 수요 자체가 유가에 탄력적이지 않음
#
# ### Permenent Effects
#
# 없음

# # Import Modules

import os
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='IPAGothic')
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from pmdarima.arima.stationarity import ADFTest
from pmdarima.arima import ndiffs
from pmdarima.metrics import smape
from sklearn.metrics import mean_squared_error
import pickle

# # Reading Data Set

# +
passenger_arrive = pd.read_excel('./data/region_data/월_승객_입국.xlsx', engine='openpyxl', index_col=[0])
passenger_depart = pd.read_excel('./data/region_data/월_승객_출국.xlsx', engine='openpyxl', index_col=[0])
passenger_total = pd.read_excel('./data/region_data/월_승객_총계.xlsx', engine='openpyxl', index_col=[0])


bts_consert = pd.read_excel('./data/월_BTS콘서트.xlsx', engine='openpyxl', usecols=[0, 1], index_col=[0])
domestic_secure = pd.read_excel('./data/월_국내안전요인.xlsx', engine='openpyxl', usecols=[0, 1], index_col=[0])
abroad_secure = pd.read_excel('./data/월_국외안전요인.xlsx', engine='openpyxl', usecols=[0, 1], index_col=[0])
pandemic = pd.read_excel('./data/월_국제팬데믹.xlsx', engine='openpyxl', usecols=[0, 1], index_col=[0])
china_holidays = pd.read_excel('./data/월_중국연휴수.xlsx', engine='openpyxl', usecols=[0, 1], index_col=[0])
korea_holidays = pd.read_excel('./data/월_한국연휴수.xlsx', engine='openpyxl', usecols=[0, 1], index_col=[0])
private_consume = pd.read_excel('./data/월_민간소비증감율.xlsx', engine='openpyxl', usecols=[0, 1], index_col=[0])
consumer_price_index = pd.read_excel('./data/월_소비자물가지수.xlsx', engine='openpyxl', usecols=[0, 2], index_col=[0])
change_rate = pd.read_excel('./data/월_환율.xlsx', engine='openpyxl', usecols=[0, 2], index_col=[0])
# -



# # Train & Test Split

# +
# Train : Test = 8 : 2
tr_start, tr_end = '2003-01', '2017-06'
te_start, te_end = '2017-07', '2020-12'

buf_arrive = passenger_arrive.copy()
buf_depart = passenger_depart.copy()
buf_total = passenger_total.copy()

train_arrive = buf_arrive[tr_start:tr_end]
train_depart = buf_depart[tr_start:tr_end]
train_total = buf_total[tr_start:tr_end]

test_arrive = buf_arrive[te_start:te_end]
test_depart = buf_depart[te_start:te_end]
test_total = buf_total[te_start:te_end]

# -

# # SARIMA Model

# +
sarima_arrive = {}
sarima_depart = {}
sarima_total = {}

for region in buf_arrive.columns:
    print('@@@@@{}@@@@@'.format(region))
    print("***** ARRIVE *****\n")
    sarima_arrive[region] = auto_sarima(train_arrive[region])
    print("***** DEPART *****\n")
    sarima_depart[region] = auto_sarima(train_depart[region])
    print("***** TOTAL *****\n")
    sarima_total[region] = auto_sarima(train_total[region])
    
# -


# Save Models
for region in buf_arrive.columns:
    with open('./models/sarima_arrive_{}.pkl'.format(region), 'wb') as arrive_pkl:
        pickle.dump(sarima_arrive[region], arrive_pkl)

    with open('./models/sarima_depart_{}.pkl'.format(region), 'wb') as depart_pkl:
        pickle.dump(sarima_depart[region], depart_pkl)

    with open('./models/sarima_total_{}.pkl'.format(region), 'wb') as total_pkl:
        pickle.dump(sarima_total[region], total_pkl)









# +
# Load Models
sarima_arrive = {}
sarima_depart = {}
sarima_total = {}

for region in buf_arrive.columns:
    with open('./models/sarima_arrive_{}.pkl'.format(region), 'rb') as arrive_pkl:
        sarima_arrive[region] = pickle.load(arrive_pkl).fit(train_arrive[region])

    with open('./models/sarima_depart_{}.pkl'.format(region), 'rb') as depart_pkl:
        sarima_depart[region] = pickle.load(depart_pkl).fit(train_depart[region])

    with open('./models/sarima_total_{}.pkl'.format(region), 'rb') as total_pkl:
        sarima_total[region] = pickle.load(total_pkl).fit(train_total[region])


# +
# Predict
sarima_arrive_pred = {}
sarima_depart_pred = {}
sarima_total_pred = {}

sarima_arrive_conf = {}
sarima_depart_conf = {}
sarima_total_conf = {}

for region in buf_arrive.columns:
    # Arrive
    arrive_pred, arrive_conf = update_model(sarima_arrive[region], test_arrive[region])
    sarima_arrive_pred[region] = arrive_pred
    sarima_arrive_conf[region] = arrive_conf
    # Depart
    depart_pred, depart_conf = update_model(sarima_depart[region], test_depart[region])
    sarima_depart_pred[region] = depart_pred
    sarima_depart_conf[region] = depart_conf
    # total
    total_pred, total_conf = update_model(sarima_total[region], test_total[region])
    sarima_total_pred[region] = total_pred
    sarima_total_conf[region] = total_conf
# -

for region in buf_arrive.columns:
    print_result(sarima_arrive_pred[region], test_arrive[region])

for region in buf_depart.columns:
    print_result(sarima_depart_pred[region], test_depart[region])

for region in buf_total.columns:
    print_result(sarima_total_pred[region], test_total[region])

# +
# bound_private_consume = 9
bound_china_holidays = 7
bound_korea_holidays = 4
lower_bound_chage = -6
upper_bound_chage = 9
lower_bound_consumer_price_index = -0.5 
bound_bts_concert = 1


# 데이터 변환
trans_value(korea_holidays, upper_bound=bound_korea_holidays)
trans_value(china_holidays, upper_bound=bound_china_holidays)
trans_value(consumer_price_index, lower_bound=lower_bound_consumer_price_index)
trans_value(change_rate, upper_bound=upper_bound_chage, lower_bound=lower_bound_chage)
trans_value(bts_consert, upper_bound=bound_bts_concert)

# -

# # SARIMAX Model

# +
files = [file.split('.')[0] for file in os.listdir('./data/slided')]

slied_data = {}

for file in files:
    slied_data[file.split('_')[1]] = pd.read_excel('./data/slided/' + file +".xlsx", engine='openpyxl', index_col=[0])
    

incheon_arrive = pd.read_excel('./data/월_입국승객.xlsx', engine='openpyxl', index_col=[0])
incheon_depart = pd.read_excel('./data/월_출국승객.xlsx', engine='openpyxl', index_col=[0])
incheon_total = pd.read_excel('./data/월_전체승객.xlsx', engine='openpyxl', index_col=[0])

# Train : Test = 8 : 2
tr_start, tr_end = '2003-01', '2017-06'
te_start, te_end = '2017-07', '2020-12'

buf['arrive'] = incheon_arrive.copy()
buf['depart'] = incheon_depart.copy()
buf['total'] = incheon_total.copy()

buf_exog = incheon_arrive.copy()

exogs = ['korea_holidays' ,'china_holidays' ,'consumer_price_index' ,'exchage_rate' ,
         'bts_consert' ,'domestic_secure' ,'abroad_secure' ,'pandemic']

buf_exog['korea_holidays'] = korea_holidays.values.copy()
buf_exog['china_holidays'] = china_holidays.values.copy()
buf_exog['bts_consert'] = bts_consert.values.copy()
buf_exog['domestic_secure'] = domestic_secure.values.copy()
buf_exog['abroad_secure'] = abroad_secure.values.copy()
buf_exog['pandemic'] = pandemic.values.copy()

buf_exog = buf_exog.drop(['arrive'], axis=1)


for key, value in slied_data.items():
    value.index = buf.index
    buf_exog[key] = value.copy()


train = buf[tr_start:tr_end].dropna()
test = buf[te_start:te_end].dropna()

train_exog = buf_exog[tr_start:tr_end].dropna()
test_exog = buf_exog[te_start:te_end].dropna()
# -

print("***** ARRIVE *****\n")
sarimax_arrive = auto_sarimax(train['arrive'], train_exog)
print("***** DEPART *****\n")
sarimax_depart = auto_sarimax(train['depart'], train_exog)
print("***** TOTAL *****\n")
sarimax_total = auto_sarimax(train['total'], train_exog)

# +
# Save Models

with open('./models/sarimax_arrive.pkl', 'wb') as pkl:
    pickle.dump(sarimax_arrive, pkl)

with open('./models/sarimax_depart.pkl', 'wb') as pkl:
    pickle.dump(sarimax_depart, pkl)

with open('./models/sarimax_total.pkl', 'wb') as pkl:
    pickle.dump(sarimax_total, pkl)

# +
# Load Models
sarimax_arrive = pm.ARIMA(order=(0, 0, 0))
sarimax_depart = pm.ARIMA(order=(0, 0, 0))
sarimax_total = pm.ARIMA(order=(0, 0, 0))

with open('./models/sarimax_arrive.pkl', 'rb') as arrive_pkl:
    sarimax_arrive = pickle.load(arrive_pkl).fit(train['arrive'], train_exog)

with open('./models/sarimax_depart.pkl', 'rb') as depart_pkl:
    sarimax_depart = pickle.load(depart_pkl).fit(train['depart'], train_exog)

with open('./models/sarimax_total.pkl', 'rb') as total_pkl:
    sarimax_total = pickle.load(total_pkl).fit(train['total'], train_exog)
# +
sarimax_arrive_pred = []
sarimax_depart_pred = []
sarimax_total_pred = []

sarimax_arrive_conf = []
sarimax_depart_conf = []
sarimax_total_conf = []

sarimax_arrive_pred, sarimax_arrive_conf = update_exog_model(sarimax_arrive, test['arrive'], test_exog)
sarimax_depart_pred, sarimax_depart_conf = update_exog_model(sarimax_depart, test['depart'], test_exog)
sarimax_total_pred, sarimax_total_conf = update_exog_model(sarimax_total, test['total'], test_exog)
# -


print_sarimax_result(sarimax_arrive_pred, test['arrive'])

print_sarimax_result(sarimax_depart_pred, test['depart'])

print_sarimax_result(sarimax_total_pred, test['total'])

# ## Normalized

# +
files = [file.split('.')[0] for file in os.listdir('./data/normalized')]

slied_data = {}

for file in files:
    slied_data[file.split('_')[1]] = pd.read_excel('./data/normalized/' + file +".xlsx", engine='openpyxl', index_col=[0])
    

incheon_arrive = pd.read_excel('./data/월_입국승객.xlsx', engine='openpyxl', index_col=[0])
incheon_depart = pd.read_excel('./data/월_출국승객.xlsx', engine='openpyxl', index_col=[0])
incheon_total = pd.read_excel('./data/월_전체승객.xlsx', engine='openpyxl', index_col=[0])

# Train : Test = 8 : 2
tr_start, tr_end = '2003-01', '2017-06'
te_start, te_end = '2017-07', '2020-12'

buf = incheon_arrive.copy()
buf['depart'] = incheon_depart.copy()
buf['total'] = incheon_total.copy()

buf_exog = incheon_arrive.copy()

exogs = ['korea_holidays' ,'china_holidays' ,'consumer_price_index' ,'exchage_rate' ,
         'bts_consert' ,'domestic_secure' ,'abroad_secure' ,'pandemic']

buf_exog['korea_holidays'] = korea_holidays.values.copy()
buf_exog['china_holidays'] = china_holidays.values.copy()
buf_exog['bts_consert'] = bts_consert.values.copy()
buf_exog['domestic_secure'] = domestic_secure.values.copy()
buf_exog['abroad_secure'] = abroad_secure.values.copy()
buf_exog['pandemic'] = pandemic.values.copy()

buf_exog = buf_exog.drop(['arrive'], axis=1)


for key, value in slied_data.items():
    value.index = buf.index
    buf_exog[key] = value.copy()


train = buf[tr_start:tr_end].dropna()
test = buf[te_start:te_end].dropna()

train_exog = buf_exog[tr_start:tr_end].dropna()
test_exog = buf_exog[te_start:te_end].dropna()
# -

print("***** ARRIVE *****\n")
sarimax_arrive = auto_sarimax(train['arrive'], train_exog)
print("***** DEPART *****\n")
sarimax_depart = auto_sarimax(train['depart'], train_exog)
print("***** TOTAL *****\n")
sarimax_total = auto_sarimax(train['total'], train_exog)

# +
sarimax_arrive_pred = []
sarimax_depart_pred = []
sarimax_total_pred = []

sarimax_arrive_conf = []
sarimax_depart_conf = []
sarimax_total_conf = []

sarimax_arrive_pred, sarimax_arrive_conf = update_exog_model(sarimax_arrive, test['arrive'], test_exog)
sarimax_depart_pred, sarimax_depart_conf = update_exog_model(sarimax_depart, test['depart'], test_exog)
sarimax_total_pred, sarimax_total_conf = update_exog_model(sarimax_total, test['total'], test_exog)
# -

print_sarimax_result(sarimax_arrive_pred, test['arrive'])

print_sarimax_result(sarimax_depart_pred, test['depart'])

print_sarimax_result(sarimax_total_pred, test['total'])


# +
def auto_sarima(data):
    sarima_model = pm.auto_arima(data, seasonal=True, m=12, test='adf',
                      start_p=0, start_q=0, max_p=7, max_d=2, max_q=7,
                      start_P=0, start_Q=0, max_P=7, max_D=1, max_Q=7)
    
    print(sarima_model)
    print(sarima_model.summary())
    
    return sarima_model

def auto_sarimax(passenger_data, exog_data):
    sarimax_model = pm.auto_arima(passenger_data, exog_data, d=1, seasonal=True, m=12, test='adf',
                      start_p=0, start_q=0, max_p=7, max_d=2, max_q=7,
                      start_P=0, start_Q=0, max_P=7, max_D=1, max_Q=7)
    print(sarimax_model)
    print(sarimax_model.summary())
    
    return sarimax_model


def update_model(model, tes):
    pred = []
    # 신뢰 구간
    confidence_intervals = []

    for new_ob in tes:
        # 1 period 씩 예측을 수행한다.
        fc, conf = forecast_one_step(model)
        pred.append(fc)
        confidence_intervals.append(conf)

        # Updates the existing model with a small number of MLE steps
        model.update(new_ob)
    
    return pred, confidence_intervals

    
def forecast_one_step(model):
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])

def print_result(pred, test):    
    pred = pd.Series(pred, index=test.index)
    pd.DataFrame({'test':test,'pred':pred}).plot()
    plt.show()
    print(round(((pred-test)/test)*100, 2))

        
def trans_value(data, upper_bound=None, lower_bound=None):
    if upper_bound == None and lower_bound == None:
        return
    col = data.columns[0]
    if upper_bound == None:
        for i in range(len(data)):
            if data[col][i] <= lower_bound:
                data[col][i] = 1
            else:
                data[col][i] = 0
                
    elif lower_bound == None:
        for i in range(len(data)):
            if data[col][i] >= upper_bound:
                data[col][i] = 1
                
            else:
                data[col][i] = 0
    else:
        for i in range(len(data)):
            if data[col][i] >= upper_bound or data[col][i] <= lower_bound:
                data[col][i] = 1
            else:
                data[col][i] = 0
    data = data.astype('int')
    
def update_exog_model(model, test, exog_test):
    pred = []
    confidence_intervals = []

    for new_ob, exog_idx in zip(test, exog_test.index):
        exog_df = pd.DataFrame(exog_test.loc[exog_idx]).T
        fc, conf = forecast_exog_one_step(model, exog_df)
        pred.append(fc)
        confidence_intervals.append(conf)
#         print(new_ob, exog_df)
        # Updates the existing model with a small number of MLE steps
        model.update(new_ob,exog_df)
    
    return pred, confidence_intervals
    
def forecast_exog_one_step(model, exog):
    fc, conf_int = model.predict(n_periods=1, X=exog, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])


# -
# # Bagging

# +
from arch.bootstrap import MovingBlockBootstrap, CircularBlockBootstrap, StationaryBootstrap, optimal_block_length

stationary_size, circular_size = int(optimal_block_length(train['total'])['stationary'] + 0.5), int(optimal_block_length(train['total'])['circular']+0.5)


# +
train_decomposed = sm.tsa.seasonal_decompose(train['total'], freq=12, extrapolate_trend='freq')

train_seasonal = train_decomposed.seasonal
train_trend = train_decomposed.trend
train_resid = train_decomposed.resid

test_decomposed = sm.tsa.seasonal_decompose(test['total'], freq=12, extrapolate_trend='freq')

test_seasonal = test_decomposed.seasonal
test_trend = test_decomposed.trend
test_resid = test_decomposed.resid


# +
bs = MovingBlockBootstrap(circular_size, train_resid)

trains_resam = [data[0][0] for data in bs.bootstrap(10)]
# -

preds = []
for train_resam in trains_resam:
    tmp_model = pm.auto_arima(train_resam, start_p=0, start_q=0, max_p=7, max_q=7)
    tmp_pred, tmp_conf = map(np.array, update_model(tmp_model, test_resid))
    tmp_pred += (test_seasonal + test_trend)
    preds.append(tmp_pred)

pred = (sum(preds) / len(trains_resam))

print_result(pred, test['total'])


