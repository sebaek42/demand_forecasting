import os
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine

# MySQL Connector using pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

engine = create_engine("mysql+mysqldb://root:"+"admin"+"@localhost/airport", encoding='utf-8')
#engine = create_engine("mysql+mysqldb://사용자이름:"+"비밀번호"+"@localhost/DB이름", encoding='utf-8')

conn = engine.connect()

def region_to_sql():
    path = './data/forecasting_data/region_data/'
    region_files = os.listdir(path)
    region_df = pd.DataFrame(columns=['date', 'region', 'type', 'value'])
    for region_file in region_files:
        cur_df = pd.DataFrame(columns=['date', 'region', 'type', 'value'])
        data = pd.read_excel(path+region_file)
        _type = ''
        if '입국' in region_file:
            _type = 'arrive'
        elif '출국' in region_file:
            _type = 'departure'
        else:
            _type = 'total'
        for region in data.columns[1:]:
            temp = pd.DataFrame(columns=['date', 'region', 'type', 'value'])
            temp['date'] = data.date
            temp['value'] = data[region]
            temp['region'] = region
            temp['type'] = _type
            cur_df = pd.concat([cur_df, temp], ignore_index=True, axis=0)
        region_df = pd.concat([region_df, cur_df], ignore_index=True, axis=0)

    region_df.to_sql(name='region_data', if_exists='append', con=engine, index=False)
        

def passenger_to_sql():
    path = './data/forecasting_data/passenger_data/'
    passenger_files = os.listdir(path)
    passenger_df = pd.DataFrame(columns=['date', 'type', 'value'])

    for passenger_file in passenger_files:
        cur_df = pd.DataFrame(columns=['date', 'type', 'value'])
        data = pd.read_excel(path+passenger_file)
        _type = data.columns[1]
        cur_df['date'] = data.date
        cur_df['value'] = data[_type]
        cur_df['type'] = _type

        passenger_df = pd.concat([passenger_df, cur_df], ignore_index=True, axis=0)

    passenger_df.to_sql(name='passenger_data', if_exists='append', con=engine, index=False)

def exog_to_sql():
    path = './data/forecasting_data/exog_data/'
    exog_files = os.listdir(path)
    exog_df = pd.DataFrame(columns=['date', 'name', 'value'])

    for exog_file in exog_files:
        cur_df = pd.DataFrame(columns=['date', 'name', 'value'])
        data = pd.read_excel(path + exog_file, usecols=[0, 1])
        name = (exog_file.split('.')[0]).split('_')[1]
        cur_df['date'] = data.date
        cur_df['name'] = name
        cur_df['value'] = data['특징값']

        exog_df = pd.concat([exog_df, cur_df], axis=0, ignore_index=True)
    
    exog_df.to_sql(name='exog_data', con=engine, if_exists='append', index=False)  