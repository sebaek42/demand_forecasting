import pymysql
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from timeseries_bagging import TS_Bagging

con = pymysql.connect(
        host='localhost', 
        user='root', 
        db='airport', 
        passwd='admin', 
        charset='utf8'
    )

"""
2021-03-03: MySQL 상 데이터 가져와 데이터 프레임으로 정리까지 함
Todo:
Bagging에서 데이터 추세, 계절, 기타로 분해 후 기타 데이터 부트스트랩핑 후 전달
각 모델에 기타 데이터 예측 후 반환

Bagging 에서 반환 받은 데이터에 추세, 계절 더한 결과 반환
"""
def main():
    exog_df, exog_origin_df = load_exog_data('exog_data'), load_exog_data('exog_origin_data')
    passenger_df = load_passenger_data()
    region_arrive_df, region_total_df, region_departure_df = load_region_data()

    # 인천 국제 공항 출발, 도착, 총계 예측
    # for _type in passenger_df.columns:
    #     bagging = TS_Bagging(passenger_df[_type], exog_df)
    #     predict = bagging.forecast()
    #     _date = [passenger_df.index[-1] + relativedelta(months=i) for i in range(1, len(predict)+1)]
    #     save_data(predict, _date, _type)

    # 지역별 인천 국제 공항 출발, 도착, 총계 예측
    # for region_df, _type in zip([region_arrive_df, region_total_df, region_departure_df], ['arrive', 'total', 'departure']):
    #     for region in region_df.columns:
    #         bagging = TS_Bagging(region_df[region], exog_df)
    #         predict = bagging.forecast()
    #         _date = [region_df.index[-1] + relativedelta(months=i) for i in range(1, len(predict)+1)]
    #         save_data(predict, _date, _type, region=region)
    #         break
    bagging = TS_Bagging(passenger_df['total'], exog_df, exog_origin_df)
    bagging.forecast()


def save_data(predict, _date, _type, region=None):
    if region:
        curs = con.cursor()
        insert_sql = """INSERT INTO forecasted_data (date, value, type, region)
                VALUES (%s, %s, %s, %s)"""
        delete_sql = """DELETE FROM forecasted_data WHERE type = %s and region = %s
        """
        curs.execute(delete_sql, (_type, region))
        data = list(zip( _date, predict, [_type]*len(predict), [region] * len(predict)))
        curs.executemany(insert_sql, data)
        con.commit()
    else:
        curs = con.cursor()
        insert_sql = """INSERT INTO forecasted_data (date, value, type)
                VALUES (%s, %s, %s)"""
        delete_sql = """DELETE FROM forecasted_data WHERE type = '{}' and region IS NULL
        """.format(_type)
        curs.execute(delete_sql)
        data = list(zip( _date, predict, [_type]*len(predict)))
        curs.executemany(insert_sql, data)
        con.commit()

def load_region_data():
    # 데이터 프레임 구성
    # COLUMNS: America, China, Japan 등 각 지역
    # INDEX: 연도
    # TUPLE: 해당 년도 승객 데이터
    cursor = con.cursor(pymysql.cursors.DictCursor)
    col_sql = 'SELECT region FROM region_data GROUP BY region ORDER BY region;'
    idx_sql = 'SELECT date FROM region_data GROUP BY date;'
    type_sql = 'SELECT type FROM region_data GROUP BY type;'

    #DataFrame index, columns
    cursor.execute(col_sql)
    cols = [d['region'] for d in cursor.fetchall()]
    cursor.execute(idx_sql)
    idx = [d['date'] for d in cursor.fetchall()]
    cursor.execute(type_sql)
    _types = [d['type'] for d in cursor.fetchall()]
    region_dfs = [pd.DataFrame(index=idx, columns=cols) for _ in range(len(_types))]

    #데이터 저장
    for _type, region_df in zip(_types, region_dfs):
        for col in region_df.columns:
            tuple_sql = 'SELECT value FROM region_data WHERE region = %s and type = %s ORDER BY date;'
            cursor.execute(tuple_sql, [col, _type])
            value = [v['value'] for v in cursor.fetchall()]
            region_df[col] = value
    
    return region_dfs

def load_passenger_data():
    # 데이터 프레임 구성
    # COLUMNS: Arrive, Departure, Total
    # INDEX: 연도
    # TUPLE: 해당 년도 승객 데이터
    cursor = con.cursor(pymysql.cursors.DictCursor)
    col_sql = 'SELECT type FROM passenger_data GROUP BY type;'
    idx_sql = 'SELECT date FROM passenger_data GROUP BY date;'
    
    #DataFrame index, columns
    cursor.execute(col_sql)
    cols = [d['type'] for d in cursor.fetchall()]
    cursor.execute(idx_sql)
    idx = [d['date'] for d in cursor.fetchall()]
    passenger_df = pd.DataFrame(index=idx, columns=cols)

    #데이터 저장
    for col in passenger_df.columns:
        tuple_sql = 'SELECT value FROM passenger_data WHERE type = %s ORDER BY date;'
        cursor.execute(tuple_sql, col)
        value = [v['value'] for v in cursor.fetchall()]
        passenger_df[col] = value

    return passenger_df

def load_exog_data(table):
    # 데이터 프레임 구성
    # COLUMNS: 각 데이터 name
    # INDEX: 연도
    # TUPLE: 해당 년도 데이터의 값
    cursor = con.cursor(pymysql.cursors.DictCursor)
    col_sql = 'SELECT name FROM {} GROUP BY name;'.format(table)
    idx_sql = 'SELECT date FROM {} GROUP BY date;'.format(table)  
    
    #DataFrame index, columns
    cursor.execute(col_sql)
    cols = [d['name'] for d in cursor.fetchall()]
    cursor.execute(idx_sql)
    idx = [d['date'] for d in cursor.fetchall()]
    exog_df = pd.DataFrame(index=idx, columns=cols)
        
    #데이터 저장
    for col in exog_df.columns:
        tuple_sql = 'SELECT value FROM {} WHERE name = %s ORDER BY date;'.format(table)
        cursor.execute(tuple_sql, col)
        value = [v['value'] for v in cursor.fetchall()]
        exog_df[col] = value
    
    return exog_df

if __name__ == "__main__":
    main()