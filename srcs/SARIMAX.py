import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

class Sarimax:
    def __init__(self, train_data, test_data, train_exog, test_exog):
        self.train_data = train_data
        self.test_data = test_data
        self.train_exog = train_exog
        self.test_exog = test_exog
    
        self.train_exog['train_data_shifted_1'] = self.shift(train_data, 1)
        
        test_data_shifted_1 = self.shift(test_data, 1)
        test_data_shifted_1[0] = train_data[-1]
        self.test_exog['test_data_shifted_1'] = test_data_shifted_1
    
    def shift(self, xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = np.zeros(shape=(abs(n), 1))
            e[n:] = xs[:-n]
        else:
            e[n:] = np.zeros(shape=(abs(n), 1))
            e[:n] = xs[-n:]
        return np.array(e)

    def auto_sarimax(self):
        # 데이터에 따른 최적 AR, MA의 차수 및 계절 차수 구함
        model = pm.auto_arima(self.train_data, self.train_exog, seasonal=True, m=12, d=1,
                                start_p=0, start_q=0, max_p=7, max_q=7)

        pred, conf = self.update_model(model)
        # self.print_result(pred, self.test_data)

        return model

    def update_model(self, model):
        pred = []
        # 신뢰 구간
        confidence_intervals = []

        for new_ob, exog_idx in zip(self.test_data, self.test_exog.index):
            # 1 period 씩 예측을 수행한다.
            exog_df = pd.DataFrame(self.test_exog.loc[exog_idx]).T
            fc, conf = self.forecast_one_step(model, exog_df)
            pred.append(fc)
            confidence_intervals.append(conf)

            # Updates the existing model with a small number of MLE steps
            print('>expected=%.1f, predicted=%.1f' % (new_ob, fc))
            print('오차율=%.1f ' % (((new_ob - fc) / new_ob) * 100))
            model.update(new_ob, exog_df)
        
        return pred, confidence_intervals

    def forecast_one_step(self, model, new_exog):
        fc, conf_int = model.predict(n_periods=1, X=new_exog, return_conf_int=True)
        return (
            fc.tolist()[0],
            np.asarray(conf_int).tolist()[0])     

    def print_result(self, pred, test):    
        pred = pd.Series(pred, index=test.index)
        pd.DataFrame({'test':test,'pred':pred}).plot()
        plt.show()
        print(round(((pred-test)/test)*100, 2))


    def fit(self):
        self.model = self.auto_sarimax()

    # 최대 1년
    def predict(self, n_periods=6):
        pred_exog = self.test_exog.iloc[-n_periods:]
        pred = []
        for i in range(n_periods):
            yhat = self.model.predict(n_periods=1, X=pred_exog.iloc[i].to_frame().transpose())
            pred.append(yhat)
        print("SARIMAX Done!")
        return np.array(pred)