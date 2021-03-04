import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

class Sarimax:
    def __init__(self, train_resid, test_resid, train_other, test_other, train_exog, test_exog):
        self.train_resid = train_resid
        self.test_resid = test_resid
        self.train_other = train_other
        self.test_other = test_other
        self.train_exog = train_exog
        self.test_exog = test_exog
    
    def auto_sarimax(self):
        print("---train model---")
        model = pm.auto_arima(self.train_resid, self.train_exog,
                                start_p=0, start_q=0, max_p=7, max_q=7)

        
        pred_resid, conf = self.update_model(model)

        pred = pred_resid + self.test_other
        test = self.test_other + self.test_resid
        self.print_result(pred, test)

        return model


    def update_model(self, model):
        pred = []
        # 신뢰 구간
        confidence_intervals = []

        for new_ob in self.test_resid:
            # 1 period 씩 예측을 수행한다.
            fc, conf = self.forecast_one_step(model)
            pred.append(fc)
            confidence_intervals.append(conf)

            # Updates the existing model with a small number of MLE steps
            model.update(new_ob)
        
        return pred, confidence_intervals

    def forecast_one_step(self, model):
        fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
        return (
            fc.tolist()[0],
            np.asarray(conf_int).tolist()[0])     

    def print_result(self, pred, test):    
        pred = pd.Series(pred, index=test.index)
        pd.DataFrame({'test':test,'pred':pred}).plot()
        plt.show()
        print(round(((pred-test)/test)*100, 2))

    # 외생 변수 줄 것
    def sarimax_predict(self, n_periods=6):
        model = self.auto_sarimax()
        pred = model.predict(n_periods=n_periods)

        return pred