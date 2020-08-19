import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('WXAgg')
from matplotlib import pyplot as plt

import ta

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


from nsepy import get_history
from nsetools import Nse

import pandas_market_calendars as mcal

from datetime import date

def Stock_Features(Share_df):
    Share_df = Share_df.dropna(axis=1)
    del Share_df['Symbol']
    del Share_df['Series']
    Share_df['Awes_Osc'] = ta.momentum.ao(Share_df['High'], Share_df['Low'], s=5, len=34, fillna=False)
    Share_df['TSI'] = ta.momentum.tsi(Share_df['Close'], r=25, s=13, fillna=False)
    Share_df['ACC_DCC'] = ta.volume.acc_dist_index(Share_df['High'], Share_df['Low'], Share_df['Close'],
                                                   Share_df['Volume'], fillna=False)
    # Share_df['Ulti_Osci']=ta.momentum.uo(Share_df['High'], Share_df['Low'], Share_df['Close'], s=7, m=14, l=28, ws=4.0, wm=2.0, wl=1.0, fillna=False)
    Share_df['Trix_18'] = ta.trend.trix(Share_df['Close'], n=18, fillna=False)
    Share_df['Trix_14'] = ta.trend.trix(Share_df['Close'], n=14, fillna=False)
    Share_df['Trix_11'] = ta.trend.trix(Share_df['Close'], n=11, fillna=False)
    Share_df['Trix_8'] = ta.trend.trix(Share_df['Close'], n=8, fillna=False)
    Share_df['Trix_7'] = ta.trend.trix(Share_df['Close'], n=7, fillna=False)
    Share_df['For_Ind'] = ta.volume.force_index(Share_df['Close'], Share_df['Volume'], n=2, fillna=False)
    Share_df['Neg_Vol_Ind'] = ta.volume.negative_volume_index(Share_df['Close'], Share_df['Volume'], fillna=False)
    # Share_df['OBV']=ta.volume.on_balance_volume(Share_df['Close'], Share_df['Volume'], fillna=False)
    # Share_df['P/C']=ta.volume.put_call_ratio()
    Share_df['Vol_Pri_Tre'] = ta.volume.volume_price_trend(Share_df['Close'], Share_df['Volume'], fillna=False)
    # Share_df['MFI']=ta.momentum.money_flow_index(Share_df['High'], Share_df['Low'], Share_df['Close'], Share_df['Volume'], n=14, fillna=False)
    # Share_df['ADX']=ta.trend.adx(Share_df['High'], Share_df['Low'], Share_df['Close'], n=14, fillna=False)
    # Share_df['ADX_Neg']=ta.trend.adx_neg(Share_df['High'], Share_df['Low'], Share_df['Close'], n=14, fillna=False)
    # Share_df['ADX_Pos']=ta.trend.adx_pos(Share_df['High'], Share_df['Low'], Share_df['Close'], n=14, fillna=False)
    Share_df['AR_DO'] = ta.trend.aroon_down(Share_df['Close'], n=25, fillna=False)
    Share_df['AR_UP'] = ta.trend.aroon_up(Share_df['Close'], n=25, fillna=False)
    Share_df['COM_CHA_Ind'] = ta.trend.cci(Share_df['High'], Share_df['Low'], Share_df['Close'], n=20, c=0.015,
                                           fillna=False)
    Share_df['DPO'] = ta.trend.dpo(Share_df['Close'], n=20, fillna=False)
    Share_df['ICHI_A'] = ta.trend.ichimoku_a(Share_df['High'], Share_df['Low'], n1=9, n2=26, visual=False, fillna=False)
    Share_df['ICHI_B'] = ta.trend.ichimoku_b(Share_df['High'], Share_df['Low'], n2=26, n3=52, visual=False,
                                             fillna=False)
    Share_df['KST_OSC'] = ta.trend.kst(Share_df['Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15,
                                       fillna=False)
    Share_df['KST_Sig'] = ta.trend.kst_sig(Share_df['Close'], r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15,
                                           nsig=9, fillna=False)
    Share_df['MACD'] = ta.trend.macd(Share_df['Close'], n_fast=12, n_slow=26, fillna=False)
    Share_df['MACD_DIFF'] = ta.trend.macd_diff(Share_df['Close'], n_fast=12, n_slow=26, n_sign=9, fillna=False)
    Share_df['MACD_SIG'] = ta.trend.macd_signal(Share_df['Close'], n_fast=12, n_slow=26, n_sign=9, fillna=False)
    # Share_df['MI']=ta.trend.mass_index(high, low, n=9, n2=25, fillna=False)
    # Share_df['VI']=ta.trend.vortex_indicator_neg(Share_df['High'], Share_df['Low'], Share_df['Close'], n=14, fillna=False)
    # Share_df['VIP']=ta.trend.vortex_indicator_pos(Share_df['High'], Share_df['Low'], Share_df['Close'], n=14, fillna=False)
    Share_df['RSI'] = ta.momentum.rsi(Share_df['Close'], n=14, fillna=False)
    Share_df['Sto'] = ta.momentum.stoch(Share_df['High'], Share_df['Low'], Share_df['Close'], n=14, fillna=False)
    Share_df['Sto_Sig'] = ta.momentum.stoch_signal(Share_df['High'], Share_df['Low'], Share_df['Close'], n=14, d_n=3,
                                                   fillna=False)
    Share_df['Chai_Vol'] = ta.volume.chaikin_money_flow(Share_df['High'], Share_df['Low'], Share_df['Close'],
                                                        Share_df['Volume'], n=20, fillna=False)
    Share_df['Will_R'] = ta.momentum.wr(Share_df['High'], Share_df['Low'], Share_df['Close'], lbp=14, fillna=False)
    # Share_df['ATR']=ta.volatility.average_true_range(Share_df['High'], Share_df['Low'], Share_df['Close'], n=14, fillna=False)
    Share_df['EOM'] = ta.volume.ease_of_movement(Share_df['High'], Share_df['Low'], Share_df['Close'],
                                                 Share_df['Volume'], n=20, fillna=False)
    Share_df['Boll_Avg'] = ta.volatility.bollinger_mavg(Share_df['Close'], n=20, fillna=False)
    Share_df['EMA_3'] = ta.trend.ema_indicator(Share_df['Close'], n=3, fillna=False)
    Share_df['EMA_5'] = ta.trend.ema_indicator(Share_df['Close'], n=5, fillna=False)
    Share_df['EMA_7'] = ta.trend.ema_indicator(Share_df['Close'], n=7, fillna=False)
    Share_df['EMA_9'] = ta.trend.ema_indicator(Share_df['Close'], n=9, fillna=False)
    Share_df['EMA_12'] = ta.trend.ema_indicator(Share_df['Close'], n=12, fillna=False)
    Share_df = Share_df.dropna(axis=0)
    return Share_df

def Prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out)  # creating new column called label with the last 5 rows are nan
    X = df  # creating the feature array
    X = preprocessing.scale(X)  # processing the feature array
    X_lately = X[-forecast_out:]  # creating the column i want to use later in the predicting method
    X = X[:-forecast_out]  # X that will contain the training and testing
    label.dropna(inplace=True)  # dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size)  # cross validation
    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response

def Linear_Regression(Share_df):
    forecast_col = 'Close'  # choosing which column to forecast
    forecast_out = 10  # how far to forecast
    test_size = 0.05  # the size of my test set
    Share_df=Stock_Features(Share_df)
    X_train, X_test, Y_train, Y_test, X_lately = Prepare_data(Share_df, forecast_col, forecast_out,
                                                              test_size)  # calling the method were the cross validation and data preperation is in

    learner = linear_model.LinearRegression()  # initializing linear regression model
    learner.fit(X_train, Y_train)  # training the linear regression model
    forecast = learner.predict(X_lately)
    Acc_Pred = ((learner.score(X_train, Y_train))*100)

    Date_Range = (str(Share_df.iloc[((len(Share_df)) - 1)].name))
    Date_Plt = pd.DataFrame(pd.date_range(Date_Range, periods=11, freq='D'))
    Date_Plt = Date_Plt[1:]
    plt.plot(Date_Plt, forecast)
    plt.title("LINEAR REGRESSION")
    plt.ylabel('PRICE')
    plt.xlabel('DATE'+'\n'+'PREDICTION ACCURACY : '+str(Acc_Pred))
    plt.grid()
    plt.show()

def Lasso_Regression(Share_df):
    forecast_col = 'Close'  # choosing which column to forecast
    forecast_out = 10  # how far to forecast
    test_size = 0.05  # the size of my test set
    Share_df = Stock_Features(Share_df)
    X_train, X_test, Y_train, Y_test, X_lately = Prepare_data(Share_df, forecast_col, forecast_out,
                                                              test_size)  # calling the method were the cross validation and data preperation is in

    learner = linear_model.Lasso()  # initializing linear regression model
    learner.fit(X_train, Y_train)  # training the linear regression model
    forecast = learner.predict(X_lately)
    Acc_Pred = ((learner.score(X_train, Y_train)) * 100)

    Date_Range = (str(Share_df.iloc[((len(Share_df)) - 1)].name))
    Date_Plt = pd.DataFrame(pd.date_range(Date_Range, periods=11, freq='D'))
    Date_Plt = Date_Plt[1:]
    plt.plot(Date_Plt, forecast)
    plt.title("LASSO REGRESSION")
    plt.ylabel('PRICE')
    plt.xlabel('DATE' + '\n' + 'PREDICTION ACCURACY : ' + str(Acc_Pred))
    plt.grid()
    plt.show()

def Random_Forest(Share_df):
    forecast_col = 'Close'  # choosing which column to forecast
    forecast_out = 10  # how far to forecast
    test_size = 0.05  # the size of my test set
    Share_df = Stock_Features(Share_df)
    X_train, X_test, Y_train, Y_test, X_lately = Prepare_data(Share_df, forecast_col, forecast_out,
                                                              test_size)  # calling the method were the cross validation and data preperation is in

    learner = RandomForestRegressor(n_estimators=10, random_state=42)  # initializing random forest regressor model
    learner.fit(X_train, Y_train)  # training the linear regression model
    forecast = learner.predict(X_lately)
    Acc_Pred = ((learner.score(X_train, Y_train)) * 100)

    Date_Range = (str(Share_df.iloc[((len(Share_df)) - 1)].name))
    Date_Plt = pd.DataFrame(pd.date_range(Date_Range, periods=11, freq='D'))
    Date_Plt = Date_Plt[1:]
    plt.plot(Date_Plt, forecast)
    plt.title("RANDOM FOREST REGRESSOR")
    plt.ylabel('PRICE')
    plt.xlabel('DATE' + '\n' + 'PREDICTION ACCURACY : ' + str(Acc_Pred))
    plt.grid()
    plt.show()

def Decision_tree(Share_df):
    forecast_col = 'Close'  # choosing which column to forecast
    forecast_out = 10  # how far to forecast
    test_size = 0.05  # the size of my test set
    Share_df = Stock_Features(Share_df)
    X_train, X_test, Y_train, Y_test, X_lately = Prepare_data(Share_df, forecast_col, forecast_out,
                                                              test_size)  # calling the method were the cross validation and data preperation is in

    learner = DecisionTreeRegressor(random_state=42) # initializing decision tree regressor model
    learner.fit(X_train, Y_train)  # training the linear regression model
    forecast = learner.predict(X_lately)
    Acc_Pred = ((learner.score(X_train, Y_train)) * 100)

    Date_Range = (str(Share_df.iloc[((len(Share_df)) - 1)].name))
    Date_Plt = pd.DataFrame(pd.date_range(Date_Range, periods=11, freq='D'))
    Date_Plt = Date_Plt[1:]
    plt.plot(Date_Plt, forecast)
    plt.title("DECISION TREE REGRESSOR")
    plt.ylabel('PRICE')
    plt.xlabel('DATE' + '\n' + 'PREDICTION ACCURACY : ' + str(Acc_Pred))
    plt.grid()
    plt.show()
