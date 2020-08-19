import matplotlib
matplotlib.use('WXAgg')
from matplotlib import pyplot as plt
import pandas as pd
import ta


def ROC(Share_df):
    M = Share_df['Close'].diff(9)
    N = Share_df['Close'].shift(9)
    ROC = pd.Series(M / N, name='ROC')
    #plt.plot(Date_data, ROC)
    ROC.plot.line()
    plt.grid()
    plt.title('RATE OF CHANGE GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('RATE OF CHANGE')
    plt.show()


def Stochastic_Oscill(Share_df):
    SOk = pd.Series((Share_df['Close'] - Share_df['Low']) / (Share_df['High'] - Share_df['Low']), name='Sok')
    #plt.plot(Date_data, SOk)
    SOk.plot.line()
    plt.grid()
    plt.title('STOCHASTIC OSCILLATOR GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('STOCHASTIC')
    plt.show()


def Force_Ind(Share_df):
    F = pd.Series(Share_df['Close'].diff(13) * Share_df['Volume'].diff(13), name='Force_')
    #plt.plot(Date_data, F)
    F.plot.line()
    plt.grid()
    plt.title('FORCE INDEX GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('FORCE INDEX')
    plt.show()


def Volume(Share_df):
    #plt.plot(Date_data, Share_df.Volume)
    Share_df['Volume'].plot.line()
    plt.grid()
    plt.title('VOLUME GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('VOLUME')
    plt.show()


def Donchian(Share_df):
    i = 0
    DC_l = []
    while i < 19:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + 20 - 1 < Share_df.High.count():
        DC = max(Share_df['High'].ix[i:i + 20 - 1]) - min(Share_df['Low'].ix[i:i + 20 - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = pd.Series(DC_l, name='Donchian_')
    DonCh = DonCh.shift(20 - 1)
    #plt.plot(Date_data, DonCh)
    DonCh.plot.line()
    plt.grid()
    plt.title('DONCHIAN CHANNEL GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('DONCHIAN CHANNEL')
    plt.show()


def Acc_Dist(Share_df):
    ad = (2 * Share_df['Close'] - Share_df['High'] - Share_df['Low']) / (Share_df['High'] - Share_df['Low']) * \
         Share_df['Volume']
    M = ad.diff(1)
    N = ad.shift(1)
    ROC = M / N
    AD = pd.Series(ROC, name='Acc/Dist_ROC_')
    #plt.plot(Date_data, AD)
    AD.plot.line()
    plt.grid()
    plt.title('ACCUMULATION DISTRIBUTION GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('ACCUMULATION DISTRIBUTION')
    plt.show()


def VWAP(Share_df):
    vwap = Share_df.VWAP
    #plt.plot(Date_data, vwap)
    vwap.plot.line()
    plt.grid()
    plt.title('VOLUME WEIGHTED AVERAGE PRICE GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('VWAP')
    plt.show()


def Turnover(Share_df):
    Turnov = Share_df.Turnover
    #plt.plot(Date_data, Turnov)
    Turnov.plot.line()
    plt.grid()
    plt.title('TURNOVER GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('TURNOVER')
    plt.show()


def Trades(Share_df):
    Trade = Share_df.Trades
    #plt.plot(Date_data, Trade)
    Trade.plot.line()
    plt.grid()
    plt.title('TRADES GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('TRADES')
    plt.show()


def Delivery_Perc(Share_df):
    Deliv = Share_df['%Deliverble']
    #plt.plot(Date_data, Deliv)
    Deliv.plot.line()
    plt.grid()
    plt.title('DELIVERABLE PERCENTAGE GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('DELIVERABLE PERCENTAGE')
    plt.show()

def Trix(Share_df):
    T= ta.trend.trix(Share_df['Close'], n=18, fillna=False)
    T.plot.line()
    plt.grid()
    plt.title('TRIX GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('TRIX')
    plt.show()

def Vol_Pri_Tre(Share_df):
    Vol_Pri_Tre=ta.volume.volume_price_trend(Share_df['Close'], Share_df['Volume'], fillna=False)
    Vol_Pri_Tre.plot.line()
    plt.grid()
    plt.title('VOLUME PRICE TREND GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('VOLUME PRICE')
    plt.show()
def MACD(Share_df):
    macd=ta.trend.macd(Share_df['Close'], n_fast=12, n_slow=26, fillna=False)
    macd.plot.line()
    plt.grid()
    plt.title('MACD GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('MACD')
    plt.show()

def RSI(Share_df):
    rsi=ta.momentum.rsi(Share_df['Close'], n=14, fillna=False)
    rsi.plot.line()
    plt.grid()
    plt.title('RSI GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('RSI')
    plt.show()

def WILL_R(Share_df):
    will_r=ta.momentum.wr(Share_df['High'], Share_df['Low'], Share_df['Close'], lbp=14, fillna=False)
    will_r.plot.line()
    plt.grid()
    plt.title('WILLIAM R GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('WILLIAM R')
    plt.show()

def EOM(Share_df):
    eom=ta.volume.ease_of_movement(Share_df['High'], Share_df['Low'], Share_df['Close'],Share_df['Volume'], n=20, fillna=False)
    eom.plot.line()
    plt.grid()
    plt.title('EOM GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('EOM')
    plt.show()

def Boll_Avg(Share_df):
    boll_avg=ta.volatility.bollinger_mavg(Share_df['Close'], n=20, fillna=False)
    boll_avg.plot.line()
    plt.grid()
    plt.title('BOLLINGER AVERAGE GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('BOLLINGER AVERAGE')
    plt.show()

def AWE_OSC(Share_df):
    awe_osc=ta.momentum.ao(Share_df['High'], Share_df['Low'], s=5, len=34, fillna=False)
    awe_osc.plot.line()
    plt.grid()
    plt.title('AWESOME OSCILLATOR GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('AWESOME OSCILLATOR')
    plt.show()

def TSI(Share_df):
    tsi=ta.momentum.tsi(Share_df['Close'], r=25, s=13, fillna=False)
    tsi.plot.line()
    plt.grid()
    plt.title('TSI GRAPH')
    plt.xlabel('DATE')
    plt.ylabel('TSI')
    plt.show()