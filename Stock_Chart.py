#from matplotlib import pyplot as plt
#import mpl_finance
import matplotlib
matplotlib.use('WXAgg')
from matplotlib import pyplot as plt

def CandleStick(Share_df):
    '''fg, ax1 = plt.subplots()
    mpl_finance.candlestick2_ohlc(ax=ax1, opens=Share_df['Open'], highs=Share_df['High'], lows=Share_df['Low'],
                                  closes=Share_df['Close'], width=0.4, colorup='#77d879', colordown='#db3f3f')
    # ax1.set_xticks(np.arange(len(Share_df)))
    plt.title('CANDLE STICK GRAPH')
    plt.ylabel('PRICE')
    plt.xlabel('DATE')
    plt.show()'''

    Share_df_1=Share_df
    plt.figure()
    width = 1
    width2 = 0.1
    pricesup = Share_df_1[Share_df_1.Close >= Share_df_1.Open]
    pricesdown = Share_df_1[Share_df_1.Close < Share_df_1.Open]

    plt.bar(pricesup.index, pricesup.Close - pricesup.Open, width, bottom=pricesup.Open, color='g')
    plt.bar(pricesup.index, pricesup.High - pricesup.Close, width2, bottom=pricesup.Close, color='g')
    plt.bar(pricesup.index, pricesup.Low - pricesup.Open, width2, bottom=pricesup.Open, color='g')

    plt.bar(pricesdown.index, pricesdown.Close - pricesdown.Open, width, bottom=pricesdown.Open, color='r')
    plt.bar(pricesdown.index, pricesdown.High - pricesdown.Open, width2, bottom=pricesdown.Open, color='r')
    plt.bar(pricesdown.index, pricesdown.Low - pricesdown.Close, width2, bottom=pricesdown.Close, color='r')
    plt.grid()

    plt.title('CANDLE STICK GRAPH')
    plt.ylabel('PRICE')
    plt.xlabel('DATE')
    plt.show()

def Line_Plot(Share_df):
    Share_df['Close'].plot.line()
    plt.grid()
    plt.title('LINE GRAPH')
    plt.ylabel('PRICE')
    plt.xlabel('DATE')
    plt.show()


def Bar_Plot(Share_df):
    Share_df['Close'].plot.bar()
    plt.grid()
    plt.title('BAR GRAPH')
    plt.ylabel('PRICE')
    plt.xlabel('DATE')
    plt.show()


def Area_Plot(Share_df):
    Share_df['Close'].plot.area()
    plt.grid()
    plt.title('AREA GRAPH')
    plt.ylabel('PRICE')
    plt.xlabel('DATE')
    plt.show()
