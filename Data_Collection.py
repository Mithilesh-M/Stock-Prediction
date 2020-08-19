#import pandas_datareader as dr
from datetime import date
from nsepy import get_history
import datetime

'''def StockData_Yahoo(Share_Yahoo,Start_Date,End_Date):
        #Share = Share+'.NS'
        Share_df = dr.data.get_data_yahoo(Share_Yahoo,start=Start_Date,end=End_Date)
        Date_data=[]
        for i in range(Share_df.High.count()):
            Date_data.append((str(Share_df.iloc[i].name))[:10])
        return (Share_df,Date_data)
    '''


def StockData_NSE(Share_NSE, Start_Y, End_Y, Start_M, End_M, Start_D, End_D):
    Share_df = get_history(symbol=Share_NSE, start=date(Start_Y, Start_M, Start_D), end=date(End_Y, End_M, End_D))
    Date_data = []
    '''for i in range(Share_df.High.count()):
        Date_data.append((str(Share_df.iloc[i].name)))
    return (Share_df, Date_data)'''
    return (Share_df)


def Dates_Conv(Start_Date, End_Date):
    def Month_D(Mon):
        Dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
        Mon_int = Dict.get(Mon)
        return (Mon_int)
    Start_Date=str(Start_Date)
    End_Date=str(End_Date)
    Start_Y = int(Start_Date[20:24])
    End_Y = int(End_Date[20:24])
    Start_M = Month_D(Start_Date[4:7])
    End_M = Month_D(End_Date[4:7])
    Start_D = int(Start_Date[8:10])
    End_D = int(End_Date[8:10])
    return (Start_Y, End_Y, Start_M, End_M, Start_D, End_D)

def Today_Date():
    today = datetime.date.today()
    return (str(today.ctime()))

def Stock_df(Stock, Start_Date, End_Date):
    Start_Y, End_Y, Start_M, End_M, Start_D, End_D = Dates_Conv(Start_Date, End_Date)
    return (StockData_NSE(Stock, Start_Y, End_Y, Start_M, End_M, Start_D, End_D))

