# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:07:36 2017

@author: Aditya
"""

#%matplotlib inline

import pandas as pd 

import matplotlib.pyplot as plt

#from missing_data_treatment import show_missing_data

import numpy as np


def load_clean_dataframe ( symbol):
    
    # get the stock name and search 
    stock_name = '../input/'+ symbol+'.csv'
    
    stock_df = pd.read_csv(stock_name)
    
    print (stock_df.columns.get_values())
    
    # dataframe has null values from excel -- clean it
        
    del_rows = stock_df[stock_df['Close'] == 'null'].index.values
    
    ##### you need to genelazine this method somewhere
    
    print (del_rows)
    
    for rows_to_del in del_rows:
        #drop the rows that are not needed 
        stock_df.drop (rows_to_del, inplace = True)
        
    return stock_df


def create_right_data_types_dataframe(stock_df):
    
    stock_df['Close'] = stock_df['Close'].astype(float)
    stock_df['Volume'] = stock_df['Volume'].astype(int)
    stock_df['Adj Close'] = stock_df['Adj Close'].astype(float)
    stock_df['High'] = stock_df['High'].astype(float)
    stock_df['Low'] = stock_df['Low'].astype(float)
    stock_df['Open'] = stock_df['Open'].astype(float)
    stock_df['Date'] = stock_df['Date'].astype('datetime64[ns]')
    
    
    return stock_df
 
    
    
    

def get_max_close( stock_df ):
    
    return stock_df['Close'].max()



def get_mean_volume(stock_df ):
    
    return stock_df['Volume'].mean()
    
    

def create_main_dataframe (startdate, enddate, symbols ):
        
    # create a blank data frame from 2012 
    timeseries_date = pd.date_range(startdate, enddate)
    
    all_data = pd.DataFrame(index = timeseries_date)
    
    # get date ranges 
    for symbol in symbols:
        
        
            #print (df_tatasteel.describe())
            #print (df_tatasteel.dtypes)
            # index return the dataframe --- remember 
  
            # from there join the datafames and get all the adj close prices 
    
            #all_data = all_data.dropna()
   
            #print (all_data)
            # form a dataframe with that 
    
            # adjusted close data only 
    
            # weekend dates will be Nan and will need to be deleted 
        
        
        df_symbol  = load_clean_dataframe(symbol)
        df_symbol = create_right_data_types_dataframe(df_symbol)
        df_symbol = df_symbol.set_index(['Date'])
        all_data = all_data.join(df_symbol['Adj Close'], how = 'inner')
        all_data = all_data.rename(columns = {'Adj Close': symbol})
        
        
    return all_data
        
    
    

    
    
def run(): 
    
    startdate = '2012-12-24'
    enddate = '2017-12-21'
    
    list_of_stocks = ['TATASTEEL','ITC','LT.NS','NSE']
    
    #2012-12-24 00:00:00
    #2017-12-21 00:00:00
    
    df_all_data = create_main_dataframe(startdate, enddate,list_of_stocks)
    
    #print (df_all_data)
    
    # based on index row slicing 
    
    #print (df_all_data.loc['2017-01-02': '2017-12-21'])
    
    # different from numpy
    
    print (df_all_data.loc['2017-01-02': '2017-12-21'])
    
    print (df_all_data['ITC'])
    
    #print (df_all_data[['TATASTEEL','ITC']])
    
    print (df_all_data.loc['2017-01-02': '2017-12-21', ['ITC','TATASTEEL']])
    
    # plotting the data frame 
    
    plot_data(df_all_data)
    
    
    
    df_all_data = normalize_data( df_all_data )
    
    plot_data(df_all_data)
    
    return df_all_data
    



def normalize_data(df_all_data):
    
    return  df_all_data/df_all_data.iloc[0]



def plot_data(dataframe ):
    
    ax = dataframe.plot(title = 'Price on normalized  scale' )
    
    ax.set_xlabel('Dates')
    
    ax.set_ylabel('price on scale')
    
    plt.show()
    
    


def exp_numpy (dataframe ):
    
    ndarray_dataframe = dataframe.values
    
    print (ndarray_dataframe[0,0])
    print (ndarray_dataframe[:, 0])
    
    print (ndarray_dataframe[0 : 3 , 1:3])
    
    print (ndarray_dataframe[: , 3])
    
    print (ndarray_dataframe[ -1 , 1:3 ])
    
    
def apply_rolling_stats ():
    
    startdate = '2012-12-24'
    enddate = '2013-12-21'
    
    list_of_stocks = ['NSE']
    
    #2012-12-24 00:00:00
    #2017-12-21 00:00:00
    
    df_all_data = create_main_dataframe(startdate, enddate,list_of_stocks)
    
    rm_NIFTY = get_rolling_mean(df_all_data, 20)
    
    rstd_NIFTY = get_rolling_std(df_all_data,20)
    
    upper, lower = get_bollinger_bands(rm_NIFTY, rstd_NIFTY)
    
    ax = df_all_data.plot(label = 'NIFTY', title = 'Price on normalized  scale' )
    
    
    
    
    rm_NIFTY.plot (label = 'Rollng mean', ax = ax)
    
    upper.plot (ax = ax)
    lower.plot (ax = ax)
    
    
    plt.show()
    
    
def apply_daily_returns ():
    
    startdate = '2012-12-24'
    enddate = '2017-01-21'
    
    list_of_stocks = ['NSE','LT.NS']
    
    #2012-12-24 00:00:00
    #2017-12-21 00:00:00
    
    df_all_data = create_main_dataframe(startdate, enddate,list_of_stocks)
    
    daily_returns = calculate_daily_return(df_all_data)
    
    print (daily_returns.head(20))
    
    #print (daily_returns)
    
    daily_returns.plot ()
    plt.show ()
    
    #kx = daily_returns.hist(bins = 50)
    
    #mean = daily_returns['NSE'].mean()
    #sd = daily_returns['NSE'].std()
    
#    print ('Mean' + str(mean))
 #   print ('SD' + str(sd))
    
    #plt.axvline (mean, color ='white', linestyle = '--' , linewidth = 2 )
    #plt.axvline (mean + sd, color ='red', linestyle = 'dashed' , linewidth =2 )
    #plt.axvline (mean - sd, color ='red', linestyle = 'dashed' , linewidth = 2)
    
    
    daily_returns['NSE'].hist( bins = 30, label = 'NSE')
    daily_returns['LT.NS'].hist( bins = 30 , label = 'LT')
    
    plt.legend(loc='upper right')
    
    
    #daily_returns['NSE']
    
    plt.show()
    
    print (daily_returns.kurtosis())
    
    
    daily_returns.plot(kind = 'scatter', x = 'NSE' , y = 'LT.NS')
    
    beta, alpha = np.polyfit(daily_returns['NSE'], daily_returns [ 'LT.NS'],1)
    
    plt.plot(daily_returns['NSE'] , beta * daily_returns['NSE'] + alpha , '-', color = 'r')
    
    plt.show()
    
    corr_matrix = daily_returns.corr(method = 'pearson')
    
    print (corr_matrix)
    
    
def apply_cum_returns ():
    
    startdate = '2012-12-24'
    enddate = '2017-01-21'
    
    list_of_stocks = ['NSE', 'ITC']
    
    #2012-12-24 00:00:00
    #2017-12-21 00:00:00
    
    df_all_data = create_main_dataframe(startdate, enddate,list_of_stocks)
    
    cum_returns = calculate_cumalative_return(df_all_data)
    
    #print (cum_returns)
    
    dx = cum_returns.plot (title = 'Cumm Returns')
    
    plt.show() 
    

    
# need to fix this method    
def calculate_cumalative_return (dataframe):
    cum_return = dataframe.copy()
    
    #print (cum_return)
    
    cum_return = (dataframe / dataframe.shift() - 1).cumsum()
    cum_return.iloc[0 , :] = 0
    
    #print (cum_return)
    return cum_return


def calculate_daily_return (dataframe):
    
    daily_return = dataframe.copy()
    
    #daily_return [1:] = (dataframe[ 1: ] / dataframe [ : -1 ].values) - 1
    
    daily_return = (dataframe / dataframe.shift(1)) - 1
    daily_return.iloc[0, :] = 0
    
    return daily_return
    
def get_rolling_std (dataframe, window ):
    
    #series = pd.rolling_std(dataframe, window)
    series = dataframe.rolling (window = window ).std ()
    return series
    
    
def get_bollinger_bands (rm , rstd):
    upper = rm + 2*rstd
    lower = rm - 2*rstd
    return upper,lower

def get_rolling_mean (dataframe, window ):
    
    #series = pd.rolling_mean(dataframe, window = window)
    
    series = dataframe.rolling (window = window).mean()
    return series
    
    
    
    
    
    

    
    
    


def apply_global_stats ( ):
    
    startdate = '2012-12-24'
    enddate = '2017-12-21'
    
    list_of_stocks = ['TATASTEEL','ITC','LT.NS','NSE']
    
    #2012-12-24 00:00:00
    #2017-12-21 00:00:00
    
    df_all_data = create_main_dataframe(startdate, enddate,list_of_stocks)
    
    print (df_all_data.mean())
    print (df_all_data.std())
    print (df_all_data.median())
    
    plot_data(df_all_data)
    
    
   
dataframe = run()

#exp_numpy(dataframe)

apply_rolling_stats()

apply_daily_returns()

#apply_cum_returns()

#show_missing_data()









def test_run():
    

    
    for symbol in ['TATASTEEL','ITC','LT.NS','NSE']:
        
        
        stock_df = load_clean_dataframe(symbol)
        
        create_right_data_types_dataframe(stock_df)
        #print (stock_df.head (10))
        print (get_max_close (stock_df))
        
        print (get_mean_volume(stock_df))
        
        print (stock_df['Date'].min())
        
        print (stock_df['Date'].max())
        
        plot_adj_close(stock_df)
        plot_high_prices(stock_df)
        stock_df[['Close', 'Adj Close']].plot()
        plt.show();
    

def plot_high_prices(stock_df):
    
    stock_df['High'] = stock_df['High'].astype(float)
    
    stock_df['High'].plot()
    plt.show()
    
    
        
        
def plot_adj_close(stock_df):
    
    stock_df['Adj Close'].plot ()
    plt.show()

        
#test_run()