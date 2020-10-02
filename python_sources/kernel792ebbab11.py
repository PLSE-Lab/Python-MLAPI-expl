
import numpy as np 
import pandas as pd
import matplotlib as mtlab
import math as mt
import  random 
import pandas_datareader as pd_reader
import statsmodels as statm


from pandas_datareader import data as wb
PG=wb.DataReader('PG',data_source='yahoo',start='1995-1-1')#Procter &Gample #AAPL for Apple #MSFT microsoft
tickers= ['PG','MSFT','T','F','GE']
new_data=pd.DataFrame()
for t in tickers:
    new_data[t]=wb.DataReader(t,data_source='yahoo',start='1995-1-1')['Adj Close']
    
print(new_data.tail())
























































