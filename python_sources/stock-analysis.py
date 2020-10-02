#!/usr/bin/env python
# coding: utf-8

# # Question 1: upload .csv files into dataframe by using pd.read_csv()

# In[ ]:


# The first step import all dataset and set the Date as index except two all_stock files
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


all_stock1=pd.read_csv('../input/all_stocks_2006-01-01_to_2018-01-01.csv')


# In[ ]:


all_stock1.head()


# In[ ]:


all_stock2=pd.read_csv('../input/all_stocks_2017-01-01_to_2018-01-01.csv')


# In[ ]:


all_stock2.head()


# In[ ]:


AABA=pd.read_csv('../input/AABA_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


AABA.head()


# In[ ]:


AAPL=pd.read_csv('../input/AAPL_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


AAPL.head()


# In[ ]:


AMZN=pd.read_csv('../input/AMZN_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


AMZN.head()


# In[ ]:


AXP=pd.read_csv('../input/AXP_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


AXP.head()


# In[ ]:


BA=pd.read_csv('../input/BA_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


BA.head()


# In[ ]:


CAT=pd.read_csv('../input/CAT_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


CAT.head()


# In[ ]:


CSCO=pd.read_csv('../input/CSCO_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


CSCO.head()


# In[ ]:


CVX=pd.read_csv('../input/CVX_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


CVX.head()


# In[ ]:


DIS=pd.read_csv('../input/DIS_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


DIS.head()


# In[ ]:


GE=pd.read_csv('../input/GE_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


GE.head()


# In[ ]:


GOOGL=pd.read_csv('../input/GOOGL_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


GOOGL.head()


# In[ ]:


GS=pd.read_csv('../input/GS_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


GS.head()


# In[ ]:


HD=pd.read_csv('../input/HD_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


HD.head()


# In[ ]:


IBM=pd.read_csv('../input/IBM_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


IBM.head()


# In[ ]:


INTC=pd.read_csv('../input/INTC_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


INTC.head()


# In[ ]:


JNJ=pd.read_csv('../input/JNJ_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


JNJ.head()


# In[ ]:


JPM=pd.read_csv('../input/JPM_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


JPM.head()


# In[ ]:


KO=pd.read_csv('../input/KO_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


KO.head()


# In[ ]:


MCD=pd.read_csv('../input/MCD_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


MCD.head()


# In[ ]:


MMM=pd.read_csv('../input/MMM_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


MMM.head()


# In[ ]:


MRK=pd.read_csv('../input/MRK_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


MRK.head()


# In[ ]:


MSFT=pd.read_csv('../input/MSFT_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


MSFT.head()


# In[ ]:


NKE=pd.read_csv('../input/NKE_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


NKE.head()


# In[ ]:


PEE=pd.read_csv('../input/PFE_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


PEE.head()


# In[ ]:


PG=pd.read_csv('../input/PG_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


PG.head()


# In[ ]:


TRV=pd.read_csv('../input/TRV_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


TRV.head()


# In[ ]:


UNH=pd.read_csv('../input/UNH_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


UNH.head()


# In[ ]:


UTX=pd.read_csv('../input/UTX_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


UTX.head()


# In[ ]:


VZ=pd.read_csv('../input/VZ_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


VZ.head()


# In[ ]:


WMT=pd.read_csv('../input/WMT_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


WMT.head()


# In[ ]:


XOM=pd.read_csv('../input/XOM_2006-01-01_to_2018-01-01.csv').set_index('Date')


# In[ ]:


XOM.head()


# '''concatenate the stock dataframes together to a single 
# dataframe. Set the keys as the ticker names'''

# In[ ]:


tickers=['AABA','AAPL','AMZN','AXP','BA','CAT','CSCO','CVX'
        ,'DIS','GE','GOOGL','GS','HD','IBM','INTC','JNJ',
         'JPM','KO','MCD','MMM','MRK','MSFT','NKE','PEE',
         'PG','TRV','UNH','UTX','VZ','WMT','XOM']


# In[ ]:


all_stock=pd.concat([AABA.iloc[:,:-1],AAPL.iloc[:,:-1],AMZN.iloc[:,:-1],AXP.iloc[:,:-1],BA.iloc[:,:-1],CAT.iloc[:,:-1],
                     CSCO.iloc[:,:-1],CVX.iloc[:,:-1],DIS.iloc[:,:-1],GE.iloc[:,:-1],GOOGL.iloc[:,:-1],GS.iloc[:,:-1],
                     HD.iloc[:,:-1],IBM.iloc[:,:-1], INTC.iloc[:,:-1],JNJ.iloc[:,:-1],JPM.iloc[:,:-1],KO.iloc[:,:-1],
                     MCD.iloc[:,:-1],MMM.iloc[:,:-1],MRK.iloc[:,:-1],MSFT.iloc[:,:-1],NKE.iloc[:,:-1],PEE.iloc[:,:-1],
                     PG.iloc[:,:-1],TRV.iloc[:,:-1], UNH.iloc[:,:-1],UTX.iloc[:,:-1],VZ.iloc[:,:-1],WMT.iloc[:,:-1],
                     XOM.iloc[:,:-1]],axis=1,keys=tickers,sort=True)


# In[ ]:


all_stock.columns.names = ['Bank Ticker','Stock Info']
all_stock.head()


# Fot question 1, I uploaded all stock files by using pd.read_csv() function and concatenate all single stock files into one DataFrame by using pd.concat() function and set the Date as index.

# # Question 2 EDA

# In[ ]:


# Try to find the average open,close price for each stock from 2006-01-01 to 2018-01-01
all_stock1.groupby('Name',as_index=False).mean()[['Name','Open','Close']]


# In[ ]:


plt.figure()
L=all_stock1.groupby('Name',as_index=False).mean()[['Name','Open','Close']]
x=np.array(L['Name'])
y=np.array(L['Open'])
z=np.array(L['Close'])
plt.subplot(3,1,1)
plt.plot(x,y)
plt.xticks(L['Name'],rotation='vertical')
plt.title('Average open price for each stock')
plt.xlabel('Stock Name')
plt.ylabel('Average open price')
plt.subplot(3, 1, 3)
plt.plot(x,z)
plt.xticks(L['Name'],rotation='vertical')
plt.title('Average close price for each stock')
plt.xlabel('Stock Name')
plt.ylabel('Average close price')
plt.show()


# In[ ]:


'''for all stocks in the time period from 2006-01-01 to 2009-12-31'''
# What is the max Open price for each stock throughout the time period
new_df_allstock1=all_stock1[(all_stock1['Date']>='2006-01-01')&(all_stock1['Date']<='2009-12-31')]#Find the data in the time period
new_df_allstock1.loc[new_df_allstock1.groupby('Name')['Open'].idxmax()][['Name','Date','Open']]#show the dataframe with max open price for each stock 


# In[ ]:


L=new_df_allstock1.loc[new_df_allstock1.groupby('Name')['Open'].idxmax()][['Name','Date','Open']]
x=np.array(L['Name'])
y=np.array(L['Open'])
plt.plot(x,y,'o',color='black')
plt.xticks(L['Name'],rotation='vertical')
plt.title('Max open price for each stock from 2006 to 2009')
plt.xlabel('Stock Name')
plt.ylabel('Max open price')
plt.show()


# In[ ]:


# Try to find returns for every stock each day which means how much the stock earns all losses each day
returns=pd.DataFrame()
for tick in tickers:
    returns[tick+' Return'] = all_stock[tick]['Close'].pct_change()
returns.head()


# In[ ]:


# Shows the return of GOOGL in year 2017
sns.distplot(returns.loc['2017-01-01':'2017-12-31']['GOOGL Return'],color='blue',bins=100)


# As shown above, the return of GOOGL stock reaches the top point at around the point a little bit more than 0.00

# In[ ]:


# The trend of close price from 2006-01-01 to 2018-01-01 for all stocks
sns.set_style('whitegrid')
for tick in tickers:
    all_stock[tick]['Close'].plot(figsize=(12,11),label=tick)
plt.legend()


# For question 2, I analyzed average open,close price for each stock from 2006-01-01 to 2018-01-01 and visualized the result; 
# What is the max Open price for each stock throughout the time period from 2006-01-01 to 2009-12-31;Try to find returns for every stock each day which means how much the stock earns all losses each day and found the trend of close price from 2006-01-01 to 2018-01-01 for all stocks by visualization.

# # Q3 Perform descriptive analysis for any two stock from given dataset and compare the analysis.

# Based on the last step, I'm going to use GOOGL and AMZN stocks which have highest average open price and highest average close price.

# In[ ]:


# Combine the GOOGL and AMZN as one DataFrame
ticker1=['GOOGL','AMZN']
two_stock=pd.concat([AMZN,GOOGL],axis=1,keys=ticker1)
two_stock.columns.names=['Stock ticker','Stock Info']


# In[ ]:


two_stock.head()


# In[ ]:


# try to find the highest open price of both stocks for each year
Year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
x=np.array(Year)
AMZN_open_max=[AMZN['2006':'2007']['Open'].max(),AMZN['2007':'2008']['Open'].max(),AMZN['2008':'2009']['Open'].max(),
    AMZN['2009':'2010']['Open'].max(),AMZN['2010':'2011']['Open'].max(),AMZN['2011':'2012']['Open'].max(),
    AMZN['2012':'2013']['Open'].max(),AMZN['2013':'2014']['Open'].max(),AMZN['2014':'2015']['Open'].max(),
    AMZN['2015':'2016']['Open'].max(),AMZN['2016':'2017']['Open'].max(),AMZN['2017':'2018']['Open'].max()]
y=np.array(AMZN_open_max)
GOOGL_open_max=[GOOGL['2006':'2007']['Open'].max(),GOOGL['2007':'2008']['Open'].max(),GOOGL['2008':'2009']['Open'].max(),
    GOOGL['2009':'2010']['Open'].max(),GOOGL['2010':'2011']['Open'].max(),GOOGL['2011':'2012']['Open'].max(),
    GOOGL['2012':'2013']['Open'].max(),GOOGL['2013':'2014']['Open'].max(),GOOGL['2014':'2015']['Open'].max(),
    GOOGL['2015':'2016']['Open'].max(),GOOGL['2016':'2017']['Open'].max(),GOOGL['2017':'2018']['Open'].max()]
z=np.array(GOOGL_open_max)
plt.figure()
plt.plot(x, y, linestyle='solid',label="AMZN")
plt.plot(x, z, linestyle='dashed',label="GOOGL")
plt.title('Highest Open Price Each year')
plt.xlabel('Year')
plt.ylabel('Open Price')
plt.legend()


# As shown above, the highest open price of AMZN gradually went up over year finally exceeded GOOGL.

# In[ ]:


# try to find the lowest volume of both stocks for each year
Year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
x=np.array(Year)
AMZN_open_min=[AMZN['2006':'2007']['Volume'].min(),AMZN['2007':'2008']['Volume'].min(),AMZN['2008':'2009']['Volume'].min(),
    AMZN['2009':'2010']['Volume'].min(),AMZN['2010':'2011']['Volume'].min(),AMZN['2011':'2012']['Volume'].min(),
    AMZN['2012':'2013']['Volume'].min(),AMZN['2013':'2014']['Volume'].min(),AMZN['2014':'2015']['Volume'].min(),
    AMZN['2015':'2016']['Volume'].min(),AMZN['2016':'2017']['Volume'].min(),AMZN['2017':'2018']['Volume'].min()]
y=np.array(AMZN_open_min)
GOOGL_open_min=[GOOGL['2006':'2007']['Volume'].min(),GOOGL['2007':'2008']['Volume'].min(),GOOGL['2008':'2009']['Volume'].min(),
    GOOGL['2009':'2010']['Volume'].min(),GOOGL['2010':'2011']['Volume'].min(),GOOGL['2011':'2012']['Volume'].min(),
    GOOGL['2012':'2013']['Volume'].min(),GOOGL['2013':'2014']['Volume'].min(),GOOGL['2014':'2015']['Volume'].min(),
    GOOGL['2015':'2016']['Volume'].min(),GOOGL['2016':'2017']['Volume'].min(),GOOGL['2017':'2018']['Volume'].min()]
z=np.array(GOOGL_open_min)

plt.figure()
plt.plot(x, y, linestyle='solid',label="AMZN")
plt.plot(x, z, linestyle='dashed',label="GOOGL")
plt.title('Lowest Volume')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()


# As shown above, the lowest volume of  GOOGL over year is always lower than AMZN especially in 2009

# In[ ]:


# compare the close price
for ticker in ticker1:
        two_stock[ticker]['Close'].plot(figsize=(12,4),label=ticker)
plt.legend()


# As shown above, the Close price of AMZN is generally higher than GOOGL however GOOGL goes up at the end of the time period 
# and beyonds the AMZN

# In[ ]:


# compare the open price
for ticker in ticker1:
        two_stock[ticker]['Open'].plot(figsize=(12,4),label=ticker)
plt.legend()


# Same trend for the Open price of these two stocks in the time period

# In[ ]:


# try to find out the relationship between close price and 30-day average close price for stocks from 2010 to 2011
plt.figure(figsize=(12,6))
AMZN['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg AMZN')
AMZN['Close'].loc['2010-01-01':'2011-01-01'].plot(label='AMZN CLOSE')
GOOGL['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg GOOGL')
GOOGL['Close'].loc['2010-01-01':'2011-01-01'].plot(label='GOOGL CLOSE')
plt.legend()


# As shown above, the close price of AMZN is more stable around the 30 days average of the stock close price than GOOGL
# from 2010 to 2011

# In[ ]:


# compare the volume trend
plt.figure(figsize=(12,6))
AMZN['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='AMZN Volume')
GOOGL['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='GOOGL Volume')
plt.legend()


# As shown Above, volume increases more sharply for AMZN stock from 2017 to 2018 than GOOGL

# In[ ]:


# show the correlation between open price of stocks
sns.heatmap(two_stock.xs(key='Open',axis=1,level='Stock Info').corr(),annot=True)


# As the heatmap shown above, the correlation between the open price of these two stocks is 0.97.

# # Question4 Take five stocks of your choice and create the same type of report stated in question 3.

# I'm going to use five stocks which are GS BA IBM CVX and MMM because they had the highest open price during 2006 and 2009 except 
# GOOGL and AMZN

# In[ ]:


# Combine GS BA IBM CVX and MMM as one DataFrame
ticker2=['GS','BA','IBM','CVX','MMM']
five_stock=pd.concat([GS,BA,IBM,CVX,MMM],axis=1,keys=ticker2)
five_stock.columns.names=['Stock ticker','Stock Info']


# In[ ]:


five_stock.head()


# In[ ]:


# try to find the highest open price of both stocks for each year
Year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
x=np.array(Year)
GS_open_max=[GS['2006':'2007']['Open'].max(),GS['2007':'2008']['Open'].max(),GS['2008':'2009']['Open'].max(),
    GS['2009':'2010']['Open'].max(),GS['2010':'2011']['Open'].max(),GS['2011':'2012']['Open'].max(),
    GS['2012':'2013']['Open'].max(),GS['2013':'2014']['Open'].max(),GS['2014':'2015']['Open'].max(),
    GS['2015':'2016']['Open'].max(),GS['2016':'2017']['Open'].max(),GS['2017':'2018']['Open'].max()]
y=np.array(GS_open_max)

BA_open_max=[BA['2006':'2007']['Open'].max(),BA['2007':'2008']['Open'].max(),BA['2008':'2009']['Open'].max(),
    BA['2009':'2010']['Open'].max(),BA['2010':'2011']['Open'].max(),BA['2011':'2012']['Open'].max(),
    BA['2012':'2013']['Open'].max(),BA['2013':'2014']['Open'].max(),BA['2014':'2015']['Open'].max(),
    BA['2015':'2016']['Open'].max(),BA['2016':'2017']['Open'].max(),BA['2017':'2018']['Open'].max()]
z=np.array(BA_open_max)

IBM_open_max=[IBM['2006':'2007']['Open'].max(),IBM['2007':'2008']['Open'].max(),IBM['2008':'2009']['Open'].max(),
    IBM['2009':'2010']['Open'].max(),IBM['2010':'2011']['Open'].max(),IBM['2011':'2012']['Open'].max(),
    IBM['2012':'2013']['Open'].max(),IBM['2013':'2014']['Open'].max(),IBM['2014':'2015']['Open'].max(),
    IBM['2015':'2016']['Open'].max(),IBM['2016':'2017']['Open'].max(),IBM['2017':'2018']['Open'].max()]
m=np.array(IBM_open_max)

CVX_open_max=[CVX['2006':'2007']['Open'].max(),CVX['2007':'2008']['Open'].max(),CVX['2008':'2009']['Open'].max(),
    CVX['2009':'2010']['Open'].max(),CVX['2010':'2011']['Open'].max(),CVX['2011':'2012']['Open'].max(),
    CVX['2012':'2013']['Open'].max(),CVX['2013':'2014']['Open'].max(),CVX['2014':'2015']['Open'].max(),
    CVX['2015':'2016']['Open'].max(),CVX['2016':'2017']['Open'].max(),CVX['2017':'2018']['Open'].max()]
n=np.array(CVX_open_max)

MMM_open_max=[MMM['2006':'2007']['Open'].max(),MMM['2007':'2008']['Open'].max(),MMM['2008':'2009']['Open'].max(),
    MMM['2009':'2010']['Open'].max(),MMM['2010':'2011']['Open'].max(),MMM['2011':'2012']['Open'].max(),
    MMM['2012':'2013']['Open'].max(),MMM['2013':'2014']['Open'].max(),MMM['2014':'2015']['Open'].max(),
    MMM['2015':'2016']['Open'].max(),MMM['2016':'2017']['Open'].max(),MMM['2017':'2018']['Open'].max()]
l=np.array(MMM_open_max)

plt.figure()
plt.plot(x, y, linestyle='solid',label="GS")
plt.plot(x, z, linestyle='dashed',label="BA")
plt.plot(x, m, linestyle='dashdot',label="IBM")
plt.plot(x, n, linestyle='dotted',label="CVX")
plt.plot(x, l, linestyle='dashed',label="MMM")
plt.title('Highest Open Price Each year')
plt.xlabel('Year')
plt.ylabel('Open Price')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)


# As shown above, in 2006, the highest open price of GS in the year is highest among the five stocks, however, at the end which is 
# 2018, the highest open price of BA became the highest one.

# In[ ]:


# try to find the lowest volume of both stocks for each year
Year=[2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]
x=np.array(Year)
GS_volume_min=[GS['2006':'2007']['Volume'].min(),GS['2007':'2008']['Volume'].min(),GS['2008':'2009']['Volume'].min(),
    GS['2009':'2010']['Volume'].min(),GS['2010':'2011']['Volume'].min(),GS['2011':'2012']['Volume'].min(),
    GS['2012':'2013']['Volume'].min(),GS['2013':'2014']['Volume'].min(),GS['2014':'2015']['Volume'].min(),
    GS['2015':'2016']['Volume'].min(),GS['2016':'2017']['Volume'].min(),GS['2017':'2018']['Volume'].min()]
y=np.array(GS_volume_min)

BA_volume_min=[BA['2006':'2007']['Volume'].min(),BA['2007':'2008']['Volume'].min(),BA['2008':'2009']['Volume'].min(),
    BA['2009':'2010']['Volume'].min(),BA['2010':'2011']['Volume'].min(),BA['2011':'2012']['Volume'].min(),
    BA['2012':'2013']['Volume'].min(),BA['2013':'2014']['Volume'].min(),BA['2014':'2015']['Volume'].min(),
    BA['2015':'2016']['Volume'].min(),BA['2016':'2017']['Volume'].min(),BA['2017':'2018']['Volume'].min()]
z=np.array(GS_volume_min)

IBM_volume_min=[IBM['2006':'2007']['Volume'].min(),IBM['2007':'2008']['Volume'].min(),IBM['2008':'2009']['Volume'].min(),
    IBM['2009':'2010']['Volume'].min(),IBM['2010':'2011']['Volume'].min(),IBM['2011':'2012']['Volume'].min(),
    IBM['2012':'2013']['Volume'].min(),IBM['2013':'2014']['Volume'].min(),IBM['2014':'2015']['Volume'].min(),
    IBM['2015':'2016']['Volume'].min(),IBM['2016':'2017']['Volume'].min(),IBM['2017':'2018']['Volume'].min()]
m=np.array(IBM_volume_min)

CVX_volume_min=[CVX['2006':'2007']['Volume'].min(),CVX['2007':'2008']['Volume'].min(),CVX['2008':'2009']['Volume'].min(),
    CVX['2009':'2010']['Volume'].min(),CVX['2010':'2011']['Volume'].min(),CVX['2011':'2012']['Volume'].min(),
    CVX['2012':'2013']['Volume'].min(),CVX['2013':'2014']['Volume'].min(),CVX['2014':'2015']['Volume'].min(),
    CVX['2015':'2016']['Volume'].min(),CVX['2016':'2017']['Volume'].min(),CVX['2017':'2018']['Volume'].min()]
n=np.array(CVX_volume_min)

MMM_volume_min=[MMM['2006':'2007']['Volume'].min(),MMM['2007':'2008']['Volume'].min(),MMM['2008':'2009']['Volume'].min(),
    MMM['2009':'2010']['Volume'].min(),MMM['2010':'2011']['Volume'].min(),MMM['2011':'2012']['Volume'].min(),
    MMM['2012':'2013']['Volume'].min(),MMM['2013':'2014']['Volume'].min(),MMM['2014':'2015']['Volume'].min(),
    MMM['2015':'2016']['Volume'].min(),MMM['2016':'2017']['Volume'].min(),MMM['2017':'2018']['Volume'].min()]
l=np.array(MMM_volume_min)

plt.figure()
plt.plot(x, y, linestyle='solid',label="GS")
plt.plot(x, z, linestyle='dashed',label="BA")
plt.plot(x, m, linestyle='dashdot',label="IBM")
plt.plot(x, n, linestyle='dotted',label="CVX")
plt.plot(x, l, linestyle='dashed',label="MMM")
plt.title('Highest Open Price Each year')
plt.xlabel('Year')
plt.ylabel('Open Price')
plt.legend(loc='best',frameon=False)


# As Shown above, MMM always had the lowest volume over years and till 2018, the lowest volume of each stock in the year is almost
# same

# In[ ]:


# compare close price over year
for ticker in ticker2:
        five_stock[ticker]['Close'].plot(figsize=(12,4),label=ticker)
plt.legend()


# As shown above,the close price of GS was highest in 2006 and BA became the highest one at the end while it was the almost lowest one at the beginning.

# In[ ]:


# compare open price over year
for ticker in ticker2:
        five_stock[ticker]['Open'].plot(figsize=(12,4),label=ticker)
plt.legend()


# As shown above,the open price of GS was highest in 2006 and BA became the highest one at the end while it was the almost lowest one at the beginning.

# In[ ]:


# try to find out the relationship between close price and 30-day average close price for stocks from 2010 to 2011
plt.figure(figsize=(12,6))
GS['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg GS')
GS['Close'].loc['2010-01-01':'2011-01-01'].plot(label='GS CLOSE')
BA['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg BA')
BA['Close'].loc['2010-01-01':'2011-01-01'].plot(label='BA CLOSE')
IBM['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg IBM')
IBM['Close'].loc['2010-01-01':'2011-01-01'].plot(label='IBM CLOSE')
CVX['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg CVX')
CVX['Close'].loc['2010-01-01':'2011-01-01'].plot(label='CVX CLOSE')
MMM['Close'].loc['2010-01-01':'2011-01-01'].rolling(window=30).mean().plot(label='30 Day Avg MMM')
MMM['Close'].loc['2010-01-01':'2011-01-01'].plot(label='MMM CLOSE')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)


# As shown above, the close price of GS is less stable around average close price than others.

# In[ ]:


# compare volume trend
plt.figure(figsize=(12,6))
GS['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='GS Volume')
BA['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='BA Volume')
IBM['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='IBM Volume')
CVX['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='CVX Volume')
MMM['Volume'].loc['2017-01-01':'2018-01-01'].plot(label='MMM Volume')
plt.legend()


# As shown Above, the volume of IBM increases more sharply.

# In[ ]:


# show the correlation between open price of stocks
sns.heatmap(five_stock.xs(key='Open',axis=1,level='Stock Info').corr(),annot=True)


# As shown above, the figures illustrate the relationship between diffrent stocks' open price.

# In[ ]:




