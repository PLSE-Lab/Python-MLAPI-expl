#!/usr/bin/env python
# coding: utf-8

# Good Day to you all. I am uploading my first kernel in Kaggle.Feeling excited :)
# I have gone thru some of the kernels which covered basic EDA,outliers,missingvalues etc.
# Hence I am not including them in my kernel in order not to sound redundant .
# Hope you find this notebook informative :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
def loaddata(filename, nrows=None):

    types = {'id': 'int64',
             'item_nbr': 'int32',
             'store_nbr': 'int16',
             'unit_sales': 'float32',
             'onpromotion': bool,
    }

    data = pd.read_csv(filename, parse_dates=['date'], 
                                     dtype=types, nrows=nrows, infer_datetime_format=True)

    return data
traindata=loaddata("../input/train.csv")
testdata=pd.read_csv("../input/test.csv",sep=",")
itemdata=pd.read_csv("../input/items.csv",sep=",")
stores=pd.read_csv("../input/stores.csv",sep=",")
oil=pd.read_csv("../input/oil.csv",sep=",")
holidays=pd.read_csv("../input/holidays_events.csv",sep=",")
transactions=pd.read_csv("../input/transactions.csv",sep=",")
samplesub=pd.read_csv("../input/sample_submission.csv",sep=",")


# Since basic EDa has already been discussed in various kernels , I am just giving a brief timeline overview on all the tables.Ignore if you find it reprtitive

# In[ ]:


min(traindata['date'])


# In[ ]:


max(traindata['date'])


# Training data ranges from 2013-01-01 to '2017-08-15'

# In[ ]:


min(testdata['date'])


# In[ ]:


max(testdata['date'])


# testdata ranges from '2017-08-16 ' to '2017-08-31'

# In[ ]:


min(oil['date'])


# In[ ]:


max(oil['date'])


# Oil data ranges from '2013-01-01' to '2017-08-31'.

# In[ ]:


min(transactions['date'])


# In[ ]:


max(transactions['date'])


# Transaction data ranges from  '2013-01-01' to '2017-08-15'

# **STORE WISE - OIL PRICE DEPENDANCY ANALYSIS**

# In[ ]:


#storewise/statewise transactions ,Merging "Stores" and "Transactions" table
stores_trans=stores.merge(transactions,on='store_nbr',how='inner')
stores_trans.head()


# In[ ]:


#calculating avg no. of transactions statewise
statewise_trans=stores_trans.groupby(by=['state','date'])['transactions'].mean()
statewise_trans=pd.DataFrame(statewise_trans)
statewise_trans=statewise_trans.reset_index(level=['state','date'])

statewise_trans.head()


# In[ ]:


#Discarding first row as oil price on '2013-01-01' is NaN
oil=oil.loc[1:,]
#merging the "statewise_trans" table with oil table 
oil_state_trans=oil.merge(statewise_trans,how='outer',on='date')
oil_state_trans.head()


# Plotting trends of "Oil prices" and "Average no. of transactions" statewise.

# In[ ]:


###plotting
fig, ax = plt.subplots()
states = oil_state_trans.state.unique()
oil_state_trans.reset_index().groupby('state').plot(x='date', y='transactions',ax=ax,legend=None)
plt.show()
#oil pattern
oil_state_trans=oil_state_trans.sort_values('date',ascending=True)
oil_state_trans.plot(x='date',y='dcoilwtico')
plt.show()


# In[ ]:


#transaction patterns of state "Azuay"
oil_state_trans[oil_state_trans['state']=='Azuay'].plot('date','transactions')
plt.legend('Azuay')
#transaction patterns of state "Pastaza"
oil_state_trans[oil_state_trans['state']=='Pastaza'].plot('date','transactions')
plt.legend("Pastaza")
plt.show()


# Trend of "Avg. no of transactions" is clearly different from the oil prices. Hence effect of oil prices may not be significant on the "No.of transactions/Sales/purchasing power" at retail stores. To further support our insight, the below calculations can be shown

# In[ ]:


#first half average of oil prices: date range '2013-01-1' to '2015-01-01'
oil['dcoilwtico'][(oil['date']>'2013-01-01') & (oil['date']<'2015-01-01')].mean()


# In[ ]:


#Second half average of oil prices: date range above '2015-01-02'
oil['dcoilwtico'][oil['date']>='2015-01-01'].mean()


# In[ ]:


#transactions of statewise in firsthalf
oil_state_trans['transactions'][(oil_state_trans['date']>'2013-01-01') & (oil_state_trans['date']<'2015-01-01')].mean()


# In[ ]:


#transactions of statewise in second half
oil_state_trans['transactions'][oil_state_trans['date']>='2015-01-01'].mean()


# There has been significant decrease in oil prices from 2015 jan. But the sales/transactions decreased slightly since then. Further we can analyse which products are susceptible to oil price changes

# Some stores have transactions  starting from 2013 but for few others transaction data started from 2015.To check the date ranges the following code can be used

# In[ ]:


#storewise min and max date of transactions 
storewisemindates=[transactions.groupby(by=['store_nbr'])['date'].min()]
storewisemaxdates=[transactions.groupby(by=['store_nbr'])['date'].max()]


# In[ ]:


storewisemindates


# In[ ]:


storewisemaxdates


# **PRODUCT CATEGORY WISE ANALYSIS**

# In[ ]:


def productcatwiseanalysis(productcat):
    #Filtering only beverages from itemdata table
    itemlist=itemdata[itemdata['family']==productcat]
    itemlist=itemlist['item_nbr']
    itemlist=pd.DataFrame(itemlist)
    print("No. of items in "+productcat+ " : "+ str(itemlist['item_nbr'].nunique()))

#Joining traindata and Beverage items table
    productcat_traindata=itemlist.merge(traindata,on="item_nbr",how='inner')
    productcat_traindata=productcat_traindata[['date','store_nbr','unit_sales']]
    productcat_traindata['year'] = productcat_traindata['date'].dt.year
    productcat_traindata['month'] = productcat_traindata['date'].dt.month
    storemonthgrpby=productcat_traindata.groupby(['store_nbr','month'])['unit_sales'].mean()
    storemonthgrpby=storemonthgrpby.reset_index()

    print("Storewise- Avg.Unitsales plot")
    y=plt.bar(storemonthgrpby['store_nbr'],storemonthgrpby['unit_sales'],align="center")
    plt.legend()
    plt.show()
    print("Monthlywise- Avg.Unit_sales plot")
    x=plt.bar(storemonthgrpby['month'],storemonthgrpby['unit_sales'],align="center")
#g.plot('store_nbr','unit_sales')
    plt.legend()
    plt.show()
#Variability of avg sales within stores
    print("Difference between maximum and min values of Avg.sales for Each store")
    Maxminusmin=storemonthgrpby.groupby(['store_nbr'])['unit_sales'].max()-storemonthgrpby.groupby(['store_nbr'])['unit_sales'].min()
    Maxminusmin=Maxminusmin.reset_index()
    plt.bar(Maxminusmin['store_nbr'],Maxminusmin['unit_sales'])
    plt.legend()
    plt.show()
    del productcat_traindata,Maxminusmin,storemonthgrpby,itemlist


# In[ ]:


#Storewise, Monthlywise analysis of product category - SEAFOODs
productcatwiseanalysis('SEAFOOD')


# Inferences from "SEAFOOD" category 
# * graph1: Stores 3 and 44-48 store numbers register maximum no.of transactions
# * graph2: No clear presence of seasonaility . Monthly wise Avg.Sales vary slightly
# * graph3:Stores with max Sales exhibit high Max-min avgsales value(stoes 3 and 44-48)

# In[ ]:


#Storewise, Monthlywise analysis of product category - BEVERAGES
productcatwiseanalysis('BEVERAGES')


# Similar inferences can be drawn for product category "BEVERAGES" also. Plots for different product categories can be drawn by calling the function "**productcatwiseanalysis**"

# In[ ]:


hlist=holidays['date'].tolist()
holidaylist=[n for n in oil['date'] if n in hlist]
len(holidaylist)


# **PRODUCT CAT -SALES PATTERN ANALYSIS**

# In[ ]:


traindata['dow']=traindata['date'].dt.weekday_name
traindata['IsWeekend']=['1' if (x=='Saturday' or x=='Sunday') else '0' for x in traindata['dow']]
traindata.head()


# In[ ]:


del traindata['dow']


# In[ ]:


Nonperishableitems=itemdata[itemdata['perishable']==0]
Nonperishableitems.shape


# In[ ]:


perishableitems=itemdata[itemdata['perishable']==1]
perishableitems.shape


# In[ ]:


def ProdcatGoodsanalysis(productcat,storenumber):
    storetraindata=traindata[traindata['store_nbr']==storenumber]
    Nonperishableitems=itemdata[itemdata['perishable']==0]
    perishableitems=itemdata[itemdata['perishable']==1]

    perishableitemlist=perishableitems[perishableitems['family']==productcat]
    perishableitemlist=perishableitemlist['item_nbr']
    perishableitemlist=pd.DataFrame(perishableitemlist)
    print("No. of perishable items in  "+productcat+ ":"+ str(perishableitemlist['item_nbr'].nunique()))
    
    Nonperishableitemlist=Nonperishableitems[Nonperishableitems['family']==productcat]
    Nonperishableitems=Nonperishableitems['item_nbr']
    Nonperishableitems=pd.DataFrame(Nonperishableitems)
    print("No. of nonperishable items in  "+productcat+ ":"+ str(Nonperishableitemlist['item_nbr'].nunique()))
    
    def perishableitemanlysis():
        productcat_traindata=perishableitemlist.merge(storetraindata,on="item_nbr",how='inner')
        productcat_traindata=productcat_traindata[['date','store_nbr','unit_sales','IsWeekend']]
        storeweekendgrpby=productcat_traindata.groupby(['store_nbr','IsWeekend'])['unit_sales'].mean()
        storeweekendgrpby=storeweekendgrpby.reset_index()
        storeweekendgrpby['IsWeekend']=pd.factorize(storeweekendgrpby['IsWeekend'])[0]
        #storeweekendgrpby = storeweekendgrpby.groupby(['store_nbr','IsWeekend'])['unit_sales'].mean()
        #storeweekendgrpby.plot.bar()
        DF1 = storeweekendgrpby.groupby(['store_nbr', 'IsWeekend'])
        salesavg = DF1['unit_sales'].aggregate(np.mean).unstack()
        salesavg.plot(kind = 'bar', title = 'by store,Weekend')
        plt.ylabel('sales')
        plt.show()
        
    def Nonperishableitemanlysis():
        productcat_traindata=Nonperishableitemlist.merge(storetraindata,on="item_nbr",how='inner')
        productcat_traindata=productcat_traindata[['date','store_nbr','unit_sales','IsWeekend']]
        storeweekendgrpby=productcat_traindata.groupby(['store_nbr','IsWeekend'])['unit_sales'].mean()
        storeweekendgrpby=storeweekendgrpby.reset_index()
        storeweekendgrpby['IsWeekend']=pd.factorize(storeweekendgrpby['IsWeekend'])[0]
        #storeweekendgrpby = storeweekendgrpby.groupby(['store_nbr','IsWeekend'])['unit_sales'].mean()
        #storeweekendgrpby.plot.bar()
        DF1 = storeweekendgrpby.groupby(['store_nbr', 'IsWeekend'])
        salesavg = DF1['unit_sales'].aggregate(np.mean).unstack()
        salesavg.plot(kind = 'bar', title = 'by store,Weekend')
        plt.ylabel('sales')
        plt.show()
        del DF1,productcat_traindata
    if (perishableitemlist['item_nbr'].nunique()==0):
        print("No perishable items in the product cat")
        Nonperishableitemanlysis()
    else:
        perishableitemanlysis()
    del Nonperishableitemlist,perishableitemlist,storetraindata


# In[ ]:


ProdcatGoodsanalysis('BEVERAGES',24)


# I created this function to view detailed product categories analysis **ProdcatGoodsanalysis** with arguements Product cat and store number
# A product cat belongs to either "Perishable goods" or "nonperishable"  category.
# Above graph shows "Avg sales of stores" of that particular category  in weekend and weekday 
# 

# **Further Analysis**:
# I am planing to draw patterns,timeseries analysis and product interactions across prodcat.
# To maintain brevity of the Notebook , I am limiting my analysis till product category, further analysis will be uploaded soon
# Suggestions and comments are welcomed.Like the kernel if you find it informative and useful
# Thanks, Happy learning :) :)
# 
