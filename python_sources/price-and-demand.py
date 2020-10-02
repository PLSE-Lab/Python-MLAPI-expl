#!/usr/bin/env python
# coding: utf-8

# # Price Demand Curve

# <img src="https://www.investopedia.com/thmb/OeU-W_PH3kN8cTLz-Fg1Le23Zfw=/1000x1000/smart/filters:no_upscale()/demand_curve2-1a87890730a044e79de897ddb61ccc76.PNG" width=500/>

# ## I had done a little work recently on price elasticity so wanted to see if i could create a price-demand curve for some of these products

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
submission_file = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')


# ## function creates the price demand curve, attempts a linear fit of the data and returns the mean-squared-error

# In[ ]:


def disp_price_demand(product,plot=False):
    
    # create series to map the day index to the week number
    df = pd.Series(calendar.wm_yr_wk.values,index=calendar.d)

    # keep only sale data for product
    item1 = sell_prices[sell_prices.item_id.isin([product])]
    
    # create series to map week number to price
    df2 = item1.groupby('wm_yr_wk').mean().sell_price

    # keep only data for product
    sale_item = train_sales[train_sales.item_id.isin([product])]
    
    # sum over all stores
    sales = sale_item.sum()[6:]
    
    # create dataframe and use df and df2 to map over the day of sales into the sale price
    total = pd.DataFrame()
    total["d"] = sales.index
    total["sales"] = sales.values.astype(int)
    total["wk"] = total.d.map(df).fillna(0).astype(int)
    total["price"] = total.wk.map(df2)


    # drop nans and round to 1 decimal place
    tdf = total.dropna()
    tdf.price = tdf.price.round(1)
    
    #print('total_sales: ',tdf.sales.sum())
    
    # get the average number of sales for a given price
    tdf1 = tdf.groupby('price').mean()
    
    # get the std of sales for a given price
    error = tdf.groupby('price').std().sales
    
    # performing a fit so want at least 3 different prices
    if len(tdf1)>2:
        
        # get linear fit
        m,c = np.polyfit(tdf1.sales,tdf1.index,1)
        
        # create values for plot
        xvalues=np.linspace(tdf1.sales.min()-error.mean(),tdf1.sales.max()+error.mean(),10,)
        yvalues =  m*xvalues + c
        
        # get predictions for mse
        y_pred = m*(tdf1.sales) + c
                
        mse = mean_squared_error(y_pred,tdf1.index)

        if plot:
            # plot fit
            plt.plot(xvalues,yvalues)
            
            # plot data with error bars
            plt.errorbar(tdf1.sales,tdf1.index,xerr=error,marker='o')
            plt.xlabel('quantity sold /day')
            plt.ylabel('price')
            
            plt.show()

        return mse
    else:
        return 1000


# In[ ]:


# get a subsection of data
names = sorted(train_sales.item_id.unique()[::20])
num = len(names)
mse_list=np.zeros(num)

for idx in tqdm(range(num)):
    product = names[idx]
    #print('product: ',product)
    mse = disp_price_demand(product)
    mse_list[idx] = mse


# ## Display the best fits

# In[ ]:


for i in np.argsort(mse_list)[:10]:
    product = names[i]
    print(product)
    mse = disp_price_demand(product,plot=True)
    print(mse_list[i])


# In[ ]:


for i in np.argsort(mse_list)[:20]:
    product = names[i]
    print(product)


# ## Display some bad ones

# In[ ]:


for i in np.argsort(mse_list)[90:100]:
    product = names[i]
    print(product)
    mse = disp_price_demand(product,plot=True)
    print(mse_list[i])


# * ## Most products do not show a clear demand curve
# 
# * ## Many also show a reverse trend with quantity and price
# 
# * ## There is a large deviation in quantity sold for a given price
# 
# # As expected, quantity sold is not merely a function of price

# In[ ]:




