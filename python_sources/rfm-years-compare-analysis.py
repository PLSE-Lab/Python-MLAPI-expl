#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Set Options
pd.options.display.max_columns
pd.options.display.max_rows = 500


# In[ ]:


#Import Data
_2010_2011_data =pd.read_excel("/kaggle/input/uci-online-retail-ii-data-set/online_retail_II.xlsx",sheet_name= "Year 2010-2011" )
_2009_2010_data =pd.read_excel("/kaggle/input/uci-online-retail-ii-data-set/online_retail_II.xlsx",sheet_name= "Year 2009-2010" )


# In[ ]:


#Get copy of dataFrame
_2010_2011_df = _2010_2011_data.copy()
_2009_2010_df = _2009_2010_data.copy()


# In[ ]:


def setTotalPrice(data):
    #Add a column for total price to calculate monetary attritube
    data["TotalPrice"] = data["Quantity"]*data["Price"]


# In[ ]:


setTotalPrice(_2010_2011_df)
setTotalPrice(_2009_2010_df)


# In[ ]:


_2010_2011_df.head()


# In[ ]:


_2009_2010_df


# In[ ]:


_2009_2010_df


# ## CHECK MISSING VALUE 

# In[ ]:


#Remove missing Values From CustomerID
_2010_2011_df.dropna(subset= ["Customer ID"],inplace= True)
#Remove zero negative quantity
deleteRows = _2010_2011_df[~_2010_2011_df['Quantity'] > 0].index
_2010_2011_df.drop(deleteRows, axis=0,inplace=True)
#Some rows start with C means refund so we will remove them
deleteRows =  _2010_2011_df[_2010_2011_df["Invoice"].str.contains("C", na=False)].index
_2010_2011_df.drop(deleteRows, axis=0,inplace=True)
#Remove POSTAGE
#deleteRows =  _2010_2011_df[_2010_2011_df["Description"].str.contains("POSTAGE", na=False)].index
#_2010_2011_df.drop(deleteRows, axis=0,inplace=True)


# In[ ]:


#Remove missing Values From CustomerID
_2009_2010_df.dropna(subset= ["Customer ID"],inplace= True)
#Remove zero negative quantity
deleteRows = _2009_2010_df[~_2009_2010_df['Quantity'] > 0].index
_2009_2010_df.drop(deleteRows, axis=0,inplace=True)
#Some rows start with C means refund so we will remove them
deleteRows =  _2009_2010_df[_2009_2010_df["Invoice"].str.contains("C", na=False)].index
_2009_2010_df.drop(deleteRows, axis=0,inplace=True)
#Remove POSTAGE
#deleteRows =  _2009_2010_df[_2009_2010_df["Description"].str.contains("POSTAGE", na=False)].index
#_2009_2010_df.drop(deleteRows, axis=0,inplace=True)


# ## RFM Customer Segmentation

# # Definition
# 
# **RFM Analysis** is a marketing technique used to determine quantitatively which customers are the best ones.
# 
# 
# Customers is rated on this three parameters **from 1 to 5**. 5 for most likely to purchase 1 for least  likely to purchase
# 
# ![rfm-segments.png](attachment:rfm-segments.png)
# 
# ##  RFM Expansion
#    *  Recency - Innovation (R) - Time since last sale purchase
#    *  Frequency - Frequency (F) - Total number of purchases
#    *  Monetary - Monetary (M) - Monetary total of all purchases
# 
# 

# In[ ]:


#return RFM DataFrame
def CalculateRFM(data):
    #Calculate recency
    #Find out the first and last order dates in the data.
    max_date = data['InvoiceDate'].max()
    import datetime as dt
    today_date = dt.datetime(max_date.year,max_date.month,max_date.day)
    recency_df = data.groupby("Customer ID").agg({'InvoiceDate': lambda x: (today_date - x.max()).days})
    recency_df.rename(columns={"InvoiceDate":"Recency"}, inplace= True)
    #calculate Frequency
    temp_df =  data.groupby(['Customer ID','Invoice']).agg({'Invoice': "count"}).groupby(['Customer ID']).agg({"Invoice": "count"})
    freq_df = temp_df.rename(columns={"Invoice": "Frequency"})
    monetary_df=data.groupby("Customer ID").agg({'TotalPrice': "sum"})
    monetary_df.rename(columns={"TotalPrice": "Monetary"}, inplace=True)
    rfm = pd.concat([recency_df,freq_df,monetary_df], axis = 1)
    return rfm 


# In[ ]:


_2010_2011_rfm = CalculateRFM(_2010_2011_df)
_2009_2010_rfm = CalculateRFM(_2009_2010_df)


# In[ ]:


_2010_2011_df.head()


# In[ ]:


#Set RFM Score
def setRFMScore(rfm):
    # Get RFM scores for 3 attribute
    rfm["RecencyScore"] = pd.qcut(rfm['Recency'],5, labels=[5,4,3,2,1])
    #if you calculate only transaction operations(unique invoice per customer) add rank(method="first")
    #if you sum all operations in per invoice no need to add rank method
    rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'].rank(method="first"),5, labels=[1,2,3,4,5])
    rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'],5, labels=[1,2,3,4,5])
    rfm["RFM_SCORE"] = rfm["RecencyScore"].astype(str) +rfm["FrequencyScore"].astype(str)+rfm["MonetaryScore"].astype(str) 
    


# In[ ]:


setRFMScore(_2010_2011_rfm)
setRFMScore(_2009_2010_rfm)


# In[ ]:


def setSegment(rfm):
    seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Loose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'}
    rfm["Segment"] = rfm["RecencyScore"].astype(str) + rfm["FrequencyScore"].astype(str)
    rfm["Segment"] = rfm["Segment"].replace(seg_map,regex=True)
    
    


# In[ ]:


setSegment(_2010_2011_rfm)
setSegment(_2009_2010_rfm)


# In[ ]:


_2009_2010_rfm


# In[ ]:


_2010_2011_rfm


# ## Data Visualiation

# In[ ]:


new_2009_2010_rfm = _2009_2010_rfm.copy()
new_2010_2011_rfm = _2010_2011_rfm.copy()


# In[ ]:


# count the number of customers in each segment


def displayBar(data):
    segments_counts = data.value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots()

    bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
    ax.set_frame_on(False)
    ax.tick_params(left=False,
                   bottom=False,
                   labelbottom=False)
    ax.set_yticks(range(len(segments_counts)))
    ax.set_yticklabels(segments_counts.index)
    
    for i, bar in enumerate(bars):
            value = bar.get_width()
            if segments_counts.index[i] in ['Champions', 'Loyal Customers']:
                bar.set_color('firebrick')
            ax.text(value,
                    bar.get_y() + bar.get_height()/2,
                    '{:,} ({:}%)'.format(int(value),
                                       int(value*100/segments_counts.sum())),
                    va='center',
                    ha='left',
                    fontsize=14,
                      weight='bold'
                   )
    
    plt.show()


# In[ ]:


displayBar(new_2009_2010_rfm['Segment'])


# In[ ]:


displayBar(new_2010_2011_rfm['Segment'])



# In[ ]:


new_2009_2010_rfm.rename(columns={"Segment": "Segment_2009_2010"}, inplace=True)
new_2010_2011_rfm.rename(columns={"Segment": "Segment_2010_2011"}, inplace=True)


# In[ ]:


new_2009_2010_rfm


# In[ ]:


len(new_2009_2010_rfm[new_2009_2010_rfm["Segment_2009_2010"] == "Champions"].index)


# In[ ]:


len(new_2010_2011_rfm[new_2010_2011_rfm["Segment_2010_2011"] == "About to Sleep"].index)


# In[ ]:


new_2010_2011_rfm


# In[ ]:


new_2010_2011_rfm.loc[:, ["Segment_2010_2011"]] 


# In[ ]:


new_2009_2010_rfm.loc[:, ["Segment_2009_2010"]] 


# In[ ]:


merge1 = new_2009_2010_rfm.loc[:, ["Segment_2009_2010"]]
merge2 = new_2010_2011_rfm.loc[:, ["Segment_2010_2011"]]

merge1.merge(merge2, how="left",on ="Customer ID").head(100)


# In[ ]:


pd.crosstab(index=merge1["Segment_2009_2010"], columns= merge2["Segment_2010_2011"])


# In[ ]:


ids =new_2009_2010_rfm[new_2009_2010_rfm["Segment_2009_2010"] == "Champions"].index
displayBar(new_2010_2011_rfm[new_2010_2011_rfm.index.get_level_values('Customer ID').isin(ids)].loc[:,"Segment_2010_2011"])


# In[ ]:


ids =new_2009_2010_rfm[new_2009_2010_rfm["Segment_2009_2010"] == "Need Attention"].index
displayBar(new_2010_2011_rfm[new_2010_2011_rfm.index.get_level_values('Customer ID').isin(ids)].loc[:,"Segment_2010_2011"])

