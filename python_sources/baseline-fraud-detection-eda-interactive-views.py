#!/usr/bin/env python
# coding: utf-8

# ## Welcome to my fraud detection Kernel. 
# 
# 

# 
# ![](http://technosavvy.co.ke/wp-content/uploads/2015/12/fraud.jpg)

# # Competition Objective is to detect fraud in transactions; 
# 
# ## Data
# 
# 
# In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target ```isFraud```.
# 
# The data is broken into two files **identity** and **transaction**, which are joined by ```TransactionID```. 
# 
# > Note: Not all transactions have corresponding identity information.
# 
# **Categorical Features - Transaction**
# 
# - ProductCD
# - emaildomain
# - card1 - card6
# - addr1, addr2
# - P_emaildomain
# - R_emaildomain
# - M1 - M9
# 
# **Categorical Features - Identity**
# 
# - DeviceType
# - DeviceInfo
# - id_12 - id_38
# 
# **The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).**
# 
# ## Questions
# I will start exploring based on Categorical Features and Transaction Amounts.
# The aim is answer some questions like:
# - What type of data we have on our data?
# - How many cols, rows, missing values we have?
# - Whats the target distribution?
# - What's the Transactions values distribution of fraud and no fraud transactions?
# - We have predominant fraudulent products? 
# - What features or target shows some interesting patterns? 
# - And a lot of more questions that will raise trought the exploration. 
# 
# 
# I hope you enjoy my kernel and if it be useful for you, <b>upvote</b> the kernel

# ## <font color="red">Please if this kernel were useful for you, please <b>UPVOTE</b> the kernel and give me your feedback =)</font>
# 

# ## Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)

import os
print(os.listdir("../input"))


# ## Reading dataset

# In[ ]:


df_train_id = pd.read_csv("../input/train_identity.csv", nrows=50000)
df_train_trans = pd.read_csv("../input/train_transaction.csv", nrows=50000)
#df_test_id = pd.read_csv("../input/test_identity.csv")
#df_test_trans = pd.read_csv("../input/test_transaction.csv")


# ## Functions to epxlore the data

# In[ ]:


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def plot_distribution(df, var_select=None, title=None, bins=1.0): 
    # Calculate the correlation coefficient between the new variable and the target
    tmp_fraud = df[df['isFraud'] == 1]
    tmp_no_fraud = df[df['isFraud'] == 0]    
    corr = df['isFraud'].corr(df[var_select])
    corr = np.round(corr,3)
    tmp1 = tmp_fraud[var_select].dropna()
    tmp2 = tmp_no_fraud[var_select].dropna()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Fraud', 'No Fraud']
    colors = ['seagreen','indianred', ]

    fig = ff.create_distplot(hist_data,
                             group_labels,
                             colors = colors, 
                             show_hist = True,
                             curve_type='kde', 
                             bin_size = bins
                            )
    
    fig['layout'].update(title = title+' '+'(corr target ='+ str(corr)+')')

    iplot(fig, filename = 'Density plot')
    
def plot_dist_churn(df, col, binary=None):
    tmp_churn = df[df[binary] == 1]
    tmp_no_churn = df[df[binary] == 0]
    tmp_attr = round(tmp_churn[col].value_counts().sort_index() / df[col].value_counts().sort_index(),2)*100
    print(f'Distribution of {col}: ')
    trace1 = go.Bar(
        x=tmp_churn[col].value_counts().sort_index().index,
        y=tmp_churn[col].value_counts().sort_index().values, 
        name='Fraud',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[col].value_counts().sort_index().index,
        y=tmp_no_churn[col].value_counts().sort_index().values,
        name='No Fraud', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace3 =  go.Scatter(   
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        yaxis = 'y2', 
        name='% Fraud', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=2 )
        )
    )
    
    layout = dict(title =  f'Distribution of {str(col)} feature by %Fraud',
              xaxis=dict(type='category'), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [0, 15], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= 'Percentual Fraud Transactions'
                         ))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    iplot(fig)
    
## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## REducing memory
df_train_trans = reduce_mem_usage(df_train_trans)
df_train_id = reduce_mem_usage(df_train_id)


# ## As we have a high dimensional data, I will reduce the memory usage

# In[ ]:


## REducing memory
df_train_trans = reduce_mem_usage(df_train_trans)
df_train_id = reduce_mem_usage(df_train_id)


# ## Knowning the Identity dataset
# - What type of data we have on our data?

# In[ ]:


resumetable(df_train_id)


# We have shape 144.2 rows by 41 columns. <br>
# Also, we can see that almost all features has missing values. We will need to work with that. <br>
# Let's see the transactions table and see the details 

# ## Knowing the transactions
# - What type of data we have on our data?

# In[ ]:


resumetable(df_train_trans)[:25]


# Wow, We have a bizarre high dimension. The shape of Transactions is: 506691, 393<br>
# I will need some time to explore it further. The first aim is start simple. 

# ## Understanding the Target Distribution

# In[ ]:


print("Transactions % Fraud:")
print(round(df_train_trans[['isFraud', 'TransactionID']]['isFraud'].value_counts(normalize=True) * 100,2))
# df_train.groupby('Churn')['customerID'].count().iplot(kind='bar', title='Churn (Target) Distribution', 
#                                                      xTitle='Customer Churn?', yTitle='Count')

trace0 = go.Bar(
    x=df_train_trans[['isFraud', 'TransactionID']].groupby('isFraud')['TransactionID'].count().index,
    y=df_train_trans[['isFraud', 'TransactionID']].groupby('isFraud')['TransactionID'].count().values,
    marker=dict(
        color=['indianred', 'seagreen']),
)

data = [trace0] 
layout = go.Layout(
    title='Fraud (Target) Distribution <br>## 0: No Fraud | 1: Is Fraud ##', 
    xaxis=dict(
        title='Transaction is fraud', 
        type='category'),
    yaxis=dict(
        title='Count')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Nice. <br>
# We have only 3.5% of positive values in our target. It's an unbalanced data and we will keep investiganting the data to find some insight.

# In[ ]:





# In[ ]:


def print_trans(tmp, num_col='TransactionAmt'):
    print(f"The mininum value in Transaction Amount is {tmp[num_col].min()}, median is {round(tmp[num_col].median(),2)}, and the maximum is {df_train_trans[num_col].max()}")
    print(f"The mean Transaction Amount of Fraudulent Transactions is {round(tmp[tmp['isFraud'] == 1][num_col].median(),2)}          \nThe mean Transaction Amount of No-Fraudulent Transactions is {round(tmp[tmp['isFraud'] == 0][num_col].median(),2)}")
    
print_trans(df_train_trans[['isFraud', 'TransactionAmt']], 'TransactionAmt')


# We can see that fraudulent transactions has a higher mean than No-Fraudulent Transactions

# In[ ]:


print("Transaction Amount Quantiles: ")
print(df_train_trans['TransactionAmt'].quantile([0.01, .025, .1, .25, .5, .75, .975, .99]))


# To avoid us of outliers and a better view of distribution, I will filter the data and get only values equal or lower than 800

# ## Ploting and Knowing Transaction Amount distribution 
# 

# In[ ]:


# df_train_trans['TransactionAmt_log'] = df_train_trans['TransactionAmt'].apply(np.log)
tmp = df_train_trans[['TransactionAmt', 'isFraud']]
tmp['TransactionAmt_log'] = tmp['TransactionAmt'].apply(np.log)
## Calling the function
plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt', 'Transaction Amount Distribution', bins=10.0,)
plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt_log', 'Transaction Amount Log Distribution', bins=0.1)


# We don't have a high correlation between Transaction Amount and Fraud Transactions. <br>
# Also, we can see many cases of fraud transactions with values between 5 to 14 and other peak in 75 -  85.
# 
# Let's keep investigating this data

# ## Knowing the Product feature
# - We have predominant fraudulent products? 

# In[ ]:


plot_dist_churn(df_train_trans[['ProductCD', 'isFraud']], 'ProductCD', 'isFraud')


# Cool!!! I think that this chart is very insightful. <br>
# Altought the W is the most frequent Product we can see higher values in C, R and S products altought we have many lowest values in these categories. 

# ## Exploring Card Features 
# We have 6 columns that are about the Card of the transaction.<br>
# I will start by the categoricals and after it, I will explore the continuous 

# In[ ]:


for col in ['card4', 'card6']:
    df_train_trans[col] = df_train_trans[col].fillna('NoInf')
    plot_dist_churn(df_train_trans, col, 'isFraud')


# Cool!! Again, we can clearly see that card4 we can't see different patterns, but in Card6 we can note that Credit has higher incidence of fraud than Debit payment

# In[ ]:


print("Card Features Quantiles: ")
print(df_train_trans[['card1', 'card2', 'card3', 'card5']].quantile([0.01, .025, .1, .25, .5, .75, .975, .99]))


# I will transform Card1 and Card2 to Logarithm scale to we better understand the distribution 

# In[ ]:


for col in ['card1', 'card2', 'card3', 'card5']:
    df_train_trans[str(col)+'_log'] = np.log(df_train_trans[col])


# ## Card1 feature by Target

# In[ ]:


## Calling the function
plot_distribution(df_train_trans[['isFraud','card1_log']], 'card1_log', 'Card 1 Feature Log Distribution by Target', bins=0.05,)


# ## Card2 feature by Target

# In[ ]:


## Calling the function
plot_distribution(df_train_trans[['isFraud','card2_log']], 'card2_log', 'Card 2 Feature Log Distribution by Target', bins=0.05)


# ## Card3 feature by Target
# - As we have many values with low frequency, I will set all values with frequency lower than 10 as -99

# In[ ]:


df_train_trans.loc[df_train_trans.card3.isin(df_train_trans['card3'].value_counts()[df_train_trans['card3'].value_counts() < 10].index), 'card3'] = -99


# In[ ]:


plot_dist_churn(df_train_trans[['card3', 'isFraud']], 'card3', 'isFraud')


# ## Card5 feature by Target
# - Again, as we have many values with low frequency I will set all values with frequency lower than 20 as -99

# In[ ]:


df_train_trans.loc[df_train_trans.card5.isin(df_train_trans['card5']                                             .value_counts()                                             [df_train_trans['card5']                                              .value_counts() < 20]                                             .index), 'card5'] = -99

plot_dist_churn(df_train_trans[['card5', 'isFraud']], 'card5', 'isFraud')


# ## Exploring the M2-M9 features
# - Seen

# In[ ]:


tmp = df_train_trans[['M1','M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'isFraud']]
for col in ['M1','M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
    tmp[col] = tmp[col].fillna('NoInf')
    plot_dist_churn(tmp, col, 'isFraud')


# 

# ## ScatterPollar of Binary Features

# from sklearn.preprocessing import LabelEncoder
# tmp = df_train_trans[]
# #Label encoding Binary columns
# le = LabelEncoder()
# 
# tmp_churn = df_train_trans[df_train_trans['isFraud'] == 1]
# tmp_no_churn = df_train_trans[df_train_trans['isFraud'] == 0]
# 
# bi_cs = df_train_trans.nunique()[df_train_trans.nunique() == 2].keys()
# dat_rad = df_train_trans[bi_cs]
# 
# for cols in bi_cs :
#     tmp_churn[cols] = le.fit_transform(tmp_churn[cols])
#     
# data_frame_x = tmp_churn[bi_cs].sum().reset_index()
# data_frame_x.columns  = ["feature","yes"]
# data_frame_x["no"]    = tmp_churn.shape[0]  - data_frame_x["yes"]
# data_frame_x  = data_frame_x[data_frame_x["feature"] != "Churn"]
# 
# #count of 1's(yes)
# trace1 = go.Scatterpolar(r = data_frame_x["yes"].values.tolist(), 
#                          theta = data_frame_x["feature"].tolist(),
#                          fill  = "toself",name = "Fraud 1's",
#                          mode = "markers+lines", visible=True,
#                          marker = dict(size = 5)
#                         )
# 
# #count of 0's(No)
# trace2 = go.Scatterpolar(r = data_frame_x["no"].values.tolist(),
#                          theta = data_frame_x["feature"].tolist(),
#                          fill  = "toself",name = "Fraud 0's",
#                          mode = "markers+lines", visible=True,
#                          marker = dict(size = 5)
#                         ) 
# for cols in bi_cs :
#     tmp_no_churn[cols] = le.fit_transform(tmp_no_churn[cols])
#     
# data_frame_x = tmp_no_churn[bi_cs].sum().reset_index()
# data_frame_x.columns  = ["feature","yes"]
# data_frame_x["no"]    = tmp_no_churn.shape[0]  - data_frame_x["yes"]
# data_frame_x  = data_frame_x[data_frame_x["feature"] != "Churn"]
# 
# #count of 1's(yes)
# trace3 = go.Scatterpolar(r = data_frame_x["yes"].values.tolist(),
#                          theta = data_frame_x["feature"].tolist(),
#                          fill  = "toself",name = "NoFraud 1's",
#                          mode = "markers+lines", visible=False,
#                          marker = dict(size = 5)
#                         )
# 
# #count of 0's(No)
# trace4 = go.Scatterpolar(r = data_frame_x["no"].values.tolist(),
#                          theta = data_frame_x["feature"].tolist(),
#                          fill  = "toself",name = "NoFraud 0's",
#                          mode = "markers+lines", visible=False,
#                          marker = dict(size = 5)
#                         ) 
# 
# data = [trace1, trace2, trace3, trace4]
# 
# updatemenus = list([
#     dict(active=0,
#          x=-0.15,
#          buttons=list([  
#             dict(
#                 label = 'Fraud Dist',
#                  method = 'update',
#                  args = [{'visible': [True, True, False, False]}, 
#                      {'title': 'Transaction Fraud Binary Counting Distribution'}]),
#              
#              dict(
#                   label = 'No-Fraud Dist',
#                  method = 'update',
#                  args = [{'visible': [False, False, True, True]},
#                      {'title': 'Transaction No-Fraud Binary Counting Distribution'}]),
# 
#         ]),
#     )
# ])
# 
# layout = dict(title='ScatterPolar Distribution of Fraud and No-Fraud Transactions (Select from Dropdown)', 
#               showlegend=False,
#               updatemenus=updatemenus)
# 
# fig = dict(data=data, layout=layout)
# 
# iplot(fig)

# In[ ]:





# # NOTE: THIS KERNEL IS NOT FINISHED. I WILL KEEP EXPLORING IT. 
