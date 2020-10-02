#!/usr/bin/env python
# coding: utf-8

# * In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.
# * The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.
# * Categorical Features - Transaction
#     * ProductCD
#     * card1 - card6
#     * addr1, addr2
#     * P_emaildomain
#     * R_emaildomain
#     * M1 - M9
# * Categorical Features - Identity
#     * DeviceType
#     * DeviceInfo
#     * id_12 - id_38
# * The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).
# 
# * You can read more about the data from this post by the competition host.
# * Files
# * train_{transaction, identity}.csv - the training set
# * test_{transaction, identity}.csv - the test set (you must predict the isFraud value for these observations)
# * sample_submission.csv - a sample submission file in the correct format

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.3f}'.format


# In[ ]:


train_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv", index_col= 'TransactionID')
train_trans = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv",index_col= 'TransactionID')


# In[ ]:


test_trans = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv", index_col= 'TransactionID')
test_id= pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv", index_col= 'TransactionID')


# In[ ]:


train_id.head()


# In[ ]:


TotalTransCount = train_trans.shape[0]
TotalIDCount = train_id.shape[0]
TotalValidTransCount = train_trans['isFraud'].value_counts()[0]
TotalFraudTransCount = train_trans['isFraud'].value_counts()[1]
TotalAmount = train_trans['TransactionAmt'].sum()
TotalFraudTransAmount =round( train_trans[train_trans['isFraud']==1]['TransactionAmt'].sum(),3)
TotalValidTransAmount = round(train_trans[train_trans['isFraud']==0]['TransactionAmt'].sum(), 3)
print('Total Transactions: ',TotalTransCount)
print('Total IDs Available: ',TotalIDCount)
print('Total valid Transactions: ',TotalValidTransCount)
print('Total Fraud Transactions: ',TotalFraudTransCount)
print('Total Amount: ',TotalAmount)
print('Total Amount in Fraud Transactions: ',TotalFraudTransAmount)
print('Total Amount in Valid Transactions: ',TotalValidTransAmount)


# In[ ]:


train_trans = train_trans.merge(train_id, how='left', left_index=True, right_index=True) # MERGING TRAIN TRANSACTION AND IDENTITY DATASETS


# In[ ]:


# QUANTILES GIVE US A GOOD IDEA OF HOW OUR DATA IS DISTRIBUTED
quantiles =  train_trans.quantile([.01, .20, .25, .50, .70, .75,.90, .95, .96,.99]) 
quantiles


# In[ ]:


# THIS FUNCTION TAKES A DATAFRAME AND RETURNS A DATAFRAME THAT CONTAINS INFORMATION OF COLUMNS LIKE NAME, DATA TYPE, MISSING VALUES, PERCENTAGE OF MISSING VALUES, UNIQUE VALUES
def metadata(df):
    metadata = pd.DataFrame()
    metadata['Name'] = df.columns.values
    metadata['Dtype'] = df.dtypes.values
    t =df.shape[0]
    missing =[]
    permissing = []
    Unique =[]
    for col in df.columns:
        s = df[col].isnull().sum()
        missing.append(s)
        permissing.append(round((s/t)*100, 3))
        Unique.append(df[col].nunique())
    metadata['Missing'] = missing
    metadata['Percentage_Missing'] = permissing
    metadata['Unique'] = Unique
    return metadata


# In[ ]:


# THIS FUNCTION IS USED TO SET SOME VALUES TO 'OTHER' FOR BETTER GRAPH PLOTING
def othervalues(val, others):
    if val in others:
        return val
    else:
        return 'Others'


# In[ ]:


# THIS FUNCTION TAKES A DATAFRAME AND SET LIST OF COLUMNS AND PLOTS 3 BAR PLOTS ABOUT FRAUD TRANSACTION.
def count_perFraud_plots(df, col, outliers = True, thresholdType='None', threshold = 0, fontsize =15, figsizeW =16, figsizeH=20, rotation =0):
    # Creating a new temperory dataframe for the function
    temp_df = df[[col,'isFraud','TransactionAmt']].copy()
   
    #print(temp_df.isnull().sum())
    if outliers:
        if thresholdType == 'quantile':
            #print(temp_df.isnull().sum())
            temp_df.loc[list(temp_df[temp_df[col]>quantiles.loc[threshold, col]].index.values), col] = 'Other'
        elif thresholdType == 'value count':
            others = list(temp_df[col].fillna('NaN').value_counts()[temp_df[col].fillna('NaN').value_counts()>threshold].index)
            temp_df[col] = temp_df[col].fillna('NaN').apply(othervalues, others=others)
            
    temp_df = temp_df.fillna('NaN')        
    group =pd.crosstab(temp_df[col], temp_df['isFraud'], normalize='index')*100 # Gives a dataframe with 'isFraud' column cotaining percetage of frauds for each value
    group = group.reset_index()
    #total =  len(df[col])
    
    if(rotation>0):
        xtickHaAlignment = 'right'
    else:
        xtickHaAlignment = 'center'

    plt.figure(figsize=(figsizeW,figsizeH))

    plt.subplot(421)
    plot1 = sns.countplot(data=temp_df, x=col, order = list(group[col].values), palette = 'dark')
    for bar in plot1.patches:
        plot1.text(bar.get_x()+bar.get_width()/2, 
                     bar.get_height(), 
                     '{:.2f}%'.format((bar.get_height()/TotalTransCount)*100), ha='center', va='bottom')
    plot1.set_title('Total Transactions for {}'.format(col), fontsize = fontsize)
    plot1.set_ylabel('No Of Transactions',  fontsize = fontsize-2)
    plot1.set_xlabel('{}'.format(col), fontsize = fontsize-2)
    plot1.set_xticklabels(plot1.get_xticklabels(),rotation=rotation, ha=xtickHaAlignment, va='top')
    
    plt.subplot(422)
    plot2 = sns.barplot(data=group, x=col,y = 1)
    for bar in plot2.patches:
        plot2.text(bar.get_x()+bar.get_width()/2, 
                     bar.get_height(), 
                     '{:.2f}%'.format(bar.get_height()), ha='center',va='bottom')
    plot2.set_title('Percentage of Fraud Transaction in each value of {}'.format(col), fontsize = fontsize)
    plot2.set_ylabel('Percentage', fontsize = fontsize-2)
    plot2.set_xlabel('{}'.format(col), fontsize = fontsize-2)
    plot2.set_xticklabels(plot2.get_xticklabels(),rotation=rotation, ha=xtickHaAlignment, va='top')
    
    percOfTransOutOfTotalFrauds = (pd.crosstab(temp_df[col], temp_df['isFraud'], normalize='columns')*100).reset_index() # Gives dataframe with column containing percentage fraud transactions of each values out of total fraud trasaction
    
    plt.subplot(423)
    plot3 = sns.barplot(data=percOfTransOutOfTotalFrauds, x=col, y=1)
    for bar in plot3.patches:
        plot3.text(bar.get_x()+bar.get_width()/2, 
                     bar.get_height(), 
                     '{:.2f}%'.format(bar.get_height()), ha='center',va='bottom')
    plot3.set_title('Percentge of Fraud Transaction Out Of Total Fraud Transactions', fontsize = fontsize)
    plot3.set_ylabel('Percentge', fontsize = fontsize-2)
    plot3.set_xlabel('{}'.format(col), fontsize = fontsize-2)
    plot3.set_xticklabels(plot3.get_xticklabels(),rotation=rotation, ha=xtickHaAlignment, va='top')
    
    total_amount = temp_df['TransactionAmt'].groupby(temp_df[col]).sum() # Gives a dataframe with sum of transaction amount for each values in the column. 
    total_amount = total_amount.reset_index()
    #total_amount= df['TransactionAmt'].sum()
    percent_total = (pd.crosstab(temp_df[col], temp_df['isFraud'], temp_df['TransactionAmt'], aggfunc=sum, normalize='index')*100).reset_index() # For each value in 'col' column given by user calculate total transaction Amount with respect to 'isFraud' column values and calculate perctage of valid and fraud amount out of total transaction amount of each value of 'col'
    #percent_total = temp_c1['TransactionAmt'].groupby([temp_c1[col],temp_c1['isFraud']]).sum()/total_amt_c1*100
    #percent_total =percent_total.unstack()
    #percent_total = percent_total.reset_index()
    
    plt.subplot(425)
    plot5 = sns.barplot(data=total_amount, x=col, y = 'TransactionAmt', palette = 'dark')
    plot5.ticklabel_format(style='plain', axis='y', useOffset=False)
    for bar in plot5.patches:
        plot5.text(bar.get_x()+bar.get_width()/2, 
                     bar.get_height(), 
                     '{:.2f}%'.format((bar.get_height()/TotalAmount)*100), ha='center', va='bottom')
    plot5.set_title('Total Amount in transaction for {}'.format(col), fontsize = fontsize)
    plot5.set_ylabel('Total Amount', fontsize = fontsize-2)
    plot5.set_xlabel('{}'.format(col), fontsize = fontsize-2)
    plot5.set_xticklabels(plot5.get_xticklabels(),rotation=rotation, ha=xtickHaAlignment, va='top')
    
    plt.subplot(426)
    plot6 = sns.barplot(data=percent_total, x=col,y = 1)
    for bar in plot6.patches:
        plot6.text(bar.get_x()+bar.get_width()/2, 
                     bar.get_height(), 
                     '{:.2f}%'.format(bar.get_height()), ha='center', va='bottom')
    plot6.set_title('Percentge of Fraud Transaction Amount in each value of {}'.format(col), fontsize = fontsize)
    plot6.set_ylabel('Percentge of Fraud Money', fontsize = fontsize-2)
    plot6.set_xlabel('{}'.format(col), fontsize = fontsize-2)
    plot6.set_xticklabels(plot6.get_xticklabels(),rotation=rotation, ha=xtickHaAlignment, va='top')
    
    percOfTransAmountOutOfTotalFraudsAmount = (pd.crosstab(temp_df[col], temp_df['isFraud'], temp_df['TransactionAmt'], aggfunc=sum, normalize='columns')*100).reset_index() # For each value in 'col' column given by user calculate total transaction Amount with respect to 'isFraud' column values and calculate perctage of valid and fraud amount out of total transaction amount of each values of 'isFraud'
    plt.subplot(427)
                   
    plot7 = sns.barplot(data=percOfTransAmountOutOfTotalFraudsAmount, x=col, y=1)
    for bar in plot7.patches:
        plot7.text(bar.get_x()+bar.get_width()/2, 
                     bar.get_height(), 
                     '{:.2f}%'.format(bar.get_height()), ha='center', va='bottom')
    plot7.set_title('Percentge of Fraud Transaction Amount Out Of Total Fraud Transaction Amount', fontsize = fontsize)
    plot7.set_ylabel('Percentge', fontsize = fontsize-2)
    plot7.set_xlabel('{}'.format(col), fontsize = fontsize-2)
    plot7.set_xticklabels(plot7.get_xticklabels(),rotation=rotation, ha=xtickHaAlignment, va='top')
    plt.tight_layout()
    plt.show()


# In[ ]:


# THIS FUNCTION WILL PLOT A CORRELATION HEATMAP WITH A SET THRESHOLD OF 0.9 CORRELATION.
def corrfunc(df , col):
    color = plt.get_cmap('RdYlGn') 
    color.set_bad('green') 
    correalation =df[col].corr()
    correalation[np.abs(correalation)<.9] = 0 # This will set all correlations less than 0.9 to 0
    plt.figure(figsize= (len(col),len(col)))
    sns.heatmap(correalation, yticklabels= True, annot = True, vmin=-1, vmax=1,cmap = color)


# In[ ]:


# THIS FUNCTION WILL PLOT BOXENPLOT AND VIOLIN PLOT 
def plot_boxen_violin(df, amountThreshold, col, outliers=False ,thresholdType='value count',threshold=0):
    temp_df = df[[col,'isFraud','TransactionAmt']].copy()
    if outliers:
        if thresholdType == 'quantile':
            #print(temp_df.isnull().sum())
            temp_df.loc[list(temp_df[temp_df[col]>quantiles.loc[threshold, col]].index.values), col] = 'Other'
            print('Q Here')
        elif thresholdType == 'value count':
            others = list(temp_df[col].fillna('NaN').value_counts()[temp_df[col].fillna('NaN').value_counts()>threshold].index)
            temp_df[col] = temp_df[col].fillna('NaN').apply(othervalues, others=others)
            print('V Here')
    temp_df = temp_df.fillna('NaN')   
    plt.figure(figsize=(16,6))
    plt.subplot(121) 
    sns.boxenplot(data=temp_df[temp_df['TransactionAmt']<amountThreshold], x=col, y='TransactionAmt', hue=df['isFraud'].map({0:'No Fraud', 1: 'Fraud'}))
    plt.subplot(122)
    plot = sns.violinplot(data=temp_df[temp_df['TransactionAmt']<amountThreshold], x=col, y='TransactionAmt', hue= df['isFraud'].map({0:'No Fraud', 1: 'Fraud'}),split=True, inner="quartiles")
    plot.legend(loc=1)
    plt.show()


# In[ ]:


browsermapping = {'google':'Google', 'google search application 48.0':'Google' ,  'google search application 49.0':'Google' ,
                'android webview 4.0': 'Android' , 'Generic/Android 7.0': 'Android', 'Generic/Android': 'Android', 'android browser 4.0': 'Android', 'android': 'Android',
                'samsung browser 6.2' : 'Samsung' , 'mobile safari 11.0':'Safari Mobile' , 'chrome 62.0':'Chrome PC' ,
                'chrome 62.0 for android':'Chrome Android' , 'edge 15.0':'Edge' , 'mobile safari generic':'Safari Mobile' ,
                'chrome 49.0':'Chrome PC' , 'chrome 61.0':'Chrome PC' , 'edge 16.0':'Edge' , 'safari generic':'Safari' ,
                'edge 14.0':'Edge' , 'chrome 56.0 for android':'Chrome Android' , 'firefox 57.0':'Firefox' ,
                'chrome 54.0 for android':'Chrome Android' , 'mobile safari uiwebview':'Safari Mobile' , 'chrome':'Chrome PC' ,
                'chrome 62.0 for ios':'Chrome IOS' , 'firefox':'Firefox' , 'chrome 60.0 for android':'Chrome Android' ,
                'mobile safari 10.0':'Safari Mobile' , 'chrome 61.0 for android':'Chrome Android' ,
                'ie 11.0 for desktop':'IE PC', 'Microsoft/Windows':'IE PC' , 'ie 11.0 for tablet':'IE TABLET' , 'mobile safari 9.0':'Safari Mobile',
                'chrome generic':'Chrome PC' , 'chrome 59.0 for android':'Chrome Android' ,
                'firefox 56.0':'Firefox' , 'chrome 55.0':'Chrome PC' , 'opera 49.0':'Opera' ,
                'ie':'IE PC' , 'chrome 55.0 for android':'Chrome Android' , 'firefox 52.0':'Firefox' ,
                'chrome 57.0 for android':'Chrome Android' , 'chrome 56.0':'Chrome PC' ,
                'chrome 46.0 for android':'Chrome Android' , 'chrome 58.0':'Chrome PC' , 'firefox 48.0':'Firefox' ,
                'chrome 59.0':'Chrome PC' , 'samsung browser 4.0':'Samsung', 'edge 13.0':'Edge' ,
                'chrome 53.0 for android':'Chrome Android' , 'chrome 58.0 for android':'Chrome Android' ,
                'chrome 60.0':'Chrome PC' , 'mobile safari 8.0':'Safari Mobile', 'firefox generic':'Firefox' , 'Samsung/SM-G532M':'Samsung',
                'chrome 50.0 for android':'Chrome Android' , 'chrome 51.0 for android':'Chrome Android' ,
                'chrome 63.0':'Chrome PC' , 'chrome 52.0 for android':'Chrome Android' , 'chrome 51.0':'Chrome PC' ,
                'firefox 55.0':'Firefox' , 'edge':'Edge' , 'opera':'Opera' , 'chrome generic for android':'Chrome Android' ,
                'samsung browser 5.4':'Samsung', 'Samsung/SCH':'Samsung', 'chrome 57.0':'Chrome PC' ,
                'firefox 47.0':'Firefox' , 'chrome 63.0 for android':'Chrome Android' , 'Samsung/SM-G531H':'Samsung',
                'chrome 43.0 for android':'Chrome Android' , 'chrome 63.0 for ios':'Chrome IOS' , 
                'chrome 49.0 for android':'Chrome Android' , 'safari':'Safari', 'samsung browser 5.2':'Samsung', 
                'firefox 58.0':'Firefox' , 'chrome 64.0 for android':'Chrome Android' , 'chrome 64.0':'Chrome PC' ,
                'firefox 59.0':'Firefox' , 'chrome 64.0 for ios':'Chrome IOS' , 'samsung browser generic':'Samsung', 'opera 51.0':'Opera' ,
                'samsung browser 7.0':'Samsung', 'Mozilla/Firefox':'Firefox' ,'samsung':'Samsung', 'opera generic':'Opera' ,
                'samsung browser 4.2':'Samsung', 'samsung browser 6.4':'Samsung', 'chrome 65.0':'Chrome PC' ,
                'chrome 65.0 for android':'Chrome Android' , 'chrome 65.0 for ios':'Chrome IOS' ,'edge 17.0':'Edge' , 'chrome 66.0':'Chrome PC' ,
                'chrome 66.0 for android':'Chrome Android' , 'safari 11.0':'Safari', 'safari 9.0':'Safari',
                'safari 10.0':'Safari', 'chrome 66.0 for ios':'Chrome IOS', 'opera 52.0':'Opera' , 'firefox 60.0':'Firefox' ,
                'opera 53.0':'Opera' , 'samsung browser 3.3':'Samsung', 'chrome 67.0 for ios':'Chrome IOS' ,
                'firefox mobile 61.0':'Firefox Mobile' , 'chrome 67.0':'Chrome PC' , 'chrome 69.0':'Chrome PC' ,
                'chrome 67.0 for android':'Chrome Android', 'chromium':'Chrome PC', 'chrome 39.0 for android':'Chrome Android' ,
                'chrome 68.0':'Chrome PC' , 'chrome 68.0 for android':'Chrome Android' , 'chrome 68.0 for ios':'Chrome IOS' ,
                'chrome 69.0 for android':'Chrome Android' , 'chrome 69.0 for ios':'Chrome IOS' , 'chrome 70.0':'Chrome PC' ,
                'chrome 70.0 for android':'Chrome Android' , 'chrome 70.0 for ios':'Chrome IOS' , 'chrome 71.0':'Chrome PC',
                'chrome 71.0 for android':'Chrome Android' , 'chrome 71.0 for ios':'Chrome IOS' , 'edge 18.0':'Edge',
                'firefox 61.0':'Firefox', 'firefox 62.0':'Firefox', 'firefox 63.0':'Firefox', 'firefox 64.0':'Firefox',
                'firefox mobile 62.0':'Firefox Mobile', 'firefox mobile 63.0':'Firefox Mobile','google search application 52.0':'Google' ,
                'google search application 54.0':'Google' , 'google search application 56.0':'Google' , 'google search application 58.0':'Google' ,
                'google search application 59.0':'Google' , 'google search application 60.0':'Google' , 'google search application 61.0':'Google' ,
                'google search application 62.0':'Google' , 'google search application 63.0':'Google' , 'google search application 64.0':'Google' ,
                'google search application 65.0':'Google' , 'mobile safari 12.0':'Safari Mobile', 'opera 54.0':'Opera' , 'opera 55.0':'Opera' ,
                'opera 56.0':'Opera' , 'safari 12.0':'Safari' , 'samsung browser 7.2':'Samsung',
                'samsung browser 7.4':'Samsung', 'samsung browser 8.2':'Samsung'}

OSmapping = {'Android 4.4.2':'Android' , 'Android 5.0':'Android' , 'Android 5.0.2':'Android' , 'Android 5.1.1':'Android' ,
            'Android 6.0':'Android' , 'Android 6.0.1':'Android' , 'Android 7.0':'Android' , 'Android 7.1.1':'Android' ,
            'Android 7.1.2':'Android' , 'Android 8.0.0':'Android' , 'Android 8.1.0':'Android' , 'Android 9':'Android' ,
            'Mac OS X 10.10':'Mac' , 'Mac OS X 10.11':'Mac' , 'Mac OS X 10.12':'Mac' , 'Mac OS X 10.13':'Mac' ,
            'Mac OS X 10.6':'Mac' , 'Mac OS X 10.9':'Mac' , 'Mac OS X 10_10_5':'Mac' , 'Mac OS X 10_11_3':'Mac' ,
            'Mac OS X 10_11_4':'Mac' , 'Mac OS X 10_11_5':'Mac' , 'Mac OS X 10_11_6':'Mac' , 'Mac OS X 10_12':'Mac' ,
            'Mac OS X 10_12_1':'Mac' , 'Mac OS X 10_12_2':'Mac' , 'Mac OS X 10_12_3':'Mac' , 'Mac OS X 10_12_4':'Mac' ,
            'Mac OS X 10_12_5':'Mac' , 'Mac OS X 10_12_6':'Mac' , 'Mac OS X 10_13_1':'Mac' , 'Mac OS X 10_13_2':'Mac' ,
            'Mac OS X 10_13_3':'Mac' , 'Mac OS X 10_13_4':'Mac' , 'Mac OS X 10_13_5':'Mac' , 'Mac OS X 10_6_8':'Mac' ,
            'Mac OS X 10_7_5':'Mac' , 'Mac OS X 10_8_5':'Mac' , 'Mac OS X 10_9_5':'Mac' , 'Mac OS X 10.14':'Mac' ,
            'Mac OS X 10_13_6':'Mac' , 'Mac OS X 10_14':'Mac' , 'Mac OS X 10_14_0':'Mac' , 'Mac OS X 10_14_1':'Mac' ,
            'Mac OS X 10_14_2':'Mac' , 'Windows 10':'Windows' , 'Windows 7':'Windows' , 'Windows 8':'Windows' ,
            'Windows 8.1':'Windows' , 'Windows Vista':'Windows' , 'Windows XP':'Windows' , 'iOS 10.0.2':'iOS' ,
            'iOS 10.1.1':'iOS' , 'iOS 10.2.0':'iOS' , 'iOS 10.2.1':'iOS' , 'iOS 10.3.1':'iOS' , 'iOS 10.3.2':'iOS' ,
            'iOS 10.3.3':'iOS' , 'iOS 11.0.0':'iOS' , 'iOS 11.0.1':'iOS' , 'iOS 11.0.2':'iOS' , 'iOS 11.0.3':'iOS' ,
            'iOS 11.1.0':'iOS' , 'iOS 11.1.1':'iOS' , 'iOS 11.1.2':'iOS' , 'iOS 11.2.0':'iOS' , 'iOS 11.2.1':'iOS' ,
            'iOS 11.2.2':'iOS' , 'iOS 11.2.5':'iOS' , 'iOS 11.2.6':'iOS' , 'iOS 11.3.0':'iOS' , 'iOS 11.3.1':'iOS' ,
            'iOS 11.4.0':'iOS' , 'iOS 11.4.1':'iOS' , 'iOS 9.3.5':'iOS' , 'iOS 12.0.0':'iOS' , 'iOS 12.0.1':'iOS' ,
            'iOS 12.1.0':'iOS' , 'iOS 12.1.1':'iOS' , 'iOS 12.1.2':'iOS' }

# THIS FUNCTION WILL MAP OS(ID_30) AND BROWSER(ID_31) IN GROUPS
def mappingOSandBrowser(val, mappingList):
    if val in mappingList.keys():
        return mappingList[val]
    else:
        return val


# In[ ]:


#total_count = len(train_trans)
plt.figure(figsize=(8,6))
g1 = sns.countplot(data = train_trans, x=train_trans['isFraud'].map({0:'No Fraud', 1: 'Fraud'}))
g1.set_title('Total Count Distribution',fontsize= 16)
g1.set_ylabel('Count',fontsize= 14)
g1.set_xlabel('Transactin Type' ,fontsize= 14)
for p in g1.patches:
    height = p.get_height()
    #print(p.get_x())
    g1.text(p.get_x()+p.get_width()/2.,height+5000,'{} ({:1.2f}%)'.format(height ,height/TotalTransCount * 100),ha="center", va='bottom', fontsize=10) 


# **Highly** unbalanced data. 

# In[ ]:


print(np.sum(train_trans.index.unique().isin(train_id.index.unique())))# Check how many indexes from TRAIN_TRANS dataset are present in TRAIN_ID dataset.
print(np.sum(test_trans.index.unique().isin(test_id.index.unique())))


# In[ ]:


print(train_trans.shape)
print(train_id.shape)

print(test_trans.shape)
print(test_id.shape)


# **Not all transaction ids have entry in identity table**

# In[ ]:


## Grouping columns according to there names
card_col = ['card1','card2','card3','card4','card5','card6']
add_col = ['addr1','addr2']
dist_col=['dist1', 'dist2']
c_col =['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']
d_col = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15']
m_col = ['M1','M2','M3','M4','M5','M6','M7','M8','M9'] 
other_col = ['isFraud','TransactionDT','TransactionAmt','ProductCD']
email_col = [ 'P_emaildomain','R_emaildomain']
v_col =[]
for v in range (1, 340):
    v_col.append('V'+str(v))
    #print( v)


# In[ ]:


train_trans[other_col].head()


# In[ ]:


metadata(train_trans[other_col])


# # TransactionAmt

# In[ ]:


quantiles[['TransactionAmt']]


# In[ ]:


train_trans[['TransactionAmt']].describe()


# In[ ]:


metadata(train_trans[['TransactionAmt']])


# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(121)
sns.distplot(train_trans[train_trans['isFraud']==1]['TransactionAmt'].dropna(), norm_hist=True)
plt.subplot(122)
sns.distplot(train_trans[train_trans['isFraud']==0]['TransactionAmt'].dropna())
plt.show()


# In[ ]:


plt.figure(figsize=(16,6))
plt.subplot(121)
sns.distplot(np.log(train_trans[train_trans['isFraud']==1]['TransactionAmt'].dropna()), norm_hist=True)
plt.subplot(122)
sns.distplot(np.log(train_trans[train_trans['isFraud']==0]['TransactionAmt'].dropna()), norm_hist=True)
plt.show()


# # ProductCD Column

# In[ ]:


metadata(train_trans[['ProductCD']])


# In[ ]:


count_perFraud_plots(train_trans,'ProductCD', False, fontsize=12, figsizeW=16, figsizeH=20)
plt.figure(figsize=(7,6))
sns.boxenplot(data=train_trans, x='ProductCD', y='TransactionAmt')
plot_boxen_violin(train_trans,2500,'ProductCD')
plt.show()


# # Card Columns

# In[ ]:


train_trans[card_col].head()


# In[ ]:


metadata(train_trans[card_col])


# In[ ]:


train_trans[card_col].describe()


# In[ ]:


quantiles[['card1','card2','card3','card5']]


# In[ ]:


corrfunc(train_trans,card_col)


# In[ ]:


plt.figure(figsize=(18,6))
plt.subplot(141)
plot1 = sns.distplot(train_trans[train_trans['isFraud']==1]['card1'])
plot1 = sns.distplot(train_trans[train_trans['isFraud']==0]['card1'])
plt.subplot(142)
plot2 = sns.distplot(train_trans[train_trans['isFraud']==1]['card2'].dropna())
plot2 = sns.distplot(train_trans[train_trans['isFraud']==0]['card2'].dropna())
plt.subplot(143)
plot3 = sns.distplot(train_trans[train_trans['isFraud']==1]['card3'].dropna())
plot3 = sns.distplot(train_trans[train_trans['isFraud']==0]['card3'].dropna())
plt.subplot(144)
plot4 = sns.distplot(train_trans[train_trans['isFraud']==1]['card5'].dropna())
plot4 = sns.distplot(train_trans[train_trans['isFraud']==0]['card5'].dropna())
plt.show()


# OBSERVATIONS:
# * Card1 has no null values 
# * CArd1 and Card2 are well distributed.
# * Card2 to Car6 have null values.
# * from Quantiles and distplot of Card3 and Card5 we can see that some values have high frequency and some have low frequency

# In card3 and card5 distribution of values in not even. we will use value count to seperate the values in card3 and card5

# ## Card3 Columns

# In[ ]:


count_perFraud_plots(train_trans,'card3', True, 'value count',250, 14, 24, 24)


# ## Card4 Column

# In[ ]:


count_perFraud_plots(train_trans,'card4', False,fontsize= 14,figsizeW= 16,figsizeH=22)
plot_boxen_violin(train_trans, 1000, 'card4')


# In[ ]:


sns.boxenplot(data=train_trans[['card4','TransactionAmt']].fillna('NaN'), x='card4', y='TransactionAmt')


# ## Card5 Columns

# In[ ]:


count_perFraud_plots(train_trans,'card5', True, 'value count',350, 14, 34, 28)


# ## Card6 Column

# In[ ]:


count_perFraud_plots(train_trans,'card6', False,fontsize= 14,figsizeW= 16,figsizeH=24)
plot_boxen_violin(train_trans, 1000, 'card6')


# # Address Columns

# In[ ]:


train_trans[add_col].head()


# In[ ]:


metadata(train_trans[add_col])


# In[ ]:


train_trans[add_col].describe()


# In[ ]:


quantiles[add_col]


# In[ ]:


corrfunc(train_trans,add_col)


# In[ ]:


plt.figure(figsize=(12,5))
plt.subplot(121)
plot1 = sns.distplot(train_trans[train_trans['isFraud'] == 1]['addr1'].dropna())
plot1 = sns.distplot(train_trans[train_trans['isFraud'] == 0]['addr1'].dropna())
plt.subplot(122)
plot2 = sns.distplot(train_trans[train_trans['isFraud'] == 1]['addr2'].dropna())
plot2 = sns.distplot(train_trans[train_trans['isFraud'] == 0]['addr2'].dropna())
plt.show()


# ## Add1 Columns

# In[ ]:


count_perFraud_plots(train_trans,'addr1', True,'value count',7000,fontsize= 16,figsizeW= 30,figsizeH=33,rotation=45)


# ## Addr2 Columns

# In[ ]:


count_perFraud_plots(train_trans,'addr2', True,'value count',60,fontsize= 14,figsizeW= 16,figsizeH=22)


# In[ ]:


plot_boxen_violin(train_trans, 5000, 'addr2', True, 'value count', 60)


# **OBSERVATIONS**
# * Addr1 is well distributed but Addr2 is concentrated at 87
# * Even 11.13% entries in addr2 are NaN values they(NaN values) represent 37.46% of fraud transaction. so droping NaN from addr1 is not a right decision.
# * from graphs we can a pattern with NaN, though. 
# * Fraud transaction with NaN are 37% of total fraud transaction but the transaction money is only 12% 

# # Dist columns

# In[ ]:


train_trans[dist_col].head()


# In[ ]:


metadata(train_trans[dist_col])


# In[ ]:


train_trans[dist_col].describe()


# In[ ]:


quantiles[dist_col]


# In[ ]:


corrfunc(train_trans, dist_col)


# In[ ]:


plt.figure(figsize=(10,6))
plt.subplot(121)
plot1 = sns.distplot(train_trans[train_trans['isFraud'] == 1]['dist1'].dropna())
plot1 = sns.distplot(train_trans[train_trans['isFraud'] == 0]['dist1'].dropna())
plt.subplot(122)
plot2 = sns.distplot(train_trans[train_trans['isFraud'] == 1]['dist2'].dropna())
plot2 = sns.distplot(train_trans[train_trans['isFraud'] == 0]['dist2'].dropna())
plt.show()


# ## Dist1 Column

# In[ ]:


count_perFraud_plots(train_trans,'dist1', True,'value count',5000,fontsize= 14,figsizeW= 22,figsizeH=24)


# ## Dist2 Column

# In[ ]:


train_trans['dist2'].value_counts()[:10]


# In[ ]:


count_perFraud_plots(train_trans,'dist2', True,'value count',300,fontsize= 14,figsizeW= 22,figsizeH=24)


# **OBSERVATION**
# * NaN values have majority of fraud transaction and fraud money.

# # Email Columns

# In[ ]:


train_trans[email_col].head()


# In[ ]:


metadata(train_trans[email_col])


# In[ ]:


train_trans[email_col].describe()


# In[ ]:


def emailprovider(emailid):
    if emailid in mapping.keys():
        #print(mapping[emailid])
        return mapping[emailid]
    else:
        return emailid.split('.')[0]

mapping= {'frontier.com':'frontier','frontiernet.net':'frontier','gmail':'gmail','gmail.com':'gmail','hotmail.co.uk':'hotmail','hotmail.com':'Microsoft','hotmail.de':'Microsoft','hotmail.es':'Microsoft','hotmail.fr':'Microsoft',
          'icloud.com':'Apple','live.com':'Microsoft','live.com.mx':'Microsoft','live.fr':'Microsoft','mac.com':'Apple','netzero.com':'Netzero','netzero.net':'Netzero','outlook.com':'Microsoft','outlook.es':'Microsoft',
          'yahoo.co.jp':'Yahoo','yahoo.co.uk':'Yahoo','yahoo.com':'Yahoo','yahoo.com.mx':'Yahoo','yahoo.de':'Yahoo','yahoo.es':'Yahoo','yahoo.fr':'Yahoo','ymail.com':'Yahoo', 'scranton.edu':'Scranton'}


# In[ ]:


temp_mail= train_trans[['P_emaildomain','R_emaildomain','isFraud', 'TransactionAmt']].copy()
temp_mail['P_Domain'] = temp_mail['P_emaildomain'].dropna().map(lambda x: x.split('.')[-1]) # This Only give the domain eg.
temp_mail['P_Provider'] = temp_mail['P_emaildomain'].dropna().apply(emailprovider)
temp_mail['R_Domain'] = temp_mail['R_emaildomain'].dropna().map(lambda x: x.split('.')[-1])
temp_mail['R_Provider'] = temp_mail['R_emaildomain'].dropna().apply(emailprovider)


# ## P_Provider Column

# In[ ]:


count_perFraud_plots(temp_mail,'P_Provider', True,'value count',300,fontsize= 16,figsizeW= 28,figsizeH=28,rotation=45)


# ## P_Domain Column

# In[ ]:


count_perFraud_plots(temp_mail,'P_Domain', False,fontsize= 14,figsizeW= 24,figsizeH=26)


# ## R_Provider Column

# In[ ]:


count_perFraud_plots(temp_mail,'R_Provider', True,'value count',200,fontsize= 14,figsizeW= 24,figsizeH=36,rotation=45)


# ## R_Domain Columns

# In[ ]:


count_perFraud_plots(temp_mail,'R_Domain', False,fontsize= 14,figsizeW= 24,figsizeH=26)


# **OBSERVATION**
# * NaN values in R_emaildomain have major part of fraud transaction and fraud amount.

# # M columns

# In[ ]:


train_trans[m_col].head()


# In[ ]:


metadata(train_trans[m_col])


# In[ ]:


train_trans[m_col].describe()


# In[ ]:


temp_mcol = train_trans[['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9','isFraud', 'TransactionAmt']].copy()


# ### M1

# In[ ]:


count_perFraud_plots(train_trans,'M1', False,fontsize= 12,figsizeW= 12,figsizeH=22)
plot_boxen_violin(train_trans, 5000, 'M1')


# ### M2

# In[ ]:


count_perFraud_plots(train_trans,'M2', False,fontsize= 12,figsizeW= 12,figsizeH=22)
plot_boxen_violin(train_trans, 5000, 'M2')


# ### M4

# In[ ]:


count_perFraud_plots(train_trans,'M4', False,fontsize= 12,figsizeW= 12,figsizeH=22)
plot_boxen_violin(train_trans, 5000, 'M4')


# ### M7

# In[ ]:


count_perFraud_plots(train_trans,'M7', False,fontsize= 12,figsizeW= 12,figsizeH=22)
plot_boxen_violin(train_trans, 5000, 'M7')


# ### M8

# In[ ]:


count_perFraud_plots(train_trans,'M8', False,fontsize= 12,figsizeW= 12,figsizeH=22)
plot_boxen_violin(train_trans, 5000, 'M8')


# # C Columns

# In[ ]:


train_trans[c_col].head()


# In[ ]:


metadata(train_trans[c_col])


# In[ ]:


train_trans[c_col].quantile([.01, .20, .50, .70, .90, .95, .96,.99])


# In[ ]:


train_trans[c_col].describe()


# OBSERVATIONS
# * There are outliers in most of the columns except C3. 
# * We can see a large diffrence in quatiles of 0.90, 0.95, 0.96 and 0.99.
# * All the values are float and no categories

# In[ ]:


corrfunc(train_trans,c_col)


# OBSERVATIONS
# 
# * We can see columns are highly correlated with many having correaltion higher than 0.90 and some having corr. of 1.
# 
# * C1, C2, C11 having a correlation of 1 so we can say they are the same.
# 
# * C7, C12 with 1 correlation and C7, C10 have 0.99 correlation (We can say almost 1)**.
# 
# * C6 and C11 have correaltion of 0.99 (We can say almost 1).
# 
# * C8, C10 with 1 correlation.
# 
# we can set a threshold for correlation and include only columns which are not highly correlated.  

# Columns to Plot C1, C13, C9, C5, C3

# ### C1 

# In[ ]:


train_trans['C1'].value_counts()


# In[ ]:


count_perFraud_plots(train_trans, 'C1', True, 'quantile',0.950, fontsize=14, figsizeH=28, figsizeW=32)


# ### C3

# In[ ]:


count_perFraud_plots(train_trans, 'C3', True, 'quantile',0.990, 14, 12, 22)
#(df, col, outliers = True, thresholdType='None', threshold = 0, fontsize =15, figsizeW =16, figsizeH=20, rotate =True):


# ### C5

# In[ ]:


count_perFraud_plots(train_trans, 'C5', True, 'quantile',0.950, 12, 16, 22)


# ### C9

# In[ ]:


count_perFraud_plots(train_trans, 'C9', True, 'quantile',0.950, 12, 18, 22)


# ## D Columns

# In[ ]:


train_trans[d_col].head()


# In[ ]:


metadata(train_trans[d_col])


# In[ ]:


train_trans[d_col].describe()


# In[ ]:


quantiles[d_col]


# In[ ]:


corrfunc(train_trans, d_col)


# In[ ]:


plt.figure(figsize=(10,8))
plot1 = sns.distplot(train_trans[train_trans['isFraud']==1]['D15'].dropna(), label = 'Fraud')
plot1 = sns.distplot(train_trans[train_trans['isFraud']==0]['D15'].dropna(), label = 'No Fraud')
plot1.legend()
plt.show()


# ### D5

# In[ ]:


count_perFraud_plots(train_trans,'D5',True, 'quantile',0.750, 14, 36, 30)


# ### D7

# In[ ]:


count_perFraud_plots(train_trans,'D7',True, 'quantile',0.750, 14, 26, 26)


# ### D12

# In[ ]:


count_perFraud_plots(train_trans,'D12',True, 'quantile',0.750, 14, 24, 26)


# ### D13

# In[ ]:


count_perFraud_plots(train_trans,'D13', True, 'quantile',0.750, 12, 12, 22)


# ### D14

# In[ ]:


count_perFraud_plots(train_trans,'D14', True, 'quantile',0.750, 12, 14, 22)


# In[ ]:


plt.figure(figsize=(20, 28))
plt.subplot(621)
sns.countplot(x= train_trans[train_trans['D4']<0]['D4'], hue = train_trans['isFraud'])
plt.subplot(622)
sns.countplot(x= train_trans[train_trans['D6']<0]['D6'], hue = train_trans['isFraud'])
plt.subplot(623)
sns.countplot(x= train_trans[train_trans['D11']<0]['D11'], hue = train_trans['isFraud'])
plt.subplot(624)
sns.countplot(x= train_trans[train_trans['D12']<0]['D12'], hue = train_trans['isFraud'])
plt.subplot(625)
sns.countplot(x= train_trans[train_trans['D14']<0]['D14'], hue = train_trans['isFraud'])
plt.subplot(626)
sns.countplot(x= train_trans[train_trans['D15']<0]['D15'], hue = train_trans['isFraud'])
plt.show()


# * In Columnss with Negative values, all negative values are non fraud entries

# ## V Columns

# In[ ]:


train_trans[v_col].head()


# In[ ]:


v_info = metadata(train_trans[v_col])


# In[ ]:


v_info['Dtype'].value_counts()


# All V columns are float.
# 
# There are 339 V columns. We will perform PCA to reduce dimensions

# # Train Identity Dataset

# In[ ]:


train_id.head()


# In[ ]:


train_id.shape


# In[ ]:


#metadata(train_id).sort_values('Unique')
metadata(train_id)


# In[ ]:


train_id.describe()


# In[ ]:


quantile_ID = train_id.quantile([.01, .20, .25, .50, .70, .75,.90, .95, .96,.99])
quantile_ID


# In[ ]:


train_id.columns


# In[ ]:


corrfunc(train_id,['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08','id_09', 'id_10', 'id_11'])


# In[ ]:


quantiles


# In[ ]:


plt.figure(figsize=(14,4))
plt.subplot(121)
plot1 = sns.distplot(train_trans[train_trans['isFraud'] == 1]['id_01'].dropna(), label = 'Fraud')
plot1 = sns.distplot(train_trans[train_trans['isFraud'] == 0]['id_01'].dropna(), label = 'No Fraud')
plot1.legend()
plt.subplot(122)
plot2 = sns.distplot(train_trans[train_trans['isFraud'] == 1]['id_02'].dropna(),label = 'Fraud')
plot2 = sns.distplot(train_trans[train_trans['isFraud'] == 0]['id_02'].dropna(),label = 'No Fraud')
plot2.legend()
plt.show()


# ### ID03

# In[ ]:


count_perFraud_plots(train_trans,'id_03',True, 'quantile',0.750, 14, 22, 24)


# ### ID04 

# In[ ]:


count_perFraud_plots(train_trans,'id_04',False,fontsize=14, figsizeW= 22, figsizeH= 24)


# ### ID07

# In[ ]:


count_perFraud_plots(train_trans,'id_07',True,'value count', 1000,fontsize=14, figsizeW= 12, figsizeH= 22)


# ### ID08

# In[ ]:


count_perFraud_plots(train_trans,'id_08',True,'value count', 1000,fontsize=14, figsizeW= 12, figsizeH= 22)


# ### ID12

# In[ ]:


count_perFraud_plots(train_trans,'id_12',False,fontsize=14, figsizeW= 12, figsizeH= 22)


# ### ID21

# In[ ]:


count_perFraud_plots(train_trans,'id_21',True,'value count', 1000,fontsize=14, figsizeW= 12, figsizeH= 22)


# ### ID22

# In[ ]:


count_perFraud_plots(train_trans,'id_22',True,'value count', 1000,fontsize=14, figsizeW= 12, figsizeH= 22)


# ### ID23

# In[ ]:


count_perFraud_plots(train_trans,'id_23',False,fontsize=14, figsizeW= 16, figsizeH= 22)


# ### ID33

# In[ ]:


count_perFraud_plots(train_trans,'id_33',True,'value count', 2000,fontsize=14, figsizeW= 24, figsizeH= 20)


# ### ID35

# In[ ]:


count_perFraud_plots(train_trans,'id_35',False,fontsize=14, figsizeW= 12, figsizeH= 20)


# ### ID30

# In[ ]:


count_perFraud_plots(train_trans,'id_30',True,'value count', 2000,fontsize=14, figsizeW= 22, figsizeH= 22)


# In[ ]:


train_trans['id_30'].unique()


# In[ ]:


tes1 =set(train_trans['id_30'].unique())
tes2= set(test_id['id-30'].unique())
tes2-tes1


# In[ ]:


temp_id3031 = train_trans[['id_30','id_31', 'isFraud', 'TransactionAmt']].copy()


# In[ ]:


temp_id3031['id_30'].value_counts()


# In[ ]:


temp_id3031['id_30'] = temp_id3031['id_30'].apply(mappingOSandBrowser, mappingList = OSmapping)


# In[ ]:


temp_id3031['id_30'].value_counts()


# ### ID30

# In[ ]:


count_perFraud_plots(temp_id3031,'id_30',False, fontsize=14, figsizeW= 18, figsizeH= 22)


# In[ ]:


temp_id3031['id_31'].value_counts()


# In[ ]:


temp_id3031['id_31'] = temp_id3031['id_31'].apply(mappingOSandBrowser, mappingList = browsermapping)


# In[ ]:


temp_id3031['id_31'].value_counts()


# ### ID31

# In[ ]:


count_perFraud_plots(temp_id3031,'id_31',True,'value count', 100,fontsize=12, figsizeW= 24, figsizeH= 28, rotation=45)


# ### ID35

# In[ ]:


count_perFraud_plots(train_trans,'id_35',False, fontsize=12, figsizeW= 12, figsizeH= 22)


# ### DeviceType

# In[ ]:


count_perFraud_plots(train_trans,'DeviceType',False, fontsize=12, figsizeW= 12, figsizeH= 22)


# In[ ]:


train_trans['DeviceInfo'].value_counts()[train_trans['DeviceInfo'].value_counts()<1000]


# ### DeviceInfo

# In[ ]:


count_perFraud_plots(train_trans,'DeviceInfo',True,'value count', 900,fontsize=14, figsizeW= 18, figsizeH= 24)


# In[ ]:


train_trans.columns[54:393]


# # Feature Engineering

# In[ ]:


test_data = test_trans.merge(test_id, how='left', left_index=True, right_index=True)
train_data = train_trans.copy()


# In[ ]:


test_data.head()


# In[ ]:


train_data.head()


# In[ ]:


print(train_data.shape)
print(test_data.shape)


# In[ ]:


# ID columns in the test dataset have a different naming convention. '-' is used instead of '_'. We replace '-' with '_'  
for col in test_data.columns[392:]:
    #print(col.replace('-','_'))
    test_data.rename(columns={col:col.replace('-','_')}, inplace=True)


# ####  We will drop Highly correalated columns. we will keep 0.90 correaltion as a threshold.
# 

# In[ ]:


# Outhut of this cell is the columnsToDrop list. I am not running this because it takes time to run.
#corr_matrix = train_trans.corr().abs() 

# Select upper triangle of correlation matrix
#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
#to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
#to_drop 


# In[ ]:


columnsToDrop = ['C2',  'C4',  'C6',  'C7',  'C8',  'C9',  'C10',  'C11',  'C12',  'C14',  'D2',  'D6',
'D7',  'D12',  'V5',  'V11',  'V13',  'V16',  'V18',  'V20',  'V21',  'V22',  'V28',
'V30',  'V31',  'V32',  'V33',  'V34',  'V36',  'V40',  'V42',  'V43',  'V45',  'V48',
'V49',  'V50',  'V51',  'V52',  'V54',  'V57',  'V58',  'V59',  'V60',  'V63',  'V64',
'V68',  'V69',  'V70',  'V71',  'V72',  'V73',  'V74',  'V76',  'V79',  'V80',  'V81',
'V84',  'V85',  'V88',  'V89',  'V90',  'V91',  'V92',  'V93',  'V94',  'V96',  'V97',
'V101',  'V102',  'V103',  'V105',  'V106',  'V113',  'V126',  'V127',  'V128',  'V132',
'V133',  'V134',  'V137',  'V140',  'V143',  'V145',  'V147',  'V149',  'V150',  'V151',
'V152',  'V153',  'V154',  'V155',  'V156',  'V157',  'V158',  'V159',  'V160',  'V163',
'V164',  'V167',  'V168',  'V177',  'V178',  'V179',  'V182',  'V183',  'V190',  'V192',
'V193',  'V196',  'V197',  'V198',  'V199',  'V201',  'V202',  'V203',  'V204',  'V206',
'V207',  'V211',  'V212',  'V213',  'V216',  'V217',  'V218',  'V219',  'V222',  'V225',
'V231',  'V232',  'V233',  'V235',  'V236',  'V237',  'V239',  'V243',  'V244',  'V245',
'V249',  'V251',  'V253',  'V254',  'V256',  'V257',  'V259',  'V263',  'V265',  'V266',
'V269',  'V271',  'V272',  'V273',  'V274',  'V275',  'V276',  'V277',  'V278',  'V279',
'V280',  'V292',  'V293',  'V294',  'V295',  'V296',  'V297',  'V298',  'V299',  'V301',
'V302',  'V304',  'V306',  'V307',  'V308',  'V309',  'V311',  'V312',  'V315',  'V316',
'V317',  'V318',  'V321',  'V322',  'V323',  'V324',  'V325',  'V326',  'V327',  'V328',
'V329',  'V330',  'V331',  'V332',  'V333',  'V334',  'V336',  'V339']


# ####  We will drop columns with large amount off null values we will keep threshold of 85%  

# In[ ]:


permissing = []
for col in train_trans.columns[:54]:
    perc_null = round((train_trans[col].isnull().sum()/TotalTransCount)*100, 3)
    permissing.append(perc_null)
null_val_perc = pd.DataFrame()
null_val_perc['Column']= train_trans.columns[:54]
null_val_perc['Missing Percentage'] = permissing

# Sort columns based on missing values percentage 
null_val_perc.sort_values(by=['Missing Percentage'] ,ascending=False)[:10]


# In[ ]:


null_val_perc[null_val_perc['Missing Percentage']>85]['Column'].values


# In[ ]:


train_data.columns[40:198]


# In[ ]:


train_data.shape


# In[ ]:


#metadata(test_data).sort_values(by='Unique')[:1]


# In[ ]:


#metadata(train_data[['V107']])
#V107 in test dataset has only 1 values this could be a error so we will drop this column


# In[ ]:


def emailprovider(emailid):
    if emailid in mapping.keys():
        #print(mapping[emailid])
        return mapping[emailid]
    else:
        return emailid.split('.')[0]

mapping= {'frontier.com':'frontier','frontiernet.net':'frontier','gmail':'gmail','gmail.com':'gmail','hotmail.co.uk':'hotmail','hotmail.com':'Microsoft','hotmail.de':'Microsoft','hotmail.es':'Microsoft','hotmail.fr':'Microsoft',
          'icloud.com':'Apple','live.com':'Microsoft','live.com.mx':'Microsoft','live.fr':'Microsoft','mac.com':'Apple','netzero.com':'Netzero','netzero.net':'Netzero','outlook.com':'Microsoft','outlook.es':'Microsoft',
          'yahoo.co.jp':'Yahoo','yahoo.co.uk':'Yahoo','yahoo.com':'Yahoo','yahoo.com.mx':'Yahoo','yahoo.de':'Yahoo','yahoo.es':'Yahoo','yahoo.fr':'Yahoo','ymail.com':'Yahoo'}


train_data['P_Domain'] = train_data['P_emaildomain'].dropna().map(lambda x: x.split('.')[-1])
train_data['P_Provider'] = train_data['P_emaildomain'].dropna().apply(emailprovider)
train_data['R_Domain'] = train_data['R_emaildomain'].dropna().map(lambda x: x.split('.')[-1])
train_data['R_Provider'] = train_data['R_emaildomain'].dropna().apply(emailprovider)

test_data['P_Domain'] = test_data['P_emaildomain'].dropna().map(lambda x: x.split('.')[-1])
test_data['P_Provider'] = test_data['P_emaildomain'].dropna().apply(emailprovider)
test_data['R_Domain'] = test_data['R_emaildomain'].dropna().map(lambda x: x.split('.')[-1])
test_data['R_Provider'] = test_data['R_emaildomain'].dropna().apply(emailprovider)
columnsToDrop.extend(('P_emaildomain','R_emaildomain'))


# In[ ]:


train_data['id_30'] = train_data['id_30'].apply(mappingOSandBrowser, mappingList = OSmapping)
train_data['id_31'] = train_data['id_31'].apply(mappingOSandBrowser, mappingList = browsermapping)
test_data['id_30'] = test_data['id_30'].apply(mappingOSandBrowser, mappingList = OSmapping)
test_data['id_31'] = test_data['id_31'].apply(mappingOSandBrowser, mappingList = browsermapping)
#columnsToDrop.extend(('id_30','id_31'))


# In[ ]:


# We will add columns to drop in columnsToDrop list
train_data.drop(columnsToDrop , axis =1, inplace=True)
test_data.drop(columnsToDrop , axis =1, inplace=True)
print(train_data.shape)
print(test_data.shape)


# In[ ]:


list(test_data.select_dtypes(include='object').columns)


# In[ ]:


list(train_data.select_dtypes(include='object').columns)


# In[ ]:



# LABEL ENCODER FOR ALL CATEGORICAL FEATURES
objectColumns = list(train_data.select_dtypes(include='object').columns)
from sklearn.preprocessing import LabelEncoder
for col in objectColumns:
    train_index = ~train_data[col].isnull()
    test_index = ~test_data[col].isnull()
    labelEncoder = LabelEncoder()
    labelEncoder.fit(list(train_data[col].dropna().values) + list(test_data[col].dropna().values))
    train_data.loc[train_index, col] = labelEncoder.transform(list(train_data.loc[train_index, col].values))
    test_data.loc[test_index, col] = labelEncoder.transform(list(test_data.loc[test_index, col].values)) 


# In[ ]:


print(train_data.shape)
print(test_data.shape)


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCA_V_Columns = list(train_data.columns[38:196])

def PCAFunction(df):
    for col in PCA_V_Columns:
        df[col].fillna(df[col].min()-1, inplace=True)
        df[[col]] = StandardScaler().fit_transform(df[[col]])

    pca = PCA(n_components=25, random_state=40)

    PCAed_V= pca.fit_transform(df[PCA_V_Columns])
    PCAed_V_DF= pd.DataFrame(PCAed_V)
    PCAed_V_DF.rename(columns  = lambda x: str('PCA_V')+str(x), inplace= True)
    print(PCAed_V_DF.shape)
    df.drop(columns=PCA_V_Columns, inplace =True)
    df = pd.concat([df, PCAed_V_DF.set_index(df.index)], axis= 1)
    
    return df


# In[ ]:


train_data = PCAFunction(train_data)
test_data = PCAFunction(test_data)


# In[ ]:


test_data.columns[81:]


# In[ ]:


print(train_data.shape)
print(test_data.shape)


# In[ ]:


test_data.head()


# In[ ]:


categoricalFeatures = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5',  'card6', 
                        'addr1', 'addr2',  'M1',  'M2',  'M3',  'M4',  'M5',  'M6',  'M7',  'M8',  'M9',  
                        'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 
                        'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 
                        'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
                        'DeviceType',  'DeviceInfo',  'P_Domain',  'P_Provider',  'R_Domain',  'R_Provider']

train_data[categoricalFeatures] = train_data[categoricalFeatures].apply(pd.to_numeric) # Converting Object datatype columns to numeric datatype to fit in model
test_data[categoricalFeatures] = test_data[categoricalFeatures].apply(pd.to_numeric)


# In[ ]:


import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# In[ ]:


# Here performing RandomSearchCV to get a starting point for the parameters. we customize the parameter in the actul model 
# param = {
#         'objective': ['binary'],
#         'boosting_type': ['gbdt'],                       #boosting
#         'n_estimators':[200,250,300,350,400,450,500,550],
#         'max_depth':[10,12,14,16,18,20],
#         'min_split_gain': uniform(loc=0, scale=4),                    #min_gain_to_split
#         'num_leaves': [200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280],
#         'min_child_samples':[50,60,70,80,100,115,130,150,170],
#         'subsample': uniform(loc=0.6, scale=0.3),                         #bagging_fraction
#         'colsample_bytree': uniform(loc=0.6, scale=0.3),                     #feature_fraction 
#         'subsample_freq':[1,2,3,4,5],
#         'learning_rate':uniform(loc=0, scale=0.5),
#         'metric':['auc'],
#         'reg_alpha': uniform(loc=0, scale=1),                       #lambda_l1   
#        'reg_lambda': uniform(loc=0, scale=1)                      #lambda_l2 
#        }

#x_test = test_data.drop(['TransactionDT'], axis=1) 
#x_train_lgbm = lgbm.Dataset(x_train)
#y_train_lgbm = lgbm.Dataset(y_train)
#lgbm_train =  lgbm.Dataset(x_train, label=y_train)
#clf = lgbm.LGBMClassifier()

#randSearchCV = RandomizedSearchCV(clf, param_distributions= param, cv=5, n_iter=10)

#randSearchCV.fit(x_train,y_train)

#randSearchCV.best_params_

#pd.DataFrame(randSearchCV.cv_results_)

#y_pred = randSearchCV.predict(x_test)


# In[ ]:


# the paramenters from RandomSearchCV 

#param = {'boosting_type': 'gbdt',
# 'colsample_bytree': 0.660201224714178,
# 'learning_rate': 0.18558795195977706,
# 'max_depth': 20,
# 'metric': 'auc',
# 'min_child_samples': 150,
# 'min_split_gain': 0.7447489311593505,
# 'n_estimators': 450,
# 'num_leaves': 230,
# 'objective': 'binary',
# 'reg_alpha': 0.3866729149893172,
# 'reg_lambda': 0.6716281805093148,
# 'subsample': 0.870092664400306,
# 'subsample_freq': 4}


# In[ ]:


#'early_stopping_round':[100,120,140,160,180,200],


# In[ ]:


x_train = train_data.drop(['isFraud', 'TransactionDT'], axis=1)
y_train = train_data['isFraud']
x_test = test_data.drop(['TransactionDT'], axis=1) 
#x = train_data.drop(['isFraud', 'TransactionDT'], axis=1)
#y = train_data['isFraud']
#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=40)


# In[ ]:


# Using the paramenters from RandomSearchCV with some changes in lighGBM

param = {
         'boosting_type': 'gbdt',
         'colsample_bytree': 0.660201224714178,
         'learning_rate': 0.03,
         'max_depth': 20,
         'metric': 'auc',
         'min_child_samples': 150,
         'min_split_gain': 0.7447489311593505,
         'n_estimators': 1200,
         'num_leaves': 35,
         'objective': 'binary',
         'reg_alpha': 0.3866729149893172,
         'reg_lambda': 0.6716281805093148,
         'subsample': 0.870092664400306,
         'subsample_freq': 4
        }

train_set = lgbm.Dataset(x_train, label=y_train)
lgb = lgbm.train(param, train_set)


# In[ ]:


y_predict = lgb.predict(x_test) 

# This Model gives me Private Score 0.900728 and Public Score 0.921558
#feature_importance.sort_values(by='Feature Importance', ascending=False)[:70]


# In[ ]:


feature_importance = pd.DataFrame()
feature_importance['Column'] = train_data.drop(['isFraud', 'TransactionDT'], axis=1).columns
feature_importance['Feature Importance'] = lgb.feature_importance()


# In[ ]:


#feature_importances.sort_values(by = 'Feature Importance', ascending=False)[:50]
plt.figure(figsize=(24,12))
plot = sns.barplot(data= feature_importance.sort_values(by='Feature Importance', ascending=False)[:70], x='Column',y='Feature Importance')
plot.set_title('Feature Importance of Features', fontsize = 14)
plot.set_ylabel('Feature Importance', fontsize = 12)
plot.set_xlabel('Features', fontsize = 12)
plot.set_xticklabels(plot.get_xticklabels(),rotation=30, ha='right', va='top')
plt.show()


# In[ ]:


#roc_auc_score(y_test, y_predict)


# In[ ]:


submission =pd.DataFrame()
submission['TransactionID'] = test_data.index
submission['isFraud']= y_predict


# In[ ]:


submission[submission['isFraud']>0.5]


# In[ ]:


submission.to_csv('submission.csv', index=False)

