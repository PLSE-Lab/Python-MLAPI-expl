#!/usr/bin/env python
# coding: utf-8

# # Kaggle--Avocado Prices dataset
# 
# # 2019-04-13 start

# # Summary
# 
# ## Describe Statistics
# 
# - Mean of average price is 1.4,mode is 1.15
# 
# 
# - Mode of total volume is not unique.
# 
# 
# - The other data(except average price) is dispersion,
# 
#   and large different from normal distribution
# 
# 
# - All of data is possitive distribution
# 
# 
# ## Total bags of each year
# 
# - 2017 has the most number of total bags than other year
# 
# 
# - 2018's data is not completed
# 
# 
# - Number of bags is increasing.
# 
# 
# ## Trend of average price
# 
# - The trend of average price seems to be cyclical
# 
# 
# - 2017's average price has much more fluctuation
# 
# 
# - 2017/9 has the highest average price,
# 
#   and 2016/5 has the lowest average price
# 
# 
# - Winter's price is normally low,
# 
#   and autumn's price is normally high.
#   
#   
# ## Percentage of different size bags
# 
# - percentage of differnet size bags each year
# 
# 
# Year | Bag Size | Percentage |
# ----- | -------- | ------------ |
# 2015 | Small Bags | 82.19% |
#      | Large Bags | 17.10% |
#      | XLarge Bags | 0.7% |
# 2016 | Small Bags | 75.62% |
#      | Large Bags | 23.01% |
#      | XLarge Bags | 1.37% |
# 2017 | Small Bags | 74.29% |
#      | Large Bags | 24.26% |
#      | XLarge Bags | 1.46% |
# 2018 | Small Bags | 73.39% |
#      | Large Bags | 25.14% |
#      | XLarge Bags | 1.47% |
# 
# 
# - The percentage of small bags is decreasing.
# 
# 
# - Between 2015 and 2016,XLarge bags suddenly increase.
# 
# 
# - The percentage of large bags and Xlarge bags is increasing.
# 
# ## Correlation analysis
# 
# - Category is negative correlation with average price
# 
# 
# - Region is less correlation with average price
# 
# 
# - Feature selection is feasibility
# 
# 
# ## Principal component analysis
# 
# - Finding the explained variance ratio
# 
# 
# - n_components choose 3
# 
# ## Predict
# 
# - Using feature selection is better than using PCA
# 
# 
# - With using feature selection,r2_score is 0.52
# 
# 
# - With using this data,training model is hard
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error,explained_variance_score
sns.set()


# In[2]:


original=pd.read_csv('../input/avocado.csv')


# In[3]:


print(original.head())
print(original.shape)
print(original.isnull().sum())
print(original.dtypes)
print(original.columns)
print(original['type'].unique())


# ## Data Preprocessing

# In[4]:


original.drop(columns='Unnamed: 0',inplace=True)


# In[5]:


original.columns


# In[6]:


original['Date']=pd.to_datetime(original['Date'])


# In[7]:


original.dtypes


# In[8]:


original['Month']=[i.month for i in original['Date']]


# In[9]:


original.head()


# In[10]:


original['4046']=original['4046'].astype('str')
original['4225']=original['4225'].astype('str')
original['4770']=original['4770'].astype('str')


# ## Question Defintion
# 
# 
# - Which year has the most number of bags?
# 
# 
# - What is the trend of Average price?
# 
# 
# - What is the percentage of different size bags of each year?
# 
# 
# - Can we predict the average price with this data?

# ## Describe Statistics

# In[11]:


original.describe()


# In[12]:


def cv(data):
    x=data.mean()
    y=data.std()
    coefficientofvariation=(round((y/x)*100,2))
    return coefficientofvariation


# In[13]:


print('AveragePrice cv:%1.2f%%'%(cv(original['AveragePrice'])))
print('Total Volume cv:%1.2f%%'%(cv(original['Total Volume'])))
print('Total Bags cv:%1.2f%%'%(cv(original['Total Bags'])))
print('Small Bags cv:%1.2f%%'%(cv(original['Small Bags'])))
print('Large Bags cv:%1.2f%%'%(cv(original['Large Bags'])))
print('XLarge Bags cv:%1.2f%%'%(cv(original['XLarge Bags'])))


# In[14]:


x=['AveragePrice','Total Volume','Total Bags','Small Bags','Large Bags','XLarge Bags']
print('skewness')
print('=============================')
print(original.skew())
print('=============================')
print('kurtosis')
print('=============================')
print(original.kurt())
print('=============================')
print('mode')
print('=============================')
print('average Price mode:',original['AveragePrice'].mode())
print('Total Volume mode:',original['Total Volume'].mode())
print('Total Bags mode:',original['Total Bags'].mode())
print('Small Bags mode:',original['Small Bags'].mode())
print('Large Bags mode:',original['Large Bags'].mode())
print('XLarge Bags mode:',original['XLarge Bags'].mode())


# In[15]:


z0=plt.figure(figsize=(15,14))
for i in range(len(x)):
    z0.add_subplot(3,3,i+1)
    original[str(x[i])].plot(kind='kde',title='%s distribution plot'%x[i])
else:
    pass
plt.show()


# ## Summary
# 
# - Mean of average Price is 1.4,mode is 1.15
# 
# 
# - Mode of total volume is not unique,
# 
#   and mode of Xlarge Bags is 0,
#   
#   other data has no mode.
# 
# 
# - Coefficient of variation of data except average price is big,
# 
#   and other data's coefficient of variation is over 100%,
#   
#   means other data is much more dispersive than average price.
#   
# 
# 
# - The skewness of average price is possitive distribution,
# 
#   and the kurtosis of average price is platykurtic.
#   
# 
# 
# - The skewness values of other data is big,
# 
#   which means other data's distribution has much more different from
#   
#   normal distribution.
#   
#   And skewness values are all possitive,which means all of data is 
#   
#   possitive distribution.
# 
# 
# - The kurtosis of other data is leptokurtic.
# 
#   

# ## Bags
# 
# Which year has the most number of bags?

# In[16]:


bagGroup=original.groupby(by='year')
bagsum=bagGroup.sum()
bagSum=pd.DataFrame(bagsum.loc[:,'Total Bags'])
bagSum.iloc[0:3,0]


# In[17]:


z1=bagSum.iloc[0:3].plot.bar(figsize=(15,14),legend=False)
z1.get_figure()
x=np.arange(bagsum.index.shape[0]-1)
y=np.array(bagSum.iloc[0:3,0])
plt.title('2013-2017 total bags by year bar graph',fontsize='large')
plt.ylabel('bags(hundred million)')
plt.xticks(x,bagSum.index[0:3],rotation=0)
for i,j in zip(x,y):
    plt.text(i,j+10000000,format(round(j,2),','),color='blue',ha='center')
else:
    pass
plt.show()


# ## Summary
# 
# 
# - 2017 has the most number of bags
# 
# 
# - 2018's data is not complete(below this,pricePivot has shown the 2018's data),
# 
#   so show the bar graph without 2018.
#   
#   
# - Between 2015 and 2016,rising of numbers is large.
# 
# 
# - Number of total bags is growing.

# ## Price
# 
# What is the trend of Average price?

# In[18]:


pricePivot=pd.pivot_table(original,index='year',columns='Month',values='AveragePrice',aggfunc=np.mean)
pricePivot


# In[19]:


pricePivot.index=pricePivot.index.astype('str')
pricePivot=pricePivot.drop(index='2018')


# In[20]:


pricePivot


# In[21]:


z2=plt.figure(figsize=(25,10))
y=np.array(list([pricePivot.iloc[0,:],pricePivot.iloc[1,:],pricePivot.iloc[2,:]]))
y=y.reshape((36,1))
plt.plot(y,linestyle='--',marker='o')
year1=np.array(['2015','2016','2017'])
month1=np.array([str(i) for i in range(1,13)])
yearmonth=[]
for i in range(3):
    for j in range(12):
        yearmonth.append(year1[i]+'/'+month1[j])
    else:
        pass
else:
    pass
x=np.arange(36)
for i,j in zip(x,y):
    plt.text(i,j+0.005,'%1.2f'%j,ha='center')
else:
    pass
plt.xticks(range(len(yearmonth)),yearmonth,rotation=45)
plt.title('2015-2017 average price plot',fontsize='large')
plt.show()


# ## Summary
# 
# - The average price of each month seems to be cyclical
# 
# 
# - 2015's average price is smooth,and 2017's average price has much more fluctuation
# 
# 
# - December~May's average price is normally low,and rising after this section
# 
# 
# - 2017/9 has the highest price,but declining after this month
# 
# 
# - Winter's average price is normally low,
# 
#   and autumn's average price is normally high.
# 
# 
# - 2016/5 has the lowest price,but rising after this month.
# 
# 
# - 2017's average price is normally higher than other year.

# ## Percentage of different size bags
# 
# What is the percentage of different size bags of each year?

# In[22]:


DbagSum=bagsum.loc[:,['Small Bags','Large Bags','XLarge Bags']]
DbagSum


# In[23]:


z3=plt.figure(figsize=(15,14))
labels=DbagSum.columns
explode=[0.0,0.0,0.0]
for i in range(DbagSum.index.shape[0]):
    z3.add_subplot(2,2,i+1)
    plt.title('%s Avocade bags pie chart'%DbagSum.index[i])
    plt.pie(DbagSum.iloc[i,:],labels=labels,explode=explode,autopct='%1.2f%%')
else:
    pass
plt.show()


# ## Summary
# 
# - Percentage of different size bags
# 
# Year | Bag Size | Percentage |
# ----- | -------- | ------------ |
# 2015 | Small Bags | 82.19% |
#      | Large Bags | 17.10% |
#      | XLarge Bags | 0.7% |
# 2016 | Small Bags | 75.62% |
#      | Large Bags | 23.01% |
#      | XLarge Bags | 1.37% |
# 2017 | Small Bags | 74.29% |
#      | Large Bags | 24.26% |
#      | XLarge Bags | 1.46% |
# 2018 | Small Bags | 73.39% |
#      | Large Bags | 25.14% |
#      | XLarge Bags | 1.47% |
#   
#   
# - The percentage of small bags is decreasing
# 
# 
# - The percentage of large bags is increasing fast
# 
# 
# - The percentage of XLarge bags is increasing
# 
# 
# - Between 2015 and 2016,the percentage of XLarge bags suddenly increase.
# 

# ## Predict

# ## Data preprocessing

# In[24]:


original.head()


# In[25]:


data=original.loc[:,['AveragePrice','4046','4225','4770','type','region']]


# In[26]:


data.head()


# In[27]:


print(data['type'].unique())
print(data['region'].unique())


# In[28]:


for i in range(data.index.shape[0]):
    if data.loc[i,'type']=='conventional':
        data.loc[i,'type']=0
    else:
        data.loc[i,'type']=1


# In[29]:


data.head()


# In[30]:


data['region']=LabelEncoder().fit_transform(data['region'])


# In[31]:


data.dtypes


# In[32]:


data['4046']=data['4046'].astype('float')
data['4225']=data['4225'].astype('float')
data['4770']=data['4770'].astype('float')


# ## Correlation analysis--Pearson coefficient matrix

# In[33]:


pd.DataFrame(data.corr(method='pearson'))


# ## Summary
# 
# With pearson coefficient matrix,
# 
# 4046.4225.4770 is negative correlation with average price,
# 
# and type is possitive correlation
# 
# Region has less correlation with average price.

# ## Principal component analysis

# In[34]:


Data=data.iloc[:,1:]
Target=data.loc[:,'AveragePrice']


# In[35]:


pca=PCA(n_components=5).fit(Data)
pca.explained_variance_ratio_


# In[36]:


Pca=PCA(n_components=3).fit(Data)
pData=Pca.transform(Data)
pData


# ## Summary
# 
# With using PCA,and show the explained variance ratio,
# 
# n_components is choosing 3
# 
# And transform the data.

# ## Predict
# 
# The correlation between features is normally ok,
# 
# and PCA n_components is 3
# 
# With using two data,
# 
# predicting average price with Gradient Boosting Regressor

# ## Predict--PCA

# In[37]:


data_train,data_test, target_train,target_test = train_test_split(pData,Target,train_size=0.6)


# In[38]:


GBR=GradientBoostingRegressor(learning_rate=0.2).fit(data_train,target_train)
GBR


# In[39]:


pre=GBR.predict(data_train)
print('explained_variance_score:%1.2f'%(explained_variance_score(target_train,pre)))
print('r2_score:%1.2f'%r2_score(target_train,pre))
print('mean_absolute_error:%1.2f'%mean_absolute_error(target_train,pre))
print('mean_squared_error:%1.2f'%mean_squared_error(target_train,pre))


# In[40]:


predict=GBR.predict(data_test)
print('explained_variance_score:%1.2f'%(explained_variance_score(target_test,predict)))
print('r2_score:%1.2f'%r2_score(target_test,predict))
print('mean_absolute_error:%1.2f'%mean_absolute_error(target_test,predict))
print('mean_squared_error:%1.2f'%mean_squared_error(target_test,predict))


# ## Predict--Feature Selection

# In[41]:


data.head()


# In[42]:


Fdata=data.loc[:,['4046','4225','4770','type']]
Ftarget=data.loc[:,'AveragePrice']


# In[43]:


dataTrain,dataTest, targetTrain,targetTest = train_test_split(Fdata,Ftarget,train_size=0.6)


# In[44]:


FGBR=GradientBoostingRegressor(learning_rate=0.3).fit(dataTrain,targetTrain)


# In[45]:


FGBR


# In[46]:


Fpre=FGBR.predict(dataTrain)
print('explained_variance_score:%1.2f'%(explained_variance_score(targetTrain,Fpre)))
print('r2_score:%1.2f'%r2_score(targetTrain,Fpre))
print('mean_absolute_error:%1.2f'%mean_absolute_error(targetTrain,Fpre))
print('mean_squared_error:%1.2f'%mean_squared_error(targetTrain,Fpre))


# In[47]:


Fpredict=FGBR.predict(dataTest)
print('explained_variance_score:%1.2f'%(explained_variance_score(targetTest,Fpredict)))
print('r2_score:%1.2f'%r2_score(targetTest,Fpredict))
print('mean_absolute_error:%1.2f'%mean_absolute_error(targetTest,Fpredict))
print('mean_squared_error:%1.2f'%mean_squared_error(targetTest,Fpredict))


# ## Summary
# 
# With using PCA and feature selection,
# 
# finding that using feature selection has higher r2_score
# 
# - using feature selection is better than PCA data
# 
# 
# - With using feature selection,r2_score is 0.52
# 
