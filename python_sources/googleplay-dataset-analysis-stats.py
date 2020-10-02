#!/usr/bin/env python
# coding: utf-8

# # Kaggle-Google play application dataset
# 
# # (statistical pratice)
# 
# ## 2019-04-30 start

# # Summary
# 
# ## Describe statistics
# 
# - Data except rating is dispersion
# 
# 
# - The distribution of data except rating is possitive distribution,
# 
#   and rating is negative distribution
#   
#   
# - All kurtosis of data is leptokurtic
# 
# 
# ## Category--Rating
# 
# - Education apps has the highest average rating than other apps
# 
# - Dating apps has the lowest average rating
# 
# - Game apps is 6th,which is higher than health and fitness apps
# 
# 
# ## Interval for all apps rating mean
# 
# - The 95% confidence interval for all apps rating mean is 
# 
#   (4.19129~4.19187)
# 
# 
# ## all apps rating mean v.s. sample data rating mean
# 
# - H0:mu1=4.19
# 
# 
# - H1:mu1!=4.19
# 
# 
# - all apps rating mean is different from sample data rating mean
# 
# 
# ## Correlation--Rating and Reviews
# 
# - Rating and Reviews has small correlation
# 
# 
# - If use least-square method,the residual range is large
# 
#   which means using this method has less effect
#   
#   
# - With scatter,can see correlation is small

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import math
sns.set()


# In[ ]:


original=pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


print(original.head())
print(original.shape)
print(original.columns)
print(original.isnull().sum())


# # Data Preprocessing

# In[ ]:


original['Rating']=original['Rating'].fillna(round(np.mean(original['Rating']),2))


# In[ ]:


original.dropna(inplace=True)


# In[ ]:


print(original.isnull().sum())
print(original.shape)


# In[ ]:


for i in range(original.index.shape[0]):
    original.iloc[i,4]=re.sub(r'[M]','',str(original.iloc[i,4]))


# In[ ]:


original.head()


# In[ ]:


for i in range(original.index.shape[0]):
    original.iloc[i,5]=re.sub(r'\D','',str(original.iloc[i,5]))


# In[ ]:


for i in range(original.index.shape[0]):
    original.iloc[i,4]=re.sub(r'[k]','000',str(original.iloc[i,4]))


# In[ ]:


for i in range(original.index.shape[0]):
    if original.iloc[i,4]=='Varies with device':
        original.iloc[i,4]=0


# In[ ]:


original['Size']=original['Size'].astype('float')
print(original['Size'].mean()/100)


# In[ ]:


for i in range(original.index.shape[0]):
    if original.iloc[i,4]==0:
        original.iloc[i,4]=round(np.mean(original['Size'])/100,1)


# In[ ]:


original['Reviews']=original['Reviews'].astype('int')
original['Installs']=original['Installs'].astype('int')


# In[ ]:


original['Price'].unique()[0:5]


# In[ ]:


for i in range(original.index.shape[0]):
    original.iloc[i,7]=re.sub('[$]','',str(original.iloc[i,7]))


# In[ ]:


original['Price']=original['Price'].astype('float')


# In[ ]:


original.tail()


# In[ ]:


original.dtypes


# # Describe statistics

# In[ ]:


de1=list(stats.describe(original.loc[:,'Rating']))
de2=list(stats.describe(original.loc[:,'Reviews']))
de3=list(stats.describe(original.loc[:,'Size']))
de4=list(stats.describe(original.loc[:,'Installs']))
de5=list(stats.describe(original.loc[:,'Price']))
des=pd.DataFrame()
des['Rating']=de1
des['Reviews']=de2
des['Size']=de3
des['Installs']=de4
des['Price']=de5
x=['n','(min,max)','mean','variance','skewness','kurtosis']
des.index=x
des


# In[ ]:


def cv(data):
    standard=data.std()
    xbar=data.mean()
    return round((standard/xbar)*100,2)


# In[ ]:


print('coefficient of variation')
print('=============================')
print('Rating:%1.2f%%'%cv(original['Rating']))
print('Reviews:%1.2f%%'%cv(original['Reviews']))
print('Size:%1.2f%%'%cv(original['Size']))
print('Installs:%1.2f%%'%cv(original['Installs']))
print('Price:%1.2f%%'%cv(original['Price']))
print('=============================')


# In[ ]:


x=['Rating','Reviews','Size','Installs','Price']
z0=plt.figure(figsize=(15,35))
for i in range(len(x)):
    z0.add_subplot(6,1,i+1)
    sns.distplot(original[str(x[i])])
    plt.title('%s distribution plot'%str(x[i]))
else:
    pass
plt.show()


# # Describe Statistics Summary
# 
# - data except Rating is dispersion
# 
# 
# - Rating is negative distribution
# 
#   and other data are possitive distribution
# 
# 
# - All of data kurtosis are leptokurtic
# 
# 
# - Range of data except rating is large

# # Question Definition
# 
# 
# - Which category has the highest average rating?
# 
# 
# - What is the interval of all apps rating mean?
# 
# 
# - Is the mean of all apps rating similar to data's mean?
# 
# 
# - Is rating and reviews has correlation?
# 
# 

# # Category--rating

# In[ ]:


cateRating=pd.DataFrame(original.groupby(by='Category').mean()['Rating'])


# In[ ]:


RatingSort=cateRating.sort_values(by='Rating',ascending=False)


# In[ ]:


z1=RatingSort.plot.barh(figsize=(15,14),legend=False)
z1.get_figure()
plt.title('Category average Rating bar graph',fontsize='large')
plt.xticks(np.linspace(0,5,15))
r=np.array(RatingSort['Rating'])
x=np.arange(RatingSort.index.shape[0])
for i,j in zip(x,r):
    plt.text(j+0.07,i-0.05,'%1.2f'%j,ha='center',color='blue')
else:
    pass
plt.show()


# # Category rating Summary
# 
# - Education apps has highest average rating than other category
# 
# 
# - Dating apps has lowest average rating than other category
# 
# 
# - Game apps is 6th,which is higher than health apps

# # confidence interval for population mean
# 
# alpha=0.05

# In[ ]:


print('sample mean:',original.loc[:,'Rating'].mean())
print()
print('Rating 95% confidence interval:')
stats.norm.interval(0.05,loc=original.loc[:,'Rating'].mean(),scale=stats.sem(original.loc[:,'Rating']))


# # interval for all apps mean Summary
# 
# Set the confidence coefficient to 0.05,
# 
# means 95% confidence interval,
# 
# with using stats.norm.interval,
# 
# 95% confidence interval is 
# 
# (4.19129~4.19187)
# 
# Because sample data is large,
# 
# using the z score,
# 
# so confidnece interval is small.
# 

# # Test
# 
# 
# All apps rating mean=mu1
# 
# H0:mu1=4.19
# 
# H1:mu1!=4.19
# 
# alpha=0.05
# 
# (two-side test)

# In[ ]:


def zstar(data):
    para=4.2
    up=data.mean()-para
    down=data.std()/math.sqrt(len(data))
    return up/down


# In[ ]:


alpha=0.05
teststats=zstar(original.loc[:,'Rating'])
zalpha=stats.norm.pdf(alpha/2)
if math.fabs(teststats)<=zalpha:
    print('|',teststats,'|','<=',zalpha)
    print('not reject H0')
else:
    print('|',teststats,'|','>',zalpha)
    print('reject H0')


# # Test summary
# 
# With using two-side test,
# 
# with alpha=0.05,
# 
# test statistics is in reject region,
# 
# which means population mean is different from sample mean

# # Correlation

# In[ ]:


x=original.loc[:,['Rating','Reviews']].corr()
r=x.iloc[1,0]


# In[ ]:


def b1(data1,data2,r):
    return r*(data2.std()/data1.std())
def b2(co1,data1,data2):
    return data2.mean()-(co1*data1.mean())


# In[ ]:


coe1=b1(original['Rating'],original['Reviews'],r)
coe2=b2(coe1,original['Rating'],original['Reviews'])
print(coe2,'+',coe1,'x')


# In[ ]:


x=np.array(original['Rating'])
y=np.array(original['Reviews'])
x1=np.linspace(np.min(x),np.max(x),1000)
y1=x1*coe1+coe2
z2=plt.figure(figsize=(15,14))
plt.scatter(x,y)
plt.plot(x1,y1)
plt.title('scatter(rating,reviews)')
plt.xlabel('Rating')
plt.ylabel('Reviews')
plt.show()


# In[ ]:


sampledata=original.loc[:,['Rating','Reviews']].sample(1000)
sampledata['predict']=sampledata['Rating']*coe1+coe2
sampledata['Residual']=sampledata['Reviews']-sampledata['predict']
sampledata.head()


# In[ ]:


stats.describe(sampledata['Residual'])


# # Correlation summary
# 
# - Reviews and Rating has small correlation
# 
# 
# - With using least-squares,
# 
#   the scatter shows that a lot part of data is higher than predict value
#   
#   
#   
# - With describing residual,range is large,
# 
#   and mean is -96031,
#   
#   which means using least-squares to predict is less effect.
