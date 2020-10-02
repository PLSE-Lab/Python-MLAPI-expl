#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import numpy as np # linear alegbra
import pandas as pd # data processing
import os # os commands
from datetime import datetime as dt #work with date time format
get_ipython().run_line_magic('matplotlib', 'notebook')
# initiate matplotlib backend
import seaborn as sns # work over matplotlib with improved and more graphs
import matplotlib.pyplot as plt #some easy plotting
transactions = pd.read_csv('../input/transactions.csv', engine = 'c', sep=',')#reading the transaction file


# In[ ]:


transactions.info()


# In[ ]:


transactions.describe()


# ** to reduce the size of transactions dataframe**

# In[ ]:



print("payment_plan_days min: ",transactions['payment_plan_days'].min())
print("payment_plan_days max: ",transactions['payment_plan_days'].max())

print('payment_method_id min:', transactions['payment_method_id'].min())
print('payment_method_id max:', transactions['payment_method_id'].max())


# In[ ]:


# h=change the type of these series
transactions['payment_method_id'] = transactions['payment_method_id'].astype('int8')
transactions['payment_plan_days'] = transactions['payment_plan_days'].astype('int16')


# In[ ]:


print('plan list price varies from ', transactions['plan_list_price'].min(), 'to ',transactions['plan_list_price'].max() )
print('actual amount varies from ', transactions['actual_amount_paid'].min(),'to ', transactions['actual_amount_paid'].max() )


# In[ ]:


transactions['plan_list_price'] = transactions['plan_list_price'].astype('int16')
transactions['actual_amount_paid'] = transactions['actual_amount_paid'].astype('int16')


# ** size of file has decreased by almost 33% **

# In[ ]:


transactions.info()


# In[ ]:


transactions['is_auto_renew'] = transactions['is_auto_renew'].astype('int8') # chainging the type to boolean
transactions['is_cancel'] = transactions['is_cancel'].astype('int8')#changing the type to boolean


# In[ ]:


sum(transactions.memory_usage()/1024**2) # memory usage 


# In[ ]:


transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'].astype(str), infer_datetime_format = True, exact=False)
# converting the series to string and then to datetime format for easy manipulation of dates
sum(transactions.memory_usage()/1024**2) # this wouldn't change the size of df as memory occupied by object is similar to datetime


# In[ ]:


transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'].astype(str), infer_datetime_format = True, exact=False)


# ** repeating the same process on members file/df**

# In[ ]:


members = pd.read_csv('../input/members.csv')


# In[ ]:


members.info()


# In[ ]:


members.describe()


# In[ ]:


members['city']=members['city'].astype('int8');
members['bd'] = members['bd'].astype('int16');
members['bd']=members['bd'].astype('int8');
members['registration_init_time'] = pd.to_datetime(members['registration_init_time'].astype(str), infer_datetime_format = True, exact=False)
members['expiration_date'] = pd.to_datetime(members['expiration_date'].astype(str), infer_datetime_format = True, exact=False)


# ** doing the same with train data**

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()
train['is_churn'] = train['is_churn'].astype('int8');


# ** now merging all the dataframe with inner joint as we would not want half information about users**

# In[ ]:


members_trans = members.merge(transactions, how='inner', on='msno')
data = members_trans.merge(train, how='inner', on='msno')
# deleting the previously imported df as they occupy space in memory
del transactions
del members
del train
del members_trans


# In[ ]:


#total memory consumptions by all these data frame
sum(data.memory_usage()/1024**2)


# Number of values in missing in data 

# In[ ]:


sum(data['gender'].isna())/len(data)


# ~51% of gender data is missing or the users did not provide the data. This can be ascertained by seeing whether their is signinficant difference in churn of those whose gender is available vs those whose gender data is not available.

# **EAD and dummy variables**

# In[ ]:


plt.figure()
data.groupby(['msno','gender'])['is_churn'].mean().groupby(level=1).mean().plot.bar();
#taking the mean will give us a fraction of people churning as values are 0/1


# ** There seems to be no correlation between geder and churn, but let's check for a correlation between people who provided gender and who did not**

# In[ ]:


def assign_gender(item):
    if (item == 'male')|(item == 'female'):
        return 1
    else:
        return 0
data['gender'] = data['gender'].apply(assign_gender)


# In[ ]:


plt.figure()
data.groupby(['msno','gender'])['is_churn'].mean().groupby(level=1).mean().plot.bar();


# ** we found a new dummy variable**

# In[ ]:


# plotting the correlation heatmap between the variables

correl = data.corr()

mask = np.zeros_like(correl, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correl, mask=mask, cmap=cmap, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})





# but this doesn't tell us a lot about any correlation

# **Also, there is a difference in the plan listprice and actual amount paid. So surely there is a new variable in discount. **

# In[ ]:


data['discount'] = data['plan_list_price'] - data['actual_amount_paid']


# **See how many people churn from different cities**

# In[ ]:



get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure()
data.groupby(['msno','city'])['is_churn'].mean().groupby(level=1).mean().plot.bar();
ax= plt.gca();
ax.set_xticks(np.linspace(0,22,23));


# **surely, there is a huge variation in churn proportion from different cities**

# Doing the same analysis for other variables such as registered_via, payment_method_id, is_churn

# In[ ]:


data['registered_via'].unique()


# In[ ]:


plt.figure()
data.groupby(['msno','registered_via'])['is_churn'].mean().groupby(level=1).mean().plot.bar();


# In[ ]:





# In[ ]:


plt.figure()
data.groupby(['msno','payment_method_id'])['is_churn'].mean().groupby(level=1).mean().plot.bar();


# In[ ]:


plt.figure()
data.groupby(['msno','is_cancel'])['is_churn'].mean().groupby(level=1).mean().plot.bar();


# ** now plotting a pairplot for each variable**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure();
sns.pairplot(data.head(), hue='is_churn', diag_kind='kde');
fig.canvas.draw();


# ** now creating dummy variables for Payment_method_id, gender(whether gender priovided or not), registeres_via and city**

# In[ ]:


newdf= data.join(pd.get_dummies(data['payment_method_id'])) #creating a new columns for paymenth method id dummyvariable 

payment_method_id = {} 
for i in data['payment_method_id'].unique():
    payment_method_id.update({i:'payment_method_id{}'.format(i)})   # create a dictionary to automatic renaming of columns

newdf = newdf.rename(columns=payment_method_id) #renaming the new columns
del newdf['payment_method_id']# deleting the extra columns
newdf.head()


# In[ ]:


newdf= newdf.join(pd.get_dummies(newdf['gender'])) #creating a new columns for paymenth method id dummyvariable 

gender = {} 
gender.update({True:'gender_provided'})   # create a dictionary to automatic renaming of columns
gender.update({False:'gender_not_provided'})
newdf = newdf.rename(columns=gender) #renaming the new columns
del newdf['gender']# deleting the extra columns
newdf.columns


# In[ ]:


newdf= newdf.join(pd.get_dummies(newdf['registered_via'])) #creating a new columns for paymenth method id dummyvariable 

registered_via = {} 
for i in data['registered_via'].unique(): 
    registered_via.update({i:'registered_via{}'.format(i)})   # create a dictionary to automatic renaming of columns
    

newdf = newdf.rename(columns=registered_via) #renaming the new columns
del newdf['registered_via']# deleting the extra columns
newdf.columns


# In[ ]:


newdf= newdf.join(pd.get_dummies(newdf['city'])) #creating a new columns for paymenth method id dummyvariable 

city = {} 
for i in data['city'].unique(): 
    city.update({i:'city{}'.format(i)})   # create a dictionary to automatic renaming of columns
    

newdf = newdf.rename(columns=city) #renaming the new columns
del newdf['city']# deleting the extra columns
newdf.head(10)


# In[ ]:


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


# ** calculating the percentage of people aged negative or more than 100** 
# 
# 

# In[ ]:


bd_mean = np.mean(newdf['bd'])


# In[ ]:


print(len(newdf[(newdf['bd']<0)|(newdf['bd']>100)])/len(newdf)*100,'%') 


# In[ ]:


newdf[(newdf['bd']<0)|(newdf['bd']>100)].loc[:,'bd'] = bd_mean # filling the odd aged people with value = mean of age of users


# In[ ]:


newdf['count_of_recharge'] = 1


# In[ ]:


newdf = newdf.groupby('msno').agg({ 'bd':np.mean, 'registration_init_time':min, 'expiration_date':max,
       'payment_plan_days':np.mean, 'plan_list_price':np.mean,'count_of_recharge':'sum', 'actual_amount_paid':np.mean,
       'is_auto_renew':np.mean, 'transaction_date':min, 'membership_expire_date':max,
       'is_cancel':np.mean, 'is_churn':min, 'discount':'sum', 'payment_method_id2':np.mean,
       'payment_method_id3':np.mean, 'payment_method_id4':np.mean, 'payment_method_id5':np.mean,
       'payment_method_id6':np.mean, 'payment_method_id8':np.mean, 'payment_method_id10':np.mean,
       'payment_method_id11':np.mean, 'payment_method_id12':np.mean, 'payment_method_id13':np.mean,
       'payment_method_id14':np.mean, 'payment_method_id15':np.mean, 'payment_method_id16':np.mean,
       'payment_method_id17':np.mean, 'payment_method_id18':np.mean, 'payment_method_id19':np.mean,
       'payment_method_id20':np.mean, 'payment_method_id21':np.mean, 'payment_method_id22':np.mean,
       'payment_method_id23':np.mean, 'payment_method_id24':np.mean, 'payment_method_id25':np.mean,
       'payment_method_id26':np.mean, 'payment_method_id27':np.mean, 'payment_method_id28':np.mean,
       'payment_method_id29':np.mean, 'payment_method_id30':np.mean, 'payment_method_id31':np.mean,
       'payment_method_id32':np.mean, 'payment_method_id33':np.mean, 'payment_method_id34':np.mean,
       'payment_method_id35':np.mean, 'payment_method_id36':np.mean, 'payment_method_id37':np.mean,
       'payment_method_id38':np.mean, 'payment_method_id39':np.mean, 'payment_method_id40':np.mean,
       'payment_method_id41':np.mean, 'gender_not_provided':np.mean, 'gender_provided':np.mean,
       'registered_via3':np.mean, 'registered_via4':np.mean, 'registered_via7':np.mean,
       'registered_via9':np.mean, 'registered_via13':np.mean, 'city1':'sum', 'city3':'sum', 
       'city4':'sum','city5':'sum', 'city6':'sum', 'city7':'sum', 'city8':'sum',
       'city9':'sum', 'city10':'sum', 'city11':'sum', 'city12':'sum', 'city13':'sum', 
       'city14':'sum', 'city15':'sum', 'city16':'sum', 'city17':'sum', 'city18':'sum',
       'city19':'sum', 'city20':'sum', 'city21':'sum', 'city22':'sum'})


# In[ ]:


newdf.head(10)


# In[ ]:


newdf.columns


# In[ ]:


newdf[newdf.columns[-21:]] = newdf[newdf.columns[-21:]].applymap(lambda x: 1 if x>0 else 0).apply(lambda x: x.astype('int8'))  # converting 0/1 for city


# In[ ]:


newdf[newdf.columns[12]].describe()


# In[ ]:


newdf[newdf.columns[13:-21]] = newdf[newdf.columns[13:-21]].apply(lambda x:x.astype('float16'))


# In[ ]:


newdf['discount'] = newdf['discount'].astype('int16')


# In[ ]:


newdf[newdf.columns[3:6]].describe()


# In[ ]:


newdf[newdf.columns[3:6]] = newdf[newdf.columns[3:6]].apply(lambda x: round(x).astype('int16'))


# In[ ]:


#churn is 7.7% which is not as bad.
np.divide(np.sum(newdf['is_churn']),newdf.index.nunique())*100


# In[ ]:


newdf.head()


# In[ ]:





# In[ ]:


newdf['days_to_buy_membership'] = newdf['transaction_date'] - newdf['registration_init_time']


# In[ ]:


del newdf['expiration_date']


# In[ ]:


newdf.hist('actual_amount_paid')


# In[ ]:


np.sum(newdf['membership_expire_date']>'2017-02-28')


# In[ ]:


newdf.info


# In[ ]:


from sklearn import linear_model as lm


# In[ ]:


model = lm.logistic.LogisticRegression()


# In[ ]:


mat = np.matrix(newdf[['bd', 'payment_plan_days', 'plan_list_price',
       'count_of_recharge', 'actual_amount_paid', 'is_auto_renew',
        'is_cancel',
       'discount', 'payment_method_id2', 'payment_method_id3',
       'payment_method_id4', 'payment_method_id5', 'payment_method_id6',
       'payment_method_id8', 'payment_method_id10', 'payment_method_id11',
       'payment_method_id12', 'payment_method_id13', 'payment_method_id14',
       'payment_method_id15', 'payment_method_id16', 'payment_method_id17',
       'payment_method_id18', 'payment_method_id19', 'payment_method_id20',
       'payment_method_id21', 'payment_method_id22', 'payment_method_id23',
       'payment_method_id24', 'payment_method_id25', 'payment_method_id26',
       'payment_method_id27', 'payment_method_id28', 'payment_method_id29',
       'payment_method_id30', 'payment_method_id31', 'payment_method_id32',
       'payment_method_id33', 'payment_method_id34', 'payment_method_id35',
       'payment_method_id36', 'payment_method_id37', 'payment_method_id38',
       'payment_method_id39', 'payment_method_id40', 'payment_method_id41',
       'gender_not_provided', 'gender_provided', 'registered_via3',
       'registered_via4', 'registered_via7', 'registered_via9',
       'registered_via13', 'city1', 'city3', 'city4', 'city5', 'city6',
       'city7', 'city8', 'city9', 'city10', 'city11', 'city12', 'city13',
       'city14', 'city15', 'city16', 'city17', 'city18', 'city19', 'city20',
       'city21', 'city22', 'days_to_buy_membership', 'is_churn']])


# In[ ]:


np.random.shuffle(mat)


# In[ ]:


train = mat[:-100000,]
test = mat[-100000:,]


# In[ ]:


xtrain= train[:,:-2]
ytrain = train[:,-1]


# In[ ]:


xtest = test[:,:-2]
ytest = test[:,-1]


# In[ ]:


ytrain.tolist()


# In[ ]:


model.fit(xtrain,ytrain.tolist())


# In[ ]:


model.score(xtest, ytest.tolist())


# In[ ]:


ytest_pred = model.predict(xtest)

sum(ytest_pred)/len(ytest_pred)*100


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.figure(figsize=(20,7))
plt.plot(int(ytest_pred))


# In[ ]:


ytest_pred


# In[ ]:


newdf.to_csv("abc.csv")


# In[ ]:


newdf.head(10)


# In[ ]:




