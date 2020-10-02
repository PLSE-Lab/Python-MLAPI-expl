#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
weather=pd.read_csv('../input/seattleWeather_1948-2017.csv')


# In[5]:


weather.head()


# In[6]:


len(weather)


# In[7]:


weather.dtypes


# In[8]:


weather.describe()


# In[9]:


weather1=weather.dropna() #drops 3 rows
len(weather1)


# In[10]:


# convert date from object to 'datetime' data type
weather1['DATE'] = pd.to_datetime(weather1['DATE'])


# In[11]:


weather1.dtypes


# In[12]:


#pull out year and month from date
weather1['YEAR']=weather1['DATE'].map(lambda x: x.year)
weather1['MONTH']=weather1['DATE'].map(lambda x: x.month)
weather1.head()


# In[13]:


#converting from farenheit to celsius
weather1["TMAX_CELSIUS"]=weather1["TMAX"].apply(lambda x: (x-32)*(5.0/9.0))
weather1["TMIN_CELSIUS"]=weather1["TMIN"].apply(lambda x: (x-32)*(5.0/9.0))
weather1["PRCP_MM"]=weather1["PRCP"].apply(lambda x: x/0.039370078740157)


# In[14]:


weather1.head()


# In[15]:


# drop the farenheit temp and PRCP columns now that we have the temp in celsius and PRCP in MM
weather2=weather1.drop(['TMAX','TMIN','PRCP'],axis='columns')
weather2.head()


# In[16]:


#convert true/false to 1/0 for RAIN
weather2['RAIN'] = weather2['RAIN'].apply(lambda x: 1 if x == True else 0)
weather2.head()


# In[17]:


# Reorder the columns
weather2=pd.DataFrame(weather2,columns=['DATE','YEAR','MONTH','TMAX_CELSIUS','TMIN_CELSIUS','PRCP_MM','RAIN'])
weather2.head()


# In[18]:


weather2.dtypes


# In[19]:


weather2.groupby('YEAR').size()
# shows every year has full daily data (365 or 366 days) except 2017


# In[20]:


# create new dataframe excluding 2017 which does not have full year's data
weather2=weather2[weather2.YEAR != 2017]
weather2.tail() #new df ends at 2016


# In[21]:


len(weather2)


# In[22]:


#Add season - assumption - Mar to May=Spring; June to Aug=Summer; Sep to Nov=Autumn; Dec to Feb=Winter
def get_season(row):
    if row['DATE'].month >= 3 and row['DATE'].month <= 5:
        return '1 Spring'
    elif row['DATE'].month >= 6 and row['DATE'].month <= 8:
        return '2 Summer'
    elif row['DATE'].month >= 9 and row['DATE'].month <= 11:
        return '3 Autumn'
    else:
        return '4 Winter'

weather2['SEASON'] = weather2.apply(get_season, axis=1)
weather2.head()


# In[23]:


weather2.dtypes


# In[25]:


sns.pairplot(weather2,x_vars=['TMAX_CELSIUS','TMIN_CELSIUS','PRCP_MM'],y_vars=['TMAX_CELSIUS','TMIN_CELSIUS','PRCP_MM'],kind='reg')


# In[28]:


#of days with rain in last 10 years
plt.figure(figsize=(20,5))
ax = sns.countplot(data = weather2[(weather2['RAIN'] == True) & (weather2['YEAR'] >= 2007)], x='MONTH', hue='YEAR')
plt.xlabel('MONTH')
plt.ylabel('Number of Days of Rain')
plt.title('Number of Days of Rain by Month During Last 10 Years 2007 to 2016')


# In[33]:


rainy=weather2[(weather2['RAIN'] == True)]
plt.figure(figsize=(20,5))
rainyplot=rainy.groupby('YEAR')['PRCP_MM'].mean()
rainyplot.plot()
plt.xlabel('YEAR')
plt.ylabel('Average Amount of Rain in Millimetres')
plt.title('Yearly Average Amount of Rain 1948-2016')
print (rainyplot)
rainyplot.describe()


# In[36]:


plt.figure(figsize=(20,5))
d=weather2.groupby('YEAR')['PRCP_MM'].sum()
d.plot()
plt.xlabel('YEAR')
plt.ylabel('Total Amount of Rain in Millimetres')
plt.title('Yearly Total Amount of Rain 1948-2016')
d.head()


# In[37]:


c=weather2.groupby('SEASON')['PRCP_MM'].sum()
plt.xlabel('SEASON')
plt.ylabel('Total Amount of Rain in Millimetres')
plt.title('Total Amount of Rain by Season')
c.plot()


# In[38]:


aa=weather2.groupby('YEAR')['TMAX_CELSIUS','TMIN_CELSIUS'].mean()
aa.plot()
plt.xlabel('YEAR')
plt.ylabel('Average Max and Min Temperature')
plt.title('Yearly Average Max and Min Temperature 1948-2016')
aa.describe()


# In[39]:


bb=weather2.groupby('MONTH')['TMAX_CELSIUS','TMIN_CELSIUS'].mean()
bb.plot()
plt.xlabel('MONTH')
plt.ylabel('Average Max and Min Temperature')
plt.title('Monthly Average Max and Min Temperature 1948-2016')


# In[40]:


sns.countplot(data=weather2, x='RAIN')
plt.xlabel('Did it Rain?  0=NO, 1=YES')
plt.ylabel('Number of Days')
plt.title('Number of Days of Rain/No Rain 1948-2016')


# In[41]:


plt.figure(figsize=(20,10))
sns.boxplot(data=weather2.drop(['DATE','YEAR','MONTH','RAIN','SEASON'],axis='columns'))


# MODELLING

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(weather2.drop(['DATE', 'YEAR', 'RAIN','SEASON'],axis=1), weather2['RAIN'], test_size=0.30)


# In[27]:


print (X_train.head())
print (len(X_train)) #70% training=17640 training data points

print (y_train.head())
print (len(y_train))

print (X_test.head())
print (len(X_test)) #30% test=7560 test data points

print (y_test.head())
print (len(y_test))

#total train+test=25200 which is the total # rows in weather2 dataframe


# LOGISTIC REGRESSION

# In[28]:


rainmodel = LogisticRegression()
rainmodel.fit(X_train,y_train)


# In[30]:


rainprediction = rainmodel.predict(X_test)
print (rainprediction)


# In[31]:


print(accuracy_score(y_test,rainprediction))
print(confusion_matrix(y_test,rainprediction))
print(classification_report(y_test,rainprediction))


# logistic regression without PRCP_MM (including it gives 100% accuracy)

# In[33]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(weather2.drop(['DATE', 'YEAR', 'RAIN','SEASON','PRCP_MM'],axis=1), weather2['RAIN'], test_size=0.30)


# In[34]:


rainmodel2 = LogisticRegression()
rainmodel2.fit(X_train2,y_train2)


# In[35]:


rainprediction2 = rainmodel2.predict(X_test2)
print(accuracy_score(y_test2,rainprediction2)) #accuracy drops to 76%
print(confusion_matrix(y_test2,rainprediction2))
print(classification_report(y_test2,rainprediction2))


# DECISION TREE

# In[38]:


dtree2 = DecisionTreeClassifier()
dtree2.fit(X_train2,y_train2)
dtree_prediction2 = dtree2.predict(X_test2)
print(accuracy_score(y_test2,dtree_prediction2)) #72% using decision tree
print(classification_report(y_test2,dtree_prediction2))
print(confusion_matrix(y_test2,dtree_prediction2))


# In[39]:


#Drop "outliers" to see if can improve accuracy
weather3=weather2.drop(weather2[weather2['TMIN_CELSIUS']<-8].index)
weather3=weather3.drop(weather3[(weather3['TMAX_CELSIUS']>36.5) | (weather3['TMAX_CELSIUS']<-6)].index)
weather3=weather3.drop(['DATE','YEAR','PRCP_MM','SEASON'],axis='columns')
weather3.head()


# In[40]:


plt.figure(figsize=(20,10))
sns.boxplot(data=weather3)


# MODELLING AFTER OUTLIERS REMOVED

# In[41]:


X_train3, X_test3, y_train3, y_test3 = train_test_split(weather3.drop(['RAIN'],axis=1), weather3['RAIN'], test_size=0.30)


# In[43]:


rainmodel3 = LogisticRegression()
rainmodel3.fit(X_train3,y_train3)
rainprediction3 = rainmodel3.predict(X_test3)
print (X_train3.head())
print (y_train3.head())
print (X_test3.head())
print (y_test3.head())
print(accuracy_score(y_test3,rainprediction3)) #accuracy score is 75%
print(confusion_matrix(y_test3,rainprediction3))
print(classification_report(y_test3,rainprediction3))


# accuracy score didn't increase so conclusion is those values likely weren't true outliers
