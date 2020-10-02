#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Getting stated : load nessesory libraries.
import pandas as pd # For working with data
import numpy as np # For stats and faster array calculations
import seaborn as sns # For visualization
import matplotlib.pyplot as plt
df1 = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv") # Loading the csv file into dataframe


# In[ ]:


df = df1.copy() # Making a copy of data always helps.


# In[ ]:


df.columns 


# In[ ]:


df.info()


# ## All the featues are numericals.
# ### 1.This problem can be persued as Regression problem as well as classification problem. 2.All the numerical features helps as we don't need to spend much time in data preprocessing,converting categorical data to numerical

# In[ ]:


df.isnull().sum() # Do we have any nyll values or not


# In[ ]:


df.describe() # Getting an overview of data, how the feature are spreaded.


# ### Looking at the 50%,75% and maximum value, It seems that some of the features need outlier treatment.[residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide]
# #### Also we don't have any null values. This will save time from missing-value treatment.

# In[ ]:


df['residual sugar'].plot(kind='box')# take a look at destribution


# In[ ]:


sns.distplot(df['residual sugar'])


# #### most of the data lies below 5. Values greater than 5 can be treated as outliers. 

# In[ ]:


temp = df[df['residual sugar']<5]['residual sugar']
temp = temp.mean() #we can take mode() but the distribution is a skewed normal distribution so, won't make much difference.

boolean = df['residual sugar']>5

df.loc[boolean,'residual sugar'] = df.loc[boolean,'residual sugar'].apply(lambda x : temp)


# In[ ]:


sns.distplot(df['residual sugar']) #after outlier treatment


# ### We'll apply the above technique to all the features having outliers problem.

# In[ ]:


sns.distplot(df['chlorides'])


# In[ ]:


df['chlorides'].plot(kind='box')


# In[ ]:


temp = df[df['chlorides']<0.2]['chlorides']
temp = temp.mean()

boolean = df['chlorides']>0.2

df.loc[boolean,'chlorides'] = df.loc[boolean,'chlorides'].apply(lambda x : temp)


# In[ ]:


sns.distplot(df['chlorides'])


# In[ ]:


sns.distplot(df['free sulfur dioxide'])


# In[ ]:


df['free sulfur dioxide'].plot(kind='box')


# In[ ]:


temp = df[df['free sulfur dioxide']<40]['free sulfur dioxide']
temp = temp.mean()

boolean = df['free sulfur dioxide']>40

df.loc[boolean,'free sulfur dioxide'] = df.loc[boolean,'free sulfur dioxide'].apply(lambda x : temp)


# In[ ]:


sns.distplot(df['free sulfur dioxide'])


# In[ ]:


sns.distplot(df['sulphates'])


# In[ ]:


temp = df[df['sulphates']<1.25]['sulphates']
temp = temp.mean()

boolean = df['sulphates']>1.25

df.loc[boolean,'sulphates'] = df.loc[boolean,'sulphates'].apply(lambda x : temp)


# In[ ]:


sns.distplot(df['sulphates'])


# In[ ]:


sns.distplot(df['total sulfur dioxide'])


# In[ ]:


temp = df[df['total sulfur dioxide']<150]['total sulfur dioxide']
temp = temp.mean()

boolean = df['total sulfur dioxide']>150

df.loc[boolean,'total sulfur dioxide'] = df.loc[boolean,'total sulfur dioxide'].apply(lambda x : temp)


# In[ ]:


sns.distplot(df['total sulfur dioxide'])


# In[ ]:


df.describe()


# ### Our data and target values were in the same dataframe, now that the data preprocessing is done we can move forward to split our data into Data(X) and target(Y).
# 
# #### Also, looking at the data, we need to do some featurescaling as well, as the feature have different range of values

# In[ ]:


x = df.drop('quality',axis=1)
y = df.quality
y = np.array(y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)


# In[ ]:


# Loading ML regression models.

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()


# In[ ]:


from sklearn.model_selection import cross_validate # cross validation is used to measure how godd a ML model performes on data


# In[ ]:


cross_validate(lr,x,y)


# In[ ]:


cross_validate(dtr,x,y)


# In[ ]:


cross_validate(rfr,x,y)


# ### No ML model has performed as per our expectation.
# #### Let's see if we can achive better results with RandomForestRegressor

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0,stratify=y)

score_list =[]
for i in range(2,100,5):
    rfr = RandomForestRegressor(n_estimators = i)
    rfr.fit(x_train,y_train)
    pred = rfr.predict(x_test)
    score_list.append(mean_absolute_error(y_test,pred))


# In[ ]:


fig = plt.figure(figsize = (10,6))
plt.plot(list(range(2,100,5)),score_list)


# #### After increasing n_estimators(trees in random forest) in RFR model, we get mean_absolute_error around 0.38-0.40 Which means accuracy around 60-62%. This can be considered as a good regression model for this data.

# In[ ]:


rfr = RandomForestRegressor(n_estimators=35)
rfr.fit(x_train,y_train)
pred = rfr.predict(x_test)
mean_absolute_error(y_test,pred)


# In[ ]:





# ## Now We'll Move to the Classificaion problem.
# ### For this, we can describe wine qualities as good, bad or normal.

# In[ ]:


sns.distplot(df['quality']) # To get an idea about the distibution of wine quality data


# #### Wine having quality score <4.5 can be considered as poor and >6.5 as good quality wine.

# In[ ]:


### Generating Categorical Column.

def applier(x):
    if x<4.5:
        return 'bad'
    elif 4.5<x<6.5:
        return 'normal'
    else :
        return 'good'

df['type'] = df['quality'].apply(lambda x : applier(x))


# In[ ]:


### We have to again specify our data(x) and target(x) for models. 

x = df.drop(columns=['quality','type'])
y = df[['type']]


# In[ ]:


### We need to do go through the feature scaling process again, as we have new unprocessed data.

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
scaler1 = StandardScaler()
x = scaler1.fit_transform(x)


# In[ ]:


### importing ML classification models.

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


# In[ ]:


### We'll implement the cross validation process from skretch.

lr_pred = []
dtc_pred = []
rfc_pred = []
gnb_pred = []

from sklearn.model_selection import KFold
kf = KFold(5)

kf.get_n_splits(x)
for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    lr.fit(X_train,y_train)
    dtc.fit(X_train,y_train)
    rfc.fit(X_train,y_train)
    gnb.fit(X_train,y_train)
    
    lr_pred.append(lr.score(X_test,y_test))
    dtc_pred.append(dtc.score(X_test,y_test))
    rfc_pred.append(rfc.score(X_test,y_test))
    gnb_pred.append(gnb.score(X_test,y_test))
print(lr_pred,dtc_pred,rfc_pred,gnb_pred,sep='\n')


# ## As we can see, LogisticRegression and RandomForest performs better than DecisionTree or NaiveBayes
# ### This is because, DecisionTree useually overfits the training data and NaiveBayes expects features to be naive(unrelated to each other) bacause it works on bayes theorem which is only applied on a set of independent events.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




