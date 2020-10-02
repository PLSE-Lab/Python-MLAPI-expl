#!/usr/bin/env python
# coding: utf-8

# In[456]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser # for parsing date, time values
import time 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import os
print(os.listdir("../input"))


# * **Importing and Exploring the dataset**

# We are given a dataset that includes the data about a mobile App,this is a free app with subscription mode<br>
# our main goal here is to predict which users are UNLIKELY to subscribe to the app, so we would focus most of<br>
# our advertisement efforts on them

# In[411]:


dataset = pd.read_csv("../input/subscription-train/train.csv")


# In[412]:


dataset.head()


# As can be seen above we have data from when users first open the app, what age they are , what features of the app<br> 
# they used(screen_list), number of this used features,did they ever used premium trial, did they like the app and most<br> importnantly if they ever enrolled and if they did, when did it happen
# * my first impression is that screen_list, used_premium_features, liked would be the most 3 important features of the data set.

# In[413]:


dataset.describe()   # we have 50K rows in our dataset.


# In[414]:


dataset.hour.head() # this is when the user first opened the app

# turn the hour column into int from string
dataset.hour = dataset.hour.str.slice(1,3).astype(int)


# In[415]:


dataset.hour.head(3)


# In[416]:


# we create a second dataset for visualization and drop id and string data
ds = dataset.copy().drop(columns=['user','screen_list', 'enrolled_date', 'first_open', 'enrolled'])


# In[417]:


ds.head()


# In[418]:


sns.pairplot(ds)


# In[419]:


sns.countplot(ds['numscreens'])


# In[420]:


plt.hist(ds['age'])


# In[421]:


sns.countplot(ds['hour'])


# In[422]:


sns.scatterplot(x = 'age', y = 'used_premium_feature',data = ds)


# In[423]:


# figuring out the correlation between our numerical features and the enrollment value
ds.corrwith(dataset.enrolled).plot.bar(figsize= (20, 10), fontsize=15, grid=True)


# As it can be seen:
# * increase in age has negative effect on enrollment (younger people have higher chance of enrolling in the product)
# * increase in numscreens has BIG positive effect on enrollment
# * while day of the week or liked have almost no effect on if user enrolled or not
# * *one interesting correlation is that when users try the premium features, the chance of them enrolling actually Decreases.*

# In[424]:


# Correlation Matrix

plt.figure(figsize=(20, 10))
sns.heatmap(ds.corr(), annot= True)


# The correlation heatmap helps us find what features (if any) are dependant on each other This is important because we have this idea that all features are independant of one another and if some of them are actually have linear relationship it will make problems during our training and predictions.

# In[425]:


dataset.dtypes


# In[426]:


# change data type of first_open from string to date type
dataset['first_open'] = [parser.parse(row_data) for row_data in dataset['first_open']]


# In[427]:


dataset['enrolled_date'] = [parser.parse(row_data) if isinstance(row_data, str) else row_data for row_data in dataset['enrolled_date']]


# Now that both our first_open and enrolled_date are numerical features we want to find the difference between them
# and set a limit for the max time distance between the two

# In[428]:


# astype('timedelta64[h]') : this will make the difference into hours
dataset["difference"] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[m]')


# In[429]:


print(dataset["difference"].dropna().mean())
print(dataset["difference"].dropna().median())


# In[430]:


# anyone who enrolled more than 6 hours after enrolling, wolud be considered not enrolled 
dataset.loc[dataset.difference > 360, 'enrolled'] = 0


# In[431]:


# we are not gonna use this dates anymore so we drop them
dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])


# In[432]:


dataset = dataset.drop(columns = ['user'])


# In[433]:


dataset.head()


# We have this screen_list column which is text.
# the way that we process it here is that we import list of the most prominent screen list activities and assign each of them a column so that in this way we can interpret this as a numerical value.

# In[434]:


top_screens = pd.read_csv("../input/top-screens/top_screens.csv")


# In[435]:


top_screens.head(10)


# In[436]:


dataset['screen_list'] = dataset.screen_list.astype(str) + ', '


# In[437]:


top_screens = top_screens.values  # turn it into a nympy array from a dataframe
ls = top_screens[:,1]
for sc in ls:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset["screen_list"] = dataset.screen_list.str.replace(sc+',', "")


# Here we turn the top_screen dataframe into a list of string values<br>
# then we add each of them to the main dataset:<br>
#     *-we search each row of 'screen_list' and if it contains the string same as the one of top_screens values<br>
#     -we put 1 for the column of that value otherwise we put zero<br>
#     -then we ommit that text from scree_list of that row.<br>*

# In[438]:


dataset.head(3)


# for all the other screen_text activities we make another column and put added up number of all of them there

# In[439]:


dataset['other'] = dataset.screen_list.str.count(",")


# In[440]:


dataset.head(3)


# In[441]:


# now we can drop the screen_list column
dataset = dataset.drop(columns = ['screen_list'])


# In[442]:


plt.figure(figsize=(20, 10))
ds_2 = dataset[["Saving1", "Saving2", "Saving2Amount", "Saving1","Saving4","Saving5","Saving6", 
                  "Saving7","Saving8","Saving9","Saving10"]]
sns.heatmap(ds_2.corr(), annot= True)


# In[443]:


# these columns are all highly corelated so we can add them all up into one column
savings_screen = ["Saving1", "Saving2", "Saving2Amount", "Saving1","Saving4","Saving5","Saving6", 
                  "Saving7","Saving8","Saving9","Saving10"]


# In[444]:


dataset.head()


# In[445]:


# add up the values from all columns from savings_screen and then put them in SavingsCount  
dataset['SavingsCount'] = dataset[savings_screen].sum(axis=1)
# next drop all columns of list savings_screen
dataset = dataset.drop(columns=savings_screen)


# In[446]:


cm_screens = ["Credit1", "Credit2", "Credit3", "Credit3Container","Credit3Dashboard"]

dataset['CMCount'] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)


# In[447]:


cc_screens = ["CC1", "CC1Category", "CC3"]

dataset['CCCount'] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)


# In[448]:


loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]

dataset['LoanCount'] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)


# In[449]:


dataset.head()


# In[450]:


Y = dataset['enrolled']
X = dataset.drop(columns="enrolled")


# In[451]:


SC = StandardScaler()
X = SC.fit_transform(X)


# In[452]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[455]:


print(len(X_train))
print(len(X_test))


# In[457]:


# here we have a binary classification (0/1) so we can use a logistic regression classifer to predict the results
# we apply L1 penalty for regularization of our model to prevent overfitting
classifier = LogisticRegression(random_state=0, penalty='l1')
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)


# In[463]:


cm = confusion_matrix(Y_test, y_pred)
print(cm)


# In[461]:


print(classification_report(Y_test, y_pred))


# In[466]:


param_grid = {'C':[0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}


grid = GridSearchCV(LogisticRegression(), param_grid, verbose= 4, cv=3, refit=True)
grid.fit(X_train, Y_train)


# In[467]:


grid.best_params_


# In[468]:


y_predict = grid.predict(X_test)
print(classification_report(Y_test, y_predict))


# In[469]:


cm = confusion_matrix(Y_test, y_predict)
print(cm)


# In[ ]:




