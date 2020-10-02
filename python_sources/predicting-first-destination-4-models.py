#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
plt.style.use('ggplot')


# In[ ]:


#read files, join - MUST HAVE FILES IN LOCAL DIRECTORY
train = pd.read_csv('../input/train_users_2.csv')
countries = pd.read_csv('../input/countries.csv')
demographics = pd.read_csv('../input/age_gender_bkts.csv')


# In[ ]:


train.date_account_created = train.date_account_created.astype('datetime64')


# In[ ]:


train['date_first_booking'] = pd.to_datetime(train['date_first_booking'], errors='coerce')


# ### Data Exploration

# In[ ]:


#Accounts created over time - Appendix [4]
train.groupby([train["date_account_created"].dt.year,train["date_account_created"].dt.month,])['id'].count().plot(kind='bar')
plt.xlabel("created date")
plt.title("Accounts created over time")
plt.xticks([])
#plt.savefig("CreatedOverTime")
plt.show()


# In[ ]:


#Bookings by Month - appendix [5]
train.groupby(train["date_first_booking"].dt.month)['id'].count().plot(kind='bar')
plt.xlabel("Booked Month")
plt.title("Bookings by Month")
plt.tight_layout()
plt.savefig("Monthly")
plt.show()


# In[ ]:


#check baseline for desintation country - Appendix[3]
train.groupby(train["country_destination"])['id'].count().sort_values().plot(kind='bar')
plt.title("Outcome Counts")
#plt.savefig("outcomecoountry")
plt.show()

#which algorithms do best with unbalanced datasets?


# In[ ]:


train.head()


# In[ ]:


#reduce the gender categories to 3 values from 4
train.gender[train.gender == 'OTHER'] = '-unknown-'


# In[ ]:


#set erroneous age values to NaN
train.age[train.age<18] = np.nan
train.age[train.age>100] = np.nan


# In[ ]:


#Plot the distribution of genders
train.gender.value_counts().plot(kind='bar')
plt.title('Histogram of Gender Variable')
plt.tight_layout()
#plt.savefig("genderDist")
plt.show()


# In[ ]:


#Plot the distribution of age
train.age.plot(kind='hist',bins=40)
plt.title('Histogram of Age Variable')
#plt.savefig("ageDist")
plt.show()


# ### Data Preparation for Modelling

# In[ ]:


#Create Age Buckets to add to our onehot encoded data frame

train['Age_Over40'] = (train.age >39).map({True:1,False:0})
train['Age_31-39'] = ((train.age<40) & (train.age>30)).map({True:1,False:0})
train['Age_Under31'] = (train.age <31).map({True:1,False:0})
train['Age_unknown'] = (train.age.isnull()).map({True:1,False:0})


# In[ ]:


#reorder columns to have all features to encode side by side
cols = train.columns.tolist()
cols.insert(0,cols.pop(5))
cols.insert(0,cols.pop(15))
train = train[cols]

#one hot encoding to prepare for modelling
encoding = pd.get_dummies(train.iloc[:,6:16],columns =train.iloc[:,6:16].columns, prefix=list(train.columns[6:16]))
onehot = pd.concat([train.iloc[:,:6],encoding,train.iloc[:,16:20]],axis=1)


# In[ ]:


#train test split dataset to measure performance. Original test_users dataset provided by airbnb does not come with labels
# so we need to create our own test set
x_train,x_test,y_train,y_test = train_test_split(onehot,onehot['country_destination'],test_size=0.25,random_state=1)


# ### Modelling

# ##### Random Forest

# In[ ]:


#Random forest classification - instantiate classifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()


# In[ ]:


#fit RF classifier 

clf.fit(x_train.iloc[:,6:156],y_train)


# In[ ]:


#predict and store predictions in a series 

preds = clf.predict(x_test.iloc[:,6:156])


# In[ ]:


#add the series to our dataframe

x_test['predicted_country'] = preds


# In[ ]:


#import packages used for model evaluation

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score
def score(true,pred):
    return(precision_score(true,pred,average='weighted'),
          recall_score(true,pred,average='weighted'),
          f1_score(true,pred,average='weighted'))


# In[ ]:


#get overall accuracy score for the RF model

accuracy_score(x_test['country_destination'],x_test['predicted_country'])


# In[ ]:


#FEATURE SELECTION - check most important features to the random forest algorithm

featureImportance = pd.DataFrame(clf.feature_importances_,onehot.columns[6:158],columns=['feature_importance']).sort_values(["feature_importance"],ascending=False)
featureImportance.head(10)


# In[ ]:


#split the data again using only the top 27 features

x_train,x_test,y_train,y_test = train_test_split(onehot.loc[:,list(featureImportance[:27].index)],onehot['country_destination'],test_size=0.25,random_state=1)


# In[ ]:


#Generate Predictions for RF classifier with 27 features

clf = RandomForestClassifier()
clf.fit(x_train,y_train)
preds = clf.predict(x_test)


# In[ ]:


#New accuracy score - .4% improvement in classification, and a simpler model !

RF = accuracy_score(y_test,preds)
RF


# ##### Naive Bayes

# In[ ]:


#import NB package

from sklearn.naive_bayes import BernoulliNB


# In[ ]:


#Generate predictions using Naive Bayes

clf = BernoulliNB()
clf.fit(x_train,y_train)
preds = clf.predict(x_test)


# In[ ]:


#NB accuracy score - 56.6%, lower than baseline

BNB = accuracy_score(y_test,preds)
BNB


# ##### Neural Network

# In[ ]:


#import NN package

from sklearn.neural_network import MLPClassifier


# In[ ]:


#Generate prediction using Neural Net

clf = MLPClassifier()
clf.fit(x_train,y_train)
preds = clf.predict(x_test)


# In[ ]:


#Neural Network accuracy score - 63.6%

NN = accuracy_score(y_test,preds)
NN


# ## Wrap Up

# In[ ]:


#Accuracy of each model

pd.DataFrame({'Random Forest':RF,'Bernoulli Naive Bayes':BNB,'Neural Network':NN,'Baseline':0.58},index=[0])

