#!/usr/bin/env python
# coding: utf-8

# **Before the trip begin**
# 
# If you like my work, please hit upvote since it will keep me motivated

# At this kernel we're going to analyse dataframe we have, and after removing the outliers and duplicated rows and at the end we're going to use LG to predict.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.tail(10)


# In[ ]:


data.shape
b=data.shape[0]#we need this later


# In[ ]:


data.columns
#data.shape


# Before we move forward let's see if there any Null variable

# In[ ]:


data.info()


# GOOD! all the dataframe contains non-null and the only categorical features is the target, even that we should check for duplicated rows and remove them to make the data clean. 

# In[ ]:


data.drop_duplicates(subset=data.columns.values[:-1], keep='first',inplace=True)
print(b-data.shape[0]," duplicated Rows has been removed")


# In[ ]:


data.shape


# Let's see what target data hide from us .

# In[ ]:


sns.countplot(x='Class',data=data)


# As you see there just a few fraud transaction in our data, it may effect the result of the accuracy but we are going to avoid that as much as we can.

# In[ ]:


data.Class.value_counts()


# **Visualisation**

# In[ ]:


g=sns.FacetGrid(data,col='Class')
g.map(plt.hist,'Time', bins=20)


# In[ ]:


g=sns.FacetGrid(data,col='Class')
g.map(plt.hist,'Amount', bins=20)


# In[ ]:


#sns.pairplot(data)


# In[ ]:


plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), vmax=.8 , square=True,annot=True,fmt='.2f')


# We notice that most of the features are correlated with target data 'Class', so no need to drop any feature.

# In[ ]:


data.corr().nlargest(31,'Class')['Class']


# Here is a function which show us every feature's distribution 

# In[ ]:


def feature_dist(df0,df1,label0,label1,features):
    plt.figure()
    fig,ax=plt.subplots(6,5,figsize=(30,45))
    i=0
    for ft in features:
        i+=1
        plt.subplot(6,5,i)
        # plt.figure()
        sns.distplot(df0[ft], hist=False,label=label0)
        sns.distplot(df1[ft], hist=False,label=label1)
        plt.xlabel(ft, fontsize=11)
        #locs, labels = plt.xticks()
        plt.tick_params(axis='x', labelsize=9)
        plt.tick_params(axis='y', labelsize=9)
    plt.show()

t0 = data.loc[data['Class'] == 0]
t1 = data.loc[data['Class'] == 1]
features = data.columns.values[:30]
feature_dist(t0,t1 ,'Normal', 'Busted', features)


# Now let's move to removing outliers section 
# but before let see if there any usng boxplot 

# In[ ]:


def showboxplot(df,features):
    melted=[]
    plt.figure()
    fig,ax=plt.subplots(5,6,figsize=(30,20))
    i=0
    for n in features:
        melted.insert(i,pd.melt(df,id_vars = "Class",value_vars = [n]))
        i+=1
    for s in np.arange(1,len(melted)):
        plt.subplot(5,6,s)
        sns.boxplot(x = "variable", y = "value", hue="Class",data= melted[s-1])
    plt.show()


showboxplot(data,data.columns.values[:-1])


# As u can see a lot of outliers and it will effect our study if we dont remove them 
# if you dont know what is an Outlier here's an example explain it :
# A value that "lies outside" (is much smaller or larger than) most of the other values in a set of data. For example in the scores 25,29,3,32,85,33,27,28 both 3 and 85 are "outliers".
# we'arnt going to remove them until we split the data so it will be removed just in train data not test data .

# In[ ]:


X=data.drop(['Class'],axis=1)
y=data['Class']


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40, shuffle =True)


# In[ ]:


#we combine the train data here for the function of removing outliers 
X_train['Class']=y_train


# In[ ]:



def Remove_Outliers(df,features):
    
    
    Positive_df = df[df["Class"] == 1]#1
    Negative_df = df[df["Class"] == 0]#0
    before=df.shape[0]

    for n in features:
        
        desc1 = Positive_df[n].describe()
        lower_bound1 = desc1[4] - 1.5*(desc1[6]-desc1[4])
        upper_bound1 = desc1[6] + 1.5*(desc1[6]-desc1[4])
        
        desc0 = Negative_df[n].describe()
        lower_bound0 = desc0[4] - 1.5*(desc0[6]-desc0[4])
        upper_bound0 = desc0[6] + 1.5*(desc0[6]-desc0[4])

        df=df.drop(df[(((df[n]<lower_bound1) | (df[n]>upper_bound1))
                      &
                      (df['Class']==1))
                      |
                      (((df[n]<lower_bound0) | (df[n]>upper_bound0))
                      &
                      (df['Class']== 0))].index)

    after=df.shape[0]
    print("number of deleted outiers :",before-after)
    return df


a=Remove_Outliers(X_train,X_train.columns.values[:-1])
X_train=a.iloc[:,:-1]
y_train=a.iloc[:,-1]


# That's huge number it may be half of the data we have but it's okay !
# let's see now the boxplot again

# In[ ]:


def showboxplot(df,features):
    melted=[]
    plt.figure()
    fig,ax=plt.subplots(5,6,figsize=(30,20))
    i=0
    for n in features:
        melted.insert(i,pd.melt(df,id_vars = "Class",value_vars = [n]))
        i+=1
    #print(melted[29])
    # print(len(melted))
    #print(np.arange(len(melted)+1))
    for s in np.arange(1,len(melted)):
        plt.subplot(5,6,s)
        sns.boxplot(x = "variable", y = "value", hue="Class",data= melted[s-1])
    plt.show()


showboxplot(a,a.columns.values[:-1])


# Better than before 
# Great!!

# **Let's predict**

# But before let's not forget to transform the train data and test data to make the algorithm work faster 

# In[ ]:


from sklearn.preprocessing import StandardScaler

X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


#logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


Train_acc_log = round(logreg.score(X_train, y_train) * 100, 3)
Test_acc_log = round(logreg.score(X_test, y_test) * 100, 3)
acc_logreg=round(accuracy_score(y_test, y_pred)*100,3)

print("Score : ",Test_acc_log)


# In[ ]:


sns.heatmap(confusion_matrix(y_test , y_pred), center=True,annot=True,fmt='.1f')


# That's all.
# 
# Again if you have any question leave it in comment section 
# ,and dont forget the UPVOTE 
# thanks

# check other notebooks here
# [https://www.kaggle.com/abdilatifssou/notebooks](http://)
