#!/usr/bin/env python
# coding: utf-8

# ## I read other Kaggle scripts claiming 99% plus accuracy. 
# ##To be clear, accuracy is not the correct metric in this case. ##
# ***Reasons:***
# 
#  1. There are very less fraud transactions. Any model will learn on the non-fraud transactions and predict with more that 90% accuracy.
# 
# 
# ----------
# 
# 
# 2 In banking its more important to not to miss out a fraud transaction. So the focus needs to be on this metric. So the accuracy would be (true frauds detected)/(total frauds predicted by model).
# 
# Lets see now, if Naive Bayes is really naive and how it does against fraud. 

# In[ ]:


import numpy as np # linear algebra
from math import log
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[ ]:


credit_data=pd.read_csv('../input/creditcard.csv')


# In[ ]:


credit_data.head()


# In[ ]:


credit_data['Time_in_hours']=credit_data['Time']/3600
credit_data['Log_Amount']=np.log(credit_data['Amount']+1)


# Looking into the time variable and looking what were the peak hours of transactions.

# In[ ]:


sns.plt.hist(data=credit_data,x='Time_in_hours')


# ***The trend seems fair. There are lower number of transactions during the start of the days and transactions increase later on which seems reasonable. Do fraud transactions have a pattern as such at which time there is increase in fraudulent attempts?***

# In[ ]:


sns.FacetGrid(data=credit_data[credit_data['Class']==1],col='Class').map(sns.plt.hist,'Time_in_hours')


# There is no reasonable inference that can be made about the pattern of fraudulent transactions regarding time. We'll see whether to keep this variable or drop it. Lets see if Amount variable has some insights.

# In[ ]:


plt.figure(1)
plt.subplot(211)
sns.plt.hist(data=credit_data[credit_data['Class']==0],x='Amount',label='Normal Transactions')
plt.legend(loc='best')
plt.subplot(212)
sns.plt.hist(data=credit_data[credit_data['Class']==1],x='Amount',label='Fraud Transactions')
plt.legend(loc='best')


# In[ ]:


#Function to find in the correlation matrix the values which are of significant use for us.
def is_high(x):
    for i in range(len(x)):
        if (x.iloc[i]<0.5 and x.iloc[i]>-0.5):
            x.iloc[i]=0
    return x


# In[ ]:


a=credit_data.corr(method='spearman')
a=a.apply(is_high)
#To display the whole correlation matrix
pd.options.display.max_columns = 50
a


# In[ ]:


#To split the data into test and train dataset.
def split_data(dataset,ratio):
    sample=np.random.rand(len(dataset))<ratio
    return(dataset[sample],dataset[~sample])


# Getting the list of all the columns in the credits dataset. So to begin I will initially add all the variables to the model and remove one by one variable and see removing which ones increases the accuracy the most. Also I read in other Kaggle scripts claiming 99% plus accuracy. 
# To be clear, accuracy is not the correct metric in this case. Reasons:
# 
#  1. There are very less fraud transactions. Any model will learn on the non-fraud transactions and predict with more that 90% accuracy.
# 2. In banking its more important to not to miss out a fraud transaction. So the focus needs to be on this metric. So the accuracy would be (true frauds detected)/(total frauds predicted by model).
# 
# Lets see now, if Naive Bayes is really naive.

# In[ ]:


col=list(credit_data.columns.values)


# In[ ]:


#Function to classify based on Naive Bayes. The algorithm runs 10 times and gives the mean of 
#predicted accuracy for each time.And it also tell which variable I removed from the total variable
#list so that I come to know which ones have to be removed.
def NB_Classify(ratio,drop_var):
    print('You dropped:',drop_var)
    #print (train.groupby('Class').count()['V1'])
    #print (test.groupby('Class').count()['V1'])
    pred_acc=[]
    for i in range(10):
        train,test=split_data(credit_data,ratio)
        clf=GaussianNB()
        clf.fit(train.drop(drop_var,axis=1),train['Class'])
        pred=clf.predict(test.drop(drop_var,axis=1))
        #print(pd.crosstab(test['Class'],pred))
        #print('You dropped:',drop_var)
        #print(accuracy_score(test['Class'],pred))
        pred_acc.append([pd.crosstab(test['Class'],pred).iloc[1,1]/(pd.crosstab(test['Class'],pred).iloc[1,0]+pd.crosstab(test['Class'],pred).iloc[1,1])])
    #' and got an accuracy of: ',np.mean(pred_acc)) 
    print(np.mean(pred_acc))


# In[ ]:


for var in col:
    NB_Classify(0.6,['Class','Log_Amount',var])


# **As it can be seen above time is not a good predictor in this case and this clearly supports our inference from the histograms that time had nothing to do with the fraud transactions. So on removing time from the list of predictors**

# In[ ]:


NB_Classify(0.6,['Class','Time','Log_Amount'])


# ## So Naive Bayes classifier is able to detect more than 80% of the fraud transactions. This is good, given the simplicity of the algorithm and its pretty basic and not hardware hungry. ##

# ***Next step would be to increase the accuracy of Naive Bayes but I think it did pretty fine. So will move on to some other classifier such as SVM and see how it performs compared to Naive Bayes.***

# In[ ]:




