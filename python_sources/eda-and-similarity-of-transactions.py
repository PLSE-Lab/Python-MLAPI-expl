#!/usr/bin/env python
# coding: utf-8

# ## Kaggle Data Set - Credit Card Fraud
# ### The datasets contains transactions made by credit cards in September 2013 by european cardholders. 
# ## Dataset Information:
#     * Number of Instances: 284,807
#     * Number of Attributes: 31 (including the class attribute)
#     * Attribute Information:
#     * Features V1, V2, ... V28 are the principal components obtained with PCA.    
#     * The only features which have not been transformed with PCA are 'Time' and 'Amount'.
#     * Feature 'Time' contains the seconds elapsed between each transaction and the first 
#       transaction in the   dataset.
# ### Class (class attribute):
#     * Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 
#       otherwise. 
#       1 = Fraud Transaction
#       0 = Normal Transaction
# ### All the remaining details regarding the data set can be found in the below link.
# ### [CreditCardFraud](https://www.kaggle.com/dalpozz/creditcardfraud/data)

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#download the data set from  
#https://www.kaggle.com/dalpozz/creditcardfraud/data
# load the data set
url = "../input/creditcard.csv"
data=pd.read_csv(url)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


#how many transactions are fraud 
data["Class"].value_counts()


# #####  **Class label**:
#    * label 0 - **Normal transactions** count is  **284315**.
#    * label 1 - **Fraud transactions** count is **492**.
#    
# Clearly it is an **imbalanced dataset**.   

# In[ ]:


# statistics
data.describe()


# In[ ]:


# lets plot plain scatter plot considering Amount and Class
data.plot(kind='scatter', x='Amount', y='Class',title ='Amount verus Transactions type');
plt.show()


# ### Observation:
#  1) We can see from the above Scatter plot that **most of the transaction amounts** are **between 0 to 2500 for both normal and fraud**.

# #### Let us check fraud Amount versus normal Amount distrubutions

# In[ ]:


g=sns.FacetGrid(data, hue='Class', size=8)
plot=g.map(sns.distplot,"Amount").add_legend()
g = g.set(xlim=(0,3000))


# In[ ]:


#Divide the dataset according to the label FraudTransactions and Normal Transactions
# Fraud means Class=1 and Normal means status =0
fraud=data.loc[data["Class"]==1]
normal=data.loc[data["Class"]==0]


# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(121)
fraud.Amount.plot.hist(title="Histogram of Fraud transactions")
plt.subplot(122)
normal.Amount.plot.hist(title="Histogram of Normal transactions")


# In[ ]:


print("Summary Statistics of fraud transactions:")
fraud.describe().Amount


# In[ ]:


print("Summary Statistics of Normal transactions:")
normal.describe().Amount


# ### Observation:
# From the above plots and Statistics we can see that  **fraud transactions amount on average is higher than normal transactions amount though absolute amount for normal transactions is high.** Based on this **we cannot simply come up with a condition on amount** to detect a fraud transaction.
# 

# ### Let us Analyze fraud and normal transactions with respect to time - Though each transaction is different just out of curiosity, am checking fraud transactions occurence with respect to time on two days

# In[ ]:


# DataSet contains two days transactions. 
# Feature 'Time' contains the seconds elapsed between each transaction and the first 
# transaction in the dataset.let us convert time in seconds to hours of a day
dataSubset = data[['Time', 'Amount', 'Class']].copy()


# In[ ]:


# Get rid of $ and , in the SAL-RATE, then convert it to a float
def seconds_Hour_Coversion(seconds):
      hours = seconds/(60*60) ## for conversion of seconds to hours.
      if hours>24: 
    ## if it is more than 24 hours then divide it by 2 as max number of hours is 48.
        hours= hours/2 
        return int(hours)
      else:
        return int(hours)


# In[ ]:


# Save the result in a new column
dataSubset['Hours'] = dataSubset['Time'].apply(seconds_Hour_Coversion)


# In[ ]:


g=sns.FacetGrid(dataSubset, hue='Class', size=10)
plot=g.map(sns.distplot,"Hours").add_legend()
g = g.set(xlim=(0,24))


# In[ ]:


#Divide the data set according to the label FraudTransactions and Normal Transactions
# Fraud means Class=1 and Normal means status =0
frauddata=dataSubset.loc[data["Class"]==1]
normaldata=dataSubset.loc[data["Class"]==0]


# In[ ]:


frauddata.describe()


# ### Observation:
# 1) During the **early hours** i.e at (2 to 3 AM) there are **more fraud transactions when compared with normal transactions** - may be more chance of occuring during that time.

# In[ ]:


#let us plot a heat map for correlation of the features
sns.heatmap(data.corr())


# ### Observation:
# 1) All most all the features are **uncorrelated**.

# ## Cosine Similarity of Transactions:
# ####  1)Extract 100 Samples from the dataSet. 
# ####  2)For every transaction in the sample  top 10 transactions in the dataset which have the lowest similarity(i,j).

# In[ ]:


## let us take 100 samples from the dataset using train_test_split without missing 
# class distrubution in the original dataset.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data.loc[:, data.columns != 'Class'],                data['Class'], test_size=0.00035, random_state=42)
sample = pd.concat([X_test, y_test], axis=1) 
sample.shape


# In[ ]:


# Computing the Similarity
similarity=cosine_similarity(sample)


# In[ ]:


# rename the index and name it as TransactionId
sample.index.name = 'TransactionId'
sample.head()


# In[ ]:


def printResult(transaction1,first10pairs,sample,dict):
     x=transaction1[1][30]
     y=transaction1[0]
     s1='For the transaction id = '+ '{:d}'.format(y) + ', and Class = ' +         '{0:.5g}'.format(x)
     print (s1 +"\n")   
     print ('Similar transactions are :'+ '\n')
     for k in first10pairs:
        printSimilarity(k,dict[k],sample)
     print ('--------------------------------------------------------'+"\n")   
            
def printSimilarity(transactionId,similarity,sample):
   
     for transaction in sample.iterrows(): 
            if transaction[0] == transactionId:
              x=transaction[1][30]
              s=similarity
              y=transactionId  
              print ("Class = " + '{0:.5g}'.format(x) + ", Similarity = "+                 '{:f}'.format(s)+ ", transactionId = "+'{:d}'.format(y)+"\n")
              


# In[ ]:


import operator
import itertools
i=-1;
dict={}
for transaction1 in sample.iterrows() :
        i=i+1
        j=0
        for transaction2 in sample.iterrows():
            if i is not j:
              dict[transaction2[0]] = similarity[i][j]
            j=j+1
                   
        if dict : 
            sorted_dict = sorted(dict, key=dict.__getitem__)[:10]
            printResult(transaction1,sorted_dict,sample,dict)
            dict.clear()    
        


# In[ ]:




