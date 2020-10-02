#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


train_data = pd.read_csv("../input/titanic/train.csv")
test_data =pd.read_csv("../input/titanic/test.csv")
print("Number of rows and columns present in the train data is ", train_data.shape)
print("Number of rows and columns present in the test data is ", test_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


train_data.drop(['Name', 'Ticket'],axis=1,inplace=True)
test_data.drop(['Name', 'Ticket'],axis=1,inplace=True)


# In[ ]:


train_data.head()


# #  EDA 

# ## Visualization of the data 

# In[ ]:


sns.countplot(train_data["Survived"])

# we can see that count of surived and not survied is diffrent and 
#we can say that data is imbalance and we need to balance the data in preprocessing 


# In[ ]:


sns.countplot(train_data["Pclass"])
# From the below graph we can say that number of people in 3rd class is highest with 500 people and then 1st class with 200 


# In[ ]:


sns.countplot(train_data["Sex"])


# In[ ]:


plt.hist(train_data["Age"],bins=5)


# In[ ]:


plt.hist(train_data["Fare"],bins=10)


# ## Preprocessing

# ### Null value Treatement 

# In[ ]:


def TreatNullValue(data):
    NulValPerdf=pd.DataFrame(data.isna().sum()/data.shape[0])*100
    threshold=float(input('Enter a value of threshold for null values: '))
    nUnique=int(input('Enter the number of unique values to decide the categorical variable: '))
    dropCol=[]
    con=[]
    cat=[]
    disnum=[]
    for i in NulValPerdf.index:
        if NulValPerdf.at[i,0]>=threshold:
            dropCol.append(i)
            print("drop col",dropCol)
    data.drop(dropCol,axis=1,inplace=True)
    for j in data.columns:
        if (data[j].nunique()>=nUnique) and (data[j].dtype=='int64' or data[j].dtype=='float64'):
            con.append(j)
            data[j].fillna(value=data[j].median(), inplace=True)
        elif (data[j].nunique()<nUnique) and (data[j].dtype=='int64' or data[j].dtype=='float64'):
            disnum.append(j)
        else:
            cat.append(j)
            data[j].fillna(value=data[j].value_counts().index[0],inplace=True)
    return (con,cat,disnum,data)


# In[ ]:


(con,cat,disnum,data)=TreatNullValue(train_data)
(con_test,cat_test,disnum_test,data)=TreatNullValue(test_data)


# In[ ]:


print("continuous data",con)
print("categorical data",cat)
print("discreate numerical",disnum)
print("shape of the dataset",train_data.shape)


# 
# ### Encoding of the data

# In[ ]:


def encode(data):
    dummies_dataset=pd.get_dummies(data[cat])
    print("Shape of dummy dataset",dummies_dataset.shape)
    #dropping the cat data from main data set 
    data.drop(cat,axis=1,inplace=True)
    for i in dummies_dataset:
        data[i]=dummies_dataset[i]
    print("shape of the train data after dummy encoding ",data.shape)
    return(data)


# In[ ]:


train_data=encode(train_data)
test_data=encode(test_data)


# In[ ]:


test_data.head()


# ### Balancing of data

# In[ ]:


from imblearn.over_sampling import SMOTE
x_train =train_data.drop("Survived",axis=1)
y_train=train_data["Survived"]
print('Before oversampling, counts of label 1',sum(y_train==1))
print('Before oversampling, counts of label 0',sum(y_train==0))
sm = SMOTE (random_state=2)
X_train_res,y_train_res = sm.fit_sample(x_train,y_train.ravel())
print("After sampling shape of X train and Ytrain ", X_train_res.shape,y_train_res.shape)
print('After oversampling, counts of label 1',sum(y_train_res==1))
print('After oversampling, counts of label 0',sum(y_train_res==0))


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_res, y_train_res)
y_pred_class=logreg.predict(test_data)
y_pred_class


# In[ ]:


df = pd.DataFrame(y_pred_class,index=None,columns=['Survived'])


# In[ ]:


export_csv = df.to_csv("gender_submission1.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:




