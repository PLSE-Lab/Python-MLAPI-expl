#!/usr/bin/env python
# coding: utf-8

# # House pricing-Kaggle 
# ## by: Guillermo Campollo
# ### 5/16/2020

# ## Importing our libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ## Importing our datasets and splitting X and Y

# In[ ]:


data=pd.read_csv('../data/raw/train.csv')
data_predict=pd.read_csv('../data/raw/test.csv')
#data.columns = data.columns.str.replace(' ', '')
X=data.drop(["SalePrice"],1)
y=data.SalePrice.values
#Dropping the 5 lines with NaNs before appending our test set
X=X.dropna(subset=["MSZoning", "SaleType"],axis=0)
sep=len(X) #this should give us our separator
X=X.append(data_predict)
X


# # Data Preprocessing and Feature Scaling

# In[ ]:


#Here we might take drop NaN columns to see if it improves
X=X.drop(columns=["Alley","PoolQC","Fence","MiscFeature"])


# ### Dummies for categorical

# In[ ]:


#Initial number of NaNs
columns=list(X.columns) #All column names
objects=X.select_dtypes(include='object').columns
numbers=X.select_dtypes(exclude='object').columns#All the object columns needed to be encoded
for i in objects:
    dummy=pd.get_dummies(X[i])
    X=pd.concat([X,dummy],axis=1)
X=X.drop(columns=objects,axis=1)


# ### Dealing with numeric variables

# In[ ]:


#Now we take care of our numerical varaibles by using imputescaler
for i in numbers[1:]:
    sc=SimpleImputer(missing_values=np.nan, strategy='mean')
    X[i]=sc.fit_transform(X[i].values.reshape(-1,1))
sum(X.isnull().sum()) #Now we have no missing values


# In[ ]:


#We drop de Id class
X=X.drop(columns="Id")


# ### Feature Scaling

# In[ ]:


for i in numbers[1:]:
    sc=StandardScaler()
    X[i]=sc.fit_transform(X[i].values.reshape(-1,1))
X


# In[ ]:


#Feature scaling our y variable
sc_y=StandardScaler()
y=sc_y.fit_transform(y.reshape(-1,1))
y


# ### Now we split again our data test and separate our train data

# In[ ]:


data_train=X.iloc[:1460,:]
data_test=X.iloc[1460:,:]


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(data_train, y, test_size = 0.2, random_state = 0)


# # Creating our SVM Regressor model

# In[ ]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)


# In[ ]:


y_pred=sc_y.inverse_transform(regressor.predict(X_test))


# In[ ]:


regressor.score(X_test,y_test)


# #### We got 82% Accuracy... Not bad

# # Here we create our predictions for submitting to kaggle

# In[ ]:


y_pred=sc_y.inverse_transform(regressor.predict(data_test)) #Predictions created


# In[ ]:


results=pd.DataFrame({"Id":data_predict.Id.values,"SalePrice":y_pred}) #Format for csv


# In[ ]:


results.to_csv('../predictions/house_pricing.csv') #CSV export

