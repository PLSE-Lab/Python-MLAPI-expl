#!/usr/bin/env python
# coding: utf-8

# # Wrapper Method Feature Selection
# * In this method, a subset of features are selected and train a model using them. Based on the inference that we draw from the previous model, we decide to add or remove features from subset.
# 

# **I am going to use House Prices data set for Wrapper Feature Selection. So, lets import the data and pre process it before applying feature selection techniques.**

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


# In[2]:


#Loading House prices dataset.
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[3]:


df.shape
#this dataset contain 1460 rows and 81 columns.


# In[4]:


df.info()


# In[5]:


#Feature selection should be done after data preprocessing.
#Ideally  all the categorical variables should be encoded into numbers, so that we can assess how deterministic they are for target.
#Currently I will be dealling with numerical columns only.
colType = ['int64','float64']
#Select the columns which are either int64 or float64.
numCols=list(df.select_dtypes(include=colType).columns)
#Assigning numerical columns from df to data variable. We can use the same variable as well.
data=df[numCols]


# In[6]:


#Lets check the shape.
data.shape
#So there are 38 rows which are numerical.


# In[7]:


#Lets split the data in training set and test set.
X_train,X_test,y_train,y_test=train_test_split(data.drop('SalePrice',axis=1),data['SalePrice'],test_size=.2,random_state=1)

X_train.shape,X_test.shape


# In[8]:


def correlation(dataset,threshold):
    col_corr=set() # set will contains unique values.
    corr_matrix=dataset.corr() #finding the correlation between columns.
    for i in range(len(corr_matrix.columns)): #number of columns
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold: #checking the correlation between columns.
                colName=corr_matrix.columns[i] #getting the column name
                col_corr.add(colName) #adding the correlated column name heigher than threshold value.
    return col_corr #returning set of column names
col=correlation(X_train,0.8)
print('Correlated columns:',col)    
    


# In[9]:


#remove correlated columns
X_train.drop(columns=col,axis=1,inplace=True)
X_test.drop(columns=col,axis=1,inplace=True)
#lets check the shape of training set and test set.
X_train.shape,X_test.shape


# In[10]:


#Filling null values with 0.
X_train.fillna(0,inplace=True)


# In[11]:


#Checking if there is null values.
X_train.isnull().sum().max()


# # Forward feature selection

# In[12]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
#I am going to use RandomForestRegressor algoritham as an estimator. Your can select other regression alogritham as well.
from sklearn.ensemble import RandomForestRegressor
#k_features=10 (It will get top 10 features best suited for prediction)
#forward=True (Forward feature selection model)
#verbose=2 (It will show details output as shown below.)
#cv=5 (Kfold cross valiation: it will split the training set in 5 set and 4 will be using for training the model and 1 will using as validation)
#n_jobs=-1 (Number of cores it will use for execution.-1 means it will use all the cores of CPU for execution.)
#scoring='r2'(R-squared is a statistical measure of how close the data are to the fitted regression line)
model=sfs(RandomForestRegressor(),k_features=10,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')
model.fit(X_train,y_train)


# In[13]:


#Get the selected feature index.
model.k_feature_idx_


# In[14]:


#Get the column name for the selected feature.
model.k_feature_names_


# **These are the best suited columns for prediction as per Forward Feature Selection.**

# # Backward Feature Selection

# **We will be using the same training data for Backward Feature Selection**

# In[15]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestRegressor
#k_features=10 (It will get top 10 features best suited for prediction)
#forward=False (Backward feature selection model)
#verbose=2 (It will show details output as shown below.)
#cv=5 (Kfold cross valiation: it will split the training set in 5 set and 4 will be using for training the model and 1 will using as validation)
#n_jobs=-1 (Number of cores it will use for execution.-1 means it will use all the cores of CPU for execution.)
#scoring='r2'(R-squared is a statistical measure of how close the data are to the fitted regression line)
backwardModel=sfs(RandomForestRegressor(),k_features=10,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')
#We will convert our training data into numpy array. If we will not convert it, model is not able to read some of the column names. 
backwardModel.fit(np.array(X_train),y_train)


# In[16]:


#Get the selected feature index.
backwardModel.k_feature_idx_


# In[17]:


#Get the column name for the selected feature.
X_train.columns[list(backwardModel.k_feature_idx_)]


# # Exhaustive Feature Selection
#  In this the best subset of feature is selected, over all possible feature subsets. For example, if a dataset contains 4 features, the algorithm will evaluate all the feature combinations(15) as follows:
#  
#  Conbination Formula= n!/(r!(n-r)!) where r objects taken from set of n object. 
#  
# * All possible combinations of 1  feature.  Out of 4 features 1 feature can be selected in 4 different ways.
# * All possible combinations of 2 features. Out of 4 features 2 features can be selected in 6 different ways.
# * All possible combinations of 3 features. Out of 4 features 3 features can be selected in  4 different ways.
# * All possible combinations of 4 features. Out of 4 features 4 features can be selected in  1 way.<br>
# Total features= 4+6+4+1=15

# **I tried to use same training data for Exhaustive Feature Selection but it is getting stucked since it has 33 features. So we will try to find out top 5 features out of 10 features which we got  from backward feature selection.**

# In[ ]:


from mlxtend.feature_selection import ExhaustiveFeatureSelector as efs
#min_features=1 (minimum number of feature)
#max_features=5 (maximum number of feature)
#n_jobs=-1 (Number of cores it will use for execution.-1 means it will use all the cores of CPU for execution.)
#scoring='r2'(R-squared is a statistical measure of how close the data are to the fitted regression line)
emodel=efs(RandomForestRegressor(),min_features=1,max_features=5,scoring='r2',n_jobs=-1)
#Lets take only 10 features which we got from backward feature selection.
miniData=X_train[X_train.columns[list(backwardModel.k_feature_idx_)]]

emodel.fit(np.array(miniData),y_train)
#If you see below the model creates 637 feature combinations from 10 features.Thats why its computationally very expensive.


# In[ ]:


#Get the selected feature index.
emodel.best_idx_


# In[ ]:


#Get the column name for the selected feature.
miniData.columns[list(emodel.best_idx_)]


# **Please checkout [Feature Selection Main Page](https://www.kaggle.com/raviprakash438/feature-selection-technique-in-machine-learning)**

# ***Please share your comments,likes or dislikes so that I can improve the post.***

# In[ ]:




