#!/usr/bin/env python
# coding: utf-8

# # Is There a Cat in the Dat ? ( Kernel 2 )
# By : Hesham Asem
# 
# ______
# 
# after using LGB Model & gaining only 77% accuracy , let's try using OneHotEncoder method then Logistic Regression 
# 
# let's start by importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


# then read the data

# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat/train.csv')  
test = pd.read_csv('../input/cat-in-the-dat/test.csv')  

print(f'Train data Shape is {train.shape}')
print(f'Test data Shape is {test.shape}')


# 300K sample size for training & 200K for testing , great . 
# 
# now to define needed functions

# In[ ]:


def Drop(feature) :
    global data
    data.drop([feature],axis=1, inplace=True)
    data.head()
    
def UniqueAll(show_value = True) : 
    global data
    for col in data.columns : 
        print(f'Length of unique data for   {col}   is    {len(data[col].unique())} ')
        if show_value == True  : 
            print(f'unique values ae {data[col].unique()}' )
            print('-----------------------------')
            
def Encoder(feature , new_feature, drop = True) : 
    global data
    enc  = LabelEncoder()
    enc.fit(data[feature])
    data[new_feature] = enc.transform(data[feature])
    if drop == True : 
        data.drop([feature],axis=1, inplace=True)
        
def CPlot(feature) : 
    global data
    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))
    
def Mapp(feature , new_feature ,f_dict, drop_feature = True) : 
    global data
    data[new_feature] = data[feature].map(f_dict)
    if drop_feature == True : 
        data.drop([feature],axis=1, inplace=True)
    else :
        data.head()
def Unique(feature) : 
    global data
    print(f'Number of unique vaure are {len(list(data[feature].unique()))} which are : \n {list(data[feature].unique())}')


# ____
# 
# as usual , start with heading data to have a look to it 

# In[ ]:


train.head()


# & here is test data

# In[ ]:


test.head()


# ____
# 
# # Forming the Data
# 
# since this example depend on categorical data , we have to slice features (X) from output (y) from training data , then concatenate X from training data to features from text data . 
# 
# & this step to make same data processing (like label encoder & so ) for all features 
# 
# so first to slice X_train & X_test

# In[ ]:


X_train = train.drop(['id' , 'target'], axis=1, inplace=False)
X_test = test.drop(['id'], axis=1, inplace=False)

X_train.shape , X_test.shape


# now to concatenate them together into X

# In[ ]:


X = pd.concat([X_train , X_test])
del X_train
del X_test
X.shape


# how it looks ? 

# In[ ]:


X.head()


# ______
# 
# # Data Processing
# 
# 
# no we'll call it data , so it be suitable for all functions we define , which depend on global data

# In[ ]:


data = X


# now for plotting some features , to be sure its values are well represented

# In[ ]:


CPlot('bin_0')


# In[ ]:


CPlot('bin_1')


# In[ ]:


CPlot('bin_2')


# In[ ]:


CPlot('bin_3')


# In[ ]:


CPlot('bin_4')


# In[ ]:


CPlot('nom_0')


# In[ ]:


CPlot('nom_1')


# In[ ]:


CPlot('nom_2')


# In[ ]:


CPlot('nom_3')


# In[ ]:


CPlot('nom_4')


# In[ ]:


CPlot('ord_0')


# In[ ]:


CPlot('ord_1')


# In[ ]:


CPlot('ord_2')


# In[ ]:


CPlot('ord_3')


# In[ ]:


CPlot('ord_4')


# In[ ]:


CPlot('ord_5')


# so now we are ready for Build the model & train the data 
# 
# ______
# 
# # Build the Model
# 
# first to prepare the data for training by defining trainging & testing data again 
# 

# In[ ]:


data.head()


# _____
# 
# 
# now to use OneHotEncoder to transform the whole data into data_dummies

# In[ ]:


OHE  = OneHotEncoder()
data_dummies = OHE.fit_transform(data)


# what is the shape ? 

# In[ ]:


data_dummies.shape


# great , now to redefine train_data & test_data

# In[ ]:


train_data = data_dummies[:train.shape[0],:]
test_data=  data_dummies[train.shape[0]:,:]
train_data.shape , test_data.shape


# and now to define X & y 

# In[ ]:


X = train_data
y = train['target']
X.shape , y.shape


# then to split it into training & testing data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# now we can use Logistic Regression Model to traing our data

# In[ ]:


LogisticRegressionModel = LogisticRegression(penalty='l2',solver='lbfgs',C=1.0,random_state=33)
LogisticRegressionModel.fit(X_train, y_train)


# how is the accuracy ? 

# In[ ]:


print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))
print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))


# although ccuracy here is about 76% , but it make a better score at the real test data which is about 80%

# 
# _____
# 
# now to predict test data , but first we have to apply same scaler model to test data

# ok , now predicting testing data , using predic_proba method , to calculate the probability of having a cat 

# In[ ]:


y_pred = LogisticRegressionModel.predict_proba(test_data)
y_pred.shape


# how it looks like ? 

# In[ ]:


y_pred[:,1]


# great , now to open sample_submission , to read id columns from it

# In[ ]:


data = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')  

print(f'Test data Shape is {data.shape}')
data.head()


# at last we concatenate id column with the result

# In[ ]:


idd = data['id']
FinalResults = pd.DataFrame(y_pred[:,1],columns= ['target'])
FinalResults.insert(0,'id',idd)
FinalResults.head()


# & export the result file

# In[ ]:


FinalResults.to_csv("sample_submission.csv",index=False)


# _____
# 
# hope you find it helpful !

# In[ ]:




