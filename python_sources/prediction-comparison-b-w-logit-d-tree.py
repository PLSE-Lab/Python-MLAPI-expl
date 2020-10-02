#!/usr/bin/env python
# coding: utf-8

# # LOGISTIC REGRESSION
# Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of regression). https://en.wikipedia.org/wiki/Logistic_regression#:~:text=Logistic%20regression%20is%20a%20statistical,a%20form%20of%20binary%20regression).
# 
# A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. A common example of a sigmoid function is the logistic function. A sigmoid function is a bounded, differentiable, real function that is defined for all real input values and has a non-negative derivative at each point and exactly one inflection point. A sigmoid "function" and a sigmoid "curve" refer to the same object.
# https://en.wikipedia.org/wiki/Sigmoid_function#:~:text=A%20sigmoid%20function%20is%20a,given%20in%20the%20Examples%20section.
# 
# # DECISION TREE
# Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
# https://en.wikipedia.org/wiki/Decision_tree_learning
# 
# Use max_depth=3 as an initial tree depth to get a feel for how the tree is fitting to your data, and then increase the depth.
# 
# Remember that the number of samples required to populate the tree doubles for each additional level the tree grows to. Use max_depth to control the size of the tree to prevent overfitting.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import pprint
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv(r'../input/adult-income-dataset/adult.csv')
pprint.pprint(df.head())
print('\n\n')
pprint.pprint(df.shape)


# **pd.isnull() returns 0 which shows that there are no Nan values in data**

# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


cat_df=df.select_dtypes('object')
cat_df.head()


# In[ ]:


print('unique elements of education column: \n')
pprint.pprint(df.education.unique())
print('\n\n')
print('unique elements of marital-status column: \n')
pprint.pprint(df['marital-status'].unique())
print('\n\n')
print('unique elements of occupation: \n')
pprint.pprint(df['occupation'].unique())
print('\n\n')
print('unique elements of workclass: \n')
pprint.pprint(df['workclass'].unique())
print('\n\n')
print('unique elements of relationship: \n')
pprint.pprint(df['relationship'].unique())
print('\n\n')
print('unique elements of race: \n')
pprint.pprint(df['race'].unique())
print('\n\n')
print('unique elements of gender: \n')
pprint.pprint(df['gender'].unique())
print('\n\n')
print('unique elements of native-country: \n')
pprint.pprint(df['native-country'].unique())
print('\n\n')
print('unique elements of income: \n')
pprint.pprint(df['income'].unique())
print('\n\n')


# In[ ]:


arr1=[]
for item in cat_df['workclass']:
    if (item == '?'):
        arr1.append(item)
print('Length of missing vals in workclass column:')
print(len(arr1))
print('\n')
arr2=[]
for item in cat_df['occupation']:
    if (item == '?'):
        arr2.append(item)
print('Length of missing vals in occupation column:')
print(len(arr2))


# **12.971% of data is missing from the dataset.**

# In[ ]:


null_data=((2809+2799)/(48842-(2809+2799)))*100
print(null_data)


# In[ ]:


x=df.select_dtypes(object)


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder


# In[ ]:


oe=OrdinalEncoder()
cat_df=oe.fit_transform(cat_df)


# In[ ]:


cat_df


# In[ ]:


cat_df1=pd.DataFrame(data=cat_df,columns=x.columns)
cat_df1


# In[ ]:


num_df1=df.select_dtypes(int)
num_df1


# In[ ]:


final_df=pd.concat([num_df1,cat_df1],axis=1)
final_df


# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


X=final_df.drop('income',axis=1)
y=final_df['income']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=50)


# # **USING LOGISTIC REGRESSION**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression(solver='lbfgs',max_iter=200)


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


prediction=logmodel.predict(X_test)
prediction


# In[ ]:


pred=pd.DataFrame(data=prediction,columns=['prediction'])
pred


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


result=classification_report(pred,y_test)
print(result)


# #  **USING DECISION TREE**

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


tree=DecisionTreeRegressor(max_depth=7)


# In[ ]:


tree.fit(X_train,y_train)


# In[ ]:


predictions=tree.predict(X_test)
print(predictions)


# In[ ]:


pred2=pd.DataFrame(data=predictions,columns=['predictions'])
pred2['predictions']


# In[ ]:


def num(n):
    if(n < 0.5):
        return 0
    else:
        return 1


# In[ ]:


x=pred2['predictions'].apply(num)
x.unique()


# In[ ]:


result2=classification_report(x,y_test)
print(result2)


# A decision tree with max_depth of 7 gives better prediction accuracy than logistic regression with solver 'lbfgs' 

# # **THE END**
