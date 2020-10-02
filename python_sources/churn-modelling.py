#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# Ayush Modi
# #general_discussions
# https://bertelsmannai.slack.com/archives/CQEPP77BP/p1579177141420800
# 
# Hello everyone!
# So today, instead of working further through the lessons, I decided to take a break and build a model on this Churn Modeling dataset.
# So you're given a dataset from a bank of 10,000 customers, with details about them like Credit Score, Age, Salaries, Balance, No. of products(like whether the customer has a credit card, whether they have a loan, etc), Geography, etc
# What's the motive? Well, the bank's been seeing unusual Churn rates(Churn's when people leave the company), so you wanna look into the data collected and give them some insights. The 'exited' column indicates whether a person left the bank or not, 1 indicating the person did leave the bank, 0 indicating that they didn't. So your task would be given some information about a customer, predict whether or not they end up leaving the bank.
# The reason I'm sharing this here is that I was able to do the problem with just the knowledge I've gained from lesson 3 and lesson 5 (I didn't know anything about PyTorch prior to the course). Therefore, if I can, then so can you.
# I'd like to encourage all of you to give the problem a try. If you've completed everything in the course till lesson 5, you should be able to do this! To break it down, it's a simple binary classification task(but unlike the notebooks, this one isn't on images. That shouldn't be a problem though, the same basic idea still works :))
# (As heads up, the notebooks I had to go back and refer to, primarily were the ones in lesson 3 - 3.35 for data manipulation using pandas, and a couple of notebooks from lesson 5 for defining my model using nn.Sequential(), and that's pretty much it! Easy innit?)
# So if you'd like to give it a try, well go ahead! Perhaps we could compare our accuracies to get the game goin! Drop in yours by replying to the thread!
# I'm attaching the dataset, which you'll need to download and then work on it. Good luck! Tag everyone who you think might be interested <3
# PS. The best accuracy I could achieve was 87.7% .. Time for you to beat that ;) (edited) 

# In[ ]:


import pandas as pd
df = pd.read_csv("../input/churn-predictions-personal/Churn_Predictions.csv", index_col=0)
df.head()


# In[ ]:


print(df.shape)
df.tail()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def Normalize(df):
    cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    return (df[cols] - df[cols].mean()) / df[cols].std()

def Categorize(df):
    cols = ['Geography', 'Gender']
    return df[cols].astype('category')
    
def OneHot(df):
#     cols = ['NumOfProducts']
    cols = ['Geography', 'Gender', 'NumOfProducts']
    return pd.get_dummies(df[cols], drop_first=True)


# In[ ]:


FunctionTransformer(OneHot, validate=False).fit_transform(df).head()


# In[ ]:


preprocess = FeatureUnion([
    ('normalize', FunctionTransformer(Normalize, validate=False)),
#     ('categorize', FunctionTransformer(Categorize, validate=False)),
    ('onehot', FunctionTransformer(OneHot, validate=False)),
])

pipe = Pipeline([
    ('union', preprocess),
    ('clf', LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(df, df[['Exited']], test_size=0.3)
pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)
print('acc: {:.2f}'.format(accuracy))


# In[ ]:




