#!/usr/bin/env python
# coding: utf-8

# <font size="20">Discount Prediction</font>

# The objective of this "Discount Prediction" Competition was to build a machine learning model to Predict Medical Wholesales Discount to their customers. In this notebook, we will walk through a complete machine learning solution, try out multiple machine learning models, select a model, work to optimize the model, and finally, inspect the outputs of the model and draw conclusions. We would like to thank everyone for this hackathon.<br><br>
# This notebook is majorly divided into three two parts.They are:
# <ol>
# <li>Exploratory Data Analysis and Preprocessing</li>
# <li>Modeling</li><ul>
# <li><b>LightGBM<b></li>
# <li>XGBOOST</li>
# <li>Random Forest</li>

# <h1>Importing the Libraries</h1>

# ![](http://)We are gonna start of by importing the generic packages that everyone uses in their kernels.

# In[ ]:


import pandas as pd #Data Analysis
import numpy as np #Linear Algebra
import seaborn as sns #Data Visualization
import matplotlib.pyplot as plt #Data Visualization\
get_ipython().run_line_magic('matplotlib', 'inline')


# These are import statements for plotply which is also a visualization library.

# In[ ]:


import json
import string
from pandas.io.json import json_normalize
color = sns.color_palette()
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[ ]:


import os
print(os.listdir("../input"))


# <h1>Importing the datasets</h1>

# In[ ]:


#This is the Product_sales_train_and_test dataset but without the "[]" in the Customer Basket.
df=pd.read_csv("../input/removed//data.csv")


# In[ ]:


train=pd.read_csv("../input/discount-prediction/Train.csv")


# In[ ]:


test=pd.read_csv("../input/discount-prediction/test.csv")


# In[ ]:


product=pd.read_csv("../input/discount-prediction/product_details.csv",encoding="mac-roman")


# In[ ]:


#Removing the front and trailing spaces
df['Customer_Basket']=df['Customer_Basket'].str.lstrip()
df['Customer_Basket']=df['Customer_Basket'].str.rstrip()


# In[ ]:


#The count of the number of Product Id's in the Customer Basket
df['Length']=df['Customer_Basket'].str.split(' ').apply(len)


# <h1>Exploratory Data Analysis with Preprocessing</h1>

# In[ ]:


df.head()


# In[ ]:


train.head()


# In[ ]:


#We can see a lot of null values in the train dataset
train.info()


# In[ ]:


#Let us see number of null values there are
train.isnull().sum()


# In[ ]:


test.isnull().sum()


# No Null Values in the test dataset.

# In[ ]:


train[train['BillNo'].isna()].head(10)


# From this we can tell that there are null values for the entire row present in the train dataset. The below code is to drop the entire row only when the entire row are "NaN" or null values.

# In[ ]:


train.dropna(axis=0,how='all',inplace=True)


# In[ ]:


train.isnull().sum()


# Now we can see that the majority of the null values are present in the Target variables. But we can impute these values with 0.

# In[ ]:


train.fillna(float(0.0),inplace=True)


# In[ ]:


train.isnull().sum()


# Finally we have no more null values.<br>

# In[ ]:


train['Customer'].value_counts().head()


# In[ ]:


len(set(test['Customer']).difference(set(train['Customer'])))


# In[ ]:


train['Discount 5%'].value_counts()


# We can see that the classes are highly unbalanced. Let us visualise it for the other classes to understand this better.<br>
# For the "Discount 5%" here the "1" represents the condition when discount is given and "0" is when the discount is not given.

# In[ ]:


sns.set_style("whitegrid")
sns.countplot(x='Discount 5%',data=train)


# It seems that there are very few 5% discounts. Now let us view it for the 12% Discount.

# In[ ]:


sns.set_style("whitegrid")
sns.countplot(x='Discount 12%',data=train)


# From this we can tell that the majority of the Discounts were 12%. Let us look at the 18% Discount.

# In[ ]:


sns.set_style("whitegrid")
sns.countplot(x='Discount 18%',data=train)


# Again we see the same pattern as in the 5% Discount but just a little more. Let us look at the final class i.e, 28% Discount.

# In[ ]:


sns.set_style("whitegrid")
sns.countplot(x='Discount 28%',data=train)


# We can see more discounts here and therefore is the second most occuring class after 12%. Since the class labels are so imbalanced we will be using SMOTE later on.<br><br>
# 
# We initially used the Customer variable but later on in our predictions we realised that it was actually degrading our model therefore we ended up not using it.

# In[ ]:


#lol=df2["Customer"].str.split(", ",n=1,expand=True)
#df2['CustomerName']=lol[0]
#df2["Location"]=lol[1]


# In[ ]:


#lol1=df3["Customer"].str.split(", ",n=1,expand=True)
#df3['CustomerName']=lol1[0]
#df3["Location"]=lol1[1]


# In[ ]:


#set(df3['Location']).difference(set(df2["Location"]))


# In[ ]:


#df3[df3["Location"]=='T.M.M. HOSPITAL, THIRUVALLA.']


# In[ ]:


#sns.countplot(x="discount",hue="Location",data=df3)


# In[ ]:


#df2["discount"].value_counts()


# In[ ]:


#len(set(trailtest['Customer']).difference(set(trailtrain['Customer'])))


# Now we will create a function that will combine all the class labels into one Target variable.

# In[ ]:


discount=[]
for i, row in train.iterrows():
    if row["Discount 5%"]==1.0:
        discount.append(1)
    elif row["Discount 12%"]==1.0:
        discount.append(2)
    elif row["Discount 18%"]==1.0:
        discount.append(3)
    elif row["Discount 28%"]==1.0:
        discount.append(4)
    else:
        discount.append(5)        


# In[ ]:


train["discount"]=discount


# Let us now plot the word count of "Customer" for each Discount Class.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
train1_df = train[train["discount"]==1]
train2_df = train[train["discount"]==2]
train3_df = train[train["discount"]==3]
train4_df = train[train["discount"]==4]
train5_df = train[train["discount"]==5]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train1_df["Customer"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train2_df["Customer"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

freq_dict = defaultdict(int)
for sent in train3_df["Customer"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

freq_dict = defaultdict(int)
for sent in train4_df["Customer"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace3 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

freq_dict = defaultdict(int)
for sent in train5_df["Customer"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace4 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of Discount 5%", 
                                          "Frequent words of Discount 12%",
                                         "Frequent words of Discount 18%",
                                         "Frequent words of Discount 28%","Frequent words of No Discount"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 2)
fig.append_trace(trace4, 3, 1)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')

#plt.figure(figsize=(10,16))
#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")
#plt.title("Frequent words for Insincere Questions", fontsize=16)
#plt.show()


# From these plots we can see that not any customer was given an extra likelihood of discounts.

# Let us drop the Discount columns now.

# In[ ]:


train.drop(['Discount 5%','Discount 12%','Discount 18%','Discount 28%'],axis=1,inplace=True)


# Since to differentiate the Customer Basket is an NLP Problem we will be using CountVectoriser. It converts a collection of text documents to a matrix of token counts. 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer(max_features=500)
y = cv1.fit_transform(df["Customer_Basket"]).toarray()


# In[ ]:


len(cv1.vocabulary_)


# In[ ]:


thirty= list(y)
thirty1=pd.DataFrame(thirty)


# In[ ]:


final=pd.concat([df,thirty1],axis=1)


# In[ ]:


finaltrain=pd.merge(train,final,on="BillNo",how="inner")
finaltest=pd.merge(test,final,on="BillNo",how="inner")


# In[ ]:


finaltrain.head()


# In[ ]:


finaltest.head()


# In[ ]:


#df2=df2[df2["BillNo"]!=float(0.0)]


# In[ ]:


finaltrain.drop(["BillNo","Customer_Basket","Customer","Date"],axis=1,inplace=True)
finaltest.drop(["BillNo","Customer_Basket","Customer","Date"],axis=1,inplace=True)


# In[ ]:


X=finaltrain.drop("discount",axis=1)
y=finaltrain["discount"]


# We will be using SMOTE here to balance the classes. It achieves this by oversampling. 

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


sm = SMOTE(random_state=2)


# In[ ]:


X_train_res, y_train_res = sm.fit_sample(X, y.ravel())


# In[ ]:


X_train=pd.DataFrame(X_train_res)


# In[ ]:


y_train=pd.DataFrame(y_train_res)


# In[ ]:


X_train["smote"]=y_train_res


# In[ ]:


X1=X_train.drop(["smote"],axis=1)
y1=X_train["smote"]


# <font size="18">Modeling</font>

# This is the modeling section of our notebook we will be using various machine learning models to perform our predictions. We have performed the submission file creation only for one of the models but did implement it for the rest of the models.

# <h1>1. LightGBM</h1>

# In[ ]:


import lightgbm as lgb


# In[ ]:


model = lgb.LGBMClassifier( class_weight = 'balanced',
                               objective = 'multiclass', n_jobs = -1, n_estimators = 400)


# In[ ]:


model.fit(X1,y1)


# In[ ]:


pred_lg=model.predict(finaltest)


# In[ ]:


pred_lg


# <h1>2. XGBOOST</h1>

# We commented the xgboost out becuase in the kernel it would show a long output on the kernel. The code definitely works for multiclass classification so you guys are free to run it.

# In[ ]:


import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[ ]:


xgb = XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=300,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=0, silent=0)                  


# In[ ]:


#xgb.fit(X1, y1)


# In[ ]:


#pred_xg=xgb.predict(finaltest)


# <h1>3.Random Forest Classifier</h1>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier(n_estimators=500)


# In[ ]:


rfc.fit(X1,y1)


# In[ ]:


rfcpredict=rfc.predict(finaltest)


# In[ ]:


rfcpredict


# <h1>Cross Validation for Hyperparameter tuning</h1>

# ![](http://)We commented this out as it was taking too much time to commit and we didn't want our submission to be late.

# In[ ]:


#from sklearn.model_selection import StratifiedKFold
#kfold = 5
#skf = StratifiedKFold(n_splits=5)


# In[ ]:


#folds = 3
#param_comb = 5

#skf = KFold(n_splits=folds, shuffle = True, random_state = 1001)

#random_search = RandomizedSearchCV(rfc, param_distributions=params, n_iter=param_comb, scoring=rmsle, n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )


# In[ ]:


#random_search.fit(X, y)


# In[ ]:


#print('\n All results:')
#print(random_search.cv_results_)
#print('\n Best estimator:')
#print(random_search.best_estimator_)
#print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
#print(random_search.best_score_ * 2 - 1)
#print('\n Best hyperparameters:')
#print(random_search.best_params_)


# <h1>Result</h1>

# At the end we were able to discern that LightGBM gave us the best results and that was what we submitted finally.
# Once again we would like to thank everyone for making this hackathon really enjoyable and educational !!
