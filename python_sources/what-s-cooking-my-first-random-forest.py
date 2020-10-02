#!/usr/bin/env python
# coding: utf-8

# 

# In this Kernel, I am exploring random forest classifications with this simple case study. I will try to figure out the impact of the model parameters on the model score as well as on the computational time.

# In[ ]:


import pandas
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt
import numpy as np
import random
import gc


# Data visualization & imbalance check
# =============

# The train data, when converted to a dataframe, contains the following information: Each row represents a recipe, characterized by a cuisine type, an id and a list of ingredients.
# 
# Let's visualize how imbalanced the cuisines are in this data set: 

# In[ ]:


train_df=pandas.read_json('../input/train.json')
submission_df=pandas.read_json('../input/test.json')


# In[ ]:


ax = train_df['cuisine'].value_counts().plot(kind='bar', title ="Cuisine types", figsize=(8, 3), legend=True, fontsize=12)
ax.set_ylabel("recipes", fontsize=12)
plt.show()


# It is very obvious that we have over-represented cuisines (Italian and Mexican for example). It is known that it could bring some fake accuracy on classification models because these will try to fit on the most represented cuisines.

# Data pre-processing
# ===================

# The first thing to do is to separate a train set of data from a test set. The original "Test" set provided as an input file is only for submission, so we need to create our own. Let's make the following hypotheses:
# Train/Test ratio: The literature suggest a ratio between 60/40 and 80/20, so let's choose 70/30...
# 

# In[ ]:


# We divide the original train data into 3 sets: 
# test_df will contain 30% recipes for each cuisine type
# train_df will contain 70% recipes for each cuisine type
# cut_df will contain the remaining data (could be used as another test set...) if exist

new_test_df=pandas.DataFrame()
new_train_df=pandas.DataFrame()
cut_df=pandas.DataFrame()
cut_percentage=0.01  
for cuisine in train_df['cuisine'].drop_duplicates().values :
    temp=pandas.DataFrame()
    temp=train_df[train_df['cuisine']==cuisine]
    rows_test = random.sample(list(temp.index), round(0.3*(1-cut_percentage)*len(train_df[train_df['cuisine']==cuisine])))
    new_test_df=new_test_df.append(temp.ix[rows_test])
    rows_train= random.sample(list(temp.drop(rows_test).index), round(0.7*(1-cut_percentage)*len(train_df[train_df['cuisine']==cuisine])))
    new_train_df=new_train_df.append(temp.ix[rows_train])
    rows=rows_test+rows_train
    cut_df=cut_df.append(temp.drop(rows))
    del temp

ax=plt.subplot()
CuisineCall = list(range(0,len(cut_df['cuisine'].value_counts().index)))
LABELS=cut_df['cuisine'].value_counts().index
ax.bar(CuisineCall,cut_df['cuisine'].value_counts(),width=0.5,color='r',align='center',label='cut data')
ax.bar(CuisineCall,new_train_df['cuisine'].value_counts(),width=0.5,color='b',align='center', label='new train data')
ax.bar(CuisineCall,new_test_df['cuisine'].value_counts(),width=0.5,color='g',align='center',label='new test data')
plt.xticks(CuisineCall, LABELS,rotation=85)
ax.autoscale(tight=True)
plt.legend()

plt.show()


# We now need to convert the last column containing the list of ingredients into n columns, n being the number of existing ingredients in this study. The goal is to have for each recipe, a combination of 0 and 1 telling us which ingredients are used.

# In[ ]:


try: 
    del train_df
except:pass;gc.collect()

# Get the ingredients column from the new train data and create the list of all existing ingredients
new_ingredients=new_train_df.ingredients
rawlist=[item for sublist in new_ingredients.ravel() for item in sublist] #convert the ingredients list of lists into a list
ingredients=list(set(rawlist)) #remove duplicates

for ing in ingredients:
    vector=[]
    # loop for train data
    for recipe in new_train_df.ingredients: 
        if ing in recipe:
            vector.append(1)
        else:
            vector.append(0)
    new_train_df[ing]=pandas.Series(vector,index=new_train_df.index) # Adds column containing 0 and 1's for this ingredient
    
    # loop for test data
    vector=[]
    for recipe in new_test_df.ingredients:
        if ing in recipe:
            vector.append(1)
        else:
            vector.append(0)
    new_test_df[ing]=pandas.Series(vector,index=new_test_df.index) # Adds column containing 0 and 1's for this ingredient
   
    # loop for cut data
    vector=[]
    for recipe in cut_df.ingredients:
        if ing in recipe:
            vector.append(1)
        else:
            vector.append(0)
    cut_df[ing]=pandas.Series(vector,index=cut_df.index) # Adds column containing 0 and 1's for this ingredient

    # While we are here, let's build also the submission data
    vector=[]
    for recipe in submission_df.ingredients:
        if ing in recipe:
            vector.append(1)
        else:
            vector.append(0)
    submission_df[ing]=pandas.Series(vector,index=submission_df.index) # Adds column containing 0 and 1's for this ingredient

# useless columns removal
new_train_df=new_train_df.drop('ingredients',1)
new_train_df=new_train_df.drop('id',1)

new_test_df=new_test_df.drop('ingredients',1)
new_test_df=new_test_df.drop('id',1)

cut_df=cut_df.drop('ingredients',1)
cut_df=cut_df.drop('id',1)

submission_df=submission_df.drop('ingredients',1)

new_train_df.head()


# Learning
# ========

# Let's define train and test tables first and clear some variables

# In[ ]:


try: 
    X_train=new_train_df.drop('cuisine',axis=1)
    Y_train=new_train_df['cuisine']
    X_test=new_test_df.drop('cuisine',axis=1)
    Y_test=new_test_df['cuisine']
    X_cut=cut_df.drop('cuisine',axis=1)
    Y_cut=cut_df['cuisine']
    del new_train_df
    del new_test_df
    del new_ingredients
    del rawlist
    del ingredients
    del vector
except:pass;gc.collect()


# train

# In[ ]:


from sklearn import metrics
forest=RFC(n_estimators=10,max_features=10)
forest.fit(X_train,Y_train)
output=forest.predict(X_test)
metrics.accuracy_score(Y_test, output)


# Method optimization
# ===================

# Now let's explore the importance of the number of trees, maximum number of features

# Number of trees
# --------------------

# We are going to plot the fitting score VS the number of estimators for a given number of maximum features (5 only to reduce computation time).

# In[ ]:


opt_table_estimators=list()
n_features=5
n_estimators=50
for i in range(1,n_estimators):
    forest=RFC(n_estimators=i,max_features=n_features)
    forest.fit(X_train,Y_train)
    output=forest.predict(X_test)
    opt_table_estimators.append(metrics.accuracy_score(Y_test, output))
plt.plot(range(1,n_estimators), opt_table_estimators)
plt.xlabel('Number of trees')
plt.ylabel('Random Forest Score')
plt.title('Random Forest Score VS Number of trees (5 features)')
plt.show()


# The plot above shows that in our case the importance of the number of trees is very important until 20. Then, the slope is getting weaker...

# Number of features
# ------------------

# In[ ]:


opt_table_n_features=list()
n_estimators=5
n_features=50
for i in range(1,n_features):
    forest=RFC(n_estimators=i,max_features=n_features)
    forest.fit(X_train,Y_train)
    output=forest.predict(X_test)
    opt_table_n_features.append(metrics.accuracy_score(Y_test, output))
plt.plot(range(1,n_features), opt_table_n_features)
plt.xlabel('Number of features')
plt.ylabel('Random Forest Score')
plt.title('Random Forest Score VS Number of features (5 trees)')
plt.show()


# The importance of the number of features has a little less effect on the prediction than the number of trees. 

# First optimized model
# ---------------------

# At this step, and thanks to the previous optimization study, we are sure to get better result with a large number of trees and with at least 5 features.

# In[ ]:


forest=RFC(n_estimators=40,max_features=10)
forest.fit(X_train,Y_train)
output=forest.predict(X_test)
metrics.accuracy_score(Y_test, output)


# Most important ingredients
# ========================================

# In[ ]:


importance = forest.feature_importances_
importance = pandas.DataFrame(importance, index=X_train.columns, columns=["Importance"])
importance_plot=importance.sort_values('Importance',ascending=False ).loc[importance['Importance']>0.004,:]
x = np.arange(len(importance_plot.index.values))
y = importance_plot.ix[:, 0]
plt.bar(x, y,align='center')
plt.xticks(x,importance_plot.index.values,rotation=85)
plt.ylabel('Importance')
plt.title('Main ingredients importance')
plt.autoscale(tight=True)
plt.show()


# In[ ]:


#FOR SUBMISSION
X_submission=submission_df.drop('id',1)
pred=forest.predict(X_submission)
Output=pandas.DataFrame(submission_df['id'],index=submission_df.index)
Output['cuisine']=pandas.Series(pred,index=submission_df.index)
Output.to_csv('output.csv',index=False)

