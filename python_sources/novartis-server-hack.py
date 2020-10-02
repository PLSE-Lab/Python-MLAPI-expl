#!/usr/bin/env python
# coding: utf-8

# # Novartis Data science challenge :
# 
# To predict whether the server is hacked or not.

# # Approach That I took:
# 
# 1. Initially I tried using Binary Classification using keras but in that case The score was almost 55
# 2. After that I tried out Decision Tree Classifier, Gradient Boosting Classification & Adaboost Classification. From these cases I saw that GBC is working best so I tried that out and got a score of 99.48
# 3. After some fine tuning in GBC I got the score of 99.52 and after that whatever I do, It was not improving
# 4. I tried out AutoMl after that. It took almost 8 Hours to train and got a score of 99.58
# 5. After that I came to know about Catboost classification so tried that out and boom!! I got a score of 99.99! But in this case even after finetuning the score was not improving
# 6. As the score was not improving I went to the EDA section again and figured out that 'X_7', 'X_9', 'X_12' & 'X_14' are reducing the accuracy. So after removing those and Training on CBC I got the score of 100.00

# In[ ]:


#import basic libraries
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns


# # Basic EDA

# In[ ]:


#loading the train data
train = pd.read_csv("../input/novartis-data/Train.csv")


# In[ ]:


train.info()


# In[ ]:


#loading the test data
test = pd.read_csv("../input/novartis-data/Test.csv")


# In[ ]:


test.info()


# In[ ]:


#Dropping the Incident_ID and Date from train data
train = train.drop(['INCIDENT_ID','DATE'], axis=1)


# In[ ]:


#Verifying all the columns that has the null values in train data
null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()


# In[ ]:


#Filled NaN values with "0" using fillna()
train["X_12"].fillna(0,inplace = True)


# In[ ]:


#Verifying all the columns that has the null values in test data
null_columns=test.columns[test.isnull().any()]
test[null_columns].isnull().sum()


# In[ ]:


#Filled NaN values with "0" using fillna()
test["X_12"].fillna(0,inplace = True)
test.isnull().sum()


# We can view in the train and test info that X_12 is float64 let's convert it into int64

# In[ ]:


train["X_12"] = train["X_12"].astype(np.int64)
train.info()


# In[ ]:


test["X_12"] = test["X_12"].astype(np.int64)
test.info()


# In[ ]:


#Removing the duplicated rows from train data
print("Shape of train before removing the duplicates :" , train.shape)
train.drop_duplicates(keep='first', inplace=True)
print("Shape of train After removing the duplicates :" , train.shape)


# In[ ]:


#Skewness of the train data
train.skew()


# # Data Analysis on Train data :

# ### Checking Skewness & Kurtosis:

# In[ ]:


f,ax = plt.subplots(1,1,figsize=(15,5))
sns.violinplot(train['MULTIPLE_OFFENSE'])
plt.show()
#skewness and kurtosis
print("Skewness: {}".format(train['MULTIPLE_OFFENSE'].skew()))
print("Kurtosis: {}".format(train['MULTIPLE_OFFENSE'].kurt()))


# ### Histogram Plot:

# In[ ]:


train.hist(figsize=(15,15))
plt.show()


# ### Pair Plot:

# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(train)
plt.show()


# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(train,kind="reg")
plt.show()


# ### Heatmap

# In[ ]:


sns.heatmap(train.corr(),annot=True, cmap='BuPu', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(14,14)
plt.show()


# In[ ]:


print("Number of training Mutiple Offence : {} ".format(len(train)))
print("Offense Rate {:.4}%".format(train["MULTIPLE_OFFENSE"].mean()*100))


# ### Let's visualize the Multiple Offense rate using pie chart

# In[ ]:


#Creating Pie Chart for the target variable
labels = ['Hacked', 'Genuine']
plt.title('Multiple Offense')
train['MULTIPLE_OFFENSE'].value_counts().plot.pie(explode=[0,0.3],autopct='%1.2f%%',shadow=True,labels=labels,fontsize=15)


# # Data Analysis on Test

# ### Histogram Plot:

# In[ ]:


test.hist(figsize=(15,15))
plt.show()


# ### Pair Plot:

# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(test)
plt.show()


# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(test,kind="reg")
plt.show()


# # DATA PROCESSING FOR PREDICTION :

# From the above plots we can see that 'X_7', 'X_9', 'X_12' & 'X_14' will reduce the acuracy. So let's drop those columns

# In[ ]:


train=train.drop(['X_7','X_9','X_12','X_14'], axis=1)
test=test.drop(['X_7','X_9','X_12','X_14'], axis=1)


# In[ ]:


X_train = train.iloc[:,:-1]
y_train = train["MULTIPLE_OFFENSE"]
#Dropping the Incident_ID and Date from test data
X_test = test.drop(['INCIDENT_ID','DATE'], axis=1)


# In[ ]:


print("Shape of X_train : ",X_train.shape)
print("Shape of y_train : ",y_train.shape)
print("Shape of X_test : ",X_test.shape)


# # SMOTE:
# 

# SMOTE stands for Synthetic Minority Oversampling Technique. This is a statistical technique for increasing the number of cases in your dataset in a balanced way. The module works by generating new instances from existing minority cases that you supply as input. This implementation of SMOTE does not change the number of majority cases.                              
# 
# To read further about SMOTE: [Link](https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/smote) 
# 

# In[ ]:


print('Before OverSampling, the shape of X_train: {}'.format(X_train.shape))
print('Before OverSampling, the shape of y_train: {} \n'.format(y_train.shape))
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))


# In[ ]:


from imblearn.over_sampling import SMOTE
sampler = SMOTE(sampling_strategy='minority')
X_train_sm, y_train_sm = sampler.fit_sample(X_train, y_train)
print('After OverSampling, the shape of X_train: {}'.format(X_train_sm.shape))
print('After OverSampling, the shape of y_train: {} \n'.format(y_train_sm.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_sm==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_sm==0)))


# In the above output we can cleary see that the target variable is well balanced.

# In[ ]:


#Spltting the data into train and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_sm, y_train_sm ,test_size=0.3, random_state=100)


# In[ ]:


#Scaling the data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss  = ss.transform(X_test)
X_val_ss   = ss.transform(X_val)


# # DATA MODELLING FOR PREDICTION :

# ## DecisionTree

# In[ ]:


#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

decisiontree = DecisionTreeClassifier(random_state=0)
decisiontree.fit(X_train_ss, y_train)

y_pred = decisiontree.predict(X_val_ss)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
recall_decisiontree = recall_score(y_pred, y_val)

print("Decision Tree Accuracy Score:",acc_decisiontree)
print('Decision Tree Recall Score:',recall_decisiontree)


# In[ ]:


from sklearn.model_selection import cross_val_score

cross_val_score(decisiontree, X_train_ss, y_train, cv=10)


# ## GradientBoosting

# In[ ]:


#Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

gbc = GradientBoostingClassifier(n_estimators=1000,learning_rate=1.0, random_state=100)
gbc.fit(X_train_ss, y_train)

y_pred = gbc.predict(X_val_ss)

acc_gbc = round(accuracy_score(y_pred, y_val) * 100, 2)
recall_gbc = recall_score(y_pred, y_val)

print("Gradient Boosting Classifier Accuracy Score:",acc_gbc)
print("Gradient Boosting Classifier Recall Score:",recall_gbc)


# In[ ]:


from sklearn.model_selection import cross_val_score

CV = cross_val_score(gbc, X_train_ss, y_train, cv=5)
print(CV.mean())


# ## AdaBoost

# In[ ]:


#AdaBoost Classifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

abc = AdaBoostClassifier(n_estimators=1000, random_state=1000,learning_rate=0.3)
abc.fit(X_train_ss, y_train)

y_pred = abc.predict(X_val_ss)

acc_abc = round(accuracy_score(y_pred, y_val) * 100, 2)
recall_abc = recall_score(y_pred, y_val)

print("AdaBoost Classifier Accuracy Score:",acc_abc)
print("AdaBoost Classifier Recall Score:",recall_abc)


# ## CatBoost

# In[ ]:


cat_features = list(range(0, X_train_sm.shape[1]))
print(cat_features)


# In[ ]:


# Catboost Classifier

#Just to check if the model is working on the data properly or not

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train_sm, y_train_sm, test_size=0.05, random_state=42)

clf = CatBoostClassifier(
    iterations=5, 
    learning_rate=0.1, 
    #loss_function='CrossEntropy'
)

clf.fit(X_train, y_train, 
        cat_features=cat_features, 
        eval_set=(X_val, y_val), 
        verbose=False
)

print('CatBoost model is fitted: ' + str(clf.is_fitted()))
print('CatBoost model parameters:')
print(clf.get_params())


# In[ ]:


#Model Training

clf = CatBoostClassifier(
    iterations=1000,
    random_seed=42,
    depth=10,
    learning_rate=0.1,
    model_size_reg=0
)

clf.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
)


# In[ ]:


print(clf.predict_proba(data=X_val))


# In[ ]:


print(clf.predict(data=X_val))


# ## Automl

# To run the automl for me it took almost 8 hours! So I'm commenting this part for now. To get the result in Automl you can remove the commenting and run!

# In[ ]:


#!pip install automl


# In[ ]:


#import h2o
#from h2o.automl import H2OAutoML


# In[ ]:


"""
h2o.init()
X_y_train_h = h2o.H2OFrame(pd.concat([X_train_sm, y_train_sm], axis='columns'))
X_y_train_h['MULTIPLE_OFFENSE'] = X_y_train_h['MULTIPLE_OFFENSE'].asfactor()
# ^ the target column should have categorical type for classification tasks
#   (numerical type for regression tasks)

X_test_h = h2o.H2OFrame(X_test)

X_y_train_h.describe()
"""


# In[ ]:


"""
aml = H2OAutoML(
    max_runtime_secs=(3600 * 8),  # 8 hours
    max_models=None,  # no limit
    seed=44
)
"""


# In[ ]:


#feature_cols = ['X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10','X_11','X_12','X_14','X_14','X_15']


# In[ ]:


"""
%%time

aml.train(
    x=feature_cols,
    y='MULTIPLE_OFFENSE',
    training_frame=X_y_train_h
)

lb = aml.leaderboard
model_ids = list(lb['model_id'].as_data_frame().iloc[:,0])
out_path = "."

for m_id in model_ids:
    mdl = h2o.get_model(m_id)
    h2o.save_model(model=mdl, path=out_path, force=True)

h2o.export_file(lb, os.path.join(out_path, 'aml_leaderboard.h2o'), force=True)
"""


# In[ ]:


#lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
#lb


# In[ ]:


#y_pred = aml.leader.predict(X_test_h)


# # Submission

# In[ ]:


y_pred=clf.predict(data=X_test)

submission_df = pd.DataFrame({'INCIDENT_ID':test['INCIDENT_ID'], 'MULTIPLE_OFFENSE':y_pred})
submission_df.to_csv('Submission CBC.csv', index=False)


# # Hope this notebook was helpful
