#!/usr/bin/env python
# coding: utf-8

# **Background**: When I started working on this kernel, I wanted to do some EDA and explore a few classification methods to see how I could achieve a decent score. Now that there are so many other excellent kernels out there with very high scores, I want to devote this kernel to shed some light one why some approaches worked better the others. I hope this will be useful for others who are confused by all these new records being set every other day in this competition. 

# ### Overview: 
# We will start with linear classifiers like logistic regression and see why or why not they are better than blind guessing for this data and then go with more complicated classifiers. Emphasis would be in understanding the results of each iteration rather than getting the best performance. 

# | Model  | Validation Set Accuracy   | Comments |  
# |---|---|---|---|
# | Blind Guessing  | 50%  |Based-line Model |
# |  Logistic Regress with all columns as numeric | 52%    |   |  
# |  Logistic Regression with one column as categorical | 51%  |   No Interactions between columns |  
# |  Logistic Regression with one column as categorical, with interactions | 80%  |  With Interactions between columns |  

# In[1]:


import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings('ignore')


# Let us first read the input data, and display the first few rows of the dataframes with titles. 

# In[2]:


# Read the test and train data sets. 
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[3]:


print("Shapes of training and test datasets:");print(df_train.shape,df_test.shape)
print("Training Data Sample");display(df_train.head())
print("Test Data Sample");display(df_test.head())


# **It looks like the data is mostly numerical, with random column names, and we need to predict the "target" column**.

# **Let us also make sure the training and test column names are same, other than the target column**

# In[4]:


# The following will be true if the column names in the training and test data sets are identical
set(df_train.columns[df_train.columns != 'target']) == set(df_test.columns)


# **Next step is too look at distributions of various columns in the test and training data sets. You have to scroll from left right to see all the columns.**

# In[5]:


print("Training Data Summary");df_train.describe()


# In[6]:


print("Test Data Summary");df_test.describe()


# **It looks like all the training and test, columns are very similar in distribution with mean value of about 0, and standard deviation of about 1.7 **
# **Let us confirm that by looking at distbrutions of 16 randomly sampled columns**

# In[7]:


fig, ax = plt.subplots()
random_cols = np.random.choice(range(1,df_train.shape[1]-1),16)
for col in df_train.columns[random_cols]:
    sns.kdeplot(df_train[col], ax = ax)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Distribution of features in the Training Set (Sample of 16 features)')


# **Great, we can see tha the distributions of the variables are very close to each other, centered around zero. There shouldn't be much need for normalizing the variables. ** 
# 
# 
# **Let us also look at the test set to make sure.**

# In[8]:


fig, ax = plt.subplots()
random_cols = np.random.choice(range(1,df_test.shape[1]),16)
for col in df_test.columns[random_cols]:
    sns.kdeplot(df_train[col], ax = ax)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Distribution of features in the Test Set (Sample of 16 features)')


# **It was identified by others that wheezy-copper-turtle-magic appears to be a categorical variable, with a distribution that is diferent from rest of the variables**. How do we go about finding such a thing ? 
# 
# Since we expected all the columns to have similar median / standard deviation values, let us plot all the medians vs. standard deviations to see if we can find an outliers. 
# 

# In[9]:


sns.set()
ax = sns.scatterplot(
    x = df_train.loc[:,(df_train.columns != 'target') & (df_train.columns != 'id') ].median(axis = 0),
    y = df_train.loc[:,(df_train.columns != 'target') & (df_train.columns != 'id') ].std(axis = 0), alpha = 0.8)
ax.set(xlabel='Median Values', ylabel='standard deviations')
plt.title('Medians vs. Standard Deviations of all the columns in the training data')


# We can clearly see that there are one or more points with Median value close to 250 and standard deviation close to 140. Let us filter these columns to figure out what they are. 

# In[10]:


median_vals = df_train.loc[:,(df_train.columns != 'target') & (df_train.columns != 'id') ].median(axis = 0)
median_vals[median_vals > 100]


# We can see from the above that only "wheezy-copper-turtle-magic" is the outlier column. Let us take a look at its distribution. We can see that its distribution is signficantly different from other columns

# In[11]:


col = 'wheezy-copper-turtle-magic'
sns.distplot(df_train[col])


# # Comparing Various Models:
# 
# ## Model 0: Blind guessing
# 

# Let us first divide the data and training and validation data sets

# In[12]:


X_train, X_val, y_train, y_val = train_test_split(df_train.drop(columns = ['id','target']),df_train['target'], test_size=0.15, random_state=2)


# Let us first start with the blind guess model where we select each row as either 1 or 0 randomly. Based on the disribution of the target variable as shown below, we expect about 50% accuracy for the blind guess model. 

# In[13]:


sns.countplot(df_train['target'])
plt.title('Distribution of target variable in the training data')


# In[14]:


# Randomly choose either True or False for each row. 
y_blind_guess = np.random.randint(2, size = len(y_val))
print("Accuracy with blind guessing: %.2f" % accuracy_score(y_val, y_blind_guess, normalize=True))


# **As expected, blind guessing gave us a 50% accuracy based on the distribution of the data. Let us see how quickly we can improve the score beyond the 50%.**

# ## Model 1: Logistic Regression with all features considered as numerical columns
# Let us first start with a linear model for logistic regression using all the features

# In[15]:


logreg = LogisticRegression().fit(X_train,y_train)
#y_pred = logreg.predict(X_val)
print("Accuracy of logisic Regression on Validation Set: %.2f" % logreg.score(X_val, y_val))


# **So, our model got a 52% accuracy. That is very slightly better than blind guessing, which should give us 50% accuracy as seen in the previous guess. There is lot of room for improvement !**
# 
# ### Question 1: Why did our logistic regression give only 2% improvement on the baseline blind guess model ? 
# 
# Other than saying that probably the data (without any feature engineering) didn't have a clear linear correlation with the target variable, we cannot conclude much at this point. But for a better understanding, let us take a look at how target variable is encoded with respect to a few randomly selected columns. 

# In[18]:


fig, axs = plt.subplots(ncols = 2 ,nrows=2, figsize=(8,8))
random_cols = df_train.columns[np.random.choice(range(1,df_train.shape[1]-1),8)]
sns.scatterplot(x = random_cols[0], y = random_cols[1], hue = 'target', data = df_train, ax=axs[0,0])
sns.scatterplot(x = random_cols[2], y = random_cols[3], hue = 'target', data = df_train, ax=axs[0,1])
sns.scatterplot(x = random_cols[4], y = random_cols[5], hue = 'target', data = df_train, ax=axs[1,0])
sns.scatterplot(x = random_cols[6], y = random_cols[7], hue = 'target', data = df_train, ax=axs[1,1])


# From the above we can imagine that it might be hard to classify the target variable based on the linear combinations of the columns. 

# ## Model 1: Logistic regression with wheezy-copper-turtle-magic as a categorical variable
# 
# As mentioned earlier, it was identified by others that wheezy-copper-turtle-magic appears to be a categorical variable, with a distribution that is diferent from rest of the variables, so converting it to a categorical, column and using one-hot encoding might help improve accuracy. 
# 
# Before we convert this column to a categorical column, let us first make sure all the values of this column in the test data are a subset of this column's values in the training data

# In[19]:


set(df_test['wheezy-copper-turtle-magic']) - set(df_train['wheezy-copper-turtle-magic'])


# Above operation shows that the column "wheezy-copper-turtle-magic", doesn't take any additional values in the test data. 

# In[20]:


df_train_2 = pd.get_dummies(df_train, columns = ['wheezy-copper-turtle-magic'], prefix='wctm-')
df_test_2 = pd.get_dummies(df_test, columns = ['wheezy-copper-turtle-magic'], prefix='wctm-')
X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(df_train_2.drop(columns = ['id','target']),df_train_2['target'], test_size=0.15, random_state=2)


# In[21]:


logreg_2 = LogisticRegression().fit(X_train_2,y_train_2)
#y_pred = logreg.predict(X_val)
print("Accuracy of logisic Regression on Validation Set: %.2f" % logreg_2.score(X_val_2, y_val_2))


# ### Question2 : Why did changing the "wheezy-copper-turtle-magic" to categorical didn't improve the accuracy as we were expecting ?
# 
# **Answer**: As mentioned in this discussion post (https://www.kaggle.com/c/instant-gratification/discussion/92930#latest-537424), it looks like the data consists of several subsets each of which is indicated by a value of the "wheezy-copper-turtle-magic" column. Sometimes changing such a column to category will be enough, but some times, we have to develop a separate model for each of the values in the category. 
# 
# For example if you are developing a pricing model for a store that sells groceries and antique items, one can imagine that antiques will have a completely different price vs. age model, compared to perishable items. So, just having the item category as a categorical variable is not enough, we need to have a separate model for each category. 
# 
# 

# So, as highlighted in Chris's excellent Kernel here: https://www.kaggle.com/cdeotte/logistic-regression-0-800, let us build a logistic regression model for each category to see if that improves the score significantly. 

# In[22]:


# Code from: https://www.kaggle.com/cdeotte/logistic-regression-0-800
    
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target']]
oof = np.zeros(len(train))
skf = StratifiedKFold(n_splits=5, random_state=42)

# INITIALIZE VARIABLES
cols.remove('wheezy-copper-turtle-magic')
interactions = np.zeros((512,255))
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for i in range(512):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    test2.reset_index(drop=True,inplace=True)
    
    skf = StratifiedKFold(n_splits=25, random_state=42)
    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):
        # LOGISTIC REGRESSION MODEL
        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)
        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]
        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / 25.0
        # RECORD INTERACTIONS
        for j in range(255):
            if clf.coef_[0][j]>0: interactions[i,j] = 1
            elif clf.coef_[0][j]<0: interactions[i,j] = -1
    #if i%25==0: print(i)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('LR with interactions scores CV =',round(auc,5))


# We can see that the accuracy greatly improved once we identified that we needed to have separate models for each category, and that just having the "magic" column as a categorical column was not enough. 

# ### Model 3: Neural Network (in-progress)

# Let us try a neural network with Tensorflow, to see if we can achieve better accuracy. 
# 
# 
# **Note:** Some of this model was inspired by: https://www.kaggle.com/dimitreoliveira/instant-gratification-deep-learning

# In[ ]:


nn_model = keras.Sequential([
    keras.layers.Dense(1024, input_dim = X_train_2.shape[1]),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer= 'adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


nn_model.summary()


# In[ ]:


nn_history = nn_model.fit(X_train_2.values, y_train_2.values,
                                  epochs=5,
                                  batch_size=128,
                                  validation_data=(X_val_2.values, y_val_2.values),
                                  verbose=2)


# **It looks like our neural network model is giving better accuracy than the simple logistic regression/random forest models**

# ## Let us submit the best model so far

# In[23]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)

