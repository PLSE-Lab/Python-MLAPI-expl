#!/usr/bin/env python
# coding: utf-8

# # Kernel Objectives
# Exploring the Data and checking out what to really expect from different Classifiers and determining what really matters to each classifier.
# 
# ### Also, Checking out why our Deep Learning Model Failed on the same features on which our random Forest Classifier excelled
# 
# ![](https://www.york.ac.uk/media/study/courses/undergraduate/biology/Genetics-bsc-banner.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/clinvar_conflicting.csv')
#df = df.dropna()
#df.info()


# # Exploratory Data Analysis
#  

# As you can observe in the graph given below, the dataset happens to be heavily biased towards the **non - conflicting** genes and that too with the **CHROM == 2** standing out as the clear bias winner.
# 
# ### What to take out from this graph?
# 
# Since the incidents where the genes are recorded to be **conflicting**, we can assume that our classifiers won't be doing much of a great job and we can assume that most of them would take **CHROM** 2 as their most important feature if we don't feature hash it into something of a lower dimension

# In[ ]:


fig = plt.figure(figsize = (10, 10))
sns.countplot(x= 'CLASS', data = df, hue = 'CHROM', palette='icefire')


# # Cleaning our data
# 
# As you can see from the graph given below, the properties in *yellow* are the null values. Simply by looking at the columns, one can judge that they are not really worth estimating due to massive information lack in our dataset. 
# 
# ### So, what are the measures that we should take?
# 
# We can simply try considering the ones with minimum losses and consider the features with  <1000 unique features

# In[ ]:


fig = plt.figure(figsize = (10, 10))
sns.heatmap(df.isnull(), cmap = 'viridis', cbar = False)


# In[ ]:


toBeConsidered = ['CHROM', 'POS', 'REF', 'ALT', 'AF_ESP', 'AF_EXAC', 'AF_TGP',
       'CLNDISDB', 'CLNDN', 'CLNHGVS', 'CLNVC','MC', 'ORIGIN', 'CLASS',
       'Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Feature_type',
       'Feature', 'BIOTYPE', 'STRAND','CADD_PHRED', 'CADD_RAW']
df2 = df[toBeConsidered]
df2 = df2.dropna()


# In[ ]:


cutdowns = []
for i in df2.columns.values:
    if df2[i].nunique() < 1000:
        cutdowns.append(i)
print("The selected Columns for training are : ", cutdowns)


# In[ ]:


df_final = df2[cutdowns]
#df_final.info()


# In[ ]:


df_final['CHROM'] = df_final['CHROM'].astype(str)


# # Feature Engineering

# We will be applying Feature Hashers on the columns with > 10 unique values and One Hot Encoding schemes otherwise.

# In[ ]:


from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features = 5, input_type = 'string')
hashed1 = fh.fit_transform(df_final['REF'])
hashed1 = hashed1.toarray()
hashedFeatures1 = pd.DataFrame(hashed1)


# In[ ]:


nameList = {}
for i in hashedFeatures1.columns.values:
    nameList[i] = "REF"+str(i+1)


hashedFeatures1.rename(columns = nameList, inplace = True)
print("The Hashed REF table is somethinng like this : \n",hashedFeatures1.head())


# In[ ]:


#df['ALT']
#fh = FeatureHasher(n_features = 5, input_type = 'string')
hashed2 = fh.fit_transform(df_final['ALT'])
hashed2 = hashed2.toarray()
hashedFeatures2 = pd.DataFrame(hashed2)

nameList2 = {}
for i in hashedFeatures2.columns.values:
    nameList2[i] = "ALT"+str(i+1)


hashedFeatures2.rename(columns = nameList2, inplace = True)
print("The Hashed ALT table is somethinng like this : \n",hashedFeatures2.head())


# In[ ]:


binaryFeature1 = pd.get_dummies(df_final['CLNVC'])
print("While the One hot encoded matrix of CLNVC Columns is like this : \n")
binaryFeature1.head()


# In[ ]:


df_final = df_final.drop(columns=['MC'], axis = 1)


# In[ ]:


hashed0 = fh.fit_transform(df_final['CHROM'])
hashed0 = hashed0.toarray()
hashedFeatures0 = pd.DataFrame(hashed0)

nameList0 = {}
for i in hashedFeatures0.columns.values:
    nameList0[i] = "CHROM"+str(i+1)


hashedFeatures0.rename(columns = nameList0, inplace = True)
hashedFeatures0.head()


# In[ ]:


hashed3 = fh.fit_transform(df_final['Allele'])
hashed3 = hashed3.toarray()
hashedFeatures3 = pd.DataFrame(hashed3)

nameList3 = {}
for i in hashedFeatures3.columns.values:
    nameList3[i] = "Allele"+str(i+1)


hashedFeatures3.rename(columns = nameList3, inplace = True)
hashedFeatures3.head()


# In[ ]:


hashed4 = fh.fit_transform(df_final['Consequence'])
hashed4 = hashed4.toarray()
hashedFeatures4 = pd.DataFrame(hashed4)

nameList4 = {}
for i in hashedFeatures4.columns.values:
    nameList4[i] = "Consequence"+str(i+1)


hashedFeatures4.rename(columns = nameList4, inplace = True)
hashedFeatures4.head()


# In[ ]:


df_final['IMPACT'].nunique()


# In[ ]:


binaryFeature3 = pd.get_dummies(df_final['IMPACT'])
binaryFeature3.head()


# In[ ]:


df_final = df_final.drop(columns=['Feature_type'], axis = 1)


# In[ ]:


binaryFeature4 = pd.get_dummies(df_final['BIOTYPE'], drop_first=True)
binaryFeature4.head()


# In[ ]:


binaryFeature5 = pd.get_dummies(df_final['STRAND'], drop_first=True)
binaryFeature5.head()


# # Making the Final Table
# 
# We will be storing the final values in the table *df3* and you can see the new columns after Feature Engineering as follows

# In[ ]:


df3 = pd.concat([binaryFeature1, binaryFeature3, binaryFeature4, binaryFeature5, hashedFeatures1 , hashedFeatures2, hashedFeatures3, hashedFeatures4,hashedFeatures0, df_final['CLASS']], axis=1)
df3 = df3.dropna()
df3.rename(columns={1 : "one", 16 : "sixteen"}, inplace = True)
print(df3.columns.values)
df3.head()


# In[ ]:


y = df3['CLASS']
X = df3.drop(columns=['CLASS'], axis = 1)


# # Importing the Machine Learning Libraries

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# # Logistic Regression
# 
# Sadly, this provides a mere 56% accuracy in terms of total precision. 
# One more thing to notice is that it didn't crack even one case where the **genes were conflicting**

# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print( "Classification Report :\n ", classification_report(y_test, pred_lr))


# # Decision Trees
# 
# This classifier provides a way better performance with a total precision of *65%* . Thankfully, it shows significant  improvement in *Recall* score of **non conflicting genes** and the Precision in **conflicting Genes**

# In[ ]:


dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
print( "Classification Report :\n ", classification_report(y_test, pred_dt))


# # Random Forest
# 
# This classifier doesn't really performs any better, but the thing to take away from this classifier is the improvement in **Recall** and **f1 - score** for **conficting cases**

# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print( "Classification Report :\n ", classification_report(y_test, pred_rf))


# # Gradient Boost Classifier
# 
# Provides the best overall accuracy, but no support for Recall and F1-score. 

# In[ ]:


gra = GradientBoostingClassifier()
gra.fit(X_train, y_train)
pred_gra = gra.predict(X_test)
print( "Classification Report :\n ", classification_report(y_test, pred_gra))


# # Comparing Feature Importance
# 
# We should have a look at how **Random Forest** and **Logistic regression**, the best and the worst performers in this classification task came about ranking the importance of their given features!

# In[ ]:


from collections import OrderedDict


# # Inference for Logistic Regression
# From the graph below, we can easily understand that the most important features taken into consideration by our Logistic Regressor are **Inversion**,** Microsatellite** and **LOW**, while the chromosomes are given very little or no importance

# In[ ]:


feature_imp = {}
for i in zip(X.columns, lr.coef_[0]):
    feature_imp[i[0]] = i[1]
final_imp = OrderedDict(feature_imp)
df_features = pd.DataFrame(final_imp, index = range(1)).T
df_features.rename(columns={0: "Importance_lr"}, inplace = True)

my_colors = ['g', 'b']*5

df_features.plot(kind='bar',figsize = (20,5), color = my_colors)
#list(feature_imp.values())


# # Inference for Random Forest Classifier
# 
# We can easily see that the feature importance is way different for Random forest classifier which focusses primely on **Chromosomes (CHROM)** Hashes

# In[ ]:


feature_imp2 = {}
for i in zip(X.columns, rf.feature_importances_):
    feature_imp2[i[0]] = i[1]

final_imp2 = OrderedDict(feature_imp2)
#print(feature_imp2)
df_features2 = pd.DataFrame(final_imp2, index = range(1)).T
df_features2.rename(columns={0: "Importance_rf"}, inplace = True)
df_features2.plot(kind='bar',figsize = (15, 5), color = my_colors)


# # For a side by side comparison of features, refer to this:

# In[ ]:


df_compare = pd.concat([df_features, df_features2], axis = 1)
df_compare.plot(kind='bar',figsize = (20, 5))


# # Adding Deep Learning in the Sauce

# In[ ]:


from keras.models import Sequential
from keras.layers import (Dense, Flatten, Dropout, BatchNormalization)


# In[ ]:


model = Sequential()
model.add(Dense(128 , input_dim = 38, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid'))


# In[ ]:


model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # The model summary of what we will be using 

# In[ ]:


model.summary()


# In[ ]:


model.fit(X, y, batch_size=64, epochs = 20, verbose=1)


# In[ ]:


prediction = model.predict(X_test)


# In[ ]:


def finalPredictions(x):
    if x<0.5 : 
        return 0
    else:
        return 1
pred_deep = []
for i in prediction:
    pred_deep.append(finalPredictions(i))
    


# # What does the deep intuition say?

# In[ ]:


print(classification_report(y_test, pred_deep))


# # Conclusion : 
# By this, we can conclude that Random Forest Classifier is our best bet.
# 
# ### Why did our Deep Learning architecture failed?
# We can refer to it failing heavily because of extensive beinary features and hashing in our Dataset. also, the bias doesn't really help our cause.

# In[ ]:




