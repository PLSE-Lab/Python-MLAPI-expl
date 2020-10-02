#!/usr/bin/env python
# coding: utf-8

# <h1>1. What is the purpose of this kernel?</h1>

# > The goal is to **compare 20 different models out-of-the-box**, without special tuning. Draw performance comparison chart at the end and choose the winner model ))

# <h1>2. Import common libraries</h1>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Set random seed to make results reproducable
np.random.seed(42)
plt.style.use('seaborn')


# <h1>3. Read the data</h1>

# In[ ]:


df = pd.read_csv('../input/clinvar_conflicting.csv',dtype={0: object, 38: str, 40: object})
df.fillna(0,inplace=True)
df.head()


# <h1>4. Simple EDA</h1>

# Lightweight EDA  just to get basic understanding what the data is. We can see that target column named CLASS is skewed, so will need to rebalance the data before classification.

# In[ ]:


# Features histograms
df.drop('CLASS',axis=1).hist(figsize=(12,7))
plt.suptitle("Features histograms", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
sns.countplot(x='CLASS',data=df)
plt.title("Target label histogram")
plt.show()


# <h1>5. Classification with different models</h1>
# Balance, scale and split the data to train and test. Then create the list of models, train each and get it's metrics - **accuracy** in our case.

# <h2>5.1 Import ML libraries</h2>

# In[ ]:


import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seed to make results reproducable
np.random.seed(42)


# <h2>5.2 Prepare the data</h2>
# 
# <h3>5.2.1  Balancing by target feature</h3>
# 
# Target CLASS column has skewed distribution. Let's balance it first, then extract a sample from balanced data. Using full dataset produces memory error here.

# In[ ]:


# Balance
g = df.groupby('CLASS')
df_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
# Extract smaller sample to avoid memory error later, when training starts
df_balanced = df_balanced.sample(1000)

# Illustrate balancing results on plots
f, ax = plt.subplots(1,2)
# Before balancing plot
df.CLASS.value_counts().plot(kind='bar', ax=ax[0])
ax[0].set_title("Before")
ax[0].set_xlabel("CLASS value")
ax[0].set_ylabel("Count")
# After balanced plot
df_balanced.CLASS.value_counts().plot(kind='bar',ax=ax[1])
ax[1].set_title("After")
ax[1].set_xlabel("CLASS value")
ax[1].set_ylabel("Count")

plt.suptitle("Balancing data by CLASS column value")
plt.tight_layout()
plt.subplots_adjust(top=0.8)
plt.show()


# <h3>5.2.2 Train/test split, encode, scale input data</h3>
# Some models requre one hot encoding, some are sensible to data normalization. Do encode and normalize data to make each model happy. Then train/test split with default ratio.
# 

# In[ ]:


# Features - all columns except 'CLASS'
# Target label = 'CLASS' column
X=df_balanced.drop('CLASS',axis=1)
# One hot encoding
X=pd.get_dummies(X, drop_first=True)
y=df_balanced['CLASS']
y=pd.get_dummies(y, drop_first=True)

# Train/test split
train_X, test_X, train_y, test_y = train_test_split(X, y)

# Normalize using StandardScaler
scaler=StandardScaler()
train_X=scaler.fit_transform(train_X)
test_X=scaler.transform(test_X)

# Histogram of target labels distribution
test_y.hist()
plt.title("Target feature distribution: CLASS values")
plt.xlabel("Value")
plt.ylabel("Count")
plt.show()


# <h2>5.3 Run models</h2>
# Create models, train and evaluate them. Store  accuracy by model in dictionary. Do not adjust models, default settings used.

# In[ ]:


# Models to try
models = [LogisticRegression(), 
          LogisticRegressionCV(), 
          PassiveAggressiveClassifier(),
          RidgeClassifier(),
          RidgeClassifierCV(),
          KNeighborsClassifier(),
          #RadiusNeighborsClassifier(),
          NearestCentroid(),
          DecisionTreeClassifier(), 
          AdaBoostClassifier(), 
          BaggingClassifier(),
          ExtraTreesClassifier(),
          GradientBoostingClassifier(),
          RandomForestClassifier(), 
          SGDClassifier(),
          GaussianNB(),
          GaussianProcessClassifier(),
          LinearDiscriminantAnalysis(),
          QuadraticDiscriminantAnalysis(),
          MLPClassifier(),
          SVC()
         ]
# Gather metrics here
accuracy_by_model={}

# Train then evaluate each model
for model in models:
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    score = accuracy_score(test_y, pred_y)
    # Fill metrics dictionary
    model_name = model.__class__.__name__
    accuracy_by_model[model_name]=score  


# <h1>6. Calculating results</h1>
# <h2>6.1 Compare gathered metrics</h2>
# As we can see below, accuracy is far from 1.0 )) maybe more input data preparation could improve the metrics. But our models did their best with the data we prepared for them. Have a look at the results.

# In[ ]:


# Draw accuracy by model chart
acc_df = pd.DataFrame(list(accuracy_by_model.items()), columns=['Model', 'Accuracy']).sort_values('Accuracy', ascending=False).reset_index(drop=True)
acc_df.index=acc_df.index+1
sns.barplot(data=acc_df,y='Model',x='Accuracy')
plt.xlim(0,1)
plt.title('Accuracy of models with default settings')
plt.xticks(rotation=45)
plt.show()

# Print table
acc_df


# <h2>6.2. Choose the hero of the Kernel</h2>
# 
# Ladies and gentlemen, we have a winner! The winnerrrr iisssss...

# In[ ]:


best_model = acc_df[acc_df.Accuracy==acc_df.Accuracy.max()]
best_model


# 

# 

# 
