#!/usr/bin/env python
# coding: utf-8

# ### Table of Contents
# 
# 1. [**Importing necessary Libraries**](# 1.-Importing-necessary-Libraries)   
# 2. [**Loading Data**](#2.-Loading-Data)  
# 3. [**Data Preprocessing**](#3.-Data-Preprocessing)  
#     3.a [**Checking for missing values**](#3.a-Checking-for-missing-values)  
#     3.b [**Checking for Class Imbalance**](#3.b-Checking-for-Class-Imbalance)  
#     3.c [**Handling class imbalance**](#3.c-Handling-class-imbalance)  
# 4. [**Feature Selection**](#4.-Feature-Selection)  
#     4.a [**Forward Propagation**](#4.a-Forward-Propagation)  
#     4.b [**Splitting the data in a 80:20 ratio**](#4.b-Splitting-the-data-in-a-80:20-ratio)  
# 5. [**Fitting a Logistic Regression model**](#5.-Fitting-a-Logistic-Regression-model)  
#     5.a [**Hyper-parameter tuning**](#5.a-Hyper-parameter-tuning)  
#     5.b [**predictions against test data**](#5.b-predictions-against-test-data)  
# 6. [**Using AutoML**](#6.-Using-AutoML)  

# ### 1. Importing necessary Libraries

# In[ ]:


import numpy as np 
import pandas as pd

from sklearn.utils import resample

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,f1_score


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# ### 2. Loading Data

# In[ ]:


df = pd.read_csv("/kaggle/input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Train.csv")
print(" Dataset size : ", df.shape)


# ### 3. Data Preprocessing
# 
# #### 3.a Checking for missing values 

# In[ ]:


print("="*15,"Total no. of missing values in each column","="*15)
print(df.isna().sum())


# #### 3.b Checking for Class Imbalance

# In[ ]:


print("Frequency of churns and No churns : \n", df["labels"].value_counts())


# In[ ]:


fig = go.Figure([go.Bar(x=df["labels"].value_counts().index, y=df["labels"].value_counts().values)])
fig['layout'].update(title={"text" : 'Distribution of churn labels','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="label",yaxis_title="count")
fig.update_layout(width=500,height=500)
fig


# #### 3.c Handling class imbalance
# 
# There are multiple Sampling approaches to deal with class imbalance problem. Most commonly preferred are 
# 1. Under(down) Sampling
# 2. Over(up) Sampling
# 3. SMOTE
# 
# Alternatively, we can also use the algorithms(like Logistic Regression, SVM etc.) in a balanced mode by balancing the class weights.

# ### 4. Feature Selection
# 
# #### 4.a Forward Propagation

# In[ ]:


X = df.drop("labels", axis=1)
y = df[["labels"]]

sfs = SFS(LogisticRegression(class_weight = "balanced"),
           k_features=10,
           forward=True,
           floating=False,
           scoring = 'f1',
           cv = 0)

sfs.fit(X,y)
print("Top 10 features selected using Forward Propagation",sfs.k_feature_names_ )


# In[ ]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


# Above plot indicates that the performance measure(F1-score here) becomes stationary after the use of first 8 features. So let's use these 8 features in model training.

# #### 4.b Splitting the data in a 80:20 ratio

# In[ ]:


train,val = train_test_split(df, test_size = 0.2, random_state = 42, stratify = df['labels'])
X_train = train[['feature_0', 'feature_2', 'feature_3', 'feature_4', 'feature_6', 'feature_9', 'feature_11', 'feature_12']]
y_train = train[["labels"]]
X_val = val[['feature_0', 'feature_2', 'feature_3', 'feature_4', 'feature_6', 'feature_9', 'feature_11', 'feature_12']]
y_val = val[["labels"]]


# ### 5. Fitting a Logistic Regression model

# In[ ]:


log_clf = LogisticRegression(class_weight = "balanced")
log_clf.fit(X_train, y_train)
train_pred = log_clf.predict(X_train)
print("Training data accuracy : ", accuracy_score(train_pred,y_train))
print("Training data F1-score : ", f1_score(train_pred,y_train))


val_pred = log_clf.predict(X_val)
print("Validation data accuracy : ", accuracy_score(val_pred,y_val))
print("Validation data F1-score : ", f1_score(val_pred,y_val))


# #### 5.a Hyper-parameter tuning

# In[ ]:


c = [10 ** x for x in range(-5, 2)]
f1_score_array=[]
for i in c:
    clf = LogisticRegression(C =i, class_weight = 'balanced')
    clf.fit(X_train, y_train)
    predict_y = clf.predict(X_val)
    f1_score_array.append(f1_score(y_val, predict_y))
    print('For values of alpha = ', i, "The F1 - score is:",f1_score(y_val, predict_y))
    
print("\nThe maximum value of f1_score is {} for C = {}".format(max(f1_score_array), c[f1_score_array.index(max(f1_score_array))]))


# In[ ]:


log_clf = LogisticRegression(C = 0.001,class_weight = "balanced")
log_clf.fit(X_train, y_train)
train_pred = log_clf.predict(X_train)
print("Training data accuracy : ", accuracy_score(train_pred,y_train))
print("Training data F1-score : ", f1_score(train_pred,y_train))


val_pred = log_clf.predict(X_val)
print("Validation data accuracy : ", accuracy_score(val_pred,y_val))
print("Validation data F1-score : ", f1_score(val_pred,y_val))


# #### 5.b predictions against test data

# In[ ]:


submission = pd.read_csv("/kaggle/input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Test.csv")
predictions = log_clf.predict(submission[['feature_0', 'feature_1', 'feature_3', 'feature_4', 'feature_6', 'feature_9', 'feature_10', 'feature_11']])


# ### 6. Using AutoML 

# In[ ]:


import h2o
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='16G')


# In[ ]:


# Data loading
df = pd.read_csv("/kaggle/input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Train.csv")
df["labels"] = df["labels"].map({0: "No", 1:"Yes"})
X = h2o.H2OFrame(df)
X.describe()


# In[ ]:


#Splitting the data
splits = X.split_frame(ratios=[0.8],seed=1)
train = splits[0]
val = splits[1]


# In[ ]:


y = "labels"
x_train = train.columns
x_train.remove(y)


# In[ ]:


# Fitting the model
aml = H2OAutoML(max_runtime_secs=300, seed=1,keep_cross_validation_predictions = True,balance_classes = True, max_after_balance_size = 7934)
aml.train(x = x_train, y = y, training_frame = train)


# In[ ]:


lb = aml.leaderboard
lb


# In[ ]:


# making the predictions on validation data 
pred = aml.predict(val.drop("labels",axis=1))

#computing accuracy
print("Accuracy on validation data = ",accuracy_score(pred.as_data_frame()["predict"], val["labels"].as_data_frame()["labels"]))

#computing f1-score 
print("F1- score = ",f1_score(pred.as_data_frame()["predict"], val["labels"].as_data_frame()["labels"],pos_label = "Yes" ))

