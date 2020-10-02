#!/usr/bin/env python
# coding: utf-8

# ##### Target variables are:
#     1. Tool wear detection
#     2. Detection of inadequate clamping- "Passed visual inspection"
#     3. Machining finalised

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# > ## 1. Load Data

# In[ ]:


main_df=pd.read_csv('/kaggle/input/tool-wear-detection-in-cnc-mill/train.csv')
main_df=main_df.fillna('no')
main_df.head()


# In[ ]:


import glob
#set working directory
os.chdir('/kaggle/input')


# ### Creating the data frame 

# In[ ]:


files = list()

for i in range(1,19):
    exp_number = '0' + str(i) if i < 10 else str(i)
    file = pd.read_csv("/kaggle/input/tool-wear-detection-in-cnc-mill/experiment_{}.csv".format(exp_number))
    row = main_df[main_df['No'] == i]
    
     #add experiment settings to features
    file['feedrate']=row.iloc[0]['feedrate']
    file['clamp_pressure']=row.iloc[0]['clamp_pressure']
    
    # Having label as 'tool_conidtion'
    
    file['label'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
    files.append(file)
df = pd.concat(files, ignore_index = True)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


# Convert 'Machining_process' into numerical values
pro={'Layer 1 Up':1,'Repositioning':2,'Layer 2 Up':3,'Layer 2 Up':4,'Layer 1 Down':5,'End':6,'Layer 2 Down':7,'Layer 3 Down':8,'Prep':9,'end':10,'Starting':11}

data=[df]

for dataset in data:
    dataset['Machining_Process']=dataset['Machining_Process'].map(pro)


# In[ ]:


df=df.drop(['Z1_CurrentFeedback','Z1_DCBusVoltage','Z1_OutputCurrent','Z1_OutputVoltage','S1_SystemInertia'],axis=1)


# In[ ]:


corm=df.corr()
corm


# In[ ]:


#checking the relationship between the variables by applying the correlation 
plt.figure(figsize=(30, 25))
p = sns.heatmap(df.corr(), annot=True)


# ## 2. Building ML Model

# In[ ]:


X=df.drop(['label','Machining_Process'],axis=1)
Y=df['label']
print('The dimension of X table is: ',X.shape,'\n')
print('The dimension of Y table is: ', Y.shape)


# ### 2.1 Train/Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

#divided into testing and training
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# In[ ]:


from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# #### 2.2.1 Stochastic Gradient Descent (SGD):
# 

# In[ ]:


sgd_model=SGDClassifier()
sgd_model.fit(x_train,y_train)


# In[ ]:


sgd_model_pred=sgd_model.predict(x_test)
acc_sgd_model=round(sgd_model.score(x_train, y_train)*100,2)
acc_sgd_model


# #### 2.2.2 Random Forest:

# In[ ]:


rmf_model=RandomForestClassifier()
rmf_model.fit(x_train,y_train)


# In[ ]:


rmf_model_pred=rmf_model.predict(x_test)
acc_rmf_model=round(rmf_model.score(x_train, y_train)*100,2)
acc_rmf_model


# #### 2.2.3 Logistic Regression

# In[ ]:


log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)


# In[ ]:


log_reg_pred=log_reg.predict(x_test)
acc_log_reg=round(log_reg.score(x_train,y_train)*100,2)
acc_log_reg


# #### 2.2.4 K Nearest Neighbor

# In[ ]:


knb_model=KNeighborsClassifier()
knb_model.fit(x_train,y_train)


# In[ ]:


knb_model_pred=knb_model.predict(x_test)
acc_knb_model=round(knb_model.score(x_train,y_train)*100,2)
acc_knb_model


# #### 2.2.5 Linear Support Vector Machine

# In[ ]:


svm_model=LinearSVC()
svm_model.fit(x_train,y_train)


# In[ ]:


svm_model_pred=svm_model.predict(x_test)
acc_svm_model=round(svm_model.score(x_train,y_train)*100,2)
acc_svm_model


# ### 2.3 Which is the best Model

# In[ ]:


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest','Stochastic Gradient Decent'],
    'Score': [acc_svm_model, acc_knb_model, acc_log_reg, 
              acc_rmf_model,acc_sgd_model]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# As we can see, the Random Forest classifier goes on the first place. But first, let us check, how random-forest performs, when we use cross validation.

# In[ ]:


from sklearn.model_selection import cross_val_score
rmf_model = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rmf_model, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores,'\n')
print("Mean:", scores.mean(),'\n')
print("Standard Deviation:", scores.std())


# Evaluate Random Forest using the out-of-bag samples to estimate the generalization accuracy. I will not go into details here about how it works. Just note that out-of-bag estimate is as accurate as using a test set of the same size as the training set. Therefore, using the out-of-bag error estimate removes the need for a set aside test set

# In[ ]:


rmf_model = RandomForestClassifier(n_estimators=100, oob_score = True)
rmf_model.fit(x_train, y_train)
y_prediction = rmf_model.predict(x_test)

rmf_model.score(x_train, y_train)

acc_rmf_model = round(rmf_model.score(x_train, y_train) * 100, 2)
print(round(acc_rmf_model,2,), "%")


# In[ ]:


print("oob score:", round(rmf_model.oob_score_, 4)*100, "%")


# ### 2.4 Further Evaluation

# #### Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict

predictions = cross_val_predict(rmf_model, x_train, y_train, cv=3)
predictions[:10] # first 10 predictions


# In[ ]:


confusion_matrix(y_train,predictions)


# In[ ]:


print("Precision_score: ", precision_score(y_train,predictions),'\n')
print("Recall: ", recall_score(y_train,predictions),'\n')
print("Accruacy_score: ", accuracy_score(y_train,predictions),'\n')
print("F_score: ", f1_score(y_train, predictions))


# #### Precision Recall Curve

# In[ ]:


from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = rmf_model.predict_proba(x_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# 
