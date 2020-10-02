#!/usr/bin/env python
# coding: utf-8

# The world of electronics payment enables the card holder to  shop online with virtual money in the pocket. The growth of the online industry also increases the risk of fraudent transaction.  Every credit card provider is working hard in this area to reduce the fraud transaction.
# 
# It is important for thecredit card companies to recognize the fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
# 
# Provided dataset is PCA transformation except Time and Amount features. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# **Inspiration**
# Identify fraudulent credit card transactions.
# 
# Provided data is imbalanced, Precision-Recall Curve (AUPRC), Accuracy may not work out well. To improving the fraud trasaction detection, recall rate need to improve without much caring on  Precision and F-1 score.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from collections import Counter
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
import os

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# ## Read Dataset

# In[ ]:


import os
print('Data File ==>\t {}'.format(os.listdir("../input")[0]))
creditcard = pd.read_csv("../input/creditcard.csv")


# ## Exploratory Data Analysis

# In[ ]:


creditcard.head(3)


# In[ ]:


# Missing values
print('No of Missing values :\t{}'.format(creditcard.isnull().sum().max()))


# In[ ]:


sns.countplot(data=creditcard,x = 'Class')
plt.title('Class Variables distribution', fontsize=14)
plt.show()

creditcard['Class'].value_counts() *100 /len(creditcard)


# Imbalance class, only 0.172% fraud case data against 99.83% none fraud case data. There are a couple of mathods that cane implemented to handle the imbalanced data.
# 
# * Up-sampling
# * Down-sampling
# 
# https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis

# In[ ]:


fig,ax = plt.subplots(nrows = 7, ncols=4, figsize=(12,21))
row = 0
col = 0
for i in range(len(creditcard.columns) -3):
    if col > 3:
        row += 1
        col = 0
    axes = ax[row,col]
    sns.boxplot(x = creditcard['Class'], y = creditcard[creditcard.columns[i +1]],ax = axes)
    col += 1
plt.tight_layout()
plt.show()


# I am not intrested here to find the outliers, but there are few features those have big difference in mean and median values for fraudent and normal trasaction. I would prefer to remove the low variance variables along with Amount and Time for further analysis.

# **Remove Amount and Time**

# In[ ]:


creditcard.drop(['Time','Amount'],axis = 1,inplace=True)


# In[ ]:


X = creditcard.iloc[:,range(0,28)].values
y = creditcard['Class'].values


# **Remove Low variance features**

# In[ ]:


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = sel.fit_transform(X)
print('Remaining Features Count:\t{}'.format(X.shape[1]))


# Since data is imbalance, I would prefer under and oversampling along with normal data.
# 
# **Model Metrics Method**

# In[ ]:


def print_result(title,actual, prediction,decision):
    print('****************************************************')
    print(title)
    print('****************************************************')
    print('Accuracy Score :\t\t{:.3}'.format(metrics.accuracy_score(actual, prediction)))
    print('Recall Score :\t\t\t{:.3}'.format(metrics.recall_score(actual, prediction)))
    print('Average Precision Score :\t{:.3}'.format(metrics.average_precision_score(actual, decision)))
    print('ROC AUC Score :\t\t\t{:.3}'.format(metrics.roc_auc_score(actual, decision)))
    print()


# **Training-Test dataset split**
# 
# I will use 70% data to train model and rest 30% for validation purpose.

# In[ ]:


# split data in training set and test set
X_train, X_test, y_train, y_test = train_test_split(    X,y,test_size=0.3, random_state = 0)


# **Logistic Regression without any change in data (Normal)**
# 

# In[ ]:


# Data Distribution 
print('Normal Data Distribution {}'.format(Counter(creditcard['Class'])))
C_VALUES = [0.001,0.01,0.1,1]
# build normal Model
for c_value in C_VALUES:
    pipeline = make_pipeline(LogisticRegression(random_state=42, C = c_value))
    model = pipeline.fit(X_train,y_train)
    prediction = model.predict(X_test)
    decision = model.decision_function(X_test)
    # print(metrics.confusion_matrix(y_test,prediction))
    print_result('Normal Data Logistic -> C ={}'.                 format(c_value), y_test,prediction,decision)


# As I doubted in the begining, model achieving almost 100% accuracy but recall score is not good enough. Best value for this model C = 1 which will able to catch only 62.6% fraud transaction. I will try upsampling and downsampling to improve the recall Score. 

# **SMOTE Oversampling Model**

# In[ ]:


X_SMOTE,y_SMOTE = SMOTE().fit_sample(X,y)
print('SMOTE Data Distribution {}'.format(Counter(y_SMOTE)))

C_VALUES = [0.001,0.01,0.1,1]
# build normal Model
for c_value in C_VALUES:
    smote_pipeline = make_pipeline_imb(SMOTE(random_state=42),                                       LogisticRegression(random_state=42, C = c_value))
    smote_model = smote_pipeline.fit(X_train,y_train)
    smote_prediction = smote_model.predict(X_test)
    smote_decision = smote_model.decision_function(X_test)
    # print(metrics.confusion_matrix(y_test,smote_prediction))
    print_result('SMOTE - Oversampling data(Logistic) -> C ={}'.                 format(c_value), y_test,smote_prediction,smote_decision)


# As expected accuracy score is less as compares to normal data. Our purpose is to improve recall scrore. Oversampling with C value of 0.001 is providing recall score of 0.918 with accuracy of 0.977.
# 
# **NearMiss Undersampling Model**

# In[ ]:


X_NearMiss,y_NearMiss = NearMiss().fit_sample(X,y)
print('NearMiss Data Distribution {}'.format(Counter(y_NearMiss)))
# build moodel with  - undersampling
C_VALUES = [0.001,0.01,0.1,1]
# build normal Model
for c_value in C_VALUES:
    nearmiss =  LogisticRegression(random_state=42,C = c_value)
    nearmiss_model = nearmiss.fit(X_NearMiss,y_NearMiss)
    nearmiss_decision = nearmiss_model.decision_function(X_test)
    nearmiss_prediction = nearmiss_model.predict(X_test)
    print_result('NearMiss - Undersampling data(Logistic) -> C ={}'.                 format(c_value), y_test,nearmiss_prediction,nearmiss_decision)


# **Conclussion**
# 
# Undersampling improves recall score but it also reduces accuracy drastically, therefore more normal trasaction are calssified as fraudulent transaction. I, personally, would prefer oversampling over other three models where model predicts approximately 92% of fraudulent transactions with overall accuracy  of 97.7% 
