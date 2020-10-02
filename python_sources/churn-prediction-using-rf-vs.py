#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries ###

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# ### Read Dataset ###

# In[ ]:


rawdata=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[ ]:


data=rawdata.copy()
data.shape


# In[ ]:


data.head(5)


# In[ ]:


data.info()


# ### Convert TotalCharges Column(feature) to Numeric From Object ### 

# In[ ]:



data['TotalCharges']=data['TotalCharges'].convert_objects(convert_numeric=True)
def uni(columnname):
    print(columnname,"--" ,data[columnname].unique())


# In[ ]:


dataobject=data.select_dtypes(['object'])
len(dataobject.columns)


# In[ ]:


for i in range(1,len(dataobject.columns)):
    uni(dataobject.columns[i])
    


# In[ ]:


def labelencode(columnname):
    data[columnname] = LabelEncoder().fit_transform(data[columnname])


# In[ ]:


for i in range(1,len(dataobject.columns)):
    labelencode(dataobject.columns[i])


# In[ ]:


data.info()


# In[ ]:


for i in range(1,len(dataobject.columns)):
     uni(dataobject.columns[i])


# In[ ]:



data.info()


# In[ ]:


df=data.copy()
dfl=data.copy()


# In[ ]:


unwantedcolumnlist=["customerID","gender","MultipleLines","PaymentMethod","tenure"]


# In[ ]:


df = df.drop(unwantedcolumnlist, axis=1)
features = df.drop(["Churn"], axis=1).columns


# In[ ]:


df_train, df_val = train_test_split(df, test_size=0.30)


# In[ ]:


print(df_train.shape)
print(df_val.shape)


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_val.isnull().sum()


# ### Train and Validation Missing Data Fill by mean(imputation) ###

# In[ ]:


df_train['TotalCharges'].fillna(df_train['TotalCharges'].mean(), inplace=True)
df_val['TotalCharges'].fillna(df_val['TotalCharges'].mean(), inplace=True)


# ### Random Forest Algorithm applied on  Train and validate on Validation ###

# In[ ]:


clf = RandomForestClassifier(n_estimators=30 , oob_score = True, n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = 50)
clf.fit(df_train[features], df_train["Churn"])

# Make predictions
predictions = clf.predict(df_val[features])
probs = clf.predict_proba(df_val[features])
display(predictions)


# ### Accuracy ###

# In[ ]:


score = clf.score(df_val[features], df_val["Churn"])
print("Accuracy: ", score)


# In[ ]:


data['Churn'].value_counts()


# ### ROC Curve ###

# In[ ]:


get_ipython().magic('matplotlib inline')
confusion_matrix = pd.DataFrame(
    confusion_matrix(df_val["Churn"], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(confusion_matrix)

# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(df_val["Churn"], probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Application of Churn Analysis? ###
# 
# * To chrun - thought of conversion
# * Telcom Industry has more operator and provide verious of services with different prices so custmoer can convert from one operator to other for service/economical benifits
# * for operator it is vital to identify churn customer and hence they can convince particular
# * above model predict customer churn

# In[ ]:




