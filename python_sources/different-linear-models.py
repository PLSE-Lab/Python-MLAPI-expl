#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/train_real.csv')
test_df = pd.read_csv('/kaggle/input/cte-ml-hack-2019/test_real.csv')


# # Train-Test split
# # 
# # Here we split train_df into train and validation dataframes

# In[ ]:


train_df["Azimuthal_angle"] = np.sqrt(train_df["Azimuthal_angle"])

train_df["H_dist_Hydro"] = np.sqrt(train_df["H_dist_Hydro"])

train_df["Dist_Hydro"] = np.sqrt((train_df["H_dist_Hydro"]**2 + train_df["V_dist_Hydro"]**2))
test_df["Dist_Hydro"] = np.sqrt((test_df["H_dist_Hydro"]**2 + test_df["V_dist_Hydro"]**2))

train_df['Dist_Hydro'] = np.sqrt(train_df['Dist_Hydro'])

train_df["H_dist_Fire"] = np.sqrt(train_df["H_dist_Fire"])

train_df["H_dist_Road"] = np.sqrt(train_df["H_dist_Road"])

train_df["Incline"] = np.sqrt(train_df["Incline"])

train_df['Hillshade_9am'] = (train_df['Hillshade_9am'])**3

train_df = train_df.drop(train_df[(train_df['Azimuthal_angle']>25)].index)
train_df = train_df.reset_index(drop = True)
train_df = train_df.drop(train_df[(train_df['H_dist_Hydro']<5)].index)
train_df = train_df.reset_index(drop = True)

X = train_df.drop(['Id', 'label','Soil','V_dist_Hydro'], axis=1)
y = train_df['label']

X_test = test_df.drop(['Id','Soil','V_dist_Hydro'], axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


y_train.head()


# In[ ]:


X_train.head()


# # EDA
# # This is where we start to explore the data and the features to see if we can apply some algorithms to the features to make them more usable for our model

# In[ ]:


#Just checking features for any unnaturally high skewness

print(train_df.skew())


# In[ ]:


# Correlation tells relation between two attributes.
# Correlation requires continous data. Hence, ignore Wilderness_Area and Soil_Type as they are binary

#Index where continuous data ends
size = 10

data=train_df.iloc[:,:size] 

cols=data.columns 

data_corr = data.corr()

threshold = 0.5

corr_list = []

for i in range(0,size):
    for j in range(i+1,size):
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) 

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))


# In[ ]:


#Plotting the features with high correlation along with the labels
for v,i,j in s_corr_list:
    sns.pairplot(train_df, hue="label", height=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()


# # Basic Linear models
# # 
# # We use Random Forest and K Nearest Neighbours models and use the one which gives us  higher accuracy
# #

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 

rf = RandomForestClassifier(random_state=42, n_estimators=180,n_jobs=-1,max_depth = 8).fit(X_train, y_train)

xgb = XGBClassifier(random_state=42, n_estimators=180,n_jobs=-1,max_depth = 8).fit(X_train, y_train)

knn = KNeighborsClassifier(7).fit(X_train,y_train)


# In[ ]:


rf_validation_res=rf.predict(X_validation)
print(accuracy_score(rf_validation_res, y_validation))


# In[ ]:


xgb_validation_res=xgb.predict(X_validation)
print(accuracy_score(xgb_validation_res, y_validation))


# In[ ]:


knn_validation_res=knn.predict(X_validation)
print(accuracy_score(knn_validation_res, y_validation))


# # Combining models
# # 
# # Here we saw that the Random Forest Classifier gave us the best accuracy in our data, but these model's tend to have high biases in exchange for increased variance. Here we try to combine Random Forest models and see if it improves our accuracy.
# #

# In[ ]:


from functools import reduce
from sklearn.metrics import roc_auc_score
def generate_rf(X_train, y_train, X_validation, y_validation, x):
    rf = RandomForestClassifier(n_estimators=25*(x+1), min_samples_leaf=2, n_jobs=-1)
    rf.fit(X_train, y_train)
    print ("rf score: ", rf.score(X_validation, y_validation))
    print ("rf auc score: ", roc_auc_score(rf.predict(X_validation), y_validation))
    return rf

def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

rfs = [generate_rf(X, y, X_validation, y_validation, i) for i in range(10)]
# in this step below, we combine the list of random forest models into one giant model
rf_combined = reduce(combine_rfs, rfs)
# the combined model scores better than most of the component models
print ("rf combined score: ", rf_combined.score(X_validation, y_validation))
print ("rf combined auc score: ", roc_auc_score(rf_combined.predict(X_validation), y_validation))


# As we don't have that large of a dataset to work on, we train it on the whole dataset instead of just the training set. This obviously means that it will give bloated accuracies in the validation as it has already seen this data beofore. The only thing we are actually checking for in this accuracy is whether our model is overfitting or not, and as we can see, it is not.

# In[ ]:


test_res = rf_combined.predict(X_test)


# In[ ]:


submission_df = pd.DataFrame()
submission_df['Id'] = test_df['Id']


# In[ ]:


submission_df['Predicted'] = test_res.tolist()


# In[ ]:


submission_df.tail()


# In[ ]:


submission_df.to_csv('ml_hack_submission.csv',index=False)


# In[ ]:


get_ipython().system('ls')

