#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
INPUTDIR = "/kaggle/input/telecom-case-study/"


# In[ ]:


data = pd.read_csv(INPUTDIR + "RawdatafileV0.0.csv")
data.head()


# In[ ]:


data.isnull().sum()


# Only Var1 & Var2 has null values. Filling with a placeholder symbol 'UNK' 

# In[ ]:


data.fillna('UNK', inplace = True) # Replace nulls with a placeholder


# In[ ]:


## There are 90 rows which are duplicates, dropping them 
selCols = data.columns

data.drop_duplicates(subset = selCols[2:], inplace= True)
print(data.shape)


# In[ ]:


data.describe().T


# In[ ]:


# Copying out the index variable & the target variable
custID = data['Cust_id']
target = data['Plan_Chg_Flag']

# Drop Var10 since it has the same value (POSTPAID)
# Removing the target variable & index variable (cust_id)

data.drop(labels=['Cust_id', 'Plan_Chg_Flag', 'Var10'], inplace = True, axis =1) # Var10 is a constant (POSTPAID)


# In[ ]:


# Check the memory usage
data.info(memory_usage='deep')


# In[ ]:


# Copying all column names 
# Converting all columns to categories 

selCols = data.columns # Read the column names

data = data.apply(lambda x: x.astype('category')) # Convert all columns to category 

# Check the memory usage 
data.info(memory_usage='deep')


# Impact of converting to categorical variables. Size of the dataframe:
# 
# Before: **2.5 MB**
# 
# After converting: **166.6 Kb**

# In[ ]:


# Convert to one-hot-encoding 
df_with_dummies = pd.get_dummies( data, columns = selCols)
df_with_dummies['Cust_id'] = custID

print(df_with_dummies.shape)
df_with_dummies.head()


# In[ ]:


# just checking the size of the DF after converting to dummy values 
df_with_dummies.info()


# We will be creating a base model & then use SHAP to understand the interaction between variables

# In[ ]:


# Importing the sklearn libraries
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report

# Converting the target variable via LabelEncoding
label_encoder = LabelEncoder().fit(target)
targetLE = label_encoder.transform(target)


# In[ ]:


#Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(df_with_dummies, targetLE, test_size=0.3, shuffle=True, stratify=target, random_state= 93456)


# In[ ]:


# Removing the cust_id before creating the model
custID_train = train_x['Cust_id']
custID_valid = valid_x['Cust_id']

train_x.drop(labels= ['Cust_id'], axis= 1, inplace= True)
valid_x.drop(labels= ['Cust_id'], axis= 1, inplace= True)


# In[ ]:


#------------------------Build LightGBM Model-----------------------
import lightgbm as lgb

train_data=lgb.Dataset(train_x,label=train_y)
valid_data=lgb.Dataset(valid_x,label=valid_y)

#Select Hyper-Parameters
params = {'metric' : 'auc',
          'boosting_type' : 'gbdt',
          'colsample_bytree' : 0.9234,
          'num_leaves' : 13,
          'max_depth' : -1,
          'n_estimators' : 200,
          'min_child_samples': 399, 
          'min_child_weight': 0.1,
          'reg_alpha': 2,
          'reg_lambda': 5,
          'subsample': 0.855,
          'verbose' : -1,
          'num_threads' : 4
}


# In[ ]:


#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 valid_sets=valid_data,
                 early_stopping_rounds= 30,
                 verbose_eval= 10
                 )


# In[ ]:


lgbm.params


# Light GBM has a method which prints out the feature importance of each variable from the training dataset. We will first plot these values as a bar chart

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Feature importance 
feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importance(),train_x.columns), reverse=True), columns= ['Value', 'Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# In[ ]:


# If you are running it on your local systems then you will have to install the package using the following
# pip install shap
import shap

# Explain model predictions using shap library:
lgbm.params['objective'] = 'binary'
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(valid_x)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'shap.summary_plot')


# In[ ]:


# Plot summary_plot
shap.summary_plot(shap_values, valid_x)


# In[ ]:


# Low values of Var27 has a high impact on the prediction values. When Var27_UNK =0, it is either a GOOD ore POOR value.   
# For Plan_chg_flag = 1 customers have either a Var27 (Good/Poor) value 

valid_x.Var27_UNK.value_counts() 


# In[ ]:


# load JS visualization code to notebook
shap.initjs()


# In[ ]:


# When customer is employed in a Private organization, lower values of Var26_20 have a larger impact on the prediction 
# Var3_Private = 1
# Var26_20 = 0

shap.dependence_plot("Var3_Private", shap_values[0], valid_x)


# In[ ]:


shap.dependence_plot("Var3_Private", shap_values[1], valid_x)


# In[ ]:


# Var27_UNK = 0

shap.dependence_plot("Var27_UNK", shap_values[0], valid_x)


# In[ ]:


# Have a large impact onthe prediction (Plan_Chg_Flag = 1)
# Var27_Good = 1 
# Var3_Private = 1

shap.dependence_plot("Var27_Good", shap_values[1], valid_x)


# In[ ]:


# calculating the interaction values 

shap_interaction_values = shap.TreeExplainer(lgbm).shap_interaction_values(valid_x)
shap.summary_plot(shap_interaction_values, valid_x)


# In[ ]:


# Interaction between Var7_1200 and Var3_Private is not conclusive
# When VAr7_1200 =0 , and Var3_Private = 1 , has a higher impact on the shap values & vice versa
# When Var7_1200 =1, Var3_Private = 1 has a negative impact & vice versa

shap.dependence_plot(("Var7_1200", "Var3_Private"), shap_interaction_values, valid_x)


# In[ ]:


# Var26_20 =1 & Var3_Private =1 has a positive impact

shap.dependence_plot(("Var26_20", "Var3_Private"), shap_interaction_values, valid_x)


# In[ ]:


shap.dependence_plot(("Var4_30", "Var3_Private"), shap_interaction_values, valid_x)


# In[ ]:


import numpy as np 

shap_sum = np.abs(shap_values[1]).mean(axis=0)
importance_df = pd.DataFrame([valid_x.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df


# In[ ]:


import shap 
shap.__version__


# In[ ]:




