#!/usr/bin/env python
# coding: utf-8

# Hi guys, I'm Ivan! 
# 
# In this kernel, I'm going to introduce a simple algorithm to solve a binary classification problem. **Logistic Regression!!!**
# As this is my very first kernel regards on classification, please correct me if you spot any mistakes! :)
# This kernel is only meant for beginners, hopefully it will be useful! Now, let's start !

# **Main Content:**
# * Import important packages,  training & testing dataset
# * Data Cleaning
# * Find the best fit model
# * Submission

# **PART I: Import Packages, training & testing dataset**

# In[ ]:


#Import basic packages
import numpy as np
import pandas as pd

#Package for data visualisation
import matplotlib.pyplot as plt

#Packages for preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

#Packages for modelling 
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

#Package for evaluation
from sklearn.model_selection import cross_val_score


# In[ ]:


#Read training & testing dataset and store it as DataFrame 
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

#Check the shape of each of the dataset
print(df_train.shape)
print(df_test.shape)


# Preprocess for data cleaning
# 
# Usually, we will conduct data cleaning process for both training & testing dataset. To aviod repetitive work, we will first need to combine both training & testing dataset into one. 
# 
# ps. By looking at the shape for both dataset, testing dataset is short of one column==> 'target'. We will need to assign a new column named 'target' in the testing set. As this is a binary classification problem, the values will be either 1 or 0 under column 'target' in training test. To avoid confusion, we will need to assign a different value. 

# In[ ]:


#Assign a value and create a new column in testing set.
df_test['target'] = 10
#Combine both dataset and denote to (df_all)
df_all = df_train.append(df_test, sort = True)


# **PART II: Data Cleaning**
# 
# Data cleaning is extremely important in many of the project, and it is very tedious for most of the time. To have a good picture on the data, sometimes we need to be creative! 
# 

# In[ ]:


#Take a look at the summary of each column
df_all.info()


# By looking at the above output, it seems that we do not have any missing value across all the columns. Well, it is not as smooth as you think....
# 
# It is stated at the missing value is replace by -1. Therefore, we have to change it back to NaN and fill up with a value that is more reasonable! 

# In[ ]:


#Replace (-1) with NaN
df_all = df_all.replace(-1,np.nan)
#Let's look at the summary again
df_all.info()


# In[ ]:


#Let's list down all the categorical variables that contain NaN. 
cat_na = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_05_cat', 'ps_car_07_cat', 'ps_car_09_cat']
#Bar chart plot regards on the frequency count of each category in each of the categorical variables that contain NaN. 
for i in cat_na:
    my_tab = pd.crosstab(index = df_all[i],columns="count")    
    my_tab.plot.bar()
    plt.show()


# Interpretation on the plot:
# 
# In each of the bar chart, there is a significance difference in the frequency count of each of the catetory. It will be reasonable that if we fill up the missing value with the value that appears the most in the corresponding column. 

# In[ ]:


#Fill NaN with most frequently number
for i in cat_na:
    df_all[i] = df_all[i].fillna(df_all[i].mode()[0])


# In[ ]:


#List down all the continuous variables that contain NaN
cont_na = ['ps_reg_03', 'ps_car_11', 'ps_car_12', 'ps_car_14']
#It is not wise to have a frequency plot for continuous variable...give it a try if you want to find out. 
#Fill NaN with mean
for i in cont_na:
    df_all[i] = df_all[i].fillna(df_all[i].mean())


# Change some of the continuous variables to categorical variables. 
# 
# Reason behind this: 
# Some of the columns are classified as continuous variables, however, it makes more sense to convert it into categorical variables if the number of categories is not huge. 

# In[ ]:


#Use nunique method to determine the number of unique values in each column
def count_unique_value(dataframe):
    df = pd.DataFrame()
    df['No. of unique value'] = dataframe.nunique()
    df['DataType'] = dataframe.dtypes
    return df

print(count_unique_value(df_all))


# In[ ]:


#Change datatype to 'Category' for the columns with number of unique value <= 20. 
def change_datatype(dataframe):
    col = dataframe.columns
    for i in col:
        if dataframe[i].nunique()<=20:
            dataframe[i] = dataframe[i].astype('category')
    
change_datatype(df_all)

#Change the datatype of target to int64. 
df_all['target'] = df_all['target'].astype('int64')


# We have done most of the work in data cleaning process. Right now, let's wrap up! 
# 

# In[ ]:


#Convert categorical variables to dummy variables
df_all_dummy = pd.get_dummies(df_all, drop_first = True)


# In[ ]:


#Split the combined dataset into training set & testing set
df_train_adj = df_all_dummy[df_all_dummy['target'] != 10]
df_test_adj = df_all_dummy[df_all_dummy['target'] == 10]


# **PART III: Find the best fit model**

# In[ ]:


#Extract training data from training set
data_to_train = df_train_adj.drop(['target','id'], axis = 1)
#Extract labels from training set
labels_to_use = df_train_adj['target']


# In[ ]:


#Build different model

#Logistic Regression
logreg = make_pipeline(RobustScaler(), LogisticRegression())

#SGD Classifier
sgd = make_pipeline(RobustScaler(), SGDClassifier(loss="log"))

#Random Forest Classifier
rfc = make_pipeline(RobustScaler(), RandomForestClassifier(50))


# Of course you can build as much model as you want. It is also possible to have a more accurate result by changing the default parameters. For Random forecast classifier, the larger number of estimators, the more accurate of the model. However, the processing time is getting slower as the number of estimators increases. The maximum number of estimators should not be larger than total number of variables. 

# **Build a evaluation model**

# In[ ]:


def evaluation_auc(model):
    result= cross_val_score(model, data_to_train, labels_to_use, cv = 3, scoring = 'roc_auc')
    return(result)


# In[ ]:


#Score for Logistic Regression
score = evaluation_auc(logreg)
print("\nLogistic Regression Score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))


# In[ ]:


#Score for SGD Classifier
score = evaluation_auc(sgd)
print("\nSGD Classifier Score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))


# In[ ]:


#Score for Random Forest Classifier
score = evaluation_auc(rfc)
print("\nRandom Forest Classifier score: {:.5f} ({:.5f})\n".format(score.mean(), score.std()))


# Logistics Regression performs the best! Our final submission will be based on that! 

# **PART IV: Submission**

# In[ ]:


#Submission preparation
test_df_id = df_test['id']
test_df_x = df_test_adj.drop(['target', 'id'], axis = 1)
logreg.fit(data_to_train, labels_to_use)

#As we are predicting probability, use predict_proba instead of predict! 
test_df_y = logreg.predict_proba(test_df_x)[:,1]

submission = pd.DataFrame({'id': list(test_df_id), 'target': list(test_df_y)})
submission.to_csv('sgd_log.csv')


# *This is the end of the kernel, thank you! *
