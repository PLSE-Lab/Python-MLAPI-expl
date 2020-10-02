#!/usr/bin/env python
# coding: utf-8

# # Exploring the patients data and starter model

# In this notebook, I will be going over the train.csv file and perform EDA as well as create a basic model using the patients data. In a realistic scenario, the pictures in the data of the competition should be used for modelling.

# In[ ]:


# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Reading in the data
df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')


# In[ ]:


# Taking a look at the first 5 rows of data
df.head()


# In[ ]:


# Getting the shape of the dataset
df.shape


# There are a total of 33126 rows of data and 8 columns including 2 target columns, i.e. benign_maginant and target.

# In[ ]:


df.dtypes.value_counts().sort_values(ascending=False)


# Checking the data types, it is clear that most of the columns contain objects.

# Now, seeing how many unique categorical types of data are in the columns having 'object' as a datatype.

# In[ ]:


df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# Now, finding the number of photos present of each patient.

# In[ ]:


df['patient_id'].value_counts()


# So, the max number of images for a single patient is 115 and the minimum number of images for a single patient is 2.

# In[ ]:


df['patient_id'].value_counts().mean()


# The mean number of images for the 2056 patients is 16.

# In[ ]:


df['target'].value_counts()


# There are large number of cases in the dataset that are benign and low number of cases in the dataset that are malignant. 

# Creating a helper function for further analysis.

# In[ ]:


def plot_analysis(col_name, df, plot_kind='bar'):
    """
    Function to plot two subplots containing Joe's and non-Joe's counts of data points for a given feature.
    :param col_name: Column name of the feature to be analysed
    :param df: DataFrame containing the source of data
    :plot_kind: Line plot or Bar Plot
    :return True: Boolean indicating that the analysis has been plotted
    """
    
    df_benign = df[df['target']==0]
    df_malignant = df[df['target']!=0]
    fig, axs = plt.subplots(2,figsize=(26,8))
    fig.suptitle('Difference between ' + col_name + ' of patients in benign and malignant cases ')
    axs[0].set_title('Benign')
    axs[0].set_ylabel('Number of cases', fontsize=12)
    axs[1].set_title('Malignant')
    axs[1].set_ylabel('Number of cases', fontsize=12)
    axs[1].set_xlabel(col_name, fontsize=12)
    if plot_kind == 'line':
        axs[0].plot(df_benign[col_name].value_counts().index, df_benign[col_name].value_counts().values)
        axs[1].plot(df_malignant[col_name].value_counts().index, df_malignant[col_name].value_counts().values)
    elif plot_kind == 'bar':
        axs[0].bar(df_benign[col_name].value_counts().index, df_benign[col_name].value_counts().values)
        axs[1].bar(df_malignant[col_name].value_counts().index, df_malignant[col_name].value_counts().values)
    return True


# In[ ]:


plot_analysis('sex', df)


# Looks the proportion of male and female patients is quite similar for the benign and malignant cases

# In[ ]:


plot_analysis('anatom_site_general_challenge', df)


# The proportion of anatom_site_general_challenge is also very similar in proportion.

# In[ ]:


plot_analysis('diagnosis', df)


# There are a large number of unknown diagnosis for benign cases but in malignant cases, it is all melanoma.

# In[ ]:


plot_analysis('age_approx', df)


# The mean age of all the patients in both cases is around 40-60.

# ## Data Modelling

# Checking for null values show that there are some missing values for sex, age_approx and anatom_site_general_challenge. Filling the null values with the mostly observed value.

# In[ ]:


df.isnull().sum()


# Performing Data pre-processing on the training data

# In[ ]:


def data_preparation(df, evaluation = False):
    # Filling in the missing values
    df['sex'] = df['sex'].fillna('male')
    df['age_approx'] = df['age_approx'].fillna(df['age_approx'].mean())
    df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].fillna('torso')

    # Label encoding sex
    labelencoder = LabelEncoder()
    df['sex'] = labelencoder.fit_transform(df['sex'])

    df = pd.get_dummies(df, columns=['anatom_site_general_challenge'])

    if evaluation:
        X_return = df.drop(['image_name', 'patient_id'], axis = 1)
        return X_return
    else:
        X_return = df.drop(['image_name', 'patient_id', 'benign_malignant','target','diagnosis'], axis = 1)
        return X_return, df['target']


# In[ ]:


X, y = data_preparation(df)


# In[ ]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify = y)

# Gradient Boosting Model
clf = LGBMClassifier(
            objective='binary',
            n_estimators=100000,
            num_leaves=10,
            learning_rate=0.1,
            max_depth=16,
            subsample_for_bin= 200000,
            subsample=1,
            subsample_freq= 200,
            silent=-1,
            verbose=-1,
            min_split_gain=0.0001,
            min_child_samples=800,
            )

# Training the model
clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc', verbose=200, early_stopping_rounds=1000)


# In[ ]:


test_df  = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
test_df.head()


# Performing data pre-processing

# In[ ]:


# Preprocessing test data
eval_X = data_preparation(test_df, evaluation = True)


# In[ ]:


# Getting the prediction probability
prediction_list = clf.predict_proba(eval_X)
final_pred_list = [a[1] for a in prediction_list]


# In[ ]:


# Appending to the test dataframe
test_df['target'] = final_pred_list


# Getting the final output file

# In[ ]:


test_df[['image_name','target']].to_csv('submission.csv',index=False)


# I will try my best to use the pictures next to get better accuracy for this competition.
