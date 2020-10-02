#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#####################################
# Developed by Baptiste PICARD      #
# picard.baptiste@laposte.net       #
# Started the 06th of March 2020    #
# picard.baptiste@laposte.net       #
#                                   #
#####################################
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Utils 
import os
import time 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Modules from sklearn for preprocessing.
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Modules from sklearn for classification/regression.
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Environment 
pd.set_option('display.max_columns', 125)

# Constants 
train_set = '../input/cat-in-the-dat-ii/train.csv'
test_set = '../input/cat-in-the-dat-ii/test.csv'
submission_set = '../input/cat-in-the-dat-ii/sample_submission.csv'
STANDARD_SCALER = True
OUTLIERS = True
DUMMIES = False
GRAFS = True
MAX = False

#print("The actual path : {}.'.format(os.getcwd()))
print('The project is set up.')
print('OPTION || SCALER = {}, DUMMIES = {}, GRAFS = {}, MAX = {}, OUTLIER = {}.'.format(STANDARD_SCALER, DUMMIES, GRAFS, MAX, OUTLIERS))


# In[ ]:


df_train = pd.read_csv(train_set, index_col='id')
df_test = pd.read_csv(test_set, index_col='id')
df_sub = pd.read_csv(submission_set)
n_rows, n_columns = df_train.shape
column_names = df_train.columns
print("Initialy, the train set is composed by {} and the test set by {}.".format(df_train.shape, (df_test.shape)))

targets = df_train.target
df_train = df_train.drop('target', axis=1)
if(df_train.shape[1] == df_test.shape[1]) :
    df = pd.concat([df_train, df_test])
print("Now, I have a set composed by {}.".format(df.shape))
df.head(5)


# In[ ]:


if(GRAFS):
    plt.figure(figsize=(13,7))
    plt.subplot(121)
    plt.title("Target distribution in the set")
    ax = sns.countplot(targets)
    for patch in ax.patches : 
        ax.text(patch.get_x() + patch.get_width()/3,
                patch.get_height()*1.02 ,
                "{}%".format(patch.get_height()/len(df_train) * 100))
    plt.show()


# In[ ]:


bins = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
for index, c_bin in enumerate(bins) :
    missing_values = df[c_bin].isnull().sum()
    if(df[c_bin].dtype == np.object) :
        if(c_bin == 'bin_3') : 
            df[c_bin] = df[c_bin].apply(lambda x: 0.0 if x == 'F' else 1.0)
        elif(c_bin == 'bin_4') : 
            df[c_bin] = df[c_bin].apply(lambda x: 0.0 if x == 'N' else 1.0)
        pd.to_numeric(df[c_bin])
    if(MAX) : 
        val = df[c_bin].value_counts().index[0]
    else : 
        val = df[c_bin].median()
    df[c_bin] = df[c_bin].fillna(val)
    print("From {} to {} missing values in the column : {} | filled value = {}.".format(missing_values, df[c_bin].isnull().sum(), c_bin, val))
print("Bins filled and encoded.")
if(GRAFS) :
    for c_bin in bins :        
        plt.figure(figsize=(13,7))
        plt.subplot(121)
        plt.title("Target distribution of "+c_bin)
        ax = sns.countplot(df[c_bin])
        for patch in ax.patches : 
            ax.text(patch.get_x() + patch.get_width()/3,
                    patch.get_height()*1.02 ,
                    "{}%".format(patch.get_height()/len(df) * 100))
        plt.show()


# In[ ]:


all_names =  ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', "nom_5", "nom_6", "nom_7", "nom_8", "nom_9"]   
cat_to_show = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
for index, c_name in enumerate(all_names) :
    missing_values = df[c_name].isnull().sum()
    if(MAX) : 
        val = df[c_name].value_counts().index[0]
    else : 
        val = df[c_name].mode()[0]
    df[c_name] = df[c_name].fillna(val)
    encoder = LabelEncoder()
    df[c_name] = encoder.fit_transform(df[c_name])
    print("From {} to {} missing values in the column : {} | filled value = {}.".format(missing_values, df[c_name].isnull().sum(), c_name, val))
print("Categorical's name columns filled and encoded.\n") 
if(GRAFS) :
    for column in cat_to_show :        
        plt.figure(figsize=(13,7))
        plt.subplot(121)
        plt.title("Target distribution of "+column)
        ax = sns.countplot(df[column])
        for patch in ax.patches : 
            ax.text(patch.get_x() + patch.get_width()/3,
                    patch.get_height()*1.02 ,
                    "{}%".format(patch.get_height()/len(df) * 100))
        plt.show()


# In[ ]:


ordinal_categories = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'] # Too many values in ord_5
for c_name in ordinal_categories : 
    missing_values = df[c_name].isnull().sum()
    if(c_name == 'ord_5') : 
        items_to_keep = list(df_train[c_name].value_counts()[:15].index)
        df[c_name] = df[c_name].apply(lambda x: 'Other' if x not in items_to_keep else x) 
    if(MAX) : 
        val = df[c_name].value_counts().index[0]
    else : 
        val = df[c_name].mode()[0]
    df[c_name] = df[c_name].fillna(val)
    encoder = LabelEncoder()
    df[c_name] = encoder.fit_transform(df[c_name])
    print("From {} to {} missing values in the column : {} | filled value = {}.".format(missing_values, df[c_name].isnull().sum(), c_name, val))
print("Ordinal columns filled.\n")
if(GRAFS) :
    for column in ordinal_categories :        
        plt.figure(figsize=(13,7))
        plt.subplot(121)
        plt.title("Target distribution of "+column)
        ax = sns.countplot(df[column])
        for patch in ax.patches : 
            ax.text(patch.get_x() + patch.get_width()/3,
                    patch.get_height()*1.02 ,
                    "{}%".format(patch.get_height()/len(df) * 100))
        plt.show()


# In[ ]:


datetime_categories = ['day', "month"] 
for c_name in datetime_categories : 
    missing_values = df[c_name].isnull().sum()
    if(MAX) : 
        val = df[c_name].value_counts().index[0]
    else : 
        val = df[c_name].mode()[0]
    df[c_name] = df[c_name].fillna(val)
    print("From {} to {} missing values in the column : {} | filled value = {}.".format(missing_values, df[c_name].isnull().sum(), c_name, val))
print("Day and month columns filled.\n")
if(GRAFS) :
    for column in datetime_categories :        
        plt.figure(figsize=(13,7))
        plt.subplot(121)
        plt.title("Target distribution of "+column)
        ax = sns.countplot(df[column])
        for patch in ax.patches : 
            ax.text(patch.get_x() + patch.get_width()/3,
                    patch.get_height(),
                    "{}%".format(patch.get_height()/len(df) * 100))
        plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if(STANDARD_SCALER) :\n    print(\'Normalizing the data with scaler.\')\n    scaler = StandardScaler()\n    df[df.columns] = scaler.fit_transform(df[df.columns])\nif(DUMMIES) :\n    print(\'Normalizing the data with dummies.\')\n    columns_to_keep = [\'bin_0\', \'bin_1\', \'bin_2\',  \'bin_3\', \'bin_4\', \'nom_5\', \'nom_6\', \'nom_7\', \'nom_8\',\'nom_9\']\n    df_tokeep = df[columns_to_keep]\n    df = df.drop(columns_to_keep, axis=1)\n    \n    for col in df.columns : \n        train = pd.get_dummies(df[col])\n        for column in train.columns : \n            df_tokeep[str(col)+\'_\'+str(column)] = train[column]\n    df = df_tokeep.copy()\ndf_train = df[:n_rows]\ndf_test = df[n_rows:]\nif(OUTLIERS) : # 1 on 23 -> 36% outliers / 3 on 23 -> >1%%\n    print("Detecting outliers")\n    start_time = time.time()\n    threshold = 1.5 \n    nb_detect = 3\n    Q1 = df_train.quantile(0.25)\n    Q3 = df_train.quantile(0.75)\n    IQR = Q3 - Q1\n    outliers = []\n    outlier_matrix = (df_train < (Q1 - threshold * IQR)) |(df_train > (Q3 + threshold * IQR))\n    for index_row, row in outlier_matrix.iterrows() :\n        cmpt = 0\n        for item in row : \n            if(item==True) : \n                cmpt += 1 \n                if(index_row not in outliers and cmpt>=nb_detect) :\n                    outliers.append(index_row)\n                    break\n    print("There are {}% of outliers (exactly {} outliers) in the train set.".format((len(outliers)/len(df_train)), len(outliers)))\n    print("It takes {} minutes to detect outliers.".format((time.time() - start_time)/60))\n    try :\n        print("{} outliers were dropped.".format(len(outliers)))\n        df_train = df_train.drop(outliers)\n        targets = targets.drop(outliers)\n    except : \n        print("Can\'t drop outliers")\nX_train, X_test, y_train, y_test = train_test_split(df_train, targets, test_size=0.1, random_state=42)\nprint("X_train : {}, X_test : {}, y_train : {}, y_test : {}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'models = [\n            {\'name\' : \'LogisticRegression\', \'model\' : LogisticRegression(), \'predictions\' : None, "auc" : 0.0, \'accuracy\' : 0.0, \'training_time\' : 0.0, \'prediction_time\' : 0.0},\n            {\'name\' : \'GradientBoostingClassifier\', \'model\' : GradientBoostingClassifier(), \'predictions\' : None, "auc" : 0.0, \'accuracy\' : 0.0, \'training_time\' : 0.0, \'prediction_time\' : 0.0},\n            {\'name\' : \'XGBClassifier\', \'model\' : xgb.XGBClassifier(), \'predictions\' : None, "auc" : 0.0, \'accuracy\' : 0.0, \'training_time\' : 0.0, \'prediction_time\' : 0.0},\n            {\'name\' : \'RandomForestClassifier\', \'model\' : RandomForestClassifier(), \'predictions\' : None, "auc" : 0.0, \'accuracy\' : 0.0, \'training_time\' : 0.0, \'prediction_time\' : 0.0}\n]\nfor index_model, model in enumerate(models) : \n    print("Model {} : {}.".format(index_model, model[\'name\']))\n    training_time = time.time()\n    model[\'model\'].fit(X_train, y_train)\n    model[\'training_time\'] = (time.time() - training_time)/60\n    print("Model {} : {} completed training.".format(index_model, model[\'name\']))\n    predict_time = time.time()\n    y_pred = model[\'model\'].predict_proba(X_test)[:,1]\n    pred = model[\'model\'].predict_proba(df_test)[:,1]\n    model[\'predictions\'] = pred\n    print("Model {} : {} completed predicting.".format(index_model, model[\'name\']))\n    model[\'auc\'] = roc_auc_score(y_test, y_pred) \n    model[\'accuracy\'] = accuracy_score(y_test, y_pred.round()) \n    model[\'prediction_time\'] = (time.time() - predict_time)/60\n    print("Model {} : {} reachs the end in {} minutes.".format(index_model, model[\'name\'], (time.time() - training_time)/60))')


# In[ ]:


if(GRAFS) : 
    df_results = {'model' : [], 'auc (%)' : [], 'accuracy (%)' : [], 'training_time (minutes)' : []}
    for model in models : 
        df_results['model'].append(model['name'])
        df_results['auc (%)'].append(model['auc'])
        df_results['accuracy (%)'].append(model['accuracy'])
        df_results['training_time (minutes)'].append(model['training_time'])
    df_results = pd.DataFrame(df_results)
    print(df_results)


# In[ ]:


try : 
    if(len(predictions_reg) == df_sub.shape[0]):
        df_sub.target = models[0]['predictions']
        df_sub.to_csv(submission_set, index=False)
else : 
    print("There is a problem uploading the results in the .csv file.")
print("Ending the kernel - Thank to read.")


# In[ ]:




