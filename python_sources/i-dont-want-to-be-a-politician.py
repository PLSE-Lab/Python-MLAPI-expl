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

import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.


# In[ ]:


# data = pd.read_csv('/kaggle/input/congressional-voting-records/house-votes-84.names', error_bad_lines=False)
data = pd.read_csv('/kaggle/input/congressional-voting-records/house-votes-84.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.dtypes


# ## All the variables are categorical

# ## Lets examine them

# ### 1. Target

# In[ ]:


data['Target'] = np.where(data['Class Name'] == 'democrat', 1, 0)


# In[ ]:


data.Target.value_counts()


# We have 61 percent of Dems here

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Class Name',data=data)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)
plt.tight_layout()


# ### This imbalance is fair enough. But the size of the dataset is pretty small so it might cause some bias at the end but lets see.

# ### 2. handicapped-infants 

# In[ ]:


data[' handicapped-infants'].value_counts()


# In[ ]:


plt.figure(figsize = (20,10))
sns.set(style="darkgrid")
ax = sns.countplot(x=' handicapped-infants',data=data)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)
plt.tight_layout()


# Lets see what are the chances of them being belonging to DEMS

# In[ ]:


data.groupby(' handicapped-infants')['Target'].mean().sort_values(ascending = False)


# ### So this varible presumes that the chances of people with handicapped infants of belonging to Democrats are 83 percent. 

# In[ ]:


data['handicapped_infants_n'] = np.where(data[' handicapped-infants'] == 'n', 1, 0)
data['handicapped_infants_y'] = np.where(data[' handicapped-infants'] == 'y', 1, 0)


# ###  water-project-cost-sharing

# In[ ]:


data[' water-project-cost-sharing'].value_counts()


# In[ ]:


data.groupby(' water-project-cost-sharing')['Target'].mean().sort_values(ascending = False)


# ###  This one's pretty stable variable.

# In[ ]:


data['water-project-cost-sharing_n'] = np.where(data[' water-project-cost-sharing'] == 'n', 1, 0)
data['water-project-cost-sharing_y'] = np.where(data[' water-project-cost-sharing'] == 'y', 1, 0)


# ###  adoption-of-the-budget-resolution

# In[ ]:


data[' adoption-of-the-budget-resolution'].value_counts()


# In[ ]:


data.groupby(' adoption-of-the-budget-resolution')['Target'].mean().sort_values(ascending = False)


# This is an absolute democrat thing

# In[ ]:


data['adoption-of-the-budget-resolution_n'] = np.where(data[' adoption-of-the-budget-resolution'] == 'n', 1, 0)
data['adoption-of-the-budget-resolution_y'] = np.where(data[' adoption-of-the-budget-resolution'] == 'y', 1, 0)


# ### physician-fee-freeze

# In[ ]:


data[' physician-fee-freeze'].value_counts()


# In[ ]:


data.groupby(' physician-fee-freeze')['Target'].mean().sort_values(ascending = False)


# This is a total republic thing by a landslide

# In[ ]:


data['physician-fee-freeze_n'] = np.where(data[' physician-fee-freeze'] == 'n', 1, 0)
data['physician-fee-freeze_y'] = np.where(data[' physician-fee-freeze'] == 'y', 1, 0)


# In[ ]:


data.columns


# In[ ]:


cols_to_be_dropped = ['Class Name', ' handicapped-infants', ' water-project-cost-sharing',
       ' adoption-of-the-budget-resolution', ' physician-fee-freeze',
       ' el-salvador-aid', ' religious-groups-in-schools',
       ' anti-satellite-test-ban', ' aid-to-nicaraguan-contras', ' mx-missile',
       ' immigration', ' synfuels-corporation-cutback', ' education-spending',
       ' superfund-right-to-sue', ' crime', ' duty-free-exports',
       ' export-administration-act-south-africa']

#We will drop these cols at the end of our analysis


# ###  el-salvador-aid

# In[ ]:


data[' el-salvador-aid'].value_counts()


# In[ ]:


data.groupby(' el-salvador-aid')['Target'].mean()


# This can be said more or less what favored by both the parties

# In[ ]:


data['el-salvador-aid_n'] = np.where(data[' el-salvador-aid'] == 'n', 1, 0)
data['el-salvador-aid_y'] = np.where(data[' el-salvador-aid'] == 'y', 1, 0)


# ### religious-groups-in-schools

# In[ ]:


data[' religious-groups-in-schools'].value_counts()


# In[ ]:


data.groupby(' religious-groups-in-schools')['Target'].mean()


# Favored by republican more and also by the newutral parties.

# In[ ]:


data['religious-groups-in-schools_n'] = np.where(data[' religious-groups-in-schools'] == 'n', 1, 0)
data['religious-groups-in-schools_y'] = np.where(data[' religious-groups-in-schools'] == 'y', 1, 0)


# ###  anti-satellite-test-ban

# In[ ]:


data[' anti-satellite-test-ban'].value_counts()


# In[ ]:


data.groupby(' anti-satellite-test-ban')['Target'].mean()


# In[ ]:


data['anti-satellite-test-ban_n'] = np.where(data[' anti-satellite-test-ban'] == 'n', 1, 0)
data['anti-satellite-test-ban_y'] = np.where(data[' anti-satellite-test-ban'] == 'y', 1, 0)


# ###  aid-to-nicaraguan-contras

# In[ ]:


data[' aid-to-nicaraguan-contras'].value_counts()


# In[ ]:


data.groupby(' aid-to-nicaraguan-contras')['Target'].mean()


# Massively favored by dems

# In[ ]:


data['aid-to-nicaraguan-contras_n'] = np.where(data[' aid-to-nicaraguan-contras'] == 'n', 1, 0)
data['aid-to-nicaraguan-contras_y'] = np.where(data[' aid-to-nicaraguan-contras'] == 'y', 1, 0)


# ###  mx-missile
# 
# I think this should be a Republican favored thing lets see

# In[ ]:


data[' mx-missile'].value_counts()


# In[ ]:


data.groupby(' mx-missile')['Target'].mean()


# Ohh!! an absolute contrast here

# In[ ]:


data['mx-missile_n'] = np.where(data[' mx-missile'] == 'n', 1, 0)
data['mx-missile_y'] = np.where(data[' mx-missile'] == 'y', 1, 0)


# ### immigration

# In[ ]:


data[' immigration'].value_counts()


# In[ ]:


data.groupby(' immigration')['Target'].mean()


# In[ ]:


data['immigration_n'] = np.where(data[' immigration'] == 'n', 1, 0)
data['immigration_y'] = np.where(data[' immigration'] == 'y', 1, 0)


# ### synfuels-corporation-cutback

# In[ ]:


data[' synfuels-corporation-cutback'].value_counts()


# In[ ]:


data.groupby(' synfuels-corporation-cutback')['Target'].mean()


# Another huge support by Dems

# In[ ]:


data['synfuels-corporation-cutback_n'] = np.where(data[' synfuels-corporation-cutback'] == 'n', 1, 0)
data['synfuels-corporation-cutback_y'] = np.where(data[' synfuels-corporation-cutback'] == 'y', 1, 0)


# ###  education-spending

# In[ ]:


data[' education-spending'].value_counts()


# In[ ]:


data.groupby(' education-spending')['Target'].mean()


# WOW!! Dems dont support education spending much !! Interesting.

# In[ ]:


data['education-spending_n'] = np.where(data[' education-spending'] == 'n', 1, 0)
data['education-spending_y'] = np.where(data[' education-spending'] == 'y', 1, 0)


# ### superfund-right-to-sue'

# In[ ]:


data[' superfund-right-to-sue'].value_counts()


# In[ ]:


data.groupby(' superfund-right-to-sue')['Target'].mean()


# In[ ]:


data['superfund-right-to-sue_n'] = np.where(data[' superfund-right-to-sue'] == 'n', 1, 0)
data['superfund-right-to-sue_y'] = np.where(data[' superfund-right-to-sue'] == 'y', 1, 0)


# ### crime
# 
# This should be interesting

# In[ ]:


data[' crime'].value_counts()


# In[ ]:


data.groupby(' crime')['Target'].mean()


# Rebulicans 98 percent support to tackle crime while DEMS 36 percent. People of US, yall need to see this man!

# In[ ]:


data['crime_n'] = np.where(data[' crime'] == 'n', 1, 0)
data['crime_y'] = np.where(data[' crime'] == 'y', 1, 0)


# In[ ]:


data[' duty-free-exports'].value_counts()


# In[ ]:


data.groupby(' duty-free-exports')['Target'].mean()


# In[ ]:


data['duty-free-exports_n'] = np.where(data[' duty-free-exports'] == 'n', 1, 0)
data['duty-free-exports_y'] = np.where(data[' duty-free-exports'] == 'y', 1, 0)


# In[ ]:


data[' export-administration-act-south-africa'].value_counts()


# In[ ]:


data.groupby(' export-administration-act-south-africa')['Target'].mean()


# In[ ]:


data['export-administration-act-south-africa_n'] = np.where(data[' export-administration-act-south-africa'] == 'n', 1, 0)
data['export-administration-act-south-africa_y'] = np.where(data[' export-administration-act-south-africa'] == 'y', 1, 0)


# #### Analysis completed. Lets move towards making a model

# In[ ]:


data = data.drop(cols_to_be_dropped, axis = 1)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train, test = train_test_split(data, test_size = 0.35, random_state = 42)


# In[ ]:


train.shape, test.shape


# In[ ]:


X_train = train.drop('Target', axis = 1)
X_test = test.drop('Target', axis = 1)
y_train = train['Target']
y_test = test['Target']


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### Lets try Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# ##### Gini

# In[ ]:


def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area


# ##### Confusion Matrix

# In[ ]:


def plot_confusion_matrix(y_true, y_pred, title = 'Confusion matrix', cmap=plt.cm.Blues):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    print ('Classification Report:\n')
    print (classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    def plot_confusion_matrix_plot(cm, title = 'Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(y_test.unique()))
        plt.xticks(tick_marks, rotation=45)
        plt.yticks(tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    print (cm)
    plot_confusion_matrix_plot(cm=cm)


# In[ ]:


rf = RandomForestClassifier(criterion = 'gini', 
                            max_depth = 8,
                            max_features = 'auto',
                            min_samples_leaf = 0.01, 
                            min_samples_split = 0.01,
                            min_weight_fraction_leaf = 0.0632, 
                            n_estimators = 1000,
                            random_state = 50, 
                            warm_start = False)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


pred = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


accuracy_score(pred, y_test)*100


# In[ ]:


plot_confusion_matrix(y_test, pred)


# #### Looks like our model scored pretty well on precision and recall areas

# ### Lets see gini

# In[ ]:


predicted_probs = rf.predict_proba(X_test)


# In[ ]:


gini(predicted_probs[:,1])


# ### Gini is slightly lower but it can be improved by changing some hyperparameters or using other powerful algos like XGBoost or GBM Classifiers.

# ### Lets see if the model overfits

# In[ ]:


predicted_probs_train = rf.predict_proba(X_train)


# In[ ]:


gini(predicted_probs_train[:,1])


# ### Looks like we are slightly overfitting

# ### Lets see the confusion matrix

# In[ ]:


predicted_train = rf.predict(X_train)


# In[ ]:


accuracy_score(predicted_train, y_train)


# In[ ]:


plot_confusion_matrix(y_train, predicted_train)


# ### Well the confusion matrix doesnt suggest so. If anything, the model is performing better on the test set.

# ###### However, this can be emended by tuning the hyperparameters. I am using very strong parameters and you can lower down the points on each parameter. This would definitely dip the training time. Nevertheless, the model performs fantastic.

# ### What do you think. Which party is doing better and how good is the presentation. Please upvote if you like it. I am open for suggestions and criticism. Throw all the flak at me.
