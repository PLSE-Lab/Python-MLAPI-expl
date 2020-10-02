#!/usr/bin/env python
# coding: utf-8

# # Bank marketing data undersampling
# 
# this dataset was acquired from: <br>
# <br>
# 
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing  <br>
# <br>
# This is the full dataset, which has more features and a lot of class imbalance. <br>
# 
# I'll resample the data using PCA, because I want to undersample the majority class as uniformly as possible.

# Attribute Information:
# 
# Input variables:
# # bank client data:
# 1 - age (numeric)  <br>
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')  <br>
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)  <br>
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')  <br>
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')  <br>
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')  <br>
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')  <br>
# # related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: 'cellular','telephone')  <br> 
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  <br>
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')  <br>
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# # other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  <br>
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  <br>
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)  <br>
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')  <br>
# # social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)  <br>
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)   <br>
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)   <br>
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)  <br>
# 20 - nr.employed: number of employees - quarterly indicator (numeric)  <br>
# 
# # Output variable (desired target):  <br>
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math

pd.options.display.max_columns = None


# In[ ]:


get_ipython().system('ls ../input/')


# In[ ]:


data = pd.read_csv('../input/bank-additional-full.csv', delimiter=';')
data.tail()


# In[ ]:


'''
Renaming columns just for better understanding
'''

data.rename(columns={'housing':'housing_loan'}, inplace=True)


# In[ ]:


data.info()


# In[ ]:


data.dtypes


# In[ ]:


'''
The column duration is not supposed to be used, since we only know the call duration
after it has been finished, so it is considered data leakage
'''

data.drop(['duration'], axis=1, inplace=True)


# In[ ]:


catg_cols = ['job', 'marital', 'education', 'default', 'housing_loan',
             'loan', 'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous',
             'poutcome']


num_cols = ['age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
            'euribor3m', 'nr.employed']


# In[ ]:


data.describe()


# In[ ]:


'''
I just want to see how many of each feature there is in each column

'''


unique_df = pd.DataFrame()

index=0
for column in catg_cols:
    
    unique_df = pd.concat([unique_df,pd.DataFrame(data[column].value_counts()).reset_index()], axis=1)
    unique_df[index] = '........'
    index+=1

unique_df.fillna('')


# ### Some of these sparse categorical columns like "'campaign', 'pdays' and 'previous'" are going to be grouped

# In[ ]:


def plot_bar(col, size=(15,8), hue=True):
    fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    if hue:
        sns.countplot(col, hue='y', data=data)
    else:
        sns.countplot(col, data=data)
        
    fig.set_size_inches(size)
    plt.xlabel(col) # Set text for the x axis
    plt.ylabel('Count')# Set text for y axis
    plt.show()
    


# ### Target balance

# In[ ]:


plot_bar('y', hue=False)


# #### Very imbalanced data, we'll need to think of undersampling strategies

# ### Campaign

# In[ ]:


plot_bar('campaign')


# In[ ]:


'''
I'm gonna group more than 5 contacts all into one category
'''

data['campaign'] = data['campaign'].apply(lambda x: x if x<5 else 5) # More than 5 contacts are grouped


# In[ ]:


plot_bar('campaign')


# ### Pdays

# In[ ]:


plot_bar('pdays')


# In[ ]:


sns.countplot(data[data['pdays']!=999]['pdays'], hue=data['y'])


# In[ ]:


'''
Going to group them into 3 categories

1 = equal or less than 10 days

2 = more than 10 days

0 = not contacted before

'''


def treat_pdays(value):
    
    if value <= 10:
        return 1
    if value > 10 and value <= 27:
        return 2
    if value > 27:
        return 0

data['pdays'] = data['pdays'].apply(treat_pdays)


# In[ ]:


plot_bar('pdays')


# ### Previous

# In[ ]:


plot_bar('previous')


# In[ ]:


'''
Going to group them into 3 categories

1 = contacted once before

2 = contacted more than once before

0 = not contacted before

'''


def treat_previous(value):
    
    if value == 0:
        return 0
    if value == 1:
        return 1
    else:
        return 2


# In[ ]:


data['previous'] = data['previous'].apply(treat_previous)


# In[ ]:


plot_bar('previous')


# ### Job

# In[ ]:


plot_bar('job')


# In[ ]:


'''
Merging housemaid into serices
'''

data['job'] = data['job'].replace('housemaid', 'services')


# In[ ]:


plot_bar('job')


# ## Resampling Data
# 
# I'll try to resample data using PCA, the data is too inbalanced. <br>
# I want to resample the majority class as uniformly as possible. <br>
# So, a good approach might be clustering the PCA components and taking equal samples from each cluster.

# In[ ]:


'''
Getting dummies for the categorical columns
'''


dummy_features = pd.get_dummies(data[catg_cols])

num_features = data[num_cols]

print(dummy_features.shape)
print(num_features.shape)


# In[ ]:


'''
Scaling the numerical variables

'''

scaler = StandardScaler()

num_features = pd.DataFrame(scaler.fit_transform(num_features), columns=num_features.columns)


# In[ ]:


'''
Concatenating the scaled numerical columns with
the dummy columns
'''


preprocessed_df = pd.concat([dummy_features, num_features], axis=1)
preprocessed_df.shape


# In[ ]:


'''
Binarizing 'yes' and 'no'
values in the labels
'''

labels = data['y'].map({'no':0, 'yes':1})


# In[ ]:


pca = PCA(n_components=2)
pcs = pca.fit_transform(preprocessed_df)

pcs_df = pd.DataFrame(pcs)


# In[ ]:


def plot_2d(X, y, label='Classes'):   
 
    for l in zip(np.unique(y)):
        plt.scatter(X[y==l, 0], X[y==l, 1], label=l)
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


# In[ ]:


plot_2d(pcs, labels)


# ### Now clustering the PCA components

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


'''
Manually setting initial cluster centers
for improved accuracy
'''
n_clusters = 15 # 15 clusters from visual inspection above

cluster_centers = np.array([[-1.5,-1.5], [-1.6,-0.5], [-1.7,0.5], [-1.9,1.5], [-2,2.5],
                            [0.5,-1], [0,0], [0,1], [-0.2,2], [-0.5, 2.8],
                            [3,-1], [3,0], [2.5,1.1], [2.5,2], [2.5,3.2]])


# In[ ]:


kmeans = KMeans(n_clusters=n_clusters, max_iter=10000, verbose=1, n_jobs=4, init=cluster_centers)

clusters = kmeans.fit_predict(pcs_df)


# In[ ]:


pcs_df['cluster'] = clusters


# In[ ]:


'''
We can see that the clustering is acceptable
'''


plt.scatter(pcs_df[0], pcs_df[1], c=pcs_df['cluster'])


# In[ ]:


labels.value_counts()


# In[ ]:


'''
I'll extract 309 samples from each cluster
'''

n_samples = labels.value_counts()[1]//15
n_samples


# In[ ]:


'''
I'll select 309 random points from each cluster
But im only sampling the majority or label == 0

'''


index_list = []

for i in range(0,n_clusters):
    
    choices = pcs_df[(labels==0) & (pcs_df['cluster'] == i)].index
    
   
    
    index_list.append(np.random.choice(choices, n_samples))

    
index_list = np.ravel(index_list)


# In[ ]:


'''
Creating a new Dataframe with all the samples from the index_list which are all from the majority class
and all the samples from the minority class
'''

resampled_raw_data = pd.concat([data.iloc[index_list], data[data['y'] == 'yes']])


# In[ ]:


'''
Confirming concatenation
'''

resampled_raw_data.shape


# In[ ]:


'''
Confirming class imbalance
'''

resampled_raw_data['y'].value_counts()


# In[ ]:


'''
Saving resampled Dataframe for future classification task
'''


resampled_raw_data.to_csv('resampled_bank_data.csv')


# In[ ]:




