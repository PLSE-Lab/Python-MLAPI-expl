#!/usr/bin/env python
# coding: utf-8

# Hi there, this notebook aims to provide visualization about different features distribution and observing abnormal values in the dataset. 
# 
# # Table of Contents
# 1. Target Distribution
# 2. application_{train|test}.csv
#     * 2.1 NaN Count
#     * 2.2 [Feature Generation] Adding IS_NAN features for each column.
#     * 2.3 The importance of the missing values
#     * 2.4. Features' Distribution
#     * 2.5 Are the distributions making sense?
# 3. bureau.csv
# 4. ...pending
# 
# # 1. Target Distribution
# 
# 
# 

# In[1]:


from IPython.display import HTML
import pandas as pd
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import collections
from lightgbm import LGBMClassifier, plot_importance
import seaborn as snss
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 200})
get_ipython().run_line_magic('matplotlib', 'inline')
train_application_df = pd.read_csv('../input/application_train.csv')
test_application_df = pd.read_csv('../input/application_test.csv')
# print(train_application_df.shape, test_application_df.shape)
all_application_df = pd.concat([train_application_df, test_application_df], axis=0)
# print(all_application_df.shape)


# The target distribution is imbalanced, which indicates the company has already done a great job (some direct feature, like external scoring, in the dataset might not be that helpful)

# In[2]:


target_distribution = train_application_df['TARGET'].value_counts()
target_distribution.plot.pie(figsize=(10, 10),
                             title='Target Distribution',
                             fontsize=15, 
                             legend=True, 
                             autopct=lambda v: "{:0.1f}%".format(v))


# # 2. application_{train|test}.csv> 
# ##  2.1 NaN Count
# There are lots of NaN values in the dataset (also as discussed in the forum, the organizer also filled in some missing data with magic values). Need to handle them carefully.

# In[3]:


total_nans = all_application_df.isna().sum()
nan_precents = (all_application_df.isna().sum()/all_application_df.isna().count()*100)
feature_overview_df  = pd.concat([total_nans, nan_precents], axis=1, keys=['NaN Count', 'NaN Pencent'])
feature_overview_df['Type'] = [all_application_df[c].dtype for c in feature_overview_df.index]
pd.set_option('display.max_rows', None)
display(feature_overview_df)
pd.set_option('display.max_rows', 20)


# ## 2.2 [Feature Generation] Adding IS_NAN features for each column.

# In[4]:


all_application_is_nan_df = pd.DataFrame()
for column in all_application_df.columns:
    if all_application_df[column].isna().sum() == 0:
        continue
    all_application_is_nan_df['is_nan_'+column] = all_application_df[column].isna()
    all_application_is_nan_df['is_nan_'+column] = all_application_is_nan_df['is_nan_'+column].map(lambda v: 1 if v else 0)
all_application_is_nan_df['target'] = all_application_df['TARGET']
all_application_is_nan_df = all_application_is_nan_df[pd.notna(all_application_is_nan_df['target'])]


# In[5]:


display(all_application_is_nan_df)


# ## 2.3 The importance of the missing values

# In[6]:


Y = all_application_is_nan_df.pop('target')
X = all_application_is_nan_df

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2, random_state=2018)


# In[7]:


clf = LGBMClassifier(n_estimators=200, learning_rate=0.01)
clf.fit(
        train_X,
        train_Y,
        eval_set=[(train_X, train_Y), (valid_X, valid_Y)],
        eval_metric='auc',
        early_stopping_rounds=50,
        verbose=False
       )
plot_importance(clf, figsize=(10,10))


# ## 2.4 Features' Distribution

# In[12]:


# add noise to y axis to avoid overlapping
def rand_jitter(arr):
    nosie = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr))

def draw_feature_distribution(df, column):
    column_values = df[df[column].notna()][column]
    # group by target
    class_0_values = df[df[column].notna() & (df['TARGET']==0)][column]
    class_1_values = df[df[column].notna() & (df['TARGET']==1)][column]
    class_t_values = df[df[column].notna() & (df['TARGET'].isna())][column]        
    print('\n\n', column)
    # for features with unique values >= 10
    if len(df[column].value_counts().keys()) >= 10:
        fig, ax = plt.subplots(1, figsize=(15, 4))
        if df[column].dtype == 'object':
            label_encoder = LabelEncoder()
            label_encoder.fit(column_values)
            class_0_values = label_encoder.transform(class_0_values)
            class_1_values = label_encoder.transform(class_1_values)
            class_t_values = label_encoder.transform(class_t_values)
            column_values = label_encoder.transform(column_values)
            plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, fontsize=12, rotation='vertical')

        ax.scatter(class_0_values, rand_jitter([0]*class_0_values.shape[0]), label='Class0', s=10, marker='o', color='#7ac143', alpha=1)
        ax.scatter(class_1_values, rand_jitter([10]*class_1_values.shape[0]), label='Class1', s=10, marker='o', color='#fd5c63', alpha=1)
        ax.scatter(class_t_values, rand_jitter([20]*class_t_values.shape[0]), label='Test', s=10, marker='o', color='#037ef3', alpha=0.4)
        ax.set_title(column +' group by target', fontsize=16)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(column +' distribution', fontsize=16)
    else:      
        all_categories = list(df[df[column].notna()][column].value_counts().keys())
        bar_width = 0.25
        
        fig, ax = plt.subplots(figsize=(20, 4))
        ax.set_title(column, fontsize=16)
        plt.xlabel('Categories', fontsize=16)
        plt.ylabel('Counts', fontsize=16)

        value_counts = class_0_values.value_counts()
        x_0 = np.arange(len(all_categories))
        y_0 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_0, y_0, color='#7ac143', width=bar_width, label='class0')

        value_counts = class_1_values.value_counts()
        x_1 = np.arange(len(all_categories))
        y_1 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_1+bar_width, y_1, color='#fd5c63', width=bar_width, label='class1')
        
        value_counts = class_t_values.value_counts()
        x_2 = np.arange(len(all_categories))
        y_2 = [value_counts.get(categroy, 0) for categroy in all_categories]
        ax.bar(x_2+2*bar_width, y_2, color='#037ef3', width=bar_width, label='test')
        
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        
        for i, v in enumerate(y_0):
            if y_0[i]+y_1[i] == 0:
                ax.text(i - .08, max(y_0)//1.25,  'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i - .08, max(y_0)//1.25,  "{:0.1f}%".format(100*y_0[i]/(y_0[i]+y_1[i])), fontsize=14, rotation='vertical')
        
        for i, v in enumerate(y_1):
            if y_0[i]+y_1[i] == 0:
                ax.text(i - .08, max(y_0)//1.25,  'Missing in Train', fontsize=14, rotation='vertical')
            else:
                ax.text(i + bar_width - .08, max(y_0)//1.25, "{:0.1f}%".format(100*y_1[i]/(y_0[i]+y_1[i])), fontsize=14, rotation='vertical')
 
        for i, v in enumerate(y_2):
            if y_2[i] == 0:
                ax.text(i + 2*bar_width - .08, max(y_0)//1.25, 'Missing in Test', fontsize=14, rotation='vertical')
            else:
                ax.text(i + 2*bar_width - .08, max(y_0)//1.25, str(y_2[i]), fontsize=14, rotation='vertical')
        
        plt.xticks(x_0 + 2*bar_width/3, all_categories, fontsize=16)
        
    plt.show()


# In[13]:


print("only showing the distribution for the first few columns, edit the counter to show all distribution")
show_feature_count = 10
for column in all_application_df.columns:
    if show_feature_count == 0:
        break
    show_feature_count -= 1
    draw_feature_distribution(all_application_df, column)


# ## 2.5 Are the distributions making sense?
# ### 2.5.1 DAYS_EMPLOYED
# How many days before the application the person started current employment
# 
# Discussed in the forum, For DAYS_xxx columns, **365243 means missing value**.
# * The original distribution:

# In[14]:


draw_feature_distribution(all_application_df, 'DAYS_EMPLOYED')


# If the magic number is removed, the distribution:

# In[15]:


# the organizer used 365243 to represent missing value in this column
temp_df = all_application_df[all_application_df['DAYS_EMPLOYED'] != 365243]
draw_feature_distribution(temp_df, 'DAYS_EMPLOYED')


# ### 2.5.2 AMT_INCOME_TOTAL
# Income of the client.
# 
# There are a huge number at the right of the plot (1.170000e+08):

# In[16]:


print(all_application_df['AMT_INCOME_TOTAL'].describe())
draw_feature_distribution(all_application_df, 'AMT_INCOME_TOTAL')


# The plot makes more sense if we remove that data point:

# In[17]:


temp_df = all_application_df[all_application_df['AMT_INCOME_TOTAL'] != 1.170000e+08]
draw_feature_distribution(temp_df, 'AMT_INCOME_TOTAL')


# ### 2.5.3 AMT_REQ_CREDIT_BUREAU_QRT
# Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)
# 
# Why were there 261 enquireies about a application within 2 months? 4 calls a day?

# In[18]:


print(all_application_df['AMT_REQ_CREDIT_BUREAU_QRT'].describe())
draw_feature_distribution(all_application_df, 'AMT_REQ_CREDIT_BUREAU_QRT')


# Removing that data point:

# In[19]:


temp_df = all_application_df[all_application_df['AMT_REQ_CREDIT_BUREAU_QRT'] != 261]
draw_feature_distribution(temp_df, 'AMT_REQ_CREDIT_BUREAU_QRT')


# ### 2.5.4 Normalized information about building where the client lives
# 
# Why those Normalized information got many 0s and 1s? such as this one:

# In[20]:


draw_feature_distribution(all_application_df, 'NONLIVINGAPARTMENTS_MODE')


# ### 2.5.5 OBS_30_CNT_SOCIAL_CIRCLE
# How many observation of client's social surroundings with observable 30 DPD (days past due) default
# 
# Is it normal to have over 350 social surroundings overations?

# In[21]:


draw_feature_distribution(all_application_df, 'OBS_30_CNT_SOCIAL_CIRCLE')


# # 3 bureau.csv
# 
# Pending

# Thanks for reading, this is my first attempt to try making a relatively complete EDA & visualization. Please let me know if you have any suggestions, I am desired to learn new things.

# In[ ]:




