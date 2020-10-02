#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # **Load dataset**

# In[ ]:


df_customer_train = pd.read_csv("/kaggle/input/customer_train.csv")
df_customer_test = pd.read_csv("/kaggle/input/customer_test.csv")

df_stories_reaction_train = pd.read_csv("/kaggle/input/stories_reaction_train.csv")
df_stories_reaction_test = pd.read_csv("/kaggle/input/stories_reaction_test.csv")

df_stories_description = pd.read_csv("/kaggle/input/stories_description.csv")


# # **Data exploration**

# In[ ]:


df_customer_train.head()


# In[ ]:


df_customer_test.head()


# In[ ]:


df_stories_reaction_train.head()


# In[ ]:


df_stories_reaction_test.head()


# In[ ]:


df_stories_description.head()


# ### extract from the 'story_json' column in the 'df_stories_description' a field called 'name' to use later in our prediction

# In[ ]:


for index, row in df_stories_description.iterrows():
    i = row['story_json'].find("name")
    j = row['story_json'][i+7:].find('"')
    df_stories_description.at[index,'story_json'] = row['story_json'][i+7:][:j]


# ## label encode all columns in df_stories_description
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical = list(df_stories_description.select_dtypes(include=['object']).columns.values)
for cat in categorical:
    df_stories_description[cat]=df_stories_description[cat].astype('str')
    df_stories_description[cat] = le.fit_transform(df_stories_description[cat])
    
for colName in df_stories_description.columns:
    if(colName != "story_id"):
        df_stories_description[colName] = pd.qcut(df_stories_description[colName].rank(method="first"), 10, labels=False)


# ## Construct the final train and test set

# In[ ]:


df_first_merged = pd.merge(df_customer_train, df_stories_reaction_train, on='customer_id')
df_final = pd.merge(df_first_merged, df_stories_description, on='story_id')
df_final = df_final.drop(['first_session_dttm', 'event_dttm', 'story_id', 'customer_id'], axis=1)

for index, row in df_final.iterrows():
    if(row['event'] == 'dislike'):
        df_final.at[index,'event'] = -1
    if(row['event'] == 'skip'):
        df_final.at[index,'event'] = -0.1
    if(row['event'] == 'view'):
        df_final.at[index,'event'] = 0.1
    if(row['event'] == 'like'):
        df_final.at[index,'event'] = 1


# In[ ]:


df_first_merged_test = pd.merge(df_customer_test, df_stories_reaction_test, on='customer_id')
df_final_test = pd.merge(df_first_merged_test, df_stories_description, on='story_id')
df_final_test = df_final_test.drop(['first_session_dttm', 'event_dttm', 'story_id', 'customer_id'], axis=1)


# In[ ]:


df_final.head()


# In[ ]:


df_final.describe()


# In[ ]:


df_final.info()


# ### Numeric columns

# In[ ]:


df_final.select_dtypes(exclude=['object']).dtypes


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2)
for i, column in enumerate(df_final.select_dtypes(exclude=['object']).dropna().columns):
    g = sns.barplot(x=df_final.select_dtypes(exclude=['object']).dropna()[column],y=df_final['event'], ax=axes[i//2,i%2])
    g.set_xticklabels(g.get_xticklabels(), rotation=70)
plt.tight_layout()


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2)
for i, column in enumerate(df_final.select_dtypes(exclude=['object']).dropna().columns):
    g = sns.countplot(df_final.select_dtypes(exclude=['object']).dropna()[column],ax=axes[i//2,i%2])
    g.set_xticklabels(g.get_xticklabels(), rotation=70)
plt.tight_layout()


# ### Conclusion
# #### Age
# - Make new column from that with labels like 'young', 'medium' and 'old' (and then use one-hot encoding)
# - Fill nans with mean calculated on test+train set
# 
# #### Children_cnt
# - Make new column from that with labels like 'small', 'medium' and 'large' plus 'missing'
# - Fill nans with 'missing' on test+train set
# 
# #### Job_position_cd
# - Do nothing
# 
# #### Story_json
# - Drop

# In[ ]:


df_final.select_dtypes(include=['object']).dtypes


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='product_0', y='event', data=df_final, ax=axs[0])
sns.countplot(x='product_0', data=df_final, ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='product_1', y='event', data=df_final, ax=axs[0])
sns.countplot(x='product_1', data=df_final, ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='product_2', y='event', data=df_final, ax=axs[0])
sns.countplot(x='product_2', data=df_final, ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='product_3', y='event', data=df_final, ax=axs[0])
sns.countplot(x='product_3', data=df_final, ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='product_4', y='event', data=df_final, ax=axs[0])
sns.countplot(x='product_4', data=df_final, ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='product_5', y='event', data=df_final, ax=axs[0])
sns.countplot(x='product_5', data=df_final, ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='product_6', y='event', data=df_final, ax=axs[0])
sns.countplot(x='product_6', data=df_final, ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='marital_status_cd', y='event', data=df_final, ax=axs[0])
sns.countplot(x='marital_status_cd', data=df_final, ax=axs[1])
plt.tight_layout()


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
encoded_clmn = le.fit_transform(df_final['job_title'].fillna('missing'))

d = {'job_title': encoded_clmn, 'event': df_final['event']}
df_for_lineplot = pd.DataFrame(data=d)


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.lineplot(x='event', y='job_title', data=df_for_lineplot, ax=axs[0])
sns.distplot(encoded_clmn, ax=axs[1])
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(ncols=2)
sns.barplot(x='gender_cd', y='event', data=df_final, ax=axs[0])
sns.countplot(x='gender_cd', data=df_final, ax=axs[1])
plt.tight_layout()


# ### Conclusion
# #### Product_0
# - Use with one-hot encoding
# - nan convert to 'missing'
# 
# #### Product_1
# - nan convert to 'missing'
# - Use with one-hot encoding
# 
# #### Product_2
# - nan convert to 'missing'
# - Use with one-hot encoding
# 
# #### Product_3
# - nan convert to 'missing'
# - Use with one-hot encoding
# 
# #### Product_4
# - nan convert to 'missing'
# - Use with one-hot encoding
# 
# #### Product_5
# - nan convert to 'missing'
# - Use with one-hot encoding
# 
# #### Product_6
# - nan convert to 'missing'
# - Use with one-hot encoding
# 
# #### gender_cd
# - nan convert to 'missing'
# - Use with one-hot encoding
# 
# #### job_title
# - nan convert to 'missing'
# - Use with count-binning
# 
# #### marital_status_cd
# - nan convert to 'missing'
# - Use with one-hot encoding

# # Data preparation

# #### Numeric data preparation

# ##### *Age*

# In[ ]:


df_age = df_final_test['age'] + df_final['age']
df_age.fillna(df_age.mean(), inplace=True)

df_final['age'].fillna(df_age.mean(), inplace=True)
df_final_test['age'].fillna(df_age.mean(), inplace=True)

df_final['age'] = pd.cut(df_final['age'], 3, labels=["young", "medium", "old"])
df_final_test['age'] = pd.cut(df_final_test['age'], 3, labels=["young", "medium", "old"])


# In[ ]:


g = sns.barplot(x=df_final['age'],y=df_final['event'])


# In[ ]:


df_final = pd.concat([df_final, pd.get_dummies(df_final['age'])], axis=1, sort=False).drop(['age'], axis=1)
df_final


# In[ ]:


df_final_test = pd.concat([df_final_test, pd.get_dummies(df_final_test['age'])], axis=1, sort=False).drop(['age'], axis=1)
df_final_test


# ##### *Children_cnt*

# In[ ]:


df_final['children_cnt'].fillna(-1, inplace=True)
df_final_test['children_cnt'].fillna(-1, inplace=True)

df_final['children_cnt'] = pd.cut(df_final['children_cnt'], 4, labels=["missing", "small", "medium", "big"])
df_final_test['children_cnt'] = pd.cut(df_final_test['children_cnt'], 4, labels=["missing", "small", "medium", "big"])


# In[ ]:


df_final = pd.concat([df_final, pd.get_dummies(df_final['children_cnt'])], axis=1, sort=False).drop(['children_cnt'], axis=1)
df_final


# In[ ]:


df_final_test = pd.concat([df_final_test, pd.get_dummies(df_final_test['children_cnt'])], axis=1, sort=False).drop(['children_cnt'], axis=1)
df_final_test


# ##### *Story_json*

# In[ ]:


df_final = df_final.drop(['story_json'], axis=1)
df_final


# In[ ]:


df_final_test = df_final_test.drop(['story_json'], axis=1)
df_final_test


# #### Object data preparation

# ##### Product (0-6)

# In[ ]:


for i in range(7):
    df_final['product_'+str(i)].fillna('missing', inplace=True)
    df_final_test['product_'+str(i)].fillna('missing', inplace=True)


# In[ ]:


df_final


# In[ ]:


for i in range(7):
    df_final = pd.concat([df_final, pd.get_dummies(df_final['product_'+str(i)])], axis=1, sort=False).drop(['product_'+str(i)], axis=1)


# In[ ]:


df_final


# In[ ]:


for i in range(7):
    df_final_test = pd.concat([df_final_test, pd.get_dummies(df_final_test['product_'+str(i)])], axis=1, sort=False).drop(['product_'+str(i)], axis=1)


# In[ ]:


df_final_test


# ##### Gender_cd

# In[ ]:


df_final['gender_cd'].fillna('missing', inplace=True)
df_final_test['gender_cd'].fillna('missing', inplace=True)


# In[ ]:


df_final = pd.concat([df_final, pd.get_dummies(df_final['gender_cd'])], axis=1, sort=False).drop(['gender_cd'], axis=1)
df_final_test = pd.concat([df_final_test, pd.get_dummies(df_final_test['gender_cd'])], axis=1, sort=False).drop(['gender_cd'], axis=1)


# In[ ]:


df_final


# In[ ]:


df_final_test


# ##### Marital_status_cd

# In[ ]:


df_final['marital_status_cd'].fillna('missing', inplace=True)
df_final_test['marital_status_cd'].fillna('missing', inplace=True)


# In[ ]:


df_final = pd.concat([df_final, pd.get_dummies(df_final['marital_status_cd'])], axis=1, sort=False).drop(['marital_status_cd'], axis=1)
df_final_test = pd.concat([df_final_test, pd.get_dummies(df_final_test['marital_status_cd'])], axis=1, sort=False).drop(['marital_status_cd'], axis=1)


# In[ ]:


df_final_test['DLW'] = 0


# In[ ]:


df_final


# In[ ]:


df_final_test


# ##### Job_title

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit_transform(pd.concat([df_final['job_title'].fillna('missing'), df_final_test['job_title'].fillna('missing')]))

df_final['job_title'] = le.transform(df_final['job_title'].fillna('missing'))
df_final_test['job_title'] = le.transform(df_final_test['job_title'].fillna('missing'))


# In[ ]:


df_final


# In[ ]:


df_final_test


# ## Normalize the train and test dataframes

# In[ ]:


#df_final = df_final.sample(n=10000, random_state=1)    
label = df_final['event']
normalized_df_final = (df_final - df_final.min()) / (df_final.max() - df_final.min())
normalized_df_final.fillna(0, inplace=True)


# In[ ]:


normalized_df_final_test = (df_final_test - df_final_test.min()) / (df_final_test.max() - df_final_test.min())
normalized_df_final_test.fillna(0, inplace=True)


# # Modelling

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(normalized_df_final.drop(['event'], axis=1), label.astype('str'))


# 

# In[ ]:


prediction = clf.predict(normalized_df_final_test.drop(['answer_id'], axis=1))


# In[ ]:


np.unique(prediction)


# # Submission

# In[ ]:


result = pd.concat([df_final_test['answer_id'], pd.DataFrame(list(prediction))], axis=1, sort=False)


# In[ ]:


final_result = result.sort_values(by=['answer_id'])


# In[ ]:


sequence = range(0, 172049)
df = pd.merge(pd.DataFrame(list(sequence), columns=['answer_id']), final_result, on='answer_id', how='left')
df = df.fillna('0.0')
df.to_csv("prediction.csv", index=False, float_format='%.2f')

