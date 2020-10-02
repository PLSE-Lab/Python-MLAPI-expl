#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


m_stroke_df = pd.read_csv("../input/healthcare-dataset-stroke-data/train_2v.csv")


# In[ ]:


m_stroke_df.head()


# In[ ]:


m_stroke_df.describe()


# **Lets check null values**

# In[ ]:


sns.heatmap(m_stroke_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title("Missing values")


# **here we can see the missing values in columns (bmi,smoking_status). To do any further analysis we should handeled these missing values**

# In[ ]:


m_stroke_df['stroke'].value_counts().plot(kind='bar')


# **It seems our dependent variable has imbalanced data. We will take care this when we start building model.**

# # Feature Exploration

# In[ ]:


s_stroke_df = pd.DataFrame()
s_stroke_df = m_stroke_df


# **Feature : gender**
# 
# **Description : Male / Female** 

# In[ ]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(x='gender',data=s_stroke_df)


# In[ ]:


s_stroke_df['gender'].value_counts()


# **Feature : age**
# 
# **Description : Age of samples** 

# In[ ]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.boxplot(y='age', data=s_stroke_df)
plt.subplot(1,2,2)
sns.kdeplot(s_stroke_df.loc[(s_stroke_df['stroke']==1), 
            'age'], color='r', shade=True, Label='Stroke') 
  
sns.kdeplot(s_stroke_df.loc[(s_stroke_df['stroke']==0),  
            'age'], color='b', shade=True, Label='No Stroke') 
  
plt.xlabel('Age') 
plt.ylabel('Probability Density') 


# **Feature: hypertension**
#     

# **Description :  Is user has hypertension**

# In[ ]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(data=s_stroke_df, x='hypertension',hue='gender')
plt.title("People segregated by their hypertension condition")

plt.subplot(1,2,2)
sns.kdeplot(s_stroke_df.loc[(s_stroke_df['stroke']==1), 
            'hypertension'], color='r', shade=True, Label='Stroke') 
  
sns.kdeplot(s_stroke_df.loc[(s_stroke_df['stroke']==0),  
            'hypertension'], color='b', shade=True, Label='No Stroke') 
  
plt.xlabel('Hypertension') 
plt.ylabel('Probability Density') 


# **Feature : heart_disease**

# **Description : People with heart disease**

# In[ ]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
sns.countplot(data=s_stroke_df, x='heart_disease',hue='gender')
plt.title("People segregated by their hypertension condition")
plt.subplot(1,2,2)
sns.kdeplot(s_stroke_df.loc[(s_stroke_df['stroke']==1), 
            'heart_disease'], color='r', shade=True, Label='Stroke') 
  
sns.kdeplot(s_stroke_df.loc[(s_stroke_df['stroke']==0),  
            'heart_disease'], color='b', shade=True, Label='No Stroke') 
  
plt.xlabel('Heart Disease') 
plt.ylabel('Probability Density') 


# **Feature : ever_married**

# **Description : Maritial Status**

# In[ ]:


plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
sns.countplot(data=s_stroke_df,x='ever_married', hue='gender')
plt.title('Maritial Status')
plt.subplot(1,3,2)
plt.title('Married / Unmarried people having / or not hypertension')
sns.countplot(data=s_stroke_df,x='ever_married', hue='hypertension')
plt.subplot(1,3,3)
plt.title('Married / Unmarried people having / or not heart disease')
sns.countplot(data=s_stroke_df,x='ever_married', hue='heart_disease')
plt.show()


# **Feature : work_type**

# **Description : Work people do for living**

# In[ ]:


plt.figure(figsize=(8,4))
sns.countplot(data=s_stroke_df,x='work_type')
plt.title('')


# **Feature : Residence_type**

# **Description : Area of living**

# In[ ]:


plt.figure(figsize=(8,4))
sns.countplot(data=s_stroke_df,x='Residence_type')
plt.title('')


# **Feature : avg_glucose_level**

# In[ ]:


plt.figure(figsize=(8,4))
sns.scatterplot(data=s_stroke_df,x='bmi',y='avg_glucose_level',hue='gender')


# **Feature : bmi**

# In[ ]:


male_mean = s_stroke_df[s_stroke_df.gender=='Male']['bmi'].mean()
female_mean = s_stroke_df[s_stroke_df.gender=='Female']['bmi'].mean()
male_mean,female_mean


# In[ ]:


def fill_bmi(col):
    gender = col[0]
    bmi = col[1]
    if pd.isnull(bmi):
        if gender=='Male':
            return male_mean
        else:
            return female_mean
    else:
        return bmi
    


# In[ ]:


s_stroke_df['bmi'] = s_stroke_df[['gender','bmi']].apply(fill_bmi,axis=1)


# **Feature : smoking_status**

# In[ ]:


s_stroke_df['smoking_status'].isnull().sum()


# In[ ]:


s_stroke_df['smoking_status'].value_counts()


# # Feature Encoding 

# In[ ]:


def encode_gender(col):
    gender = col
    if gender=='Male':
        return 1
    else:
        return 0
    


# In[ ]:


def encode_married(col):
    gender = col
    if gender=='Yes':
        return 1
    else:
        return 0


# In[ ]:


s_stroke_df['gender'] = s_stroke_df['gender'].apply(encode_gender)


# In[ ]:


s_stroke_df['ever_married'] = s_stroke_df['ever_married'].apply(encode_married)


# In[ ]:


work_type_encode = pd.get_dummies(s_stroke_df['work_type'])
work_type_encode.head(2)


# In[ ]:


residence_type_encode = pd.get_dummies(s_stroke_df['Residence_type'],prefix='residence')
residence_type_encode.head(2)


# **Creating new category in 'smoking status' column 'occasionally smoked'**

# In[ ]:


s_stroke_df['smoking_status'].fillna('occasionally smoked',inplace=True)


# In[ ]:


s_stroke_df['smoking_status'].value_counts()


# In[ ]:


smoking_encoded = pd.get_dummies(s_stroke_df['smoking_status'])
smoking_encoded.head(2)


# **Lets drop the columns we have encoded**

# In[ ]:


s_stroke_df = s_stroke_df.drop(['work_type','Residence_type','smoking_status'],axis=1)


# In[ ]:


s_stroke_df.head()


# **Lets merge encoded columns**

# In[ ]:


s_stroke_encod = pd.concat([s_stroke_df,work_type_encode,residence_type_encode,smoking_encoded],axis=1)


# In[ ]:


s_stroke_encod.head()


# In[ ]:




