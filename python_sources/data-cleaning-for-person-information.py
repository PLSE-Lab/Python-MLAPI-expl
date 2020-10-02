#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


# read csv files
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
train


# In[ ]:


# understand the train dataset
train.info()


# ### **Total number of records: 33126**
# 
# 
# 
# ***Therefore columns with null entries are:***
# * sex
# * age_approx
# * anatom_site_general_challenge  

# ## 1. Sex:
# lets fill in with either male or female according to which gender is more common

# In[ ]:


#find the ratio of men to total ratio:
no_of_male = (train['sex'] == 'male').sum()
no_of_values = (train['sex'].notna()).count()
print(no_of_male/no_of_values)


# since the ratio of males > 0.5, let's fill na values with male

# In[ ]:


train['sex'] = train['sex'].fillna('male')
print('no. of na values in sex = ', train['sex'].isna().sum() )


# ## 2. age_approx:
# 
# let's follow the best practice and replace na values with median age

# In[ ]:


train['age_approx'] = train['age_approx'].fillna( train['age_approx'].median() )

print('no. of na values in age_approx = ', train['age_approx'].isna().sum() )


# ## 3. anatom_site_general_challenge:
# fill it with the most common site as it is the mst probable !

# In[ ]:


most_freq_site = train['anatom_site_general_challenge'].value_counts().idxmax()

train['anatom_site_general_challenge'].fillna( most_freq_site, inplace = True)

print('no. of na values in anatom_site_general_challenge = ', train['anatom_site_general_challenge'].isna().sum() )


# ## Diagnosis:
# This column will not be available in the test dataset, so we cannot use it for prediction directly but only for EDA, so let's drop the Diagnosis column for now

# In[ ]:


train.drop('diagnosis', axis = 1, inplace = True)
train.info()


# In[ ]:


# lets clean this table for data analysis

# 1. Lets create a separate dataframe for this purpose
p_data = train.copy()  # p_data as in person data

# 2. Lets drop the columns image_name and patient_id
#         we'll access the image_name from the train dataset 
#         and since patient_id doens't represent any probability of being malignant or benign, 
#                     - we'll treat column id as new patient id

p_data.drop(['image_name', 'patient_id'],axis = 1, inplace = True)

# if you want to preserve patient_id to be the id instead use:

# p_data.drop('image_name',axis = 1, inplace = True)
# p_data.set_index('patient_id')


# ## lets convert all categorical data into bool type!

# In[ ]:


# convert sex into a bool column
p_data['is_male'] = (p_data['sex'] == 'male')
p_data.drop('sex', axis = 1, inplace = True)

#convert benign_malignant into a bool column'
p_data['is_malignant'] = p_data['target']
p_data.drop(['benign_malignant', 'target'], axis = 1, inplace = True)


p_data


# In[ ]:


#convert anatom_site_general_challenge into categories

#note: we inclued all the sites except for the final site cause in the final dataframe if we have false for
#all the included sites it means true automatically for the last site so adding it will only ceate redundancy

regions = p_data['anatom_site_general_challenge'].unique()[:-1]

for region in regions:
    p_data['is_' + region] = (p_data['anatom_site_general_challenge'] == region)
    print(region)

p_data.drop('anatom_site_general_challenge', axis = 1,  inplace = True)
p_data


# In[ ]:


# normalize age
p_data['normalized_age'] = p_data['age_approx']/p_data['age_approx'].max()
p_data.drop('age_approx', axis = 1, inplace = True)

p_data


# In[ ]:




