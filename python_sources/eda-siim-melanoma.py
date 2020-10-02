#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# In[ ]:


data = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
data.head()


# In[ ]:


data.groupby(['patient_id']).size()


# In[ ]:


y = data['target']
x = data.drop(['image_name','patient_id','target','benign_malignant'],axis=1,inplace=False)


# In[ ]:


fig = plt.plot(figsize=(5,5))
sns.countplot(y)
B,M = data['target'].value_counts()
print('Bening: ',B)
print('Malignant: ',M)


# In[ ]:


F,M = data['sex'].value_counts()
print('Female: ',F)
print('Male: ',M)
data['anatom_site_general_challenge'].value_counts()


# In[ ]:


fig,ax = plt.subplots(3,1,figsize=(10,30))
ax[0].set_xticklabels(x['anatom_site_general_challenge'],rotation=45)
sns.countplot(x['anatom_site_general_challenge'],ax=ax[0])

ax[1].set_xticklabels(x['sex'],rotation=45)
sns.countplot(x['sex'],ax=ax[1])

ax[2].set_xticklabels(x['age_approx'],rotation=45)
sns.countplot(x['age_approx'],ax=ax[2],hue=y)


# In[ ]:


data.groupby(['age_approx', 'target']).size()


# In[ ]:


data.groupby(['sex', 'target']).size()


# In[ ]:


data.groupby(['anatom_site_general_challenge', 'target']).size()


# In[ ]:


data.isna().sum()


# ### Sex
# 
# - Total NaN: 65
# - Benign : 65
# - Malignant: 0
# 
# These all NaN values belong to only two patients IP_5205991 and IP_9835712. <br>
# For all NaN values site location are lower extremity, upper extremity, torso or head/neck.<br> We will assign the majority sex according to site location for these NaN values.

# In[ ]:


fig,ax = plt.subplots(2,2,figsize=(14,15))
na_sex = data[data['sex'].isna()]
rest_sex = data[data['sex'].isna()==False]

ax[0][0].set_xticklabels(na_sex['anatom_site_general_challenge'],rotation=45)
ax[0][0].set_title('Site (Sex NaN) Patient Id: IP_5205991')
sns.countplot(na_sex['anatom_site_general_challenge'][na_sex['patient_id']=='IP_5205991'],ax=ax[0][0])

ax[0][1].set_xticklabels(na_sex['anatom_site_general_challenge'],rotation=45)
ax[0][1].set_title('Site (Sex NaN) Patient Id: IP_9835712')
sns.countplot(na_sex['anatom_site_general_challenge'][na_sex['patient_id']=='IP_9835712'],ax=ax[0][1])

ax[1][0].set_xticklabels(rest_sex['anatom_site_general_challenge'],rotation=45)
ax[1][0].set_title('Site (Sex Rest)')
sns.countplot(rest_sex['anatom_site_general_challenge'],hue=rest_sex['sex'],ax=ax[1][0])


# In[ ]:


data['sex'] =np.where(data['patient_id']=='IP_5205991','female', data['sex']) 
data['sex'] =np.where(data['patient_id']=='IP_9835712','male', data['sex']) 


# In[ ]:


data.isna().sum()


# ### Age
# 
# - Total NaN: 68
# - Benign : 68
# - Malignant: 0
# 
# These all NaN values belong to only three patients IP_5205991, IP_9835712 and IP_0550106. <br>
# For all NaN values site location are lower extremity, upper extremity, torso or head/neck.<br> Since age looks like a normal distribution. we will assign the mean age according to the site location for these NaN values.

# In[ ]:


fig,ax = plt.subplots(2,2,figsize=(15,15))
na_age = data[data['age_approx'].isna()]
rest_age = data[data['age_approx'].isna()==False]

ax[0][0].set_xticklabels(na_age['anatom_site_general_challenge'],rotation=10)
ax[0][0].set_title('Site (Age NaN) Patient Id: IP_5205991')
sns.countplot(na_age['anatom_site_general_challenge'][na_age['patient_id']=='IP_5205991'],ax=ax[0][0])

ax[0][1].set_xticklabels(na_age['anatom_site_general_challenge'],rotation=10)
ax[0][1].set_title('Site (Age NaN) Patient Id: IP_9835712')
sns.countplot(na_age['anatom_site_general_challenge'][na_age['patient_id']=='IP_9835712'],ax=ax[0][1])

ax[1][0].set_xticklabels(na_age['anatom_site_general_challenge'],rotation=45)
ax[1][0].set_title('Site (Age NaN) Patient Id: IP_0550106')
sns.countplot(na_age['anatom_site_general_challenge'][na_age['patient_id']=='IP_0550106'],ax=ax[1][0])


ax[1][1].set_title('Site (Age Rest)')
IntToSite = {0:'head/neck',1:'lower extremity',2:'oral/genital',
             3:'palms/soles',4:'torso',5:'upper extremity'}
labels = [0]*6
for i in range(6):
    sns.distplot(rest_age['age_approx'][rest_age['anatom_site_general_challenge']==IntToSite[i]],ax=ax[1][1],
                 hist=False,rug=True)
    labels[i]=IntToSite[i]
    

ax[1][1].legend(labels=labels)

for i in range(6):
    print(f"Mean age for site {IntToSite[i]}: {rest_age['age_approx'][rest_age['anatom_site_general_challenge']==IntToSite[i]].mean()}")
    


# In[ ]:


data['age_approx'] =np.where(data['patient_id']=='IP_5205991','50.0', data['age_approx']) 
data['age_approx'] =np.where(data['patient_id']=='IP_9835712','50.0', data['age_approx']) 
data['age_approx'] =np.where(data['patient_id']=='IP_0550106','50.0', data['age_approx'])


# In[ ]:


data.isna().sum()


# ### Anatom Site
# 
# For anatom site we are assigning  'unknown' to NaN values and when we will conert anto site to one hot vector, we will give equal probability of each site to represent 'unknown' ([1/6,1/6,1/6,1/6,1/6,1/6])

# In[ ]:


data['anatom_site_general_challenge']=data['anatom_site_general_challenge'].fillna('unknown')


# In[ ]:


data['anatom_site_general_challenge'].value_counts()


# In[ ]:


data.isna().sum()


# In[ ]:


data.to_csv('modified_train.csv',index=False)


# In[ ]:


test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
test.isna().sum()


# In[ ]:


test['anatom_site_general_challenge']=test['anatom_site_general_challenge'].fillna('unknown')
test.isna().sum()


# In[ ]:


test.to_csv('modified_test.csv',index=False)


# In[ ]:




