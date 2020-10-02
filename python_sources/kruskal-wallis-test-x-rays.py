#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import mstats # Stats
from sklearn import preprocessing # Preprocessing
from sklearn.model_selection import train_test_split # Split data
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [17, 6]

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


path = '../input/chest-xrays-multi-and-single-label-with-resampling/data-chest-x-ray-singlelabel-balanced-resample-gender-proportion.plk'
path1 = '../input/chest-xray-data-multi14-all/data-chest-x-ray-multilabel-14-all.plk'


# In[ ]:


data = pd.read_pickle(path)
print(data.shape)
data1 = pd.read_pickle(path1)
print(data1.shape)
data.head()


# In[ ]:


final_data = pd.merge(data, data1[['Patient Age', 'Patient Gender']], left_index=True, right_index=True)
final_data.shape


# ## Describe data

# In[ ]:


data.describe()


# ## Preprocessing: Label Enconder -> Gender

# In[ ]:


le_gender = preprocessing.LabelEncoder()
le_age = preprocessing.LabelEncoder()
le_gender.fit(final_data['Patient Gender'])
print(list(le_gender.classes_))
final_data['Patient Gender'] = le_gender.transform(final_data['Patient Gender']) 

final_data.head()


# In[ ]:


final_data[['Patient Age', 'Patient Gender']].describe()


# ## Split data

# In[ ]:


final_data['Finding Labels'].unique()


# In[ ]:



age_No_Finding = final_data[final_data['Finding Labels']=='No Finding']['Patient Age']
age_Cardiomegaly = final_data[final_data['Finding Labels']=='Cardiomegaly']['Patient Age']
age_Hernia = final_data[final_data['Finding Labels']=='Hernia']['Patient Age']
age_Infiltration = final_data[final_data['Finding Labels']=='Infiltration']['Patient Age']
age_Nodule = final_data[final_data['Finding Labels']=='Nodule']['Patient Age']
age_Emphysema = final_data[final_data['Finding Labels']=='Emphysema']['Patient Age']
age_Effusion = final_data[final_data['Finding Labels']=='Effusion']['Patient Age']
age_Atelectasis = final_data[final_data['Finding Labels']=='Atelectasis']['Patient Age']
age_Pleural_Thickening = final_data[final_data['Finding Labels']=='Pleural_Thickening']['Patient Age']
age_Pneumothorax = final_data[final_data['Finding Labels']=='Pneumothorax']['Patient Age']
age_Mass = final_data[final_data['Finding Labels']=='Mass']['Patient Age']
age_Fibrosis = final_data[final_data['Finding Labels']=='Fibrosis']['Patient Age']
age_Consolidation = final_data[final_data['Finding Labels']=='Consolidation']['Patient Age']
age_Edema = final_data[final_data['Finding Labels']=='Edema']['Patient Age']
age_Pneumonia = final_data[final_data['Finding Labels']=='Pneumonia']['Patient Age']


gender_No_Finding = final_data[final_data['Finding Labels']=='No Finding']['Patient Gender']
gender_Cardiomegaly = final_data[final_data['Finding Labels']=='Cardiomegaly']['Patient Gender']
gender_Hernia = final_data[final_data['Finding Labels']=='Hernia']['Patient Gender']
gender_Infiltration = final_data[final_data['Finding Labels']=='Infiltration']['Patient Gender']
gender_Nodule = final_data[final_data['Finding Labels']=='Nodule']['Patient Gender']
gender_Emphysema = final_data[final_data['Finding Labels']=='Emphysema']['Patient Gender']
gender_Effusion = final_data[final_data['Finding Labels']=='Effusion']['Patient Gender']
gender_Atelectasis = final_data[final_data['Finding Labels']=='Atelectasis']['Patient Gender']
gender_Pleural_Thickening = final_data[final_data['Finding Labels']=='Pleural_Thickening']['Patient Gender']
gender_Pneumothorax = final_data[final_data['Finding Labels']=='Pneumothorax']['Patient Gender']
gender_Mass = final_data[final_data['Finding Labels']=='Mass']['Patient Gender']
gender_Fibrosis = final_data[final_data['Finding Labels']=='Fibrosis']['Patient Gender']
gender_Consolidation = final_data[final_data['Finding Labels']=='Consolidation']['Patient Gender']
gender_Edema = final_data[final_data['Finding Labels']=='Edema']['Patient Gender']
gender_Pneumonia = final_data[final_data['Finding Labels']=='Pneumonia']['Patient Gender']


# # Test Kruskal Wallis with Age and Gender

# In[ ]:


a = 0.05/15 #Bonferroni Correction 


# ### Kruskal Test: Patient Age

# In[ ]:


H, pval = mstats.kruskalwallis(
    list(age_Atelectasis),
    list(age_Cardiomegaly),
    list(age_Consolidation),
    list(age_Edema),
    list(age_Effusion),
    list(age_Emphysema),
    list(age_Fibrosis),
    list(age_Hernia),
    list(age_Infiltration),
    list(age_Mass),
    list(age_No_Finding),
    list(age_Nodule),
    list(age_Pleural_Thickening),
    list(age_Pneumonia),
    list(age_Pneumothorax)
)

print("H-statistic:\t{}\nP-value:\t{}".format(H, pval))
if pval < a:
    print("Reject NULL hypothesis - Significant differences exist between groups.")
if pval > a:
    print("Accept NULL hypothesis - No significant difference between groups.")


# ### Kruskal Test: Patient Gender

# In[ ]:


H, pval = mstats.kruskalwallis(
    list(gender_Atelectasis),
    list(gender_Cardiomegaly),
    list(gender_Consolidation),
    list(gender_Edema),
    list(gender_Effusion),
    list(gender_Emphysema),
    list(gender_Fibrosis),
    list(gender_Hernia),
    list(gender_Infiltration),
    list(gender_Mass),
    list(gender_No_Finding),
    list(gender_Nodule),
    list(gender_Pleural_Thickening),
    list(gender_Pneumonia),
    list(gender_Pneumothorax)
)

print("H-statistic:\t{}\nP-value:\t{}".format(H, pval))
if pval < a:
    print("Reject NULL hypothesis - Significant differences exist between groups.")
if pval > a:
    print("Accept NULL hypothesis - No significant difference between groups.")


# ## Columnas apiladas de enfermedades por rangos de edad

# In[ ]:


fig, ax = plt.subplots(figsize=[15,10])
final_data.groupby(['Finding Labels', pd.cut(final_data['Patient Age'], np.arange(0,100,10))])       .size()       .unstack(0)       .plot.bar(stacked=True, ax=ax)


# In[ ]:


sns.FacetGrid(final_data,hue='Finding Labels',size=8).map(sns.distplot,'Patient Age').add_legend()
plt.show()


# ## Columnas apiladas de enfermedades por genero

# In[ ]:


fig, ax = plt.subplots(figsize=[15,10])
final_data.groupby(['Patient Gender', 'Finding Labels'])       .size()       .unstack(0)       .plot.bar(stacked=True, ax=ax)
ax.legend(["Female", "Male"]);


# In[ ]:




