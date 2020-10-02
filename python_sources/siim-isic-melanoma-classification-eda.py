#!/usr/bin/env python
# coding: utf-8

# # Summary

# **[I. Generalities](#generalities)** -  Overview of the available data: format, number, values, description etc.
# 
# **[II. Tabular data](#tabular_data)** - Exploration of the tabular data/metadata and their potential predictive power
# 
# [II. a) Values, distributions & balancing](#vdb) - Inspection of each variable separately
# 
# [II. b) Interactions between tabular data](#interactions_data) - Interactions between each variables
#    
# [II. c) Interaction at the patient level](#interactions_patient) - Information shared between different images of the same patient
# 
# 
# 
# **[III. Image data](#images)** - A convenient tool to interactively show images by label
# 
# **[IV. Wrap-up](#wrapup)** - Final comments and sum-up of the most interesting findings
# 

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize']=(20,10)
import seaborn as sns
sns.set_style("dark")
import warnings
warnings.filterwarnings('ignore')
from ipywidgets import interact


# <div id="generalities">
#     
# #  I. Generalities 
#     
# </div>
# 

# In[ ]:


df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
df.head()


# Distributions of the content of the images (diagnosis):

# In[ ]:


df['diagnosis'].value_counts()


# In this competition, we have both images and contextual data. The goal is to predict from them the existence of a malignant melanome.
# 
# ### Images are available in three formats:
# * **JPEG**
# * **DICOM**, a widely used format among the health data science practitionner
# * **TFRecord**, in this case data are resized to 1024 x 1024.
#     
#     
# ### Contextual data are tabular data and include:
# * Information about the patient (the sex and the approximative age) 
# * Information about the image (what part of the bodies is it taken from)
# * Detailed diagnostic (ie other classes, beside the binary melanoma/no melanoma dichotomy that is the frame of this competition). 
# 
# Let's stop a bit to know what are the definitions corresponding to those detailed classes. More information and images are provided in the links: 
# * **[Nevus](https://dermnetnz.org/topics/mole/)**: a birthmark or a mole on the skin, especially a birthmark in the form of a raised red patch.
# * **[Melanoma](https://dermnetnz.org/topics/melanoma/)**: a tumour of melanin-forming cells, especially a malignant tumour associated with skin cancer.
# * **[Seborrheic keratosis](https://dermnetnz.org/topics/seborrhoeic-keratosis/)**: a non-cancerous (benign) skin tumour that originates from cells in the outer layer of the skin. Like liver spots, seborrheic keratoses are seen more often as people age
# * **[Lentigo NOS (Lentigo?)](https://dermnetnz.org/topics/lentigo/)**: A small pigmented spot on the skin with a clearly defined edge, surrounded by normal-appearing skin. It is a harmless (benign) hyperplasia of melanocytes which is linear in its spread.
# * **[Lichenoid keratosis](https://dermnetnz.org/topics/lichenoid-keratosis/)**:A usually small, solitary, inflamed macule or thin pigmented plaque. Multiple eruptive lichenoid keratoses in sun-exposed sites are also described. Their colour varies from an initial reddish brown to a greyish purple/brown as the lesion resolves several weeks or months later.
# * **[Solar lentigo](https://dermnetnz.org/topics/solar-lentigo/)**: Solar lentigo is a harmless patch of darkened skin. They are very common, especially in people over the age of 40 years.

# ***

# <div id="tabular_data">
#     
# # II. Tabular data
#     
# </div>
# 

# <div id="vdb">
#     
# ## a) Values, distributions & balancing
#     
# </div>

# In[ ]:


categorical_cols = ['diagnosis','sex','anatom_site_general_challenge','benign_malignant','target']

fig,ax = plt.subplots(2,(len(categorical_cols)+1)//2,figsize=(30,15))

ratio = {}
for i,col in enumerate(categorical_cols+['age_approx']):
    ratio[col] = 100*df[col].value_counts(dropna=False)/df[col].value_counts(dropna=False).sum()
    if i==5:
        ax[1][2].hist(df['age_approx'])     
    else:
        ax[i%2][i//2].bar([str(x) for x in ratio[col].index],height=ratio[col])
    ax[i%2][i//2].set_title(col,fontdict={'fontsize':20})
    for tick in ax[i%2][i//2].get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize('x-large')

fig.suptitle('Distribution of each categorical data',y=1.02,fontsize=25)
fig.tight_layout()


# ### Takeaways:
# - Most of the categorical data are (strongly) imbalanced.
# - This is in particular the case of the target variable (near 98% of not malignant lesion). This is likely to be one of the difficulties of this competition. 
# - If we look into the details of the target, in the "diagnosis" columns, we see that in most cases (80%), the status is "unknown", a probably very heterogeneous category. 
# - Patients' sex is also slightly imbalanced, with a few more men. This data is almost always knwon. 
# - Interestingly, the body part from which the picture is taken is also imbalanced, with more than 50% corresponding to the torso. It will be interesting to see how it can impact the result.

# <div id="interactions_data">
#     
# ## b) Interactions between tabular data
#     
# </div>

# In[ ]:


df_category = df.copy()
df_category['benign_malignant'] = (df_category['benign_malignant'] == 'malignant').apply(int)


# ### Relation between target, begnin_malignant & diagnosis:

# In[ ]:


print("Agreement between 'malignant' value and target:",100*((df['benign_malignant']=='malignant') == (df['target'])).sum()/len(df),'%')
print("Agreement between 'melanoma' value and target:",100*(((df['diagnosis'] =='melanoma') ) == (df['target'])).sum()/len(df),'%')


# **Remark**: We see that the *benign_malignant* & *target* is actually the same information. *diagnosis* is a bit more complex, as it details all the other cases of not-malignant lesions. We thus do not did to keep benign_malignant for the next analysis.

# ### Relation between biological data:

# In[ ]:


sex_c = df_category['sex'].value_counts()
sex_c = sex_c.sort_index()
fig,ax = plt.subplots() 
for idx in sex_c.index:
    sns.kdeplot(df.loc[df['sex']==idx,'age_approx'],shade=True,ax=ax)
ax.legend(sex_c.index)
ax.set_title('Age distribution for each sex',fontsize=20)
res = ax.set_xlabel('approx_age')


# In[ ]:


diag_c = df_category['diagnosis'].value_counts()
diag_c = diag_c.sort_index()
fig,ax = plt.subplots() 
for idx in diag_c.index:
    sns.kdeplot(df.loc[df['diagnosis']==idx,'age_approx'],shade=True,ax=ax)
ax.legend(diag_c.index)
ax.set_title('Relation between age & diagnosis',fontsize=20)
res = ax.set_xlabel('approx_age')


# In[ ]:


ax = sns.catplot(x="sex",
            y="target",
            kind="bar",
            hue='age_approx',
            data=df_category,
            height=9, 
            aspect=2.3);
_=ax.fig.suptitle('Influence of the age on likeliness that the diagnosis is melanoma',y=1.02, fontsize=20)


# In[ ]:


print("Number of men > 90 in the dataset:", len(df_category.loc[(df_category['age_approx']==90.0) & (df_category['sex']=='male'),'patient_id'].unique()))


# ### Takeaways: 
# 
# - There is not a huge difference between male and female age distributions. We should however note that since men are a bit more aged in this dataset, some slight "sex effects" could actually be explained by an "age effect". 
# - As expected, diagnostic distribution differ with regard to the age: besides nevus and "unknown" lesions, all the lesion are more likely to appear later (40 years & more). 
# - If we look more specifically at the target, we see that melanoma exist at any age, but it definitely becomes more likely with the age (> 40 years old and especially 70 years old). This correlation with age may be impacted by the sex (with for instance 40% of the diagnosis being melanoma for the most aged men - around 90 years old -  and far less for women), but this could also be an effect of data scarcity for some age range. 

# ### Relation between image location and other variables:

# In[ ]:


anatom_c = df_category['anatom_site_general_challenge'].value_counts()
anatom_c = anatom_c.sort_index()
fig,ax = plt.subplots() 
for idx in anatom_c.index:
    sns.kdeplot(df.loc[df['anatom_site_general_challenge']==idx,'age_approx'],shade=True,ax=ax)
ax.legend(anatom_c.index)
ax.set_title('Relation between age & anatomic site',fontsize=20)
res = ax.set_xlabel('approx_age')


# In[ ]:


ax = sns.catplot(x="sex",
            y="target",
            hue="anatom_site_general_challenge",
            kind="bar",
            data=df_category,
            height=9, 
            aspect=2.3);
_=ax.fig.suptitle('Influence of the image site on the target, by sex', y=1.02,fontsize=20)


# In[ ]:


print("Number of female patient with an image of the oral/genital site:", len(df.loc[(df_category['sex']=="female") & (df_category['anatom_site_general_challenge']=="oral/genital"),'patient_id'].unique()))


# ### Takeaways
# 
# The part of the body where the mole is located seems to have an important influence:
# - with head/neck moles usually more associated with malignant moles. 
# - This also apparently dependent on the sex,as the latter effect is especially significan for men. 
# - While this may be related to the relatively small sample we have, it seems taht for women, oral/genital moles are especially at risk.

# <div id="interactions_patient">
#     
# ## c) Interaction at the patient level
#     
# </div>
# 

# In[ ]:


print("Number of unique patients in the dataset:", len(df_category['patient_id'].unique()))


# Actually, there is only 2056 different patients (for more than 30k pictures). The distribution of the number of images by patient can be seen below.

# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
df_category.groupby('patient_id').count()['image_name'].hist(ax=ax,bins=100)
ax.set_title('Number of pictures by patient', fontsize=20)
ax.grid(False)


# ### Takeaways: 
# 
# - There are often more than one picture (and even more than 5) for each patient. Let's have a deeper dive into the relation that those images have between each other. 

# In[ ]:


ax= df_category.groupby('patient_id').aggregate({'age_approx':pd.Series.nunique}).hist()[0][0]
ax.set_title('Distribution of the number of different age_approx for each patient',fontsize=20)
ax.grid(False)


# Interestingly, same patient are often associated with more than one approximative age. While ths can be associated with errors in the dataset, this is also probably associatded with the fact that those images have been taken at different moments.
# 
# Another interesting question - partially influence by the previous - is the quantity of information carried by an image about another of the same patient. In other words, for a given image, could we use other images of the patient to get information on the diagnosis of the first image? 

# In[ ]:


df_category_patient_with_melanoma = df_category.groupby('patient_id').apply(lambda x: 'melanoma' in set(x['diagnosis']))
set_category_patient_with_melanoma = {x[0] for x  in df_category_patient_with_melanoma.iteritems() if x[1]}
df_category_patient_with_melanoma = df_category.loc[df_category['patient_id'].isin(set_category_patient_with_melanoma)]

# We extract the id of the first image with melanoma for each patient with at least one melanoma 
first_melanoma_pictures = df_category_patient_with_melanoma.loc[df_category_patient_with_melanoma['diagnosis']=='melanoma'].drop_duplicates(['patient_id','diagnosis'],keep='first').index

# We select all the other images
df_other_images_patient_with_melanoma = df_category_patient_with_melanoma.loc[df_category_patient_with_melanoma['diagnosis'] != 'melanoma']
df_other_images_patient_with_melanoma =  df_category_patient_with_melanoma.drop(first_melanoma_pictures)
df_other_images_patient_with_melanoma['image_set'] = 'Other images of patients with melanoma'


df_all_images = df_category.copy()
df_all_images['image_set'] = 'Full dataset'

# We can concatenate all the images with only other images of patient with melanoma, in order to build comparative statics on the risk of occurence of melanoma
concatenated = pd.concat([df_all_images,
          df_other_images_patient_with_melanoma])

ax = sns.catplot(x="sex",
            y="target",
            hue="image_set",
            kind="bar",
            data=concatenated,
            height=8, 
            aspect=2.2);
_=ax.fig.suptitle('Influence on the diagnosis of the existence of a melanoma elsewhere',fontsize=20,y=1.02)


# ### Takeaways
# 
# - Comparing all images vs images of patient with a melanoma (but not considering this image), we see that the latter category is more likely to have a melanoma.
# - This effect seems to be stable accross both sexes. 
# - This is is an important information as it shows that prediction is likely to be improved using information (data or metadata) from other images.

# <div id="images">
#     
# # III. Image data
# 
#     
# </div>

# In[ ]:


path_train_jpg = '../input/siim-isic-melanoma-classification/jpeg/train/'


# The interactive cell below can be used to see the different types of diagnosis. It takes a few seconds to run. 

# In[ ]:


diagnosis_list = ['unknown', 'nevus','melanoma','seborrheic keratosis','lentigo NOS',
 'lichenoid keratosis','solar lentigo','atypical melanocytic proliferation','cafe-au-lait macule']
def show_diagnosis(diagnosis='melanoma'):
    assert diagnosis in diagnosis_list
    fig, ax = plt.subplots(3,2,figsize=(10,10))
    samples = df.loc[df['diagnosis']==diagnosis].sample(6,replace=True)['image_name']
    ax = ax.ravel()
    for j, name in enumerate(samples): 
        ax[j].imshow(plt.imread(path_train_jpg+name+'.jpg'))
        ax[j].grid(False)
        # Hide axes ticks
        ax[j].set_xticks([])
        ax[j].set_yticks([])
    fig.suptitle("Diagnosis: "+diagnosis,fontsize=20, y=0.95)

    plt.show()
    
int = interact(show_diagnosis,diagnosis=diagnosis_list) 


# ***

# <div id="wrapup">
#     
# # IV Wrap-up
# 
# 
# This quick EDA provided some insights - mainly regarding tabular data/metadata, that will probably be useful to improve prediction performances on this dataset. Here are the most promising we found: 
# 
# Interestingly, the body part from which the picture is taken is also imbalanced, with more than 50% corresponding to the torso. It will be interesting to see how it can impact the result.
# 
# - **Dataset is generally imbalanced**, and even strongly imbalanced when it comes to the target (98% of the diagnosis are not melanoma)
# - The more aged a person, the more likely the diagnosis for a skin lesion would be melanoma. This seems to be especially true for men. 
# - **Melanoma diagnosis seems to be more likey for men** than for women (but further investigation are needed to understand if this is not only an "age effect"). 
# - **The skin site is also an important parameter**, with head/neck & upper extremity more likely to result in a melanoma diagnosis. Here as well, there is apparently a "sex effect", with head/neck sites far more likely to be associated with melanoma diagnosis for men than for women.
# - In this dataset, same patients appear usually more than once, as multiple site of lesions are pictured. Furthermore, information a site is likely to be useful for another site since **people with a melanoma are apparently more likely to have another one melanoma on another site.**
# 
# 
# I'll had new analysis later. Don't hesitate to tell me if you have any remarks or ideas that could be explored to understand how to make the best use of those data ! 
#     
#     
# </div>

# In[ ]:




