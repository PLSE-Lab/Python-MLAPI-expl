#!/usr/bin/env python
# coding: utf-8

# ## EDA of cadiovascular diseases data 
# 
# This work is my first public Kernel in Kaggle, and I would really appreciate your feedback or upvote!
# 
# The dataset consists of 70 000 records of patients data in 12 features, such as age, gender, systolic blood pressure, diastolic blood pressure, and etc. The target class "cardio" equals to 1, when patient has cardiovascular desease, and it's 0, if patient is healthy.
# 
# The task is to predict the presence or absence of cardiovascular disease (CVD) using the patient examination results. 
# 
# #### Data description
# 
# There are 3 types of input features:
# 
# - *Objective*: factual information;
# - *Examination*: results of medical examination;
# - *Subjective*: information given by the patient.
# 
# | Feature | Variable Type | Variable      | Value Type |
# |---------|--------------|---------------|------------|
# | Age | Objective Feature | age | int (days) |
# | Height | Objective Feature | height | int (cm) |
# | Weight | Objective Feature | weight | float (kg) |
# | Gender | Objective Feature | gender | categorical code |
# | Systolic blood pressure | Examination Feature | ap_hi | int |
# | Diastolic blood pressure | Examination Feature | ap_lo | int |
# | Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
# | Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
# | Smoking | Subjective Feature | smoke | binary |
# | Alcohol intake | Subjective Feature | alco | binary |
# | Physical activity | Subjective Feature | active | binary |
# | Presence or absence of cardiovascular disease | Target Variable | cardio | binary |
# 
# All of the dataset values were collected at the moment of medical examination. 
# 
# 
# ### Initial analysis
# Let's look at the dataset and given variables.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
import os
df = pd.read_csv("../input/cardio_train.csv",sep=";")


# In[ ]:


df.head()


# #### Univariate analysis
# To understand all our variables, at first, we should look at their datatypes. We can do it with `info()` function:

# In[ ]:


df.info()


# All features are numerical, 12 integers and 1 decimal number (weight). The second column gives us an idea how big is the dataset and how many non-null values are there for each field. We can use `describe()` to display sample statistics such as `min`, `max`, `mean`,`std` for each attribute:

# In[ ]:


df.describe()


# Age is measured in days, height is in centimeters. Let's look ate the numerical variables and how are they spread among target class. For example, at what age does the number of people with CVD exceed the number of people without CVD?

# In[ ]:


from matplotlib import rcParams
rcParams['figure.figsize'] = 11, 8
df['years'] = (df['age'] / 365).round().astype('int')
sns.countplot(x='years', hue='cardio', data = df, palette="Set2");


# It can be observed that people over 55 of age are more exposed to CVD. From the table above, we can see that there are outliers in `ap_hi`, `ap_lo`, `weight` and `height`. We will deal with them later.
# 
# Let's look at categorical variables in the dataset and their distribution:

# In[ ]:


df_categorical = df.loc[:,['cholesterol','gluc', 'smoke', 'alco', 'active']]
sns.countplot(x="variable", hue="value",data= pd.melt(df_categorical));


# #### Bivariate analysis
# 
# It may be useful to split categorical variables by target class:

# In[ ]:


df_long = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc', 'smoke', 'alco', 'active'])
sns.catplot(x="variable", hue="value", col="cardio",
                data=df_long, kind="count");


# It can be clearly seen that patients with CVD have higher cholesterol and blood glucose level. And, generally speaking less active.
# 
# To figure out whether "1" stands for women or men in gender column, let's calculate the mean of height per gender. We assume that men are higher than women on average.

# In[ ]:


df.groupby('gender')['height'].mean()


# Average height for "2" gender is greater, than for "1" gender, therefore "1" stands for women. Let's see how many men and women presented in the dataset:

# In[ ]:


df['gender'].value_counts()


# Who more often report consuming alcohol - men or women?

# In[ ]:


df.groupby('gender')['alco'].sum()


# So, men consume alcohol more frequently on average. 
# Next, the target variables are balanced:`

# In[ ]:


df['cardio'].value_counts(normalize=True)


# To see how the target class is distributed among men and women, we can use also `crosstab`

# In[ ]:


pd.crosstab(df['cardio'],df['gender'],normalize=True)


# ### Cleaning Data

# Are there any `NA`s or missing values in a dataset?

# In[ ]:


df.isnull().values.any()


# If we look more closely to height and weight columns, we will notice that minimum height is 55 cm and minimum weight is 10 kg. That has to be an error, since minimum age is 10798 days, which equals to 29 years. On the other hand, the maximum height is 250 cm and the highest weight is 200 kg, which might be irrelevant, when generilizing data. To deal with these errors, we can remove outliers.

# In[ ]:


df.describe()


# Let's remove weights and heights, that fall below 2.5% or above 97.5% of a given range.

# In[ ]:


df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025))].index,inplace=True)


# In addition, in some cases diastolic pressure is higher than systolic, which is also  incorrect. How many records are inaccurate in terms of blood pressure?

# In[ ]:


print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo']> df['ap_hi']].shape[0]))


# Let's get rid of the outliers, moreover blood pressure could not be negative value!

# In[ ]:


df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,inplace=True)
df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,inplace=True)


# In[ ]:


blood_pressure = df.loc[:,['ap_lo','ap_hi']]
sns.boxplot(x = 'variable',y = 'value',data = blood_pressure.melt())
print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo']> df['ap_hi']].shape[0]))


# #### Multivariate analysis
# It might be useful to consider correation matrix:

# In[ ]:


corr = df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});


# As we can see age and cholesterol have significant impact, but not very high correlated with target class.

# Let's create `violinplot` to show height distribution across gender. Looking at the mean values of height and weight for each value of the gender feature might not be enough to determin whether 1 is male or female designation.

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
df_melt = pd.melt(frame=df, value_vars=['height'], id_vars=['gender'])
plt.figure(figsize=(12, 10))
ax = sns.violinplot(
    x='variable', 
    y='value', 
    hue='gender', 
    split=True, 
    data=df_melt, 
    scale='count',
    scale_hue=False,
    palette="Set2");


# Let's create a new feature - Body Mass Index (BMI):
# 
# $$BMI = \frac {mass_{kg}} {height ^2_{m}},$$
# 
# and compare average BMI for healthy people to average BMI of ill people. Normal BMI values are said to be from 18.5 to 25.

# In[ ]:


df['BMI'] = df['weight']/((df['height']/100)**2)
sns.catplot(x="gender", y="BMI", hue="alco", col="cardio", data=df, color = "yellow",kind="box", height=10, aspect=.7);


#  Drinking women have higher risks for CVD than drinking men based on thier BMI.

# *To be continued*

# In[ ]:




