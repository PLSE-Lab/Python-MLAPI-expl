#!/usr/bin/env python
# coding: utf-8

# ## Background Information
# > Current CVD rate in USA is **48%** according to the American Heart Association  
# > source: https://www.sciencedaily.com/releases/2019/01/190131084238.htm

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.stats import sem, t
from scipy import mean


# In[28]:


df = pd.read_csv("../input/heart.csv")


# In[29]:


df.head()


# In[30]:


df.dtypes


# In[31]:


# check for null values
df.isnull().values.any()


# In[32]:


df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# In[34]:


df.target = df.target.replace({0:'No Heart Disease', 1:'Heart Disease'})
df.sex = df.sex.replace({0:'female', 1:'male'})
df.chest_pain_type = df.chest_pain_type.replace({1:'agina pectoris', 2:'atypical agina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})
df.st_slope = df.st_slope.replace({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})
df.fasting_blood_sugar = df.fasting_blood_sugar.replace({0:'lower than 120mg/ml', 1:'greater than 120mg/ml'})
df.exercises_angina = df.exercise_angina.replace({0:'no', 1:'yes'})
df.thalassemia = df.thalassemia.replace({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})


# # Understanding our Sample

# In[35]:


sns.countplot(data=df, x="target", palette="bwr")


# In[36]:


sns.countplot(data=df, x="sex", palette="bwr")


# In[44]:


sns.countplot(data=df, x="target", hue="sex", palette="bwr")


# In[38]:


sns.distplot(df[df.sex=="male"].age, color="b")
sns.distplot(df[df.sex=="female"].age, color="r")
plt.xlabel("Age Distribution (blue = male, red = female)")


# In[39]:


# correlation matrix
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm')


# # Terminology
# ### Chest Pain Type
# > **angina** is a type of chest pain caused by reduced blood flow to heart.
# > Angina pectoris or typical angina occurs when the heart must work harder and doesn't come as a surprise (lasts around 5 mins or less).
# > Atypical angina is discomfort centered in the chest that is **not** cardiac pain
# <img src="https://slideplayer.com/slide/10052722/32/images/5/Types+of+Angina+Stable+%E2%80%93chest+pain+precipitated+by+exertion+or+stress..jpg">  
# 
# ### Serum cholesterol
# > Your serum cholesterol includes:
# > - LDL level
# > - HDL level
# > - 20 percent of your triglyceride level
# > The lower the LDL level and the higher the HDL level, the better. Healthy serum cholesterol is less than 200 mg/dL
# > Calculation: HDL + LDL + 0.2 * triglycerides  
# <img src="https://hhp-blog.s3.amazonaws.com/2016/10/iStock_65528817_MEDIUM.jpg">  
# 
# ### ST Segment and ST Depression
# > The ST Segment represents an electrically neutral area of the complex between QRS Complex(main spike) and T wave(second round peak)  
# <img src="https://litfl.com/wp-content/uploads/2018/10/ST-segment-depression-upsloping-downsloping-horizontal.png">  
# 
# ### Thalassemia
# Blood disorder that causes the body to make inadequate amount of hemoglobin, the protein that carries oxygen, in which results in large numbers of red blood cells being destroyed.
# <img src="https://ghr.nlm.nih.gov/art/large/thalassemia-red-blood-cells.jpeg">  

# # Analyze Associations

# In[40]:


# chest pain type and disease association
sns.countplot(x="chest_pain_type", hue="target", data=df, palette="bwr")


# In[41]:


# relationship between fasting blood sugar and serum cholesterol
sns.catplot(x="fasting_blood_sugar", y="serum_cholesterol", kind="box", data=df, palette="bwr")


# In[ ]:


sns.countplot(x="fasting_blood_sugar", hue="target", data=df, palette="bwr")


# In[ ]:


# relationship between serum cholesterol and heart disease
sns.catplot(x="target", y="serum_cholesterol", kind="box", data=df, palette="bwr")
plt.xlabel("Heart disease (0 = no, 1 = yes)")


# In[ ]:


sns.catplot(x="target", y="max_heart_rate", hue="st_slope", kind="box", data=df, palette="bwr")


# In[ ]:


# relationship between serum cholesterol and heart disease
sns.catplot(x="target", y="resting_blood_pressure", hue="sex", kind="box", data=df, palette="bwr")
plt.xlabel("Heart disease (0 = no, 1 = yes)")


# In[ ]:


# chest pain type and thalassemia association
plt.figure(figsize=(10,8))
sns.countplot(x="chest_pain_type", hue="thalassemia", data=df, palette="bwr")


# In[ ]:


# fasting blood sugar
# exercise_induced_angina
# cholesterol
sns.catplot(x="target", y="serum_cholesterol", hue="fasting_blood_sugar", col="exercise_angina",
           capsize=.2, palette="bwr", height=6, aspect=.75, kind="point", data=df)


# # Chi-squared test for association (chest pain and target)
# > ### State:
# > Ho: No association between chest pain type and target  
# > Ha: There is an association between chest pain type and target
# > ### Plan:
# > #### Conditions:
# > - Assume random sample
# > - Sample size is less than 10% of the total population (people with heart disease in America & Europe)
# > - All expected values are >= 5 (computed below)

# In[ ]:


chestpain_target = pd.crosstab(index=df.target, columns=df.chest_pain_type, margins=True)
observed = chestpain_target.ix[0:2,0:4]
chestpain_target


# In[ ]:


chi2, p, dof, expected = stats.chi2_contingency(observed=observed)


# In[ ]:


print("="*10+" Results "+"="*10)
print("Chi-squared Statistic: "+str(chi2))
print("p-value: "+str(p))
print("Degrees of Freedom: "+str(dof))
print("Expected: "+str(expected))
print("="*29)


# ### Conclusion
# > Since p-value 1.33e-17 is less than significance level a = 0.05, we have evidence to reject
# > the null hypotheseis and conclude there is an association between chest pain type and target

# # Two-sample t test for serum cholesterol levels (with and without heart disease)

# In[ ]:


# relationship between serum cholesterol and heart disease
sns.catplot(x="target", y="resting_blood_pressure", kind="box", data=df, palette="bwr")


# ### As we can see there is not much difference between the serum cholesterol levels of patients with or without heart disease, this is why a test is necessary for us egg heads :D
# ### State:
# > Ho: mean of x1 = mean of x2
# > Ha: mean of x1 != mean of x2
# ### Plan:
# > Confidence = 95%
# > a = 0.05
# > #### Conditions:
# > - Both samples are random
# > - Both samples >= 30
# > - Both samples are independent and less than 10% of total population

# In[ ]:


# 2 samples
no_disease_sc = df[df.target=="No Heart Disease"].serum_cholesterol
disease_sc = df[df.target=="Heart Disease"].serum_cholesterol


# In[ ]:


statistic, p = stats.ttest_ind(no_disease_sc, disease_sc)


# In[ ]:


print("="*10+" Results "+"="*10)
print("Statistic "+str(statistic))
print("p-value: "+str(p))
print("="*29)


# ## Conclusion
# P-value 0.139 is bigger than our significance level 0.05, we lack of convincing evidence to reject the Ho, thus conclude there is no difference in means of serum cholesterol levels of two samples (with and without disease)

# In[ ]:




