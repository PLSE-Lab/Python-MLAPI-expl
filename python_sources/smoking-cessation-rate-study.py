#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_excel(r'/kaggle/input/african-ngo-data/Data2 (1).xls')


# In[ ]:


df.head(400)


# In[ ]:


len(df[df['w3_smk_type'].isnull()==True])
#showing the no of cases where result of w3_smk_type is not mentioned i.e Nan is present 


# In[ ]:


len(df[df['w2_age']<df['w1_age']])
#44 cases where age of person is more for w1 instead of w2


# In[ ]:


len(df[df['w3_age']<df['w2_age']])
#75 cases where age of person is more for w2 instead of w3


# In[ ]:


len(df[(df['w2_b1']=='Yes I gave up')&(df['w3_b1']=='No, I did  not')])
#showing the no of people that gave up smoking in Wave 2 but started smoking again in Wave 3
#answer to 3rd question


# In[ ]:


df[df['w2_b1']=='Yes I gave up']


# In[ ]:


df.w2_b1[df['w2_b1']=='No, I did  not']=0
df.w2_b1[df['w2_b1']=='Yes I gave up']=1
#Assigning values against strings for assessment & data cleaning


# In[ ]:


df.head(5)


# In[ ]:


plt.figure()
df.plot(x='pid',y='w2_b1',kind='kde',figsize=(15,7),title='Smoking cessation graph b/w wave 1 & 2',color='red')
plt.legend()


# In[ ]:


plt.figure(figsize=(13,7))
sns.set_style('whitegrid')
sns.countplot(x='w2_b1',data=df)
plt.title('CountPlot for Wave2 status')


# In[ ]:


len(df[df['w2_b1']==1])
#displaying no of people that gave up smoking during wave 1 & wave 2


# In[ ]:


Smoking_cessation_rate= (84/1706)*100
print(Smoking_cessation_rate)


# In[ ]:


plt.figure(figsize=(12,7))
sns.pointplot(x='pid',y='w2_b1',data=df,linestyles='-',color='purple')
plt.title('Confidence Interval')


# In[ ]:


plt.figure(figsize=(13,7))
sns.set_style('whitegrid')
sns.countplot(x='w2_b1',data=df,hue='w1_gen',palette='coolwarm',saturation=8)
plt.title('Smoking cessation rate for Men and Women')


# In[ ]:


df[(df['w1_gen']=='Male') & (df['w2_b1']==1)]['w2_b1'].value_counts()
#total 70 men quit smoking b/w first and second wave


# In[ ]:


df[(df['w1_gen']=='Female') & (df['w2_b1']==1)]['w2_b1'].value_counts()
#total 14 women quit smoking b/w first and second wave


# In[ ]:


df[df['w1_gen']=='Male']['w1_gen'].value_counts()
#total 1530 men participated


# In[ ]:


df[df['w1_gen']=='Female']['w1_gen'].value_counts()
#total 176 women participated


# In[ ]:


cessation_rates_men=(70/1530)*100
print(cessation_rates_men)


# In[ ]:


cessation_rates_women= (14/176)*100
print(cessation_rates_women)


# In[ ]:


# Smoking cessation rate b/w wave 1 & wave 2 shows that women shows a better recovery pattern in comparison to men


# In[ ]:


df.head(5)


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='w2_b1',data=df,hue='w1_region',saturation=5)
plt.title('Region based Smoking cessation graph')
plt.legend()


# In[ ]:


df[(df['w1_region']=='Tbilisi') & (df['w2_b1']==1)]['w2_b1'].value_counts()


# In[ ]:


df[(df['w1_region']=='Tbilisi')]['w1_region'].value_counts()


# In[ ]:


SC_tbilisi_rate= (25/406)*100
print(SC_tbilisi_rate)


# In[ ]:


df[(df['w1_region']=='Kutaisi') & (df['w2_b1']==1)]['w2_b1'].value_counts()


# In[ ]:


df[(df['w1_region']=='Kutaisi')]['w1_region'].value_counts()


# In[ ]:


SC_Kutaisi_rate= (10/257)*100
print(SC_Kutaisi_rate)


# In[ ]:


df[(df['w1_region']=='Gori') & (df['w2_b1']==1)]['w2_b1'].value_counts()


# In[ ]:


df[(df['w1_region']=='Gori')]['w1_region'].value_counts()


# In[ ]:


SC_gori_rate= (17/370)*100
print(SC_gori_rate)


# In[ ]:


df[(df['w1_region']=='Zugdidi') & (df['w2_b1']==1)]['w2_b1'].value_counts()


# In[ ]:


df[(df['w1_region']=='Zugdidi')]['w1_region'].value_counts()


# In[ ]:


SC_Zugdidi_rate= (13/392)*100
print(SC_Zugdidi_rate)


# In[ ]:


df[(df['w1_region']=='Akhaltsikhe') & (df['w2_b1']==1)]['w2_b1'].value_counts()


# In[ ]:


df[(df['w1_region']=='Akhaltsikhe')]['w1_region'].value_counts()


# In[ ]:


SC_Akhaltsikhe_rate= (19/281)*100
print(SC_Akhaltsikhe_rate)


# In[ ]:


#OUT OF 5 REGIONS AKHALTSIKHE REGION SHOWS BEST PERFORMANCE WITH AN IMPRESSIVE SCORE OF 6.7615% IN QUITING SMOKING


# In[ ]:


df.head(10)


# In[ ]:


print(df['w1_smk_type'].unique())


# In[ ]:


def value(x):
    if x=='Filtered cigarettes':
        return 1;
    elif x=='Both filtered & non-filtered':
        return 2;
    elif x=='Unfiltered cigarettes':
        return 3;
    elif x=='Cigarettes & RYO':
        return 4;
    elif x=='RYO only':
        return 5;
    elif x=='Only other tobacco':
        return 6;
    


# In[ ]:


df['w1_smk_type']=df['w1_smk_type'].apply(func=value)


# In[ ]:


df.head(5)


# In[ ]:


df['w1_smk_type'].dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
plt.hist(df['w1_smk_type'],bins=40,histtype='barstacked',color='red')
plt.title('WAVE 1')


# In[ ]:


#plt.plot(df['pid'],df['w1_smk_type'],df['pid'],df['w2_smk_type'],df['pid'],df['w3_smk_type'])


# In[ ]:


df['w3_smk_type'].apply (pd.to_numeric, errors='coerce')
df.dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(12,6))
plt.hist(df['w2_smk_type'],bins=40,histtype='barstacked',color='blue')
plt.title('WAVE 2')


# In[ ]:


plt.figure(figsize=(12,6))
plt.hist(df['w3_smk_type'],bins=40,histtype='barstacked',color='violet')
plt.title('WAVE 3')


# In[ ]:





# 
# 
# # Thank You
# 
