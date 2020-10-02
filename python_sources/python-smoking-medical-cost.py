#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/insurance.csv')


# In[ ]:


print('data frame shape', df.shape); df.head(5)


# In[ ]:


######   step 1) identify missing values   #######

print('Missing Values')
print('\n')
print(df.isnull().sum().sort_values(ascending = False))
# no missing values


# In[ ]:


#---------------------#
####  Date Types   ####
#---------------------#

# convert categorical data like 'sex' and 'smoker' to numeric data types to perform correlation analysis

#----  Sex  ----#

#df['sex'].value_counts
# (male = 0; female = 1)
gender = {"male" : 0, "female" : 1}
df.sex = [gender[item] for item in df.sex]

#----  Smoker  ----#

#df['smoker'].value_counts
s = {"no" : 0, "yes" : 1}
df.smoker = [s[item] for item in df.smoker]

#----  Region  ----#

#df['region'].unique()
r = {'southwest' : 0, 'southeast' : 1, 'northwest' : 2, 'northeast' : 3}
df.region = [r[item] for item in df.region]


# In[ ]:


#---------------------------------#
####   Correlation Analysis   #####
#---------------------------------#

df.corr() # shows all the correlations but we want to know
# how they are all correlated to having At Risk
c_corr = df.corr()["charges"]
corr_charges = c_corr.abs().sort_values(ascending=False)[1:]
print("####  Correlation to Medical Costs  #####\n")
print(corr_charges)
# being a smoker has a 78% correlation to charges
# let's focus on the most correlated variables (smoker, age, bmi)


# In[ ]:


# reset the data types and rename the variables of interest for presentation
df = pd.read_csv('../input/insurance.csv')
df.rename(columns = {'charges' : 'Medical Costs',
                     'smoker' : 'Smoker'}, inplace = True)
#df.dtypes


# In[ ]:


#------------------------------#
####   Data Visulaizations  ####
#------------------------------#

# how many people are in the study?

col_list = ['light lavender','denim']
col_list_palette = sns.xkcd_palette(col_list)
sns.set_palette(col_list_palette)
sns.set_palette(col_list_palette)
sns.catplot(x = 'sex', data=df, kind='count',height=5, aspect=1.5, edgecolor="black")
plt.title('Count of Individuals in Study', fontsize = 20)
plt.xlabel('Gender')
print(df['sex'].value_counts())


# In[ ]:


# set the viz parameters

col_list = ["shit","pistachio"]
col_list_palette = sns.xkcd_palette(col_list)
sns.set_palette(col_list_palette)

sns.set(rc={"figure.figsize": (10,6)},
            palette = sns.set_palette(col_list_palette),
            context="talk",
            style="ticks")

# How many smokers are in this study?
sns.catplot(x = 'Smoker', data=df, kind='count',height=5,aspect=1.5, edgecolor="black")
plt.title('Count of Smokers in Study', fontsize = 20)
print(df['Smoker'].value_counts())


# In[ ]:


# Count of Smokers by Sex
sns.catplot(x ='sex',hue='Smoker', data=df, kind="count",height=6, aspect=1.8,legend_out=False, edgecolor="black")
plt.suptitle('Count of Smokers by Gender', fontsize = 20)
plt.xlabel('Gender')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print(pd.crosstab(df['Smoker'],df['sex']))


# In[ ]:


#----------------------------------#
####  Medical Cost by Smoker   #####
#----------------------------------#

sns.boxplot(x = "Smoker", y = "Medical Costs", data=df)
plt.suptitle('Medical Costs by Smokers', fontsize = 20)
plt.xlabel('Smoker', fontsize = 20)


# In[ ]:


#------------------------------------------------------------#
####   Distribution of Medical Cost by Smoker/Non-Smoker  ####
#------------------------------------------------------------#

sns.kdeplot(df[df['Smoker']=='yes']['Medical Costs'],lw=8, shade=True, label='Smoker', alpha = 0.6)
sns.kdeplot(df[df['Smoker']=='no']['Medical Costs'],lw=8, shade=True,label='Non-Smoker', alpha = 0.7)
plt.suptitle('Medical Costs by Smoker/Non-Smoker', fontsize = 20)
plt.xlabel('Medical Cost in Dollars')#;plt.ylabel('Frequency');plt.legend()


# In[ ]:


#-------------------------------------------------------#
#####    Statistical Comparison of Medical Costs    #####
#####         for Smokers and Non-Smokers           #####
#-------------------------------------------------------#

smoker = df[df['Smoker']=='yes']['Medical Costs']
non_smoker = df[df['Smoker']=='no']['Medical Costs']


# In[ ]:


#-------------------------------------------------------#
#####    Statistical Comparison of Medical Costs    #####
#####         for Smokers and Non-Smokers           #####
#-------------------------------------------------------#

# is there a statistically significant difference in medical costs for smokers and non-smokers? 

from scipy import stats
from statistics import variance

#-------------------------------------#
####   Equal Variance and T Test  #####
#-------------------------------------#

####   are the variances equal?   #####
stats.levene(smoker, non_smoker)
# tests the null hypothesis that all input samples are from
# populations with equal variances

####    are the means the same?   #####
stats.ttest_ind(smoker, non_smoker, equal_var = False)
# two-sided test for the null hypothesis that 2 independent samples
# have identical average (expected) values.

# p-value of t-test is well below 0.05 and so we reject the NULL Hypothesis 
# that the average medical costs for smokers and non-smokers is the same


# In[ ]:


print('The average medical cost for a non_smoker is ${:0,.0f} and the average medical costs for a smoker is ${:0,.0f}'.format(non_smoker.mean(), smoker.mean()))


# In[ ]:


#------------------------------------------------------#
####    Visualization Medical Cost Distributions    ####
####          for Smokers and Non-Smokers           ####
#------------------------------------------------------#

plt.figure(figsize=(11,6))
sns.kdeplot(smoker,lw=8, shade=True,label='Smoker', alpha = 0.7)
plt.axvline(np.mean(smoker), linestyle='--', linewidth = 5,color='#7f5f00')
sns.kdeplot(non_smoker ,lw=8, shade=True, label='Non-Smoker', alpha = 0.7)
plt.axvline(np.mean(non_smoker), linestyle='--', linewidth = 5,color = '#7ebd01') 
plt.suptitle('Medical Costs for Smokers & Non-Smokers', fontsize = 20)
plt.xlabel('Medical Costs');plt.ylabel('Frequency');plt.legend()


# In[ ]:


#--------------------------------------------#
####   Medical Cost by Smoker/Non-Smoker  ####
#--------------------------------------------#

sns.catplot(x='Smoker', y='Medical Costs',data=df, kind='violin', height=7, aspect=1.5, 
            linewidth = 6, legend=False)
plt.suptitle('Medical Cost by Smoker/Non-Smoker', fontsize = 25)
plt.xlabel('Smoker'); plt.ylabel('Medical Cost in Dollars')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]);


# In[ ]:


#--------------------------------------------------#
####  Age Distribution of Study Participants   #####
#--------------------------------------------------#

print('#####  Age Statistics  #######\n')
print('      Minimum     ',int(df['age'].min()))
print('      Average     ',int(df['age'].mean()))
print('      Median      ', int(np.median(df['age'])))
print('      Max         ',int(df['age'].max()))
print('   Std Deviation  ', int(df['age'].std()))


# In[ ]:


print('#####  Medical Cost Statistics  #######\n')
print('      Minimum      ${:,.0f}'.format(df['Medical Costs'].min()))
print('      Average     ${:,.0f}'.format(df['Medical Costs'].mean()))
print('      Median       ${:,.0f}'.format(np.median(df['Medical Costs'])))
print('      Max         ${:,.0f}'.format(df['Medical Costs'].max()))
print('   Std Deviation  ${:,.0f}'.format(df['Medical Costs'].std()))


# In[ ]:


#-------------------------------------------------------------#
#####   Age and Medical Costs Associated with Smoking    ######
#-------------------------------------------------------------#


# In[ ]:


#-------------------------------#
####   Medical Cost by Age   ####
#-------------------------------#

sns.regplot(x="age",y="Medical Costs", data=df, color='#3f9b0b', fit_reg=False, ci=None)
plt.suptitle('Medical Costs by Age', fontsize = 25)
plt.ylabel('Medical Cost in Dollars'),plt.xlabel('Age')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:


#-----------------------------------------#
####   Medical Cost by Age and Smoker  ####
#-----------------------------------------#

a = sns.FacetGrid(df, col='Smoker',hue='Smoker',height =6,aspect= 0.9)                  
a.map(plt.scatter, 'age', 'Medical Costs')
a.set_axis_labels('Age', 'Medical Costs in Dollars')
plt.suptitle('Medical Costs & Age by Smoker/Non-Smoker', fontsize = 25);plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:


#--------------------------------------------------------#
####  Body Mass Distribution of Study Participants   #####
#--------------------------------------------------------#

# Distribution of Body Mass
print('#####  Body Mass Statistics  #######\n')
print('      Minimum     ',int(df['bmi'].min()))
print('      Average     ',int(df['bmi'].mean()))
print('      Median      ', int(np.median(df['bmi'])))
print('      Max         ',int(df['bmi'].max()))
print('   Std Deviation  ', int(df['bmi'].std()))


# In[ ]:


#-------------------------------------------------------------------------#
#####   Body Mass Index and Medical Costs Associated with Smoking    ######
#-------------------------------------------------------------------------#


# In[ ]:


#----------------------------------------------#
#####   Medical Costs and Body Mass Index  #####
#----------------------------------------------#

plt.subplots(figsize=(8,8))
sns.regplot(x="bmi",y="Medical Costs", data=df, color='#3f9b0b')
plt.suptitle('Medical Costs and Body Mass Index')
plt.ylabel('Medical Cost in Dollars'),plt.xlabel('Body Mass Index')


# In[ ]:


#------------------------------------------------------#
#####   Medical Costs & Body Mass Index by Smoker  #####
#------------------------------------------------------#

a = sns.lmplot(x="bmi", y="Medical Costs", col="Smoker",
           hue="Smoker", data=df, height = 6)
plt.suptitle('Medical Costs & Body Mass Index by Smoker')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
a.set_axis_labels('Body Mass Index', 'Medical Costs in Dollars')


# In[ ]:


#---------------------------------#
####   binning the age data   #####
#---------------------------------#

b = np.linspace(round(min(df.age)-3), round(max(df.age)+1), num=11).astype(int)
df['age_binned'] = pd.cut(df['age'], bins = b); df.head(10)

# Group by the bin and calculate averages
avg_age  = df.groupby('age_binned').mean()


# In[ ]:


#---------------------------------------------#
#####   Age and Associated Medical Cost   #####
#---------------------------------------------#

plt.bar(avg_age.index.astype(str), avg_age['Medical Costs'], color='#3f9b0b')
plt.xticks(rotation = 75); plt.xlabel('Age'); plt.ylabel('Average Medical Cost in Dollars')
plt.suptitle('Age and Associated Medical Cost');


# In[ ]:


#---------------------------------------#
####  Age & Medical Costs by Smoker  ####
#---------------------------------------#

plt.subplots(figsize=(10,8))
g = sns.boxplot(x = 'age_binned', y = "Medical Costs", hue = "Smoker",data=df)
plt.suptitle('Age by Medical Costs & Smoker', fontsize = 20)
plt.xlabel('Age', fontsize = 20)
plt.xticks(rotation = 70)
plt.legend(loc='upper left', title = "Smoker")


# In[ ]:


#------------------------------#
####  binning the bmi data  ####
#------------------------------#

mnb = round(min(df.bmi)-1)
mxb = round(max(df.bmi)+2)
bm = np.linspace(mnb, mxb, num=11).astype(int)
df['bmi_binned'] = pd.cut(df['bmi'], bins = bm); df.head(10)
# Group by the bin and calculate averages
avg_bmi  = df.groupby('bmi_binned').mean()


# In[ ]:


#------------------------------------------#
####  Body Mass Index & Medical Costs   ####
#------------------------------------------#

plt.bar(avg_bmi.index.astype(str), avg_bmi['Medical Costs'], color='#3f9b0b')
plt.xticks(rotation = 75); plt.xlabel('Body Mass Index'); plt.ylabel('Average Medical Cost in Dollars')
plt.suptitle('Body Mass and Associated Medical Cost');


# In[ ]:


#---------------------------------------------------#
####  Body Mass Index & Medical Costs by Smoker  ####
#---------------------------------------------------#

g = sns.boxplot(x = 'bmi_binned', y = "Medical Costs", hue = "Smoker",data=df)
                # **boxplot_kwargs, palette = 'Reds')
plt.suptitle('Body Mass Index by Medical Costs & Smoker', fontsize = 20)
plt.xlabel('Body Mass Index', fontsize = 20)
plt.xticks(rotation = 70)


# In[ ]:


print('################################################')
print('############     Report Summary    #############')
print('################################################\n')
print('Average Age')
print('--------------')
print('Smoker      %0.0f' % df[df['Smoker']=='yes']['age'].mean())
print('Non-Smoker  %0.0f' %df[df['Smoker']=='no']['age'].mean(),'\n')
print('Average Body Mass')
print('--------------------')
print('Smoker      %0.0f' % df[df['Smoker']=='yes']['bmi'].mean())
print('Non-Smoker  %0.0f' % df[df['Smoker']=='no']['bmi'].mean(),'\n')
print('Average Medical Cost by Gender')
print('-----------------------------')
print('Female      ${:,.0f}'.format(df[df['sex']=='female']['Medical Costs'].mean()))
print('Male        ${:,.0f}'.format(df[df['sex']=='male']['Medical Costs'].mean()), '\n')
print('Average Medical Cost by Smoker')
print('--------------------------------')
print('Non-Smoker   ${:,.0f}'.format(df[df['Smoker']=='no']['Medical Costs'].mean()))
print('Smoker      ${:,.0f}'.format(df[df['Smoker']=='yes']['Medical Costs'].mean()))

#print('   Std Deviation  ${:,.0f}'.format(df['charges'].std()))


# In[ ]:


smoker = {"no" : 0, "yes" : 1}
df.Smoker = [smoker[item] for item in df.Smoker]

print('##############################################')
print('###########    Correlations       ############')
print('##############################################')
print('\n')
print('Cost to Smoker   ', round(df['Medical Costs'].corr(df['Smoker'])*100,1),'%')
print('Cost to Age      ', round(df['Medical Costs'].corr(df['age'])*100,1),'%')
print('Cost to Body Mass ', round(df['Medical Costs'].corr(df['bmi'])*100,1),'%')

