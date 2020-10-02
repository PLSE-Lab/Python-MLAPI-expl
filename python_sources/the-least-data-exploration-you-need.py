#!/usr/bin/env python
# coding: utf-8

# This is interesting, I read the book "*Automating Inequality: How High-Tech Tools Profile, Police, and Punish the Poor*" by *Virginia Eubanks* a few months ago. There she discusses how predictive models can affect lives of tens of thousands of people (mostly very vulnerable) who are in need. Now that this challenge has come up on Kaggle, I can deeply feel how the models we build can affect people. It is **our responsibility** not to harm them systematically. This is a big deal and the first step is understanding the dataset so let's dive into it.
# 
# Note: no matter how much time I put into this, there is always room for improvement and let's not forget that **none of us is as smart as all of us**. Thus, any suggestion from you is appreciated and I will incorporate them into this kernel to approve its overall quality over time.
# 
# # Overview
# Let's import training data and see what we have:

# In[ ]:


import numpy as np 
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('../input/train.csv')
print(train_data.info())
train_data.describe()


# we have 143 columns, one of which is the row identifier "Id", the other one is the household identifier "idhogar", another one is the target variable "Target" which leaves us with **140 features**; we have **9557 predictors** or observations and event per variable (EPV) of approximately 68.
# Most of our variables (129) are integers, a few floats (8) and a few (3) objects.
# 
# # type=object
# Let's see what are the objects and what unique values are in there:

# In[ ]:


train_data.columns[train_data.dtypes==object]


# In[ ]:


print(train_data.dependency.unique())
print(train_data.edjefe.unique())
print(train_data.edjefa.unique())


# Our columns with object type could also be described in numbers. Great! but its not our goal in this kernel. We're here to explore.  Read the data description for these fields:
# *  dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64); we need a clarification from the database developer on this field for "yes" and "no" 
# * edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1  and no=0
# * edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# 
# 
# # households
# It's super imprtant that we have household identifier "idhogar" (another object-type column) which can be used to create household-wide features. Let's take a look at households:

# In[ ]:


households = train_data.groupby('idhogar').apply(lambda x: len(x))
print(households.describe())
plt.hist(households, bins=range(1, 13), align='left')
plt.xlabel("Number of household's members")
plt.ylabel('Number of households')
plt.grid(True)
plt.xlim([1, 13])
plt.xticks(range(1, 14))
plt.show()


# We have **2988 households** in our dataset each of which has **3 members on average** with a **maximum of 13 members**. The distribution is shown above. The distribution is close to log-normal as expected.
# 
# # Missing data
# Now, let us see how much data is missing.

# In[ ]:


train_data_na = (train_data.isnull().sum() / len(train_data)) * 100
train_data_na = train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :train_data_na})
f, ax = plt.subplots(figsize=(14, 6))
plt.xticks(rotation='90')
sns.barplot(x=train_data_na.index, y=train_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
missing_data


# Going over the data description again:
# * rez_esc, Years behind in school
# * v18q1, number of tablets household owns; *In this case all NaN data could be replaced with 0, NaN values corresponds to v18q = 0 (owns a tablet)*
# * v2a1, Monthly rent payment
# * SQBmeaned, square of the mean years of education of adults (>=18) in the household
# * meaneduc, average years of education for adults (18+)
# 
# # Target variable
# Target - the target is an categorial variable indicating groups of income levels.
# 
# 1 = extreme poverty 
# 2 = moderate poverty 
# 3 = vulnerable households 
# 4 = non vulnerable households

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].hist(train_data['Target'], bins=[0.5, 1.5, 2.5, 3.5, 4.5]);
ax[1].hist(train_data['Target'], bins=[0.5, 1.5, 2.5, 3.5, 4.5], normed=True);
ax[0].set_xlabel('Target variable')
ax[0].set_ylabel('number of people')
ax[0].set_xlim([0.5, 4.5])
ax[0].set_xticks(range(1, 5))
ax[0].grid(True)
ax[1].set_xlabel('Target variable')
ax[1].set_ylabel('percentage of people')
ax[1].set_yticks(np.arange(0.0, 0.7, 0.1))
ax[1].set_yticklabels(range(0, 70, 10))
ax[1].set_xlim([0.5, 4.5])
ax[1].set_xticks(range(1, 5))
ax[1].grid(True)
plt.show()


# # Correlation
# Let's see which features has the highest correlation with the target variable.

# In[ ]:


corrmat = train_data.dropna().corr().abs()['Target'].sort_values(ascending=False).drop('Target')
f, ax = plt.subplots(figsize=(20, 6))
plt.xticks(rotation='90')
sns.barplot(x=corrmat.head(50).index, y=corrmat.head(50))
plt.xlabel('Features', fontsize=15)
plt.ylabel('Abs correlation with Target variable', fontsize=15)
plt.show()


# The following features have the highest correlations with the Target variable:
# 1. meaneduc, average years of education for adults (18+)
# 2. hogar_nin, Number of children 0 to 19 in household
# 3. r4t1, persons younger than 12 years of age
# 4. SQBhogar_nin, hogar_nin squared
# 5. cielorazo, =1 if the house has ceiling
# 
# # Gender
# 

# In[ ]:


lables = ['younger than 12 years of age', '12 years of age and older', 'Total individuals in the household', 'gender']
men_cors = ['r4h1', 'r4h2', 'r4h3', 'male']
women_cors = ['r4m1', 'r4m2', 'r4m3', 'female']

fig, ax = plt.subplots(figsize=(12,6))
ind = np.arange(len(men_cors))
width = 0.35

p1 = ax.bar(ind, corrmat[men_cors].values, width, color='r', bottom=0)
p2 = ax.bar(ind + width, corrmat[women_cors].values, width,
            color='y', bottom=0)

ax.set_title('Correlation of variables by gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(lables)
ax.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.ylabel('correlation with target variable')
ax.autoscale_view()
plt.show()


# # Data inconsistency checks
# Let's check and see if there is any inconsistency within the data; if we get anything greater than zero below it means those rows of data are incosistent with each other.

# In[ ]:


train_data['v18q1'].fillna(0, inplace=True)
print('number of data rows with (NOT have tablet) & (number of tablets > 0) = %0.f' % len(train_data[(train_data['v18q'] == 0.0) & (train_data['v18q1'] > 0)]))
print('number of data rows with (NOT have mobile phone) & (number of phones > 0) = %0.f' % len(train_data[(train_data['mobilephone'] == 0.0) & (train_data['qmobilephone'] > 0)]))
print('sum(Total females in the household) - sum(Females younger than 12 years of age + Females 12 years of age and older) = {}'.format(train_data['r4m3'].sum() - train_data['r4m2'].sum() - train_data['r4m1'].sum()))
print('sum(Total males in the household) - sum(Males younger than 12 years of age + Males 12 years of age and older) = {}'.format(train_data['r4h3'].sum() - train_data['r4h2'].sum() - train_data['r4h1'].sum()))
print('sum(Total persons in the household) - sum(persons younger than 12 years of age + persons 12 years of age and older) = {}'.format(train_data['r4t3'].sum() - train_data['r4t2'].sum() - train_data['r4t1'].sum()))
print('number of rows for which gender is not specified = {}'.format(train_data['male'].sum() + train_data['female'].sum() - len(train_data)))

wall_material = ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother']
print('number of rows for which wall materials are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in wall_material])))
floor_material = ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']
print('number of rows for which floor materials are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in floor_material])))
roof_type = ['techozinc', 'techoentrepiso', 'techocane', 'techootro']
print('number of rows for which roof types are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in roof_type])))
toilet_status = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']
print('number of rows for which toilet status are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in toilet_status])))
cooking_energy_source = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']
print('number of rows for which source of cooking energies are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in cooking_energy_source])))
rubbish_disposal_type = ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']
print('number of rows for which rubbish disposal types are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in rubbish_disposal_type])))
marital_status = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']
print('number of rows for which marital status are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in marital_status])))
role_in_family = ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
print('number of rows for which roles in family are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in role_in_family])))
level_of_education = ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']
print('number of rows for which level of educations are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in level_of_education])))
house_ownership = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
print('number of rows for which house owenerships are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in house_ownership])))
region = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']
print('number of rows for which regions are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in region])))
area = ['area1', 'area2']
print('number of rows for which areas are not described in database = {}'.format(len(train_data) - sum([train_data[_].sum() for _ in area])))


# Data is consistent except for the roof types; there are 66 rows for which the roof type is not specified and 3 rows for which the education level is not defined. Let's check it with another method to make sure we got it right the first time.

# In[ ]:


print('number of rows with undefined roof type = {}'.format(len(train_data[(train_data['techozinc'] == 0) & (train_data['techoentrepiso'] == 0) & (train_data['techocane'] == 0) & (train_data['techootro'] == 0)])))
print('number of rows with undefined education level = {}'.format(len(train_data) - sum(train_data[level_of_education].sum(axis=1))))


# Yep. There are **66 rows with undefined roof type**; and **3 rows with undefined education level**.
# 
# # Other distributions
# Below are our categorial variables for which we want to see how they are distributed. 
# 
# * Wall type:
# paredblolad, =1 if predominant material on the outside wall is block or brick
# paredzocalo, "=1 if predominant material on the outside wall is socket (wood,  zinc or absbesto"
# paredpreb, =1 if predominant material on the outside wall is prefabricated or cement
# pareddes, =1 if predominant material on the outside wall is waste material
# paredmad, =1 if predominant material on the outside wall is wood
# paredzinc, =1 if predominant material on the outside wall is zink
# paredfibras, =1 if predominant material on the outside wall is natural fibers
# paredother, =1 if predominant material on the outside wall is other
# 
# * Floor type:
# pisomoscer, "=1 if predominant material on the floor is mosaic,  ceramic,  terrazo"
# pisocemento, =1 if predominant material on the floor is cement
# pisoother, =1 if predominant material on the floor is other
# pisonatur, =1 if predominant material on the floor is  natural material
# pisonotiene, =1 if no floor at the household
# pisomadera, =1 if predominant material on the floor is wood
# 
# * Roof type:
# techozinc, =1 if predominant material on the roof is metal foil or zink
# techoentrepiso, "=1 if predominant material on the roof is fiber cement,  mezzanine "
# techocane, =1 if predominant material on the roof is natural fibers
# techootro, =1 if predominant material on the roof is other
# 
# * Toilet type:
# sanitario1, =1 no toilet in the dwelling
# sanitario2, =1 toilet connected to sewer or cesspool
# sanitario3, =1 toilet connected to  septic tank
# sanitario5, =1 toilet connected to black hole or letrine
# sanitario6, =1 toilet connected to other system
# 
# * Energy for cooking type:
# energcocinar1, =1 no main source of energy used for cooking (no kitchen)
# energcocinar2, =1 main source of energy used for cooking electricity
# energcocinar3, =1 main source of energy used for cooking gas
# energcocinar4, =1 main source of energy used for cooking wood charcoal
# 
# * Rubbish disposal type:
# elimbasu1, =1 if rubbish disposal mainly by tanker truck
# elimbasu2, =1 if rubbish disposal mainly by botan hollow or buried
# elimbasu3, =1 if rubbish disposal mainly by burning
# elimbasu4, =1 if rubbish disposal mainly by throwing in an unoccupied space
# elimbasu5, "=1 if rubbish disposal mainly by throwing in river,  creek or sea"
# elimbasu6, =1 if rubbish disposal mainly other
# 
# * Marital status:
# estadocivil1, =1 if less than 10 years old
# estadocivil2, =1 if free or coupled uunion
# estadocivil3, =1 if married
# estadocivil4, =1 if divorced
# estadocivil5, =1 if separated
# estadocivil6, =1 if widow/er
# estadocivil7, =1 if single
# 
# * Role in household:
# parentesco1, =1 if household head
# parentesco2, =1 if spouse/partner
# parentesco3, =1 if son/doughter
# parentesco4, =1 if stepson/doughter
# parentesco5, =1 if son/doughter in law
# parentesco6, =1 if grandson/doughter
# parentesco7, =1 if mother/father
# parentesco8, =1 if father/mother in law
# parentesco9, =1 if brother/sister
# parentesco10, =1 if brother/sister in law
# parentesco11, =1 if other family member
# parentesco12, =1 if other non family member
# 
# * Education level:
# instlevel1, =1 no level of education
# instlevel2, =1 incomplete primary
# instlevel3, =1 complete primary
# instlevel4, =1 incomplete academic secondary level
# instlevel5, =1 complete academic secondary level
# instlevel6, =1 incomplete technical secondary level
# instlevel7, =1 complete technical secondary level
# instlevel8, =1 undergraduate and higher education
# instlevel9, =1 postgraduate higher education
# 
# * House ownership:
# tipovivi1, =1 own and fully paid house
# tipovivi2, "=1 own,  paying in installments"
# tipovivi3, =1 rented
# tipovivi4, =1 precarious
# tipovivi5, "=1 other(assigned,  borrowed)"
# 
# * Region:
# lugar1, =1 region Central
# lugar2, =1 region Chorotega
# lugar3, =1 region PacÃ­fico central
# lugar4, =1 region Brunca
# lugar5, =1 region Huetar AtlÃ¡ntica
# lugar6, =1 region Huetar Norte
# 
# * Urban/rural:
# area1, =1 zona urbana
# area2, =2 zona rural

# In[ ]:


f, ax = plt.subplots(12, 1, figsize=(14, 168))
types = [wall_material, floor_material, roof_type, toilet_status, cooking_energy_source, rubbish_disposal_type, marital_status, role_in_family, level_of_education, house_ownership, region, area]
types_titles = ['wall materials', 'floor materials', 'roof type', 'toilet status', 'source of cooking energy', 'rubbish disposal type', 'marital status', 'role in family', 'level of education', 'house ownership', 'region', 'area']
for i, t in enumerate(types):
    sns.barplot(x=t, y=train_data[t].sum().values, ax=ax[i])
    ax[i].set_title('distribution of ' + types_titles[i])

plt.show()


# # Age distribution
# from https://www.kaggle.com/hrmello/a-first-look-at-the-target

# In[ ]:


sns.kdeplot(train_data.age, legend=False)
plt.title('overall age distribution')
plt.xlabel("Age");


# In[ ]:


p = sns.FacetGrid(data = train_data, hue = 'Target', size = 5, legend_out=True)
p = p.map(sns.kdeplot, 'age')
plt.legend()
plt.title("Age distribution by household condition")
p;


# # Transformable variables:
# The following variables could be lumped into one variable in order to reduce number of features:
# 
# wall quality:
# * epared1, =1 if walls are bad
# * epared2, =1 if walls are regular
# * epared3, =1 if walls are good
# 
# roof quality:
# * etecho1, =1 if roof are bad
# * etecho2, =1 if roof are regular
# * etecho3, =1 if roof are good
# 
# floor quality:
# * eviv1, =1 if floor are bad
# * eviv2, =1 if floor are regular
# * eviv3, =1 if floor are good
# 
# level of education:
# * instlevel1, =1 no level of education
# * instlevel2, =1 incomplete primary
# * instlevel3, =1 complete primary
# * instlevel4, =1 incomplete academic secondary level
# * instlevel5, =1 complete academic secondary level
# * instlevel6, =1 incomplete technical secondary level
# * instlevel7, =1 complete technical secondary level
# * instlevel8, =1 undergraduate and higher education
# * instlevel9, =1 postgraduate higher education
# 
# house ownership:
# * tipovivi1, =1 own and fully paid house
# * tipovivi2, "=1 own,  paying in installments"
# * tipovivi3, =1 rented
# * tipovivi4, =1 precarious
# * tipovivi5, "=1 other(assigned,  borrowed)"
# 
# area:
# * area1, =1 zona urbana
# * area2, =2 zona rural
