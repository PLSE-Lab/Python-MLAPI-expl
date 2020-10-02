#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt

import math


# In[ ]:


def shaking(x):
    return x + np.random.random(len(x))


# # The problem
# Prediction of Costa Ricon households poverty levels. The levels are varies from 1 to 4, where 1 stands for extreme poverty households and 4 stands for non vulnerable households.

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


print("Train rows %s" % train_df.shape[0])
print("Test rows %s" % test_df.shape[0])
print("Features %s" % len(test_df.columns.values))


# In[ ]:


train_df.sample(5)


# There are lots of columns. But some of them simply **one hot encoding** representation of some feature. Lets union them back.

# In[ ]:


one_hot_columns = {
    'wall_material' : ['paredblolad', 'paredzocalo','paredpreb','pareddes','paredmad','paredzinc', 'paredfibras','paredother'],
    'floor_material' : ['pisomoscer', 'pisocemento', 'pisoother','pisonatur','pisonotiene','pisomadera'],
    'roof_material' : ['techozinc', 'techoentrepiso', 'techocane', 'techootro'],
    'water_provision' : ['abastaguadentro', 'abastaguafuera', 'abastaguano'],
    'electricity' : ['public','planpri', 'noelec', 'coopele'],
    'toilet' : ['sanitario1','sanitario2', 'sanitario3', 'sanitario5', 'sanitario6'],
    'cooking_energy_source' : ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4'],
    'rubish_disposal' : ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6'],
    'wall' : ['epared1', 'epared2', 'epared3'],
    'roof' : ['etecho1', 'etecho2', 'etecho3'], 
    'floor' : ['eviv1', 'eviv2', 'eviv3'],
    'sex' : ['male', 'female'],
    'family_membership_status' : ['parentesco1','parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8','parentesco9','parentesco10','parentesco11','parentesco12'],
    'marital_status': ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7'],
    'education' : ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'],
    'dwelling_status' : ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5'],
    'region' : ['lugar1','lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6'], 
    'urban_rural' : ['area1', 'area2']
}


# In[ ]:


def merge_one_hot(df):
    all_one_hot = sum(one_hot_columns.values(), [])
    rest_df = df.drop(columns = all_one_hot)
    
    for name, group in one_hot_columns.items():
        rest_df[name] = (df[group] == 1).idxmax(1).astype('category')
    
    return rest_df


# In[ ]:


train_df = merge_one_hot(train_df)
test_df = merge_one_hot(test_df)


# In[ ]:


print("%s features" % len(test_df.columns.values))


# In[ ]:


train_df.head()


# # Nullable columns

# In[ ]:


train_df.columns[train_df.isnull().any()].values


# **v2a1** is monthly rent payment. Lets check, how it associated with other columns.

# In[ ]:


train_df.v2a1.describe()


# In[ ]:


train_df.v2a1.isna().sum()


# Lots of unfilled values. Look how unfilled rent distributed over welling status (owned, rented, etc).

# In[ ]:


display(train_df.groupby("dwelling_status").v2a1.apply(lambda x: x.isna().sum()))
train_df.groupby("dwelling_status").v2a1.apply(lambda x: x.isna().sum()).plot.bar()


# It looks like households, that has their own house, or lives in it for some precarious reasons doesn't pay a rent. So let fill the values with zeros.

# In[ ]:


train_df.loc[train_df.v2a1.isna(), 'v2a1'] = 0.0
test_df.loc[test_df.v2a1.isna(), 'v2a1'] = 0.0


# **v18q1** number of tablets, household have. As **v18q** indicates the fact of owning the tablet, we can suppose, that NaN means there are no tablets.

# In[ ]:


((train_df.v18q == 0) == (train_df.v18q1.isna())).mean() == 1.0


# In[ ]:


train_df.v18q1.fillna(0.0, inplace=True)
test_df.v18q1.fillna(0.0, inplace=True)


# **rez_esc** is Years behind in school. It makes sense, that the column is associated with **escolari** years of schooling.
# From wikipedia:
# > Education in Costa Rica is divided in 3 cycles: pre-education (before age 7), primary education (from 6-7 to 12-13), and secondary school (from 12-13 to 17-18), which leads to higher education.
# The primary education lasts six years and is divided in two cycles.
# The secondary education is divided in two cycles of three years. The first cycle is dedicated to general education. The second cycle, while keeping a core curriculum, implies a specialization. Specializations can be academic or technical.

# In[ ]:


train_df.rez_esc.unique()


# In[ ]:


train_df[['age', 'rez_esc']].apply(lambda x: x + np.random.random(len(x))*0.5).plot.scatter(x='age', y='rez_esc', s=1)


# In[ ]:


def get_primary_school_years(escolari, education, rez_esc):
    if education not in ['instlevel1', 'instlevel2']:
        return 6 # if somebody have primary education, he studied in primary school 6 years
    if np.isnan(rez_esc):
        return escolari
    return rez_esc

def get_secondary_school_years(escolari, education, rez_esc):
    if education in ['instlevel1', 'instlevel2', 'instlevel3']:
        return 0
    if education in ['instlevel4', 'instlevel5', 'instlevel6', 'instlevel7']:
        if np.isnan(rez_esc):
            return max(escolari - 6, 0)
        return rez_esc
    return 6

def get_high_school_years(escolari, education, rez_esc):
    if education not in ['instlevel8', 'instlevel9']:
        return 0
    return max(escolari - 12, 0)

def fill_education(df):
    df['primary_school_years'] = df[['escolari', 'education', 'rez_esc']].apply(lambda x: get_primary_school_years(*x), axis = 1)
    df['secondary_school_years'] = df[['escolari', 'education', 'rez_esc']].apply(lambda x: get_secondary_school_years(*x), axis = 1)
    df['high_school_years'] = df[['escolari', 'education', 'rez_esc']].apply(lambda x: get_high_school_years(*x), axis = 1)
    df['escolari'] = df.primary_school_years + df.secondary_school_years + df.high_school_years
    return df.drop(columns = ['rez_esc'])

train_df = fill_education(train_df)
test_df = fill_education(test_df)


# **meaneduc** is average years of education. What it means?

# In[ ]:


train_df[['idhogar', 'meaneduc']].head(15)


# As we see, the column have same values inside household.

# In[ ]:


train_df[['idhogar', 'meaneduc', 'age', 'primary_school_years', 'secondary_school_years']][train_df.meaneduc.isna()]


# Look likes, value is not filled for households without adults. Lets fill it with zero.

# In[ ]:


train_df.meaneduc.fillna(0.0, inplace=True)
test_df.meaneduc.fillna(0.0, inplace=True)


# **SQBmeaned** the previous value, but squared.

# In[ ]:


train_df['SQBmeaned'] = train_df.meaneduc ** 2
test_df['SQBmeaned'] = test_df.meaneduc ** 2


# # Columns with mixed data

# In[ ]:


train_df.columns[(train_df.dtypes == object)]


# **dependency** Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)

# In[ ]:


train_df[['dependency', 'age', 'idhogar']].head(30)


# Looks like *no* stands for *0*. But what means *yes*?

# In[ ]:


train_df.loc[train_df.idhogar.isin(train_df.idhogar[train_df.dependency == 'yes'].unique()), ['idhogar', 'age']].head(10)


# Looks like simpy uncalculated values. Lets calculate it ourself.

# In[ ]:


def calc_dependency(df):
    households_dependencies = df.groupby('idhogar')        .apply(lambda household: 
               ((household.age < 19) | (household.age > 64)).sum() / 
               household.age.apply(lambda a: (a > 18) & (a < 65)).sum())
    df['dependency'] = df.idhogar.map(households_dependencies).apply(lambda v: 8 if np.isinf(v) else v)

calc_dependency(train_df)
calc_dependency(test_df)


# **edjefe** and **edjefa**
# edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0

# In[ ]:


train_df.loc[
    (train_df.family_membership_status == 'parentesco1') & 
    train_df.edjefe.isin(['yes', 'no']) & 
    train_df.edjefa.isin(['yes', 'no']) , 
    ['idhogar', 'sex', 'edjefe', 'edjefa', 'primary_school_years', 'secondary_school_years']
].sample(10)


# In[ ]:


train_df.loc[
    (train_df.family_membership_status == 'parentesco1') & 
    ((~train_df.edjefe.isin(['yes', 'no'])) |
    (~train_df.edjefa.isin(['yes', 'no']))) , 
    ['idhogar', 'sex', 'edjefe', 'edjefa', 'primary_school_years', 'secondary_school_years']
].sample(10)


# Strange columns. Replace it with education years of household head.

# In[ ]:


def calc_head_escolari(df):
    heads_escolari = df.loc[df.family_membership_status == 'parentesco1', ['idhogar', 'escolari']].set_index('idhogar').escolari
    df['heads_escolari'] = df.idhogar.map(heads_escolari)
    df.heads_escolari.fillna(0, inplace = True)
    df.drop(columns = ['edjefa', 'edjefe'], inplace = True)
    
calc_head_escolari(train_df)
calc_head_escolari(test_df)


# # Data exploring

# In[ ]:


train_df.columns.values


# **qmobilephone** per person?

# In[ ]:


import seaborn as sns
corr = train_df[['mobilephone', 'qmobilephone', 'Target']]    .assign(mobilephone_per_person = train_df.qmobilephone/train_df.hogar_total)    .assign(person_per_mobilephone = train_df.hogar_total/train_df.qmobilephone)    .corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True)


# **person_per_mobilephone** gives highest correlation with Target. Add this column.****

# In[ ]:


train_df['person_per_mobilephone'] = train_df.hogar_total / train_df.qmobilephone
test_df['person_per_mobilephone'] = test_df.hogar_total / test_df.qmobilephone


# Repeat the same with tablets.

# In[ ]:


import seaborn as sns
corr = train_df[['v18q1', 'v18q', 'Target']]    .assign(tablet_per_person = train_df.v18q1/train_df.hogar_total)    .assign(person_per_tablet = train_df.hogar_total/train_df.v18q1)    .corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True)


# As we see, it doesn't imporove any correlations.

# In[ ]:


(train_df.hogar_total != train_df.tamhog).sum()


# As we see, **tamhog** and **hogar_total** are equals. Delete one. What about **tamviv**?

# In[ ]:


train_df.drop(columns=['tamhog'], inplace = True)
test_df.drop(columns=['tamhog'], inplace = True)


# In[ ]:


(train_df.groupby('idhogar').apply(lambda df: (df.tamviv - df.hogar_total).mean()) != 0).sum()


# There are 112 households, with some stange *guests*. Add as column?

# In[ ]:


train_df.assign(guests = train_df.tamviv - train_df.hogar_total)[['tamviv', 'guests', 'hogar_total', 'Target']].corr()


# Very low correlation. Skip it.

# Lets calculate average "device quantity" per person. Devices are fridge, phones, tv, computer, tablets.

# In[ ]:


corr = train_df    .assign(devices = train_df.computer + train_df.refrig + train_df.qmobilephone + train_df.v18q1 + train_df.television)    .assign(dev_per_person = lambda df: df.devices/df.hogar_total)    .assign(person_per_dev = lambda df: df.hogar_total/df.devices)[[
        'devices', 'dev_per_person', 'person_per_dev', 'computer', 'refrig', 'qmobilephone', 'v18q1', 'television', 'Target'
    ]].corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True)


# In[ ]:





# In[ ]:


fig, ax = plt.subplots(1, 2)
fig.suptitle("Rooms and bedrooms")
train_df.rooms.plot.hist(ax=ax[0], bins = len(train_df.rooms.unique()))
train_df.bedrooms.plot.hist(ax=ax[1], bins = 8)


# In[ ]:


fig, ax = plt.subplots(1, 2)
fig.set_figwidth(14)
fig.set_figheight(7)
fig.suptitle("Rooms and bedrooms by total undividuals")
fig.legend()
colors = {1:'red', 2:'blue', 3:'green', 4:'black'}
train_df[['hogar_total', 'rooms']]    .apply(lambda x: shaking(x)*0.1)    .plot.scatter(x = 'hogar_total', y = 'rooms', ax = ax[0], s = 2, c=train_df.Target.apply(lambda x: colors[x]))
train_df[['hogar_total', 'bedrooms']]    .apply(lambda x: shaking(x)*0.1)    .plot.scatter(x = 'hogar_total', y = 'bedrooms', ax = ax[1], s = 2, c=train_df.Target.apply(lambda x: colors[x]))


# In[ ]:


import seaborn as sns
corr = train_df[['rooms', 'bedrooms', 'hogar_total','overcrowding', 'Target']].assign(rooms_per_person = lambda df: df.hogar_total / (df.rooms - df.bedrooms)).corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True)
# plt.matshow(train_df[['rooms', 'bedrooms', 'hogar_total', 'Target']].corr())
# (train_df.bedrooms/train_df.hogar_total).plot.bar()


# In[ ]:




