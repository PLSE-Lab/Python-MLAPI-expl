#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import sort

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import multiprocessing

n_jobs = multiprocessing.cpu_count()
n_jobs


# In[ ]:


#prediction and Classification Report
from sklearn.metrics import classification_report

# select features using threshold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics.scorer import make_scorer

# plot tree, importance
from xgboost import plot_tree, plot_importance


# In[ ]:


# load xgboost, test train split
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/train.csv')


# In[ ]:


num_of_cols = len(list(df.columns))
num_of_cols


# In[ ]:


pd.options.display.max_columns = num_of_cols


# In[ ]:


len(df)


# In[ ]:


# columns with null values
df_isna = pd.DataFrame(df.isnull().sum())
df_isna.loc[(df_isna.loc[:, df_isna.dtypes != object] != 0).any(1)]


# In[ ]:


nan_cols = list(df_isna.loc[(df_isna.loc[:, df_isna.dtypes != object] != 0).any(1)].T.columns)
nan_cols


# In[ ]:


df[nan_cols].describe()


# In[ ]:


df.describe(include='all')


# In[ ]:


df[nan_cols].sample(3000).describe()


# **Multiple people can be part of a single household. Only predictions for heads of household are scored.**

# In[ ]:


df['parentesco1'].loc[df.parentesco1 == 1].describe()


# In[ ]:


(df['parentesco1'].loc[df.parentesco1 == 1].describe()['count']/len(df))*100


# Nearly one-third (31%) of the ID's are heads of the household!
# 
# Need to build a dataframe of houses-- ie, all people belonging to same household in one row
# 
# Something like, ID's, house details, facilities, etc. All persons belonging to the same household will have the same Target!?***
# 
# *** to be verified!!
# 
# Apparently, it does ... as stated by "*idhogar - this is a unique identifier for each household. This can be used to create household-wide features, etc. All rows in a given household will have a matching value for this identifier.*"
# 
# 

# In[ ]:


# find number of households

df['idhogar'].describe()


# There's a difference of 15 between unique households and number of persons who're head of the houses!
# 
# 2988 - 2973
# 
# What could be the reason for this?
# 
# 

# In[ ]:


house_ids = list(df['idhogar'].unique())


# In[ ]:


df[['parentesco1','idhogar']].loc[df.parentesco1 == 1].head(5)


# In[ ]:


hid_heads = df.groupby(['idhogar'])['parentesco1'].apply(lambda x: pd.unique(x.values.ravel()).tolist()).reset_index()
len(hid_heads)


# In[ ]:


df_hid = pd.DataFrame(hid_heads, index=None, columns=['idhogar','parentesco1'])
df_hid.sample(5)


# In[ ]:


df_hid['parentesco1'] = df_hid['parentesco1'].apply(lambda x: ''.join(map(str, x)))


# In[ ]:


df_hid.sample(5)


# In[ ]:


df_hid.loc[df_hid.parentesco1 == '0']


# In[ ]:


# id's without head!
hid_wo_heads = list(df_hid['idhogar'].loc[df_hid.parentesco1 == '0'])
len(hid_wo_heads)


# In[ ]:


df_hwoh = df[df['idhogar'].isin(hid_wo_heads)]


# In[ ]:


df_hwoh[['idhogar', 'parentesco1','v2a1']]


# In[ ]:


df['v2a1'].hist()


# In[ ]:


df['v2a1'].loc[-df['idhogar'].isin(hid_wo_heads)].hist()


# In[ ]:


df_hwoh['v2a1'].hist()


# In[ ]:


len(df_hwoh)


# In[ ]:


# these 15 households (23 rows) doesn't have a head..
# we should exclude these from analysis and scoring perhaps...
df_hwoh['idhogar'].unique()


# In[ ]:


print(df[['Id','v2a1','idhogar','parentesco1','Target']].loc[df.idhogar == '09b195e7a'])
print(df[['Id','v2a1','idhogar','parentesco1','Target']].loc[df.idhogar == 'f2bfa75c4'])


# In[ ]:


# required dataframe - without households without a head!!
print("before removal: ", len(df))
df = df.loc[-df['idhogar'].isin(hid_wo_heads)]
print("after removal: ", len(df))


# # Missing Values!!

# In[ ]:


df['v2a1'].describe().plot()


# In[ ]:


df[['v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']].describe().plot()


# In[ ]:


import gc

gc.collect()


# In[ ]:


len(df['v2a1'].unique())


# In[ ]:


df['v2a1'].unique()


# In[ ]:


df['v2a1'].max()


# > ### Outliers

# In[ ]:


df[['v2a1','idhogar','parentesco1','Target']].loc[df.v2a1 > 1000000]


# In[ ]:


df[['v2a1','idhogar','parentesco1','Target']].loc[df.v2a1 >= 1000000]


# In[ ]:


# remove these two rows...
df[['v2a1','idhogar','parentesco1','Target']].loc[df.idhogar == '563cc81b7']


# In[ ]:


print("before removal: ", len(df))
df.drop(df[df.idhogar == '563cc81b7'].index, inplace=True)
print("after removal: ", len(df))


# In[ ]:


df['v2a1'].hist()


# In[ ]:


sns.kdeplot(df['v2a1'])


# In[ ]:


sns.kdeplot(df['v18q1'])


# In[ ]:


sns.kdeplot(df['rez_esc'])


# In[ ]:


sns.kdeplot(df['meaneduc'])


# In[ ]:


sns.kdeplot(df['SQBmeaned'])


# All columns with NaN values follows nearly the same distribution! 
# Let's wait to see the remaining column distributions before filling NaN values.

# In[ ]:


cols = list(df.columns)
cols


# In[ ]:


df.sample(10)


# In[ ]:


set(df.dtypes)


# Only 3 types of data; no dates!?

# In[ ]:


col_types = {}

for col in cols:
    col_types[col] = df[col].dtype
    # print(col, df[col].dtype)


# In[ ]:


# import collections

# od = collections.OrderedDict(sorted(col_types.items()))

#for k, v in od.items():
#    print(k, v)        # sorted columns by name 


# In[ ]:


# alternately we can use just sorted
# sorted(col_types)


# In[ ]:


print(len(col_types))


# In[ ]:


for key in sorted(col_types):
    print(key, col_types[key])


# In[ ]:


cat_cols = []
num_cols = []
for col in cols:
    if df[col].dtype == 'O':
        cat_cols.append(col)
        print(col, df[col].dtype)
    else:
        num_cols.append(col)


# In[ ]:


# categorical columns
cat_cols


# In[ ]:


df[cat_cols].sample(10)


# idhogar - this is a unique identifier for each household. This **can be used to create household-wide features**, etc. 
# 
# > All rows in a given household will have a matching value for this identifier.
# 
# 

# In[ ]:


len(num_cols)


# In[ ]:


# numerical columns
sorted(num_cols)


# In[ ]:


g = sns.PairGrid(df[nan_cols])
g = g.map_offdiag(plt.scatter)


# * v2a1, Monthly rent payment
# * hacdor, =1 Overcrowding by bedrooms
# * rooms,  number of all rooms in the house
# * hacapo, =1 Overcrowding by rooms
# * v14a, =1 has bathroom in the household
# * refrig, =1 if the household has refrigerator
# * v18q, owns a tablet
# * v18q1, number of tablets household owns
# * r4h1, Males younger than 12 years of age
# * r4h2, Males 12 years of age and older
# * r4h3, Total males in the household
# * r4m1, Females younger than 12 years of age
# * r4m2, Females 12 years of age and older
# * r4m3, Total females in the household
# * r4t1, persons younger than 12 years of age
# * r4t2, persons 12 years of age and older
# * r4t3, Total persons in the household
# * tamhog, size of the household
# * tamviv, number of persons living in the household
# * escolari, years of schooling
# * rez_esc, Years behind in school
# * hhsize, household size
# * paredblolad, =1 if predominant material on the outside wall is block or brick
# * paredzocalo, "=1 if predominant material on the outside wall is socket (wood,  zinc or absbesto"
# * paredpreb, =1 if predominant material on the outside wall is prefabricated or cement
# * pareddes, =1 if predominant material on the outside wall is waste material
# * paredmad, =1 if predominant material on the outside wall is wood
# * paredzinc, =1 if predominant material on the outside wall is zink
# * paredfibras, =1 if predominant material on the outside wall is natural fibers
# * paredother, =1 if predominant material on the outside wall is other
# * pisomoscer, "=1 if predominant material on the floor is mosaic,  ceramic,  terrazo"
# * pisocemento, =1 if predominant material on the floor is cement
# * pisoother, =1 if predominant material on the floor is other
# * pisonatur, =1 if predominant material on the floor is  natural material
# * pisonotiene, =1 if no floor at the household
# * pisomadera, =1 if predominant material on the floor is wood
# * techozinc, =1 if predominant material on the roof is metal foil or zink
# * techoentrepiso, "=1 if predominant material on the roof is fiber cement,  mezzanine "
# * techocane, =1 if predominant material on the roof is natural fibers
# * techootro, =1 if predominant material on the roof is other
# * cielorazo, =1 if the house has ceiling
# * abastaguadentro, =1 if water provision inside the dwelling
# * abastaguafuera, =1 if water provision outside the dwelling
# * abastaguano, =1 if no water provision
# * public, "=1 electricity from CNFL,  ICE,  ESPH/JASEC"
# * planpri, =1 electricity from private plant
# * noelec, =1 no electricity in the dwelling
# * coopele, =1 electricity from cooperative
# * sanitario1, =1 no toilet in the dwelling
# * sanitario2, =1 toilet connected to sewer or cesspool
# * sanitario3, =1 toilet connected to  septic tank
# * sanitario5, =1 toilet connected to black hole or letrine
# * sanitario6, =1 toilet connected to other system
# * energcocinar1, =1 no main source of energy used for cooking (no kitchen)
# * energcocinar2, =1 main source of energy used for cooking electricity
# * energcocinar3, =1 main source of energy used for cooking gas
# * energcocinar4, =1 main source of energy used for cooking wood charcoal
# * elimbasu1, =1 if rubbish disposal mainly by tanker truck
# * elimbasu2, =1 if rubbish disposal mainly by botan hollow or buried
# * elimbasu3, =1 if rubbish disposal mainly by burning
# * elimbasu4, =1 if rubbish disposal mainly by throwing in an unoccupied space
# * elimbasu5, "=1 if rubbish disposal mainly by throwing in river,  creek or sea"
# * elimbasu6, =1 if rubbish disposal mainly other
# * epared1, =1 if walls are bad
# * epared2, =1 if walls are regular
# * epared3, =1 if walls are good
# * etecho1, =1 if roof are bad
# * etecho2, =1 if roof are regular
# * etecho3, =1 if roof are good
# * eviv1, =1 if floor are bad
# * eviv2, =1 if floor are regular
# * eviv3, =1 if floor are good
# * dis, =1 if disable person
# * male, =1 if male
# * female, =1 if female
# * estadocivil1, =1 if less than 10 years old
# * estadocivil2, =1 if free or coupled uunion
# * estadocivil3, =1 if married
# * estadocivil4, =1 if divorced
# * estadocivil5, =1 if separated
# * estadocivil6, =1 if widow/er
# * estadocivil7, =1 if single
# * parentesco1, =1 if household head
# * parentesco2, =1 if spouse/partner
# * parentesco3, =1 if son/doughter
# * parentesco4, =1 if stepson/doughter
# * parentesco5, =1 if son/doughter in law
# * parentesco6, =1 if grandson/doughter
# * parentesco7, =1 if mother/father
# * parentesco8, =1 if father/mother in law
# * parentesco9, =1 if brother/sister
# * parentesco10, =1 if brother/sister in law
# * parentesco11, =1 if other family member
# * parentesco12, =1 if other non family member
# * idhogar, Household level identifier
# * hogar_nin, Number of children 0 to 19 in household
# * hogar_adul, Number of adults in household
# * hogar_mayor, # of individuals 65+ in the household
# * hogar_total, # of total individuals in the household
# * dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# * edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# * edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# * meaneduc,average years of education for adults (18+)
# * instlevel1, =1 no level of education
# * instlevel2, =1 incomplete primary
# * instlevel3, =1 complete primary
# * instlevel4, =1 incomplete academic secondary level
# * instlevel5, =1 complete academic secondary level
# * instlevel6, =1 incomplete technical secondary level
# * instlevel7, =1 complete technical secondary level
# * instlevel8, =1 undergraduate and higher education
# * instlevel9, =1 postgraduate higher education
# * bedrooms, number of bedrooms
# * overcrowding, # persons per room
# * tipovivi1, =1 own and fully paid house
# * tipovivi2, "=1 own,  paying in installments"
# * tipovivi3, =1 rented
# * tipovivi4, =1 precarious
# * tipovivi5, "=1 other(assigned,  borrowed)"
# * computer, =1 if the household has notebook or desktop computer
# * television, =1 if the household has TV
# * mobilephone, =1 if mobile phone
# * qmobilephone, # of mobile phones
# * lugar1, =1 region Central
# * lugar2, =1 region Chorotega
# * lugar3, =1 region PacÃ­fico central
# * lugar4, =1 region Brunca
# * lugar5, =1 region Huetar AtlÃ¡ntica
# * lugar6, =1 region Huetar Norte
# * area1, =1 zona urbana
# * area2, =2 zona rural
# * age, Age in years
# * SQBescolari, escolari squared
# * SQBage, age squared
# * SQBhogar_total, hogar_total squared
# * SQBedjefe, edjefe squared
# * SQBhogar_nin, hogar_nin squared
# * SQBovercrowding, overcrowding squared
# * SQBdependency, dependency squared
# * SQBmeaned, square of the mean years of education of adults (>=18) in the household
# * agesq, Age squared

# In[ ]:


cols_electronics = ['refrig','mobilephone','television','qmobilephone','computer', 'v18q', 'v18q1', ]
cols_house_details = ['v2a1', 'area1', 'area2', 'bedrooms','rooms', 'cielorazo', 'v14a', 
                    'tamhog', 'hacdor', 'hacapo', 'r4t3', ]
cols_person_details = ['age', 'agesq', 'female', 'male',]
cols_SQ = ['SQBage', 'SQBdependency', 'SQBedjefe', 'SQBescolari', 'SQBhogar_nin', 
           'SQBhogar_total', 'SQBmeaned', 'SQBovercrowding',]
cols_water = ['abastaguadentro', 'abastaguafuera', 'abastaguano',]

cols_h = [ 'hhsize', 'hogar_adul', 'hogar_mayor', 'hogar_nin', 'hogar_total',]
cols_r = ['r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3',]
cols_tip = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',]
cols_roof = ['techocane', 'techoentrepiso', 'techootro', 'techozinc',]
cols_floor = ['pisocemento', 'pisomadera', 'pisomoscer', 'pisonatur', 'pisonotiene', 'pisoother',]
cols_sanitary = [ 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6',]
cols_parents = [ 'parentesco1', 'parentesco10', 'parentesco11', 'parentesco12', 'parentesco2', 'parentesco3',
                'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9',]
cols_outside_wall = [ 'paredblolad', 'pareddes', 'paredfibras', 'paredmad', 'paredother', 
              'paredpreb', 'paredzinc', 'paredzocalo',]
cols_instlevel = [ 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6',
                  'instlevel7', 'instlevel8', 'instlevel9',]
cols_lugar = [ 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6',]
cols_estadoc = [ 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 
                'estadocivil5', 'estadocivil6', 'estadocivil7',]
cols_elim = ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6',]
cols_energ = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',]
cols_eviv = [ 'eviv1', 'eviv2', 'eviv3',]
cols_etech = [ 'etecho1', 'etecho2', 'etecho3',]
cols_pared = [ 'epared1', 'epared2', 'epared3',]
cols_unknown = [ 'dis', 'escolari', 'meaneduc', 
                'overcrowding', 'rez_esc', 'tamhog', 'tamviv', ]
cols_elec = ['coopele', 'noelec', 'planpri', 'public',]

total_features = cols_electronics+cols_house_details+cols_person_details+cols_SQ+cols_water+cols_h+cols_r+cols_tip+cols_roof+cols_floor+cols_sanitary+cols_parents+cols_outside_wall+cols_instlevel+cols_lugar+cols_estadoc+cols_elim+cols_energ+cols_eviv+cols_etech+cols_pared+cols_unknown+cols_elec

len(total_features)


# In[ ]:


df[cols_electronics].plot.area()


# In[ ]:


df['Target'].unique()


# In[ ]:


cols_electronics_target = cols_electronics.append('Target')
df[cols_electronics].corr()


# In[ ]:


cols_electronics.remove('Target')
cols_electronics


# In[ ]:


df.groupby('Target')[cols_electronics].sum()


# In[ ]:


df['tamhog'].unique()


# In[ ]:


# high correlation between 
# no. of persons in the household,
# persons living in the household 
# and size of the household
# we can use any one...!!
df[['tamhog','r4t3', 'tamviv']].corr()


# In[ ]:


df[['r4t3','tamviv']].corr()


# In[ ]:


total_features.remove('r4t3')
total_features.remove('tamhog')
total_features.remove('tamviv')

len(total_features)


# In[ ]:


df['escolari'].unique()


# In[ ]:


df['escolari'].hist()


# In[ ]:


df['escolari'].describe()


# In[ ]:


df['escolari'].plot.line()


# In[ ]:


sns.kdeplot(df.escolari)


# In[ ]:


correlations = df[num_cols].corr()


# In[ ]:


# correlation heatmap masking
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(17, 13))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=.3, center=0,
square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


# difficult to look into the above one
es_corr = df[num_cols].corrwith(df.escolari, axis=0)


# In[ ]:


for x,y in zip(num_cols, list(es_corr)):
    if (y >= 0.75) or (y < -0.6):
        print(x,y)


# In[ ]:


# escolari is highly correlated to SQBescolari
# let's see if SQBescolari is correlated to cols_house_details


# In[ ]:


sqbes_corr = df[num_cols].corrwith(df.SQBescolari, axis=0)


# In[ ]:


for x,y in zip(num_cols, list(sqbes_corr)):
    if (y >= 0.5) or (y < -0.6):
        print(x,y)


# In[ ]:


total_features.remove('escolari')
len(total_features)


# In[ ]:


df.loc[df.Target == 1].groupby('overcrowding').SQBescolari.value_counts().unstack().plot.bar()


# In[ ]:


df['overcrowding'].hist()


# In[ ]:


df['overcrowding'].unique()


# In[ ]:


df.plot.scatter(x='Target', y='overcrowding')


# In[ ]:


df.groupby('Target').overcrowding.value_counts().unstack().plot.bar()


# In[ ]:


df['Target'].describe()


# Target - the target is an ordinal variable indicating groups of income levels.
# 1. extreme poverty
# 1. moderate poverty
# 1. vulnerable households
# 1. non vulnerable households 

# In[ ]:


df['Target'].unique()


# In[ ]:


df['Target'].hist()


# ---

# In[ ]:


nan_cols


# In[ ]:


# filling missing values

df[nan_cols].corr()


# In[ ]:


for col in nan_cols:
    if col != 'v2a1':
        print(col, df[col].unique())


# In[ ]:


# there's a clear quadratic relation between meaneduc and SQBmeaned
# hence, we can ignore either one of these..say, meaneduc
sns.regplot(df['meaneduc'],df['SQBmeaned'], order=2)


# In[ ]:


# filling na values in meaneduc and SQBmeaned
df['meaneduc'].fillna(0, inplace=True)
df['SQBmeaned'].fillna(0, inplace=True)


# In[ ]:


total_features.remove('meaneduc')

total_features


# In[ ]:


# we can fill v18q1 (household tablets) with 0 as individual tablet count is 0 for all such columns
df[['v18q','v18q1','idhogar']].loc[df.v18q1.isna()].describe()


# In[ ]:


df['v18q1'] = df['v18q'].groupby(df['idhogar']).transform('sum')


# In[ ]:


df.sample(7)


# In[ ]:


ff = pd.DataFrame(df.isnull().sum())
ff.loc[(ff.loc[:, ff.dtypes != object] != 0).any(1)]


# In[ ]:


# rez_esc - years behind in school
df['rez_esc'].describe()


# In[ ]:


df['rez_esc'].isnull().sum()


# In[ ]:


# only these many rows has values for years behind school
len(df) - df['rez_esc'].isnull().sum()


# In[ ]:


df['v2a1'].isnull().sum()


# In[ ]:


# only these many rows has values for income
len(df) - df['v2a1'].isnull().sum()


# In[ ]:


# number of rows where income and rez_esc has values
len(df.loc[(df.v2a1 >= 0)]), len(df.loc[(df.rez_esc >= 0)])


# In[ ]:


# how many rows with nan values for both income and rez_esc 
len(df.loc[(df.v2a1 >= 0) & (df.rez_esc >= 0)])


# In[ ]:


# how many rows with nan values for either income or rez_esc 
len(df.loc[(df.v2a1 >= 0) | (df.rez_esc >= 0)])


# In[ ]:


df['rez_esc'].hist()


# In[ ]:


df[['rez_esc','v2a1']].corr()


# In[ ]:


df[['Target','rez_esc']].corr()


# In[ ]:


df[['Target','rez_esc']].fillna(0).corr()


# In[ ]:


df[['v2a1','Target']].corr()


# In[ ]:


df[['v2a1','Target']].fillna(0).corr()


# In[ ]:


df['rez_esc'].unique()


# In[ ]:


plt.figure(figsize=(13,7))
sns.kdeplot(df['rez_esc'])
sns.kdeplot(df['rez_esc'].fillna(0))


# In[ ]:


plt.figure(figsize=(13,7))
sns.kdeplot(df['v2a1'])
sns.kdeplot(df['v2a1'].fillna(0))


# In[ ]:


x, y = df['rez_esc'], df['v2a1']
plt.figure(figsize=(23,17))
g = sns.jointplot(x, y, data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")


# In[ ]:


x, y = df['rez_esc'].fillna(0), df['v2a1'].fillna(0)
sns.jointplot(x, y, data=df, kind="kde")


# How can we fill missing values for Income and Years behind?
# 
# Ignoring 'rez_esc' as there's very low correlation with Target, wheras filling with 0 for Income reduces the correlation by more than 0.1.

# In[ ]:


df['rez_esc'].fillna(0, inplace=True)


# In[ ]:


ff = pd.DataFrame(df.isnull().sum())
ff.loc[(ff.loc[:, ff.dtypes != object] != 0).any(1)]


# In[ ]:


x, y = df['Target'], df['v2a1']
plt.figure(figsize=(23,17))
g = sns.jointplot(x, y, data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")


# In[ ]:


x, y = df['Target'], df['v2a1'].fillna(0)
plt.figure(figsize=(23,17))
g = sns.jointplot(x, y, data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")


# In[ ]:


df['Target'].value_counts()


# In[ ]:


df.groupby('Target').count()['v2a1']


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.groupby('Target').count()['v2a1'].plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.fillna(0).groupby('Target').count()['v2a1'].plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.groupby(['Target','hhsize']).count()['v2a1'].unstack().plot(ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
df.fillna(0).groupby(['Target','hhsize']).count()['v2a1'].unstack().plot(ax=ax)


# In[ ]:


df['hhsize'].value_counts()


# In[ ]:


df['hogar_total'].value_counts()


# In[ ]:


# use hhsize, ignore 'hogar_total',
total_features.remove('hogar_total')

len(total_features)


# In[ ]:


df[['hhsize','hogar_adul']].corr()


# In[ ]:


df[['hhsize','Target']].corr()


# In[ ]:


df[['Target','hogar_adul']].corr()


# In[ ]:


sns.kdeplot(df['hogar_adul'])
sns.kdeplot(df['hhsize'])


# In[ ]:


sns.kdeplot(df['hogar_total'])
sns.kdeplot(df['hogar_adul'])


# In[ ]:


max(df['hogar_adul']), max(df['hogar_total'])


# In[ ]:


df.groupby('idhogar').sum()[['hogar_adul','hogar_total']].sample(10).plot.bar()


# In[ ]:


sns.kdeplot(df['hogar_total'])
sns.kdeplot(df['hogar_nin'])


# In[ ]:


df['male'].value_counts()


# In[ ]:


df['female'].value_counts()


# we need only one column among male and female as both represent the same data!

# In[ ]:


# removing female
total_features.remove('female')

len(total_features)


# r4t3, Total persons in the household
# 
# tamhog, size of the household
# 
# tamviv, number of persons living in the household
# 

# In[ ]:


df['r4t3'].value_counts()


# In[ ]:


df['tamhog'].value_counts()


# In[ ]:


df['tamviv'].value_counts()


# In[ ]:


plt.figure(figsize=(17,13))
sns.kdeplot(df['tamviv'])
sns.kdeplot(df['tamhog'])
sns.kdeplot(df['r4t3'])
sns.kdeplot(df['hhsize'])
sns.kdeplot(df['hogar_total'])
#sns.kdeplot(df['hogar_adul'])


# In[ ]:


# removing 'r4t3', as 'hhsize' is of almost same distribution
total_features.remove('r4t3')


# In[ ]:


len(total_features)


# In[ ]:


df['dependency'].describe()


# In[ ]:


df['dependency'].value_counts()


# In[ ]:


df['SQBdependency'].value_counts()


# In[ ]:


df['SQBdependency'].describe()


# In[ ]:


cat_cols


# In[ ]:


df['edjefe'].describe()


# In[ ]:


df['edjefa'].describe()


# In[ ]:


df['edjefe'].value_counts()


# In[ ]:


df['edjefa'].value_counts()


# ### Categorical columns
# 
# dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# 
# ### we can ignore dependency column
# we've SQBdependency, which is square of dependency!
# 
# edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0 
# 
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# 
# 
# for edjefa and edjefe, yes and no to be replaced with 1 and 0 respectively
# 
# 

# In[ ]:


df.loc[df.edjefa == 'yes', 'edjefa'] = 1
df.loc[df.edjefa == 'no', 'edjefa'] = 0


# In[ ]:


df.loc[df.edjefe == 'yes', 'edjefe'] = 1
df.loc[df.edjefe == 'no', 'edjefe'] = 0


# In[ ]:


df[['edjefa','edjefe']].describe()


# In[ ]:


df[['edjefa','edjefe']] = df[['edjefa','edjefe']].apply(pd.to_numeric)


# In[ ]:


df[['edjefa','edjefe']].dtypes


# In[ ]:


len(total_features)


# In[ ]:


total_features.append('edjefa')
total_features.append('edjefe')


# In[ ]:


len(total_features)


# In[ ]:


cols_water


# In[ ]:


df[cols_water].describe()


# In[ ]:


df[cols_water].corr()


# In[ ]:


df['abastaguadentro'].value_counts()


# In[ ]:


df['abastaguafuera'].value_counts()


# In[ ]:


df['abastaguano'].value_counts()


# In[ ]:


df_water_target = df.groupby('Target')[cols_water].sum().reset_index()
df_water_target


# In[ ]:


722+1496+1133+5844


# In[ ]:


df_water_target.corr()


# As we can see from the above table, provision of water has pretty low significance on the level of poverty.
# 
# Hence we'll remove water provision from features.
# 

# In[ ]:


len(total_features)


# In[ ]:


total_features.remove('abastaguano')
total_features.remove('abastaguafuera')

len(total_features)


# In[ ]:


# cols_floor
# 
# ['pisocemento', 'pisomadera', 'pisomoscer', 'pisonatur', 'pisonotiene', 'pisoother',]

df['pisocemento'].value_counts()


# In[ ]:


df_floor_target = df.groupby('Target')[cols_floor].sum().reset_index()
df_floor_target


# we can remove pisoother, pisonotiene (this doesn't follow the other trend) and pisonatur straight away from total features
# 
# pisocemento, pisomadera and pisomoscer follow the same trend.. 
# 

# In[ ]:


len(total_features)


# In[ ]:


# removing these features -> inc by 0.002
total_features.remove('pisonatur')
total_features.remove('pisonotiene')
total_features.remove('pisoother')

len(total_features)


# Outside wall material
# 
# cols_outside_wall = [ 'paredblolad', 'pareddes', 'paredfibras', 'paredmad', 'paredother', 'paredpreb', 'paredzinc', 'paredzocalo',]

# In[ ]:


# cols_outside_wall

# [ 'paredblolad', 'pareddes', 'paredfibras', 'paredmad', 'paredother', 'paredpreb', 'paredzinc', 'paredzocalo',]


# In[ ]:


df_wall_target = df.groupby('Target')[cols_outside_wall].sum().reset_index()
df_wall_target


# In[ ]:


sns.kdeplot(df['paredblolad'])


# In[ ]:


sns.kdeplot(df['paredpreb'])
sns.kdeplot(df['paredmad'])


# In[ ]:


sns.kdeplot(df['paredmad'])
sns.kdeplot(df['paredzocalo'])


# In[ ]:


sns.kdeplot(df['paredpreb'])
sns.kdeplot(df['paredmad'])
sns.kdeplot(df['paredzocalo'])


# we can remove pareddes, paredfibras, paredother, paredzinc & paredzocalo
# 
# keeping paredblolad, paredmad and paredpreb

# In[ ]:


len(total_features)


# In[ ]:


# removing these features -> reached time limit

# total_features.remove('pareddes')
# total_features.remove('paredfibras')
# total_features.remove('paredother')
# total_features.remove('paredzinc')
# total_features.remove('paredzocalo')

# len(total_features)


# In[ ]:


# from here till model -> do not submit 


# In[ ]:


# cols_roof
# ['techocane', 'techoentrepiso', 'techootro', 'techozinc',]


# In[ ]:


df_roof_target = df.groupby('Target')[cols_roof].sum().reset_index()
df_roof_target


# In[ ]:


sns.kdeplot(df['techozinc'])


# In[ ]:


sns.kdeplot(df['techozinc'])
sns.kdeplot(df['techoentrepiso'])


# In[ ]:


sns.kdeplot(df['techoentrepiso'])
sns.kdeplot(df['techocane'])


# In[ ]:


len(total_features)


# In[ ]:


# remove these features -> (1)
# total_features.remove('techootro')
# total_features.remove('techocane')

# len(total_features)


# cols_sanitary 
# [ 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6',]

# In[ ]:


# [ 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6',]

df_sani_target = df.groupby('Target')[cols_sanitary].sum().reset_index()
df_sani_target


# In[ ]:


sns.kdeplot(df['sanitario1'])
sns.kdeplot(df['sanitario6'])


# In[ ]:


sns.kdeplot(df['sanitario3'])
sns.kdeplot(df['sanitario2'])


# In[ ]:


len(total_features)


# In[ ]:


# remove these features -> (2)
# total_features.remove('sanitario1')
# total_features.remove('sanitario5')
# total_features.remove('sanitario6')

# len(total_features)


# cols_tip  ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',]
# 

# In[ ]:


# cols_tip 
# ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',]

df_tipo_target = df.groupby('Target')[cols_tip].sum().reset_index()
df_tipo_target


# In[ ]:


sns.kdeplot(df['tipovivi2'])


# In[ ]:


sns.kdeplot(df['tipovivi1'])
sns.kdeplot(df['tipovivi3'])


# In[ ]:


sns.kdeplot(df['tipovivi5'])
sns.kdeplot(df['tipovivi4'])


# In[ ]:


df['v2a1'].isna().sum()


# In[ ]:


df['tipovivi3'].value_counts()


# In[ ]:


df.loc[(df['v2a1'].isna()) & (df.tipovivi3 == 1)]


# In[ ]:


# check the value of parentesco1 and fill corresponding value
# group by idhogar -> check and fill 


# In[ ]:


df['v2a1'].loc[df.parentesco1 == 1].plot.line()


# In[ ]:


df['v2a1'].loc[df.parentesco1 == 1].plot.hist()


# In[ ]:


df['v2a1'].loc[df.parentesco1 == 1].mean(), df['v2a1'].loc[df.parentesco1 == 1].max(), df['v2a1'].loc[df.parentesco1 == 1].min()


# In[ ]:


df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 == 1) & (df.v2a1.isna())].describe(include='all')


# In[ ]:


df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 == 1) & (-df.v2a1.isna()) & (df.tipovivi3==1) & (df.tipovivi1==0)].describe(include='all')


# In[ ]:


df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 == 1) & (-df.v2a1.isna()) & (df.tipovivi3==1) & (df.tipovivi1==0)].mean()


# In[ ]:


df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 != 1) & (-df.v2a1.isna()) & (df.tipovivi3==1) & (df.tipovivi1==0)].mean()


# In[ ]:


df[['v2a1','idhogar','parentesco1']].loc[(df.parentesco1 != 1) & (-df.v2a1.isna()) & (df.tipovivi3==1) & (df.tipovivi1==0)].describe(include='all')


# In[ ]:


df[['v2a1','idhogar','parentesco1']].loc[df.parentesco1 != 1].describe(include='all')


# In[ ]:


df[['v2a1','idhogar','parentesco1']].loc[df.parentesco1 == 1].describe(include='all')


# In[ ]:


# 50% of the samples have ~120000 as the monthly rent..
# 


# In[ ]:


df['v2a1'].fillna(120000, inplace=True)


# In[ ]:


df['v2a1'].isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'Target' in total_features


# > F1-Score
# 
# In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. 
# 
# 

# ### Model

# In[ ]:


X, y = df[total_features], df['Target']


# In[ ]:


#Split the dataset into train and Test
seed = 42
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[ ]:


#Train the XGboost Model for Classification
model1 = xgb.XGBClassifier(n_jobs=n_jobs)
model1


# In[ ]:


train_model1 = model1.fit(X_train, y_train)


# In[ ]:


model2 = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5, n_jobs=n_jobs)
model2


# In[ ]:


train_model2 = model2.fit(X_train, y_train)


# In[ ]:


# predictions
pred1 = train_model1.predict(X_test)
pred2 = train_model2.predict(X_test)

print('Model 1 XGboost Report %r' % (classification_report(y_test, pred1)))
print('Model 2 XGboost Report %r' % (classification_report(y_test, pred2)))


# In[ ]:


print("Accuracy for model 1: %.2f" % (accuracy_score(y_test, pred1) * 100))
print("Accuracy for model 2: %.2f" % (accuracy_score(y_test, pred2) * 100))


# ## Hyperparameter Tunning of XGboost
# 
# ### Based on the work of [Aarshay Jain](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

# In[ ]:


#Let's do a little Gridsearch, Hyperparameter Tunning
model3 = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 n_jobs=n_jobs,
 scale_pos_weight=1,
 seed=27)


# In[ ]:


train_model3 = model3.fit(X_train, y_train)
pred3 = train_model3.predict(X_test)
print("Accuracy for model 3: %.2f" % (accuracy_score(y_test, pred3) * 100))


# In[ ]:


print('Model 3 XGboost Report %r' % (classification_report(y_test, pred3)))


# In[ ]:


gc.collect()


# In[ ]:





# In[ ]:





# parameters = {
#     'n_estimators': [100, 250, 500],
#     'max_depth': [6, 9, 12],
#     'subsample': [0.9, 1.0],
#     'colsample_bytree': [0.9, 1.0],
# }
# 

# In[ ]:


parameters = {
    'n_estimators': [100],
    'max_depth': [6, 9],
    'subsample': [0.9, 1.0],
    'colsample_bytree': [0.9, 1.0],
}


# In[ ]:


grid = GridSearchCV(model3,
                    parameters, n_jobs=n_jobs,
                    scoring="neg_log_loss",
                    cv=3)
grid


# In[ ]:


# grid.fit(X_train, y_train)
# print("Best: %f using %s" % (grid.best_score_, grid.best_params_))


# In[ ]:


#means = grid.cv_results_['mean_test_score']
#stds = grid.cv_results_['std_test_score']
#params = grid.cv_results_['params']

#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


#pred4 = grid.predict(X_test)
#classification_report(y_test, pred4)


# In[ ]:


#print("Accuracy for model 4: %.2f" % (accuracy_score(y_test, pred4) * 100))


# In[ ]:


gc.collect()


# In[ ]:


fig, ax = plt.subplots(figsize=(23, 17))
plot_importance(model3, ax=ax)


# In[ ]:


less_imp_features = ['estadocivil1','instlevel9','techocane','parentesco10','v14a',
                     'parentesco11','parentesco5','paredother','parentesco7','noelec',
                     'elimbasu4','elimbasu6']


# In[ ]:


# before removing less important features
len(total_features)


# In[ ]:


for f in less_imp_features:
    if f in total_features:
        total_features.remove(f)

len(total_features)


# In[ ]:


X, y = df[total_features], df['Target']


# In[ ]:


#Split the dataset into train and Test
seed = 43
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[ ]:


train_model5 = model3.fit(X_train, y_train)
pred5 = train_model5.predict(X_test)
print("Accuracy for model 5: %.2f" % (accuracy_score(y_test, pred5) * 100))


# In[ ]:


print('Model 5 XGboost Report %r' % (classification_report(y_test, pred5)))


# #### train model on full dataset

# In[ ]:


train_model6 = train_model5.fit(X, y)


# In[ ]:


train_model6


# In[ ]:





# In[ ]:


# plot_tree(model3)
# scoring = ['precision_macro', 'recall_macro']
# scores = cross_validate(model3, X_train, y_train, cv=5, scoring=scoring)
# scores
# sorted(scores.keys())
# scoring = {'precision_macro': 'precision_macro', 'recall_macro': make_scorer(recall_score, average='macro')}
# scores = cross_validate(model3, X_train, y_train, cv=7, scoring=scoring)
# scores
# 


# In[ ]:


# prediction using cross_val_predict()
# predicted = cross_val_predict(model3, X_test, y_test, cv=10)
# accuracy_score(y_test, predicted)


# ### Thresholds | feature importances

# In[ ]:


thresholds = sort(model3.feature_importances_)
thresholds


# In[ ]:


thresholds.shape


# In[ ]:


np.unique(thresholds).shape


# # Fit model using each importance as a threshold
# for thresh in np.unique(thresholds):
#     # select features using threshold
#     selection = SelectFromModel(model3, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X_train)
#     # train model
#     selection_model = xgb.XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1,
#                                         gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
#                                         n_jobs=n_jobs, scale_pos_weight=1, seed=27)
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(X_test)
#     y_pred = selection_model.predict(select_X_test)
#     predictions = [round(value) for value in y_pred]
#     accuracy = accuracy_score(y_test, predictions)
#     f1score = f1_score(y_test, predictions)
#     recallscore = recall_score(y_test, predictions)
#     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%, Recall Score: %.2f%%, F1 Score: %.2f%%" 
#           % (thresh, select_X_train.shape[1], accuracy*100.0, recallscore*100.0, f1score*100.0))

# # feature importances
# feature_importances = model3.feature_importances_
# feature_importances

# # helper to extract required features
# def extract_pruned_features(feature_importances, min_score=0.003):
#     column_slice = feature_importances[feature_importances['weights'] > min_score]
#     print(column_slice)
# #    return column_slice.index.values

# pruned_features = extract_pruned_features(feature_importances, min_score=0.002)
# pruned_features

# X_train_reduced = X_train[pruned_features]
# X_test_reduced = X_test[pruned_features]
# X_train_reduced.shape, X_test_reduced.shape

# # fit and train
# def fit_and_print_metrics(X_train, y_train, X_test, y_test, model):
#     model.fit(X_train, y_train)
#     predictions_proba = model.predict_proba(X_test)
#     log_loss_score = log_loss(y_test, predictions_proba)
#     print('Log loss: %.5f' % log_loss_score)
# 
# fit_and_print_metrics(X_train_reduced, y_train, X_test_reduced, y_test, model3)

# ### Random Forest Classifier

# In[ ]:


model4 = RandomForestClassifier(n_jobs=n_jobs)
model4


# In[ ]:


gc.collect()


# In[ ]:


train_model4 = model4.fit(X_train, y_train)
pred4 = train_model4.predict(X_test)
print("Accuracy for model 4: %.2f" % (accuracy_score(y_test, pred4) * 100))


# In[ ]:


print('Model 4 XGboost Report %r' % (classification_report(y_test, pred4)))


# In[ ]:


confusion_matrix(y_test, pred4)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Submission

# In[ ]:


df_test = pd.read_csv('../input/test.csv')


# In[ ]:


len(df_test)


# In[ ]:


df_test.sample(10)


# In[ ]:


# considering only head of household
# df_test = df_test.loc[df_test.parentesco1 == 1]
# 


# In[ ]:


len(df_test)


# In[ ]:


df_test.loc[df_test.edjefa == 'yes', 'edjefa'] = 1
df_test.loc[df_test.edjefa == 'no', 'edjefa'] = 0

df_test.loc[df_test.edjefe == 'yes', 'edjefe'] = 1
df_test.loc[df_test.edjefe == 'no', 'edjefe'] = 0
df_test[['edjefa','edjefe']] = df_test[['edjefa','edjefe']].apply(pd.to_numeric)
df_test[['edjefa','edjefe']].dtypes


# In[ ]:


X_actual_test = df_test[total_features]


# In[ ]:


X_actual_test.shape


# In[ ]:


pred_actual = train_model6.predict(X_actual_test)
pred_actual


# In[ ]:


pred_actual.shape


# In[ ]:


df_final = pd.DataFrame(df['Id'], pred_actual).reset_index()
df_final.columns = ['Target','Id']


# In[ ]:


cols = df_final.columns.tolist()
cols


# In[ ]:


cols = cols[-1:] + cols[:-1]
cols


# In[ ]:


df_final = df_final[cols]
df_final.head(7)


# In[ ]:


df_final.index.name = None
df_final.head(7)


# In[ ]:


df_final['Target'].value_counts()


# In[ ]:


df_final[cols].sample(4)


# In[ ]:


df_final[cols].to_csv('sample_submission.csv', index=False)


# In[ ]:


os.listdir('../input/')


# In[ ]:




