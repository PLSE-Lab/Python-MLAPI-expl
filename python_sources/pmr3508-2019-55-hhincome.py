#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


train_data = "../input/costa-rican-household-poverty-prediction/train.csv"
test_data = "../input/costa-rican-household-poverty-prediction/test.csv"


# In[ ]:


train_df = pd.read_csv(train_data)


# In[ ]:


train_df.shape


# ### Intro
# Since it should be an analysis by household: grouping by the household id

# In[ ]:


train_df = train_df.groupby('idhogar').first()


# In[ ]:


train_df.shape


# So much smaller, let's start exploring the data

# In[ ]:


train_df.head()


# In[ ]:


attributes = list(train_df)
attributes


# In[ ]:


ntrain_df = train_df.dropna()


# In[ ]:


ntrain_df.shape


# From above info, we see that applying dropna() directly to the raw data it is left with few rows, since the amount of data is not great we need to analyze and see which collumns have missing data

# In[ ]:


na_per_column = train_df.isna().sum()
na_per_column


# From the output above we can filter the collumns with missing data and see which one have too many, therefore should be discarded
# 
# Getting the indexes of the columns with missing data:

# In[ ]:


nan_indexes = na_per_column.nonzero()
nan_indexes


# Getting the collumn names and the respectives amount of missing data

# In[ ]:


na_per_column[nan_indexes[0]]


# From the data field description:
# 
# v2a1, Monthly rent payment
# 
# v18q1, number of tablets household owns
# 
# rez_esc, Years behind in school
# 
# meaneduc,average years of education for adults (18+)
# 
# SQBmeaned, square of the mean years of education of adults (>=18) in the household

# ### Dealing with missing data
# Analizying the missing data variables

# In[ ]:


train_df[['v2a1','v18q1','rez_esc','meaneduc','SQBmeaned']].describe()


# From the analysis above, for 'v18q1' we can infer that the missing data are household with 0 tablets, and for 'meaneduc' and 'SQB meaned' we can use the standard value.

# In[ ]:


train_df['v18q1'] = train_df['v18q1'].fillna(0)
train_df['meaneduc'] = train_df['meaneduc'].fillna(4)
train_df['SQBmeaned'] = train_df['SQBmeaned'].fillna(97)


# In[ ]:


train_df[['v18q1','meaneduc','SQBmeaned']].describe()


# In[ ]:


na_per_column = train_df.isna().sum()
nan_indexes = na_per_column.nonzero()
na_per_column[nan_indexes[0]]


# cleared the nan values

# ### Dealing with binary columns
# In this database we have many columns with binary data, only to express whether an attribute is present or not in the household, therefore let's combine those columns into a single one according to its characteristic, taking the walls attribute as an example

# In[ ]:


wall_col = ['paredblolad','paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc','paredfibras', 'paredother']


# In[ ]:


def join_binary_columns(df, cols, new_column_label):
    df[new_column_label] = df[cols[0]]
    for i in range(1,len(cols)):
        df[new_column_label] = df[new_column_label] + df[cols[i]]*(2**i)
    return df.drop(columns = cols)


# In[ ]:


train_df = join_binary_columns(train_df, wall_col, 'wall')


# In[ ]:


train_df['wall'].describe()


# In[ ]:


train_df['wall'].value_counts().plot(kind="bar")


# Applying this transformation to the other many binary decomposed variables:
# - Floor
# - Roof
# - Water provision
# - Electricity
# - Toilet
# - Energy cooking
# - Rubbish disposal
# - quality of walls
# - quality of roof
# - quality of floor
# - State of house
# - location of house

# In[ ]:


floor_col = ['pisonotiene', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisomadera']
roof_col = ['techozinc', 'techoentrepiso', 'techocane', 'techootro']
water_col = ['abastaguadentro', 'abastaguafuera', 'abastaguano']
electricity_col = ['public', 'planpri', 'noelec', 'coopele']
toilet_col = ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']
energy_col = ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']
rubbish_col = ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']
qua_wall_col = ['epared1', 'epared2', 'epared3']
qua_roof_col = ['etecho1', 'etecho2', 'etecho3']
qua_floor_col = ['eviv1', 'eviv2', 'eviv3']
house_state_col = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
location_col = ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']
area_col = ['area1', 'area2']
# some social variables aggregation that maybe useful to refine training (not used in this notebook)
# males_col = ['r4h1', 'r4h2', 'r4h3']
# females_col = ['r4m1', 'r4m2', 'r4m3']
# person_col = ['r4t1', 'r4t2', 'r4t3']
# civil_state_col = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']
# family_col = ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']


# In[ ]:


train_df = join_binary_columns(train_df, floor_col, 'floor')
train_df = join_binary_columns(train_df, roof_col, 'roof')
train_df = join_binary_columns(train_df, water_col, 'water')
train_df = join_binary_columns(train_df, electricity_col, 'electricity')
train_df = join_binary_columns(train_df, toilet_col, 'toilet')
train_df = join_binary_columns(train_df, energy_col, 'energy')
train_df = join_binary_columns(train_df, rubbish_col, 'rubbish')
train_df = join_binary_columns(train_df, qua_wall_col, 'qua_wall')
train_df = join_binary_columns(train_df, qua_roof_col, 'qua_roof')
train_df = join_binary_columns(train_df, qua_floor_col, 'qua_roof')
train_df = join_binary_columns(train_df, house_state_col, 'house_state')
train_df = join_binary_columns(train_df, location_col, 'location')
train_df = join_binary_columns(train_df, area_col, 'area')


# ### Data analysis
# After filtering and managing the data, we can start doing some analysis

# In[ ]:


train_df["Target"].value_counts().plot(kind="bar")


# In[ ]:


attributes = list(train_df)
attributes


# In[ ]:


from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[ ]:


drop_columns = ['v2a1', 'rez_esc']
train_df = train_df.drop(columns = ['v2a1', 'rez_esc'])
attributes.remove('Target')
attributes.remove('Id')
for e in drop_columns:
    attributes.remove(e)


# Still, many variables, let's just see the correlation between the variables and output

# In[ ]:


corr = train_df[attributes].apply(preprocessing.LabelEncoder().fit_transform).apply(lambda x: x.corr(train_df.Target))
corr


# Filtering with the best correlation we can find

# In[ ]:


corr_thresh = 0.2
corr.where(abs(corr) > corr_thresh).dropna()


# In[ ]:


train_attr = corr.where(abs(corr) > corr_thresh).dropna().index.values


# In[ ]:


train_attr


# In[ ]:


train_df_x = train_df[train_attr].apply(preprocessing.LabelEncoder().fit_transform)
train_df_y = train_df.Target


# Applying the same data preparation to test dataframe

# In[ ]:


test_df = pd.read_csv(test_data)
test_df['v18q1'] = test_df['v18q1'].fillna(0)
test_df['meaneduc'] = test_df['meaneduc'].fillna(4)
test_df['SQBmeaned'] = test_df['SQBmeaned'].fillna(97)
test_df = join_binary_columns(test_df, wall_col, 'wall')
test_df = join_binary_columns(test_df, floor_col, 'floor')
test_df = join_binary_columns(test_df, roof_col, 'roof')
test_df = join_binary_columns(test_df, water_col, 'water')
test_df = join_binary_columns(test_df, electricity_col, 'electricity')
test_df = join_binary_columns(test_df, toilet_col, 'toilet')
test_df = join_binary_columns(test_df, energy_col, 'energy')
test_df = join_binary_columns(test_df, rubbish_col, 'rubbish')
test_df = join_binary_columns(test_df, qua_wall_col, 'qua_wall')
test_df = join_binary_columns(test_df, qua_roof_col, 'qua_roof')
test_df = join_binary_columns(test_df, qua_floor_col, 'qua_roof')
test_df = join_binary_columns(test_df, house_state_col, 'house_state')
test_df = join_binary_columns(test_df, location_col, 'location')
test_df = join_binary_columns(test_df, area_col, 'area')
test_df_x = test_df[train_attr].apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


columns = ['neighbors', 'scores']
results = [columns]
for n in range (5, 40):
    neighbors = n
    cross = 5
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    scores = cross_val_score(knn, train_df_x, train_df_y, cv = cross)
    results.append([neighbors, scores])


# In[ ]:


import statistics as st

analysis = [['neighbors', 'mean', 'max', 'min']]
for i in range(1, len(results)):
    analysis.append([results[i][0], st.mean(results[i][1]), max(results[i][1]), min(results[i][1])])


# In[ ]:


analysis


# In[ ]:


neighbors = 32
cross = 5
KNeighborsClassifier(n_neighbors = neighbors)
cross_val_score(knn, train_df_x, train_df_y, cv = cross)
knn.fit(train_df_x, train_df_y)
test_pred_y = knn.predict(test_df_x)
test_pred_y


# In[ ]:


result = pd.DataFrame({'Id':test_df.Id.values, 'Target':test_pred_y})
result


# In[ ]:


result.to_csv("submission.csv", index = False)

