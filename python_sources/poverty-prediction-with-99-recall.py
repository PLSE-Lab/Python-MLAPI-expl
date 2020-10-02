#!/usr/bin/env python
# coding: utf-8

# # Part1- Data cleaning-both train and test dataset

# # (a) Reading all three datasets, their shapes, structure and searching for missing values

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[ ]:


pd.set_option('display.max_columns', 143)


# In[ ]:


HPtrain=pd.read_csv('../input/train.csv')

HPtest=pd.read_csv('../input/test.csv')

HPsample=pd.read_csv('../input/sample_submission.csv')


# In[ ]:





# In[ ]:


HPtrain.head()


# In[ ]:


HPtrain.shape


# In[ ]:


HPtrain.describe().append(HPtrain.isnull().sum().rename('isnull'))


# In[ ]:


HPtest.head()


# In[ ]:


HPtest.shape


# In[ ]:


HPtest.describe().append(HPtest.isnull().sum().rename('isnull'))


# In[ ]:


HPsample.head()


# # (b) Deduplication- removing duplicate columns
# We look for any duplicated columns and drop them
# https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns

# 'tamhog','hhsize' and 'hogar_total' are exactly duplicated and almost duplicated with 'r4t3'

# 'SQBage' and 'agesq' are duplicated

# 3 duplicated columns removed 'hhsize', 'hogar_total' and 'agesq'. Two additional columns 'tamhog'(almost same as r4t3) and 'tamviv'(misleading)

# In[ ]:


from collections import Counter
a=Counter(HPtrain['r4t3'])
b=Counter(HPtrain['tamhog'])
a-b


# In[ ]:


from collections import Counter
a0=Counter(HPtrain['tamviv'])
b0=Counter(HPtrain['tamhog'])
a0-b0


# In[ ]:


from collections import Counter
a1=Counter(HPtrain['r4t3'])
b1=Counter(HPtrain['tamviv'])
a1-b1


# In[ ]:


from collections import Counter
a2=Counter(HPtrain['r4t3'])
b2=Counter(HPtrain['hhsize'])
a2-b2


# In[ ]:


from collections import Counter
a3=Counter(HPtrain['tamviv'])
b3=Counter(HPtrain['hhsize'])
a3-b3


# In[ ]:


from collections import Counter
a4=Counter(HPtrain['tamhog'])
b4=Counter(HPtrain['hhsize'])
a4-b4


# In[ ]:


from collections import Counter
a3=Counter(HPtrain['hhsize'])
b3=Counter(HPtrain['hogar_total'])
print(a3-b3)


# In[ ]:


from collections import Counter
a5=Counter(HPtrain['tamhog'])
b5=Counter(HPtrain['hogar_total'])
print(a5-b5)


# In[ ]:


from collections import Counter
a6=Counter(HPtrain['SQBage'])
b6=Counter(HPtrain['agesq'])
print(a6-b6)

from collections import Counter
at6=Counter(HPtest['SQBage'])
bt6=Counter(HPtest['agesq'])
print(at6-bt6)


# In[ ]:


HPtrain.drop(['hhsize', 'hogar_total' and 'agesq'],axis=1,inplace=True)

HPtest.drop(['hhsize', 'hogar_total' and 'agesq'],axis=1,inplace=True)


# In[ ]:


HPtrain['r4t3'].unique()


# In[ ]:


HPtrain['tamhog'].unique()


# In[ ]:


HPtrain['tamviv'].unique()


# In[ ]:


HPtrain.loc[HPtrain['tamviv']==15]


# In[ ]:


del HPtrain['tamhog']

del HPtest['tamhog']


# In[ ]:


del HPtrain['tamviv']

del HPtest['tamviv']


# In[ ]:


HPtrain.shape


# In[ ]:


HPtrain.head()


# # (c) Missing fields

# We examine the missing values in both train and test dataset one by one.

# In[ ]:


HPtrain.describe().append(HPtrain.isnull().sum().rename('isnull'))


# In[ ]:


HPtest.describe().append(HPtest.isnull().sum().rename('isnull'))


# # let's examine the variable 'v2a1' where we have find out 17403 NaNs

# v2a1, Monthly rent payment
# tipovivi2, "=1 own,  paying in installments"
# tipovivi3, =1 rented
# 
# NaN in 'v2a1' only appear for rented house ('tipovivi3'=1) so we can replace these NaNs with 0

# In[ ]:


HPtrain.loc[((HPtrain['v2a1'].isnull())|(HPtrain['tipovivi3']==0)),['v2a1','tipovivi3','tipovivi2']]


# In[ ]:


HPtrain.loc[((HPtrain['v2a1'].notnull()) & (HPtrain['tipovivi3']==0)),['v2a1','tipovivi3','tipovivi2']]


# In[ ]:


HPtrain['v2a1']= HPtrain['v2a1'].fillna(value=0)

HPtest['v2a1']= HPtest['v2a1'].fillna(value=0)


# # Now we look for the NaNs in 'v18q1' having 18126 NaNs
# v18q1, number of tablets household owns
# v18q, owns a tablet
# idhogar, Household level identifier

# We thus see that all the NaNs in the 'v18q1' actually represent the families with no tablets in the household. So we replace these Nans with zeros in both the datasets.

# In[ ]:


HPtrain.loc[(HPtrain['v18q1'].isnull()) & (HPtrain['v18q']==0),['Id', 'v18q1', 'v18q', 'idhogar', 'age','Target']]


# In[ ]:


HPtrain.loc[HPtrain['v18q1']!=HPtrain['v18q'],['Id', 'v18q1', 'v18q',  'idhogar','age']]


# In[ ]:


HPtrain['v18q1']= HPtrain['v18q1'].fillna(value=0.0)

HPtest['v18q1']= HPtest['v18q1'].fillna(value=0.0)


# # Now we look at the NaNs in 'rez_esc' having 7928 NaNs
# rez_esc, Years behind in school

# All NaN values in the 'rez_esc' are for mature adults or underaged children which is justified. Only one NaN correspond to a child of age 10 with Id=ID_f012e4242 who is not attending school. Reason for not attending the school is disability (dis=1). Therefore we replace all NaN in this column by 0.0 in both datasets.

# In[ ]:


HPtrain.loc[((HPtrain['rez_esc'].isnull()) & (HPtrain['age']>=18) | (HPtrain['age']<=6)),['rez_esc','age']]


# In[ ]:


HPtrain.loc[((HPtrain['rez_esc'].isnull()) & (HPtrain['age']>6) & (HPtrain['age']<18)),['Id','rez_esc','age','dis']]


# In[ ]:


HPtrain['rez_esc'].unique()


# In[ ]:


HPtrain['rez_esc']= HPtrain['rez_esc'].fillna(value=0.0)

HPtest['rez_esc']= HPtest['rez_esc'].fillna(value=0.0)


# # Let's see the column 'meaneduc' which have 5 NaN's
# meaneduc,average years of education for adults (18+)

# We see that the number of adults for these 5 households are zero which results in NaN. Here we replace these NaN's by zero.

# In[ ]:


HPtrain.loc[HPtrain['meaneduc'].isnull(), ['Id', 'meaneduc', 'age', 'rez_esc','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3']]


# In[ ]:


HPtrain['meaneduc']= HPtrain['meaneduc'].fillna(value=0.0)

HPtest['meaneduc']= HPtest['meaneduc'].fillna(value=0.0)


# # Lastly we see the missing values in 'SQBmeaned'
# SQBmeaned, square of the mean years of education of adults (>=18) in the household
# The reason is clear their origin is from the NaNs in the column 'meaneduc. We also replace these NaN's with zero.

# In[ ]:


HPtrain['SQBmeaned']= HPtrain['SQBmeaned'].fillna(value=0.0)

HPtest['SQBmeaned']= HPtest['SQBmeaned'].fillna(value=0.0)


# In[ ]:


print(HPtrain.shape)

print(HPtest.shape)


# In[ ]:


print(HPtrain.isnull().sum().sum())

print(HPtest.isnull().sum().sum())


# In[ ]:


HPtrainC=HPtrain.copy()
HPtestC=HPtest.copy()


# # Part2-Feature engineering
# 
# In this section we will play with the features and will keep features or will generate new features that properly goes with the description of the datset.
# 
# (a) We will look the dataset from the perspective of each household ('idhogar') rather than persons living in the household ('Id'). So we will pick-up only those features that properly describe each household rather than features that describe individual features. This observation will help us to reduce the number of dimensions by 46 (from 139 to 93 from training data and from 138 to 92 for testing data).We also create a new dataframe with two columns 'idhogar' and 'Id' for future prediction.
# 
# (b) We drop rows with duplicate household index ('idhogar').
# 
# (C) We closely look individual features and found that some pair of features tell the same story. We drop one of the feature from the pair.
# 
# (d) Grouping features which form a group.
# 
# (e)Converting some 'object' type variable to numeric.
# 
# (f) We work on Variance inflation factor and eliminate some variables which are highly correlated.

# In[ ]:


HPtrainC.dtypes


# In[ ]:


HPtrainC.head()


# In[ ]:


print(HPtrainC.shape)

print(HPtestC.shape)


# In[ ]:





# In[ ]:


cols=list(HPtrainC.columns)


# In[ ]:


print(cols)


# Now we rearrange some of the columns so that our analysis becomes somewhat easy.

# In[ ]:


HPtrainC=HPtrainC[['idhogar','Id', 'male', 'female', 'v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6','parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12','escolari', 'rez_esc', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dis',  'dependency', 'edjefe', 'edjefa', 'meaneduc', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'v18q1', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'Target']].copy()

HPtestC=HPtestC[['idhogar','Id', 'male', 'female', 'v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6','parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12','escolari', 'rez_esc', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dis',  'dependency', 'edjefe', 'edjefa', 'meaneduc', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q', 'v18q1', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'mobilephone', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned']].copy()


# In[ ]:


print(HPtrainC.shape)

print(HPtestC.shape)


# In[ ]:


HPtrainC.head(10)


# In[ ]:


HPtestC.head(10)


# # (a) We drop following 46 features that descibe an individual rather than a householed.
# 
# 'Unnamed: 0', ------------------1
# 
# 'Id', 'male', 'female', -----------3
# 
# 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', -----------7
# 
# 'parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12',-----------------12  
# 
# 'escolari', 'rez_esc', ---------------2
# 
# 'dis',-----------1
#  
# 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 
# 'instlevel7', 'instlevel8', 'instlevel9', ----------9
#  
# 'v18q',----------1
#   
# 'mobilephone',--------1
#  
# 'age', 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 
# 'SQBdependency', 'SQBmeaned',---------9
# 

# In[ ]:


print(list(HPtrainC.columns))


# In[ ]:


HPtrainN=HPtrainC[['idhogar','v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'Target']].copy()

HPtestN=HPtestC[['idhogar','v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2','hogar_nin', 'hogar_adul', 'hogar_mayor', 'r4t3', 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6', 'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1', 'bedrooms', 'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'computer', 'television', 'qmobilephone', 'lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']].copy()

test_predict= HPtestC[['idhogar', 'Id']].copy()


# In[ ]:


print(HPtrainN.shape)

print(HPtestN.shape)

print(test_predict.shape)


# In[ ]:


HPtrainN.head()


# In[ ]:


# This give us an estimate of the number of households in the train/test dataset
print(HPtrainN['idhogar'].nunique())

print(HPtestN['idhogar'].nunique())


# # (b) We drop rows with duplicate household index ('idhogar').
# This shrinks both the dataframe to the size of number of households in that set as we can see below.

# In[ ]:


HPtrainN.drop_duplicates(subset='idhogar', keep='first', inplace=True)

HPtestN.drop_duplicates(subset='idhogar', keep='first', inplace=True)


# In[ ]:


print(HPtrainN.shape)

print(HPtestN.shape)


# In[ ]:





# # (C) We closely look individual features and found that some pair of features tell the same story. We drop one of the feature from the pair.

# In[ ]:





# 'area1' and 'area2' basically indicate the same data. For 'area1', 1(Urban) and 0(Rural). Opposite is true for 'area2'. We therefore delete 'area2'.

# In[ ]:


HPtrainN['area2'].unique()


# In[ ]:


HPtrainN['area1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['area2'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


del HPtrainN['area2']

del HPtestN['area2']


# In[ ]:





# # (d) Grouping features which may form a group

# 'lugar1' to 'lugar5' indicates different regions of the country. For 'lugar1' a region with index 1 belongs to 'region Central' and 0 indicates it doesn't belong to this region. For other lugar indices same is true.  We denote every region with a unique ID number and marge then all with 'lugar1'. 
# 
# 1:region Central, 2:region Chorotega, 3:region PacÃ­fico central, 4:region Brunca, 5:region Huetar AtlÃ¡ntica and 6:region Huetar Norte. 
# 
# This delete 5 more features.

# In[ ]:


HPtrainN['lugar1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['lugar2'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['lugar3'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['lugar4'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['lugar5'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['lugar6'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['lugar2'] = HPtrainN['lugar2'].map({0: 0, 1: 2})

HPtestN['lugar2'] = HPtestN['lugar2'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['lugar3'] = HPtrainN['lugar3'].map({0: 0, 1: 3})

HPtestN['lugar3'] = HPtestN['lugar3'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['lugar4'] = HPtrainN['lugar4'].map({0: 0, 1: 4})

HPtestN['lugar4'] = HPtestN['lugar4'].map({0: 0, 1: 4})


# In[ ]:


HPtrainN['lugar5'] = HPtrainN['lugar5'].map({0: 0, 1: 5})

HPtestN['lugar5'] = HPtestN['lugar5'].map({0: 0, 1: 5})


# In[ ]:


HPtrainN['lugar6'] = HPtrainN['lugar6'].map({0: 0, 1: 6})

HPtestN['lugar6'] = HPtestN['lugar6'].map({0: 0, 1: 6})


# In[ ]:


HPtrainN['lugar1'] = HPtrainN['lugar1']+ HPtrainN['lugar2']+ HPtrainN['lugar3']+ HPtrainN['lugar4']+ HPtrainN['lugar5']+ HPtrainN['lugar6']

HPtestN['lugar1'] = HPtestN['lugar1']+ HPtestN['lugar2']+ HPtestN['lugar3']+ HPtestN['lugar4']+ HPtestN['lugar5']+ HPtestN['lugar6']


# In[ ]:


HPtrainN['lugar1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN.drop(['lugar2','lugar3','lugar4','lugar5','lugar6'], inplace=True, axis=1)

HPtestN.drop(['lugar2','lugar3','lugar4','lugar5','lugar6'], inplace=True, axis=1)


# In[ ]:


HPtrainN.shape


# In[ ]:





# Same is true with the features 'tipoviv1' to 'tipoviv5'. We perform the same task and group them in 'tipoviv1'. 
# 
# Where, 1:own and fully paid house, 2:own,paying in installments, 3:rented, 4:precarious, 5:other(assigned,  borrowed). 
# 
# This drops 4 more characters.

# In[ ]:


HPtrainN['tipovivi1'].unique()


# In[ ]:


HPtrainN['tipovivi2'].unique()


# In[ ]:


HPtrainN['tipovivi2'].unique()


# In[ ]:


HPtrainN['tipovivi4'].unique()


# In[ ]:


HPtrainN['tipovivi5'].unique()


# In[ ]:


HPtrainN['tipovivi2'] = HPtrainN['tipovivi2'].map({0: 0, 1: 2})

HPtestN['tipovivi2'] = HPtestN['tipovivi2'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['tipovivi3'] = HPtrainN['tipovivi3'].map({0: 0, 1: 3})

HPtestN['tipovivi3'] = HPtestN['tipovivi3'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['tipovivi4'] = HPtrainN['tipovivi4'].map({0: 0, 1: 4})

HPtestN['tipovivi4'] = HPtestN['tipovivi4'].map({0: 0, 1: 4})


# In[ ]:


HPtrainN['tipovivi5'] = HPtrainN['tipovivi5'].map({0: 0, 1: 5})

HPtestN['tipovivi5'] = HPtestN['tipovivi5'].map({0: 0, 1: 5})


# In[ ]:


HPtrainN['tipovivi1'] = HPtrainN['tipovivi1'] + HPtrainN['tipovivi2'] + HPtrainN['tipovivi3'] + HPtrainN['tipovivi4'] + HPtrainN['tipovivi5']

HPtestN['tipovivi1'] = HPtestN['tipovivi1'] + HPtestN['tipovivi2'] + HPtestN['tipovivi3'] + HPtestN['tipovivi4'] + HPtestN['tipovivi5']


# In[ ]:


HPtrainN.drop(['tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',], inplace=True, axis=1)

HPtestN.drop(['tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5',], inplace=True, axis=1)


# In[ ]:


HPtrainN['tipovivi1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN.shape


# In[ ]:





# Same is true with the features 'eviv1' to 'eviv3'. We grpup them to 'eviv1'. This drops 2 more characters.
# 
# 1:if floors are bad, 2:if floors are regular, 3:if floors are good

# In[ ]:


HPtrainN['eviv2'].unique()


# In[ ]:


HPtrainN['eviv3'].unique()


# In[ ]:


HPtrainN['eviv2'] = HPtrainN['eviv2'].map({0: 0, 1: 2})

HPtestN['eviv2'] = HPtestN['eviv2'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['eviv3'] = HPtrainN['eviv3'].map({0: 0, 1: 3})

HPtestN['eviv3'] = HPtestN['eviv3'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['eviv1'] = HPtrainN['eviv1'] + HPtrainN['eviv2'] + HPtrainN['eviv3'] 

HPtestN['eviv1'] = HPtestN['eviv1'] + HPtestN['eviv2'] + HPtestN['eviv3'] 


# In[ ]:


HPtrainN.drop(['eviv2', 'eviv3'], inplace=True, axis=1)

HPtestN.drop(['eviv2', 'eviv3'], inplace=True, axis=1)


# In[ ]:


HPtrainN['eviv1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN.shape


# In[ ]:





# Same is true with the features 'etecho1' to 'etecho3'. This drops 2 more characters.
# 
# 1:if roofs are bad, 2:if roof are regular, 3:if roof are good

# In[ ]:


HPtrainN['etecho1'].unique()


# In[ ]:


HPtrainN['etecho2'].unique()


# In[ ]:


HPtrainN['etecho3'].unique()


# In[ ]:


HPtrainN['etecho2'] = HPtrainN['etecho2'].map({0: 0, 1: 2})

HPtestN['etecho2'] = HPtestN['etecho2'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['etecho3'] = HPtrainN['etecho3'].map({0: 0, 1: 3})

HPtestN['etecho3'] = HPtestN['etecho3'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['etecho1'] = HPtrainN['etecho1'] + HPtrainN['etecho2'] + HPtrainN['etecho3'] 

HPtestN['etecho1'] = HPtestN['etecho1'] + HPtestN['etecho2'] + HPtestN['etecho3'] 


# In[ ]:


HPtrainN.drop(['etecho2', 'etecho3'], inplace=True, axis=1)

HPtestN.drop(['etecho2', 'etecho3'], inplace=True, axis=1)


# In[ ]:


HPtrainN['etecho1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN.shape


# In[ ]:





# Same is true with the features 'epared1' to 'epared3'. This drops 2 more characters.
# 
# 1:walls are bad, 2:if walls are regular, 3:if walls are good

# In[ ]:


HPtrainN['epared1'].unique()


# In[ ]:


HPtrainN['epared2'].unique()


# In[ ]:


HPtrainN['epared2'].nunique()


# In[ ]:


HPtrainN['epared3'].unique()


# In[ ]:


HPtrainN['epared2'] = HPtrainN['epared2'].map({0: 0, 1: 2})

HPtestN['epared2'] = HPtestN['epared2'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['epared3'] = HPtrainN['epared3'].map({0: 0, 1: 3})

HPtestN['epared3'] = HPtestN['epared3'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['epared1'] = HPtrainN['epared1'] + HPtrainN['epared2'] + HPtrainN['epared3'] 

HPtestN['epared1'] = HPtestN['epared1'] + HPtestN['epared2'] + HPtestN['epared3'] 


# In[ ]:


HPtrainN.drop(['epared2', 'epared3'], inplace=True, axis=1)

HPtestN.drop(['epared2', 'epared3'], inplace=True, axis=1)


# In[ ]:


HPtrainN['epared1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['epared1'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# Same is true with the features 'elimbasu1' to 'elimbasu5'. This drops 4 more characters.
# 
# 1:if rubbish disposal mainly by tanker truck,
# 2:if rubbish disposal mainly by botan hollow or buried,
# 3:if rubbish disposal mainly by burning,
# 4:if rubbish disposal mainly by throwing in an unoccupied space,
# 5:if rubbish disposal mainly by throwing in river,  creek or sea",
# 6:if rubbish disposal mainly other

# In[ ]:


HPtrainN['elimbasu1'].unique()


# In[ ]:


HPtrainN['elimbasu2'].unique()


# In[ ]:


HPtrainN['elimbasu3'].unique()


# In[ ]:


HPtrainN['elimbasu4'].unique()


# In[ ]:


HPtrainN['elimbasu5'].unique()


# In[ ]:


HPtrainN['elimbasu2'] = HPtrainN['elimbasu2'].map({0: 0, 1: 2})

HPtestN['elimbasu2'] = HPtestN['elimbasu2'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['elimbasu3'] = HPtrainN['elimbasu3'].map({0: 0, 1: 3})

HPtestN['elimbasu3'] = HPtestN['elimbasu3'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['elimbasu4'] = HPtrainN['elimbasu4'].map({0: 0, 1: 4})

HPtestN['elimbasu4'] = HPtestN['elimbasu4'].map({0: 0, 1: 4})


# In[ ]:


HPtrainN['elimbasu5'] = HPtrainN['elimbasu5'].map({0: 0, 1: 5})

HPtestN['elimbasu5'] = HPtestN['elimbasu5'].map({0: 0, 1: 5})


# In[ ]:


HPtrainN['elimbasu6'] = HPtrainN['elimbasu6'].map({0: 0, 1: 6})

HPtestN['elimbasu6'] = HPtestN['elimbasu6'].map({0: 0, 1: 6})


# In[ ]:


HPtrainN['elimbasu1'] = HPtrainN['elimbasu1'] +   HPtrainN['elimbasu2'] +  HPtrainN['elimbasu3'] +  HPtrainN['elimbasu4'] +  HPtrainN['elimbasu5'] +  HPtrainN['elimbasu6'] 

HPtestN['elimbasu1'] = HPtestN['elimbasu1'] +   HPtestN['elimbasu2'] +  HPtestN['elimbasu3'] +  HPtestN['elimbasu4'] +  HPtestN['elimbasu5'] +  HPtestN['elimbasu6'] 


# In[ ]:


HPtrainN.drop(['elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6'], inplace=True, axis=1)

HPtestN.drop(['elimbasu2','elimbasu3','elimbasu4','elimbasu5','elimbasu6'], inplace=True, axis=1)


# In[ ]:


HPtrainN['elimbasu1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['elimbasu1'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# 'energcocinar1' to 'energcocinar4' marged to the column 'energcocinar1'. This reduces 3 more features.
# 
# 1:no main source of energy used for cooking (no kitchen),
# 2:main source of energy used for cooking electricity,
# 3:main source of energy used for cooking gas,
# 4:main source of energy used for cooking wood charcoal

# In[ ]:


HPtrainN['energcocinar1'].unique()


# In[ ]:


HPtrainN['energcocinar2'].unique()


# In[ ]:


HPtrainN['energcocinar3'].unique()


# In[ ]:


HPtrainN['energcocinar4'].unique()


# In[ ]:


HPtrainN['energcocinar2'] = HPtrainN['energcocinar2'].map({0: 0, 1: 2})

HPtestN['energcocinar2'] = HPtestN['energcocinar2'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['energcocinar3'] = HPtrainN['energcocinar3'].map({0: 0, 1: 3})

HPtestN['energcocinar3'] = HPtestN['energcocinar3'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['energcocinar4'] = HPtrainN['energcocinar4'].map({0: 0, 1: 4})

HPtestN['energcocinar4'] = HPtestN['energcocinar4'].map({0: 0, 1: 4})


# In[ ]:


HPtrainN['energcocinar1'] = HPtrainN['energcocinar1'] +   HPtrainN['energcocinar2'] +  HPtrainN['energcocinar3'] +  HPtrainN['energcocinar4'] 

HPtestN['energcocinar1'] = HPtestN['energcocinar1'] +   HPtestN['energcocinar2'] +  HPtestN['energcocinar3'] +  HPtestN['energcocinar4'] 


# In[ ]:


HPtrainN.drop(['energcocinar2','energcocinar3','energcocinar4'], inplace=True, axis=1)

HPtestN.drop(['energcocinar2','energcocinar3','energcocinar4'], inplace=True, axis=1)


# In[ ]:


HPtrainN['energcocinar1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['energcocinar1'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# v14a, =1 has toilet in the household
# 
# sanitario1, =1 no toilet in the dwelling
# 
# we delete 'v14a' since they are also described in 'sanitaroi1' to 'sanitaroi5','sanitaroi6'. We marge them all to 'sanitaroi1'.
# 
# 1:no toilet in the dwelling,
# 2:toilet connected to sewer or cesspool,
# 3:toilet connected to  septic tank,
# 5:toilet connected to black hole or letrine,
# 6:toilet connected to other system

# In[ ]:


HPtrainN['v14a'].value_counts()


# In[ ]:


del HPtrainN['v14a']

del HPtestN['v14a']


# In[ ]:


HPtrainN['sanitario1'].value_counts()


# In[ ]:


HPtrainN['sanitario2'] = HPtrainN['sanitario2'].map({0: 0, 1: 2})

HPtestN['sanitario2'] = HPtestN['sanitario2'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['sanitario3'] = HPtrainN['sanitario3'].map({0: 0, 1: 3})

HPtestN['sanitario3'] = HPtestN['sanitario3'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['sanitario5'] = HPtrainN['sanitario5'].map({0: 0, 1: 5})

HPtestN['sanitario5'] = HPtestN['sanitario5'].map({0: 0, 1: 5})


# In[ ]:


HPtrainN['sanitario6'] = HPtrainN['sanitario6'].map({0: 0, 1: 6})

HPtestN['sanitario6'] = HPtestN['sanitario6'].map({0: 0, 1: 6})


# In[ ]:


HPtrainN['sanitario1'] = HPtrainN['sanitario1'] +   HPtrainN['sanitario2'] +  HPtrainN['sanitario3'] +  HPtrainN['sanitario5'] +  HPtrainN['sanitario6']

HPtestN['sanitario1'] = HPtestN['sanitario1'] +   HPtestN['sanitario2'] +  HPtestN['sanitario3'] +  HPtestN['sanitario5'] +  HPtestN['sanitario6']


# In[ ]:


HPtrainN.drop(['sanitario2','sanitario3','sanitario5','sanitario6'], inplace=True, axis=1)

HPtestN.drop(['sanitario2','sanitario3','sanitario5','sanitario6'], inplace=True, axis=1)


# In[ ]:


HPtrainN['sanitario1'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['sanitario1'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# 1:electricity from CNFL, ICE, ESPH/JASEC"
# 2:electricity from private plant
# 3:no electricity in the dwelling
# 4:electricity from cooperative

# In[ ]:


HPtrainN['public'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['planpri'] = HPtrainN['planpri'].map({0: 0, 1: 2})

HPtestN['planpri'] = HPtestN['planpri'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['noelec'] = HPtrainN['noelec'].map({0: 0, 1: 3})

HPtestN['noelec'] = HPtestN['noelec'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['coopele'] = HPtrainN['coopele'].map({0: 0, 1: 4})

HPtestN['coopele'] = HPtestN['coopele'].map({0: 0, 1: 4})


# In[ ]:


HPtrainN['public'] = HPtrainN['public'] + HPtrainN['planpri'] + HPtrainN['noelec'] + HPtrainN['coopele']

HPtestN['public'] = HPtestN['public'] + HPtestN['planpri'] + HPtestN['noelec'] + HPtestN['coopele']


# In[ ]:


HPtrainN.drop(['planpri', 'noelec', 'coopele'], inplace=True, axis=1)

HPtestN.drop(['planpri', 'noelec', 'coopele'], inplace=True, axis=1)


# In[ ]:


HPtrainN['public'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['public'] = HPtrainN['public'].map({0: 3, 1:1,2:2,3:3,4:4})


# In[ ]:


HPtrainN['public'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['public'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# 1:if predominant material on the outside wall is block or brick,
# 2:if predominant material on the outside wall is socket (wood,  zinc or absbesto,
# 3:if predominant material on the outside wall is prefabricated or cement,
# 4:if predominant material on the outside wall is waste material,
# 5:if predominant material on the outside wall is wood,
# 6:if predominant material on the outside wall is zink,
# 7:if predominant material on the outside wall is natural fibers,
# 8:if predominant material on the outside wall is other

# In[ ]:


HPtrainN['paredblolad'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['paredzocalo'] = HPtrainN['paredzocalo'].map({0: 0, 1: 2})

HPtestN['paredzocalo'] = HPtestN['paredzocalo'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['paredpreb'] = HPtrainN['paredpreb'].map({0: 0, 1: 3})

HPtestN['paredpreb'] = HPtestN['paredpreb'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['pareddes'] = HPtrainN['pareddes'].map({0: 0, 1: 4})

HPtestN['pareddes'] = HPtestN['pareddes'].map({0: 0, 1: 4})


# In[ ]:


HPtrainN['paredmad'] = HPtrainN['paredmad'].map({0: 0, 1: 5})

HPtestN['paredmad'] = HPtestN['paredmad'].map({0: 0, 1: 5})


# In[ ]:


HPtrainN['paredzinc'] = HPtrainN['paredzinc'].map({0: 0, 1: 6})

HPtestN['paredzinc'] = HPtestN['paredzinc'].map({0: 0, 1: 6})


# In[ ]:


HPtrainN['paredfibras'] = HPtrainN['paredfibras'].map({0: 0, 1: 7})

HPtestN['paredfibras'] = HPtestN['paredfibras'].map({0: 0, 1: 7})


# In[ ]:


HPtrainN['paredother'] = HPtrainN['paredother'].map({0: 0, 1: 8})

HPtestN['paredother'] = HPtestN['paredother'].map({0: 0, 1: 8})


# In[ ]:


HPtrainN['paredblolad'] = HPtrainN['paredblolad'] + HPtrainN['paredzocalo'] + HPtrainN['paredpreb'] + HPtrainN['pareddes'] + HPtrainN['paredmad'] + HPtrainN['paredzinc'] + HPtrainN['paredfibras'] + HPtrainN['paredother']

HPtestN['paredblolad'] = HPtestN['paredblolad'] + HPtestN['paredzocalo'] + HPtestN['paredpreb'] + HPtestN['pareddes'] + HPtestN['paredmad'] + HPtestN['paredzinc'] + HPtestN['paredfibras'] + HPtestN['paredother']


# In[ ]:


HPtrainN.drop(['paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc','paredfibras','paredother'], inplace=True, axis=1)

HPtestN.drop(['paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc','paredfibras','paredother'], inplace=True, axis=1)


# In[ ]:


HPtrainN['paredblolad'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['paredblolad'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# 1:if predominant material on the floor is mosaic,  ceramic,  terrazo,
# 2:if predominant material on the floor is cement,
# 3:if predominant material on the floor is other,
# 4:if predominant material on the floor is  natural material,
# 5:if no floor at the household,
# 6:if predominant material on the floor is wood

# In[ ]:


HPtrainN['pisomoscer'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['pisocemento'] = HPtrainN['pisocemento'].map({0: 0, 1: 2})

HPtestN['pisocemento'] = HPtestN['pisocemento'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['pisoother'] = HPtrainN['pisoother'].map({0: 0, 1: 3})

HPtestN['pisoother'] = HPtestN['pisoother'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['pisonatur'] = HPtrainN['pisonatur'].map({0: 0, 1: 4})

HPtestN['pisonatur'] = HPtestN['pisonatur'].map({0: 0, 1: 4})


# In[ ]:


HPtrainN['pisonotiene'] = HPtrainN['pisonotiene'].map({0: 0, 1: 5})

HPtestN['pisonotiene'] = HPtestN['pisonotiene'].map({0: 0, 1: 5})


# In[ ]:


HPtrainN['pisomadera'] = HPtrainN['pisomadera'].map({0: 0, 1: 6})

HPtestN['pisomadera'] = HPtestN['pisomadera'].map({0: 0, 1: 6})


# In[ ]:


HPtrainN['pisomoscer'] = HPtrainN['pisomoscer'] + HPtrainN['pisocemento'] + HPtrainN['pisoother'] + HPtrainN['pisonatur'] + HPtrainN['pisonotiene'] + HPtrainN['pisomadera']

HPtestN['pisomoscer'] = HPtestN['pisomoscer'] + HPtestN['pisocemento'] + HPtestN['pisoother'] + HPtestN['pisonatur'] + HPtestN['pisonotiene'] + HPtestN['pisomadera']


# In[ ]:


HPtrainN.drop(['pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera'], inplace=True, axis=1)

HPtestN.drop(['pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera'], inplace=True, axis=1)


# In[ ]:


HPtrainN['pisomoscer'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['pisomoscer'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# 1:if predominant material on the roof is metal foil or zink,
# 2:if predominant material on the roof is fiber cement,  mezzanine,
# 3:if predominant material on the roof is natural fibers,
# 4:if predominant material on the roof is other

# In[ ]:


HPtrainN['techozinc'].value_counts()


# In[ ]:


HPtrainN['techoentrepiso'].value_counts()


# In[ ]:


HPtrainN['techocane'].value_counts()


# In[ ]:


HPtrainN['techootro'].value_counts()


# In[ ]:


HPtrainN['techoentrepiso'] = HPtrainN['techoentrepiso'].map({0: 0, 1: 2})

HPtestN['techoentrepiso'] = HPtestN['techoentrepiso'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['techocane'] = HPtrainN['techocane'].map({0: 0, 1: 3})

HPtestN['techocane'] = HPtestN['techocane'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['techootro'] = HPtrainN['techootro'].map({0: 0, 1: 4})

HPtestN['techootro'] = HPtestN['techootro'].map({0: 0, 1: 4})


# In[ ]:


HPtrainN['techozinc'] = HPtrainN['techozinc'] + HPtrainN['techoentrepiso'] + HPtrainN['techocane'] + HPtrainN['techootro']

HPtestN['techozinc'] = HPtestN['techozinc'] + HPtestN['techoentrepiso'] + HPtestN['techocane'] + HPtestN['techootro']


# In[ ]:


HPtrainN.drop(['techoentrepiso', 'techocane', 'techootro'], inplace=True, axis=1)

HPtestN.drop(['techoentrepiso', 'techocane', 'techootro'], inplace=True, axis=1)


# In[ ]:


HPtrainN['techozinc'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['techozinc'].value_counts()


# We see that 19 buildings has index 0 we map them to 4 (others)

# In[ ]:


HPtrainN['techozinc'] = HPtrainN['techozinc'].map({0: 4, 1: 1, 2:2, 3:3, 4:4})


# In[ ]:


HPtrainN['techozinc'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# 1:if water provision inside the dwelling,
# 2:if water provision outside the dwelling,
# 3:if no water provision

# In[ ]:


HPtrainN['abastaguadentro'].value_counts()


# In[ ]:


HPtrainN['abastaguafuera'] = HPtrainN['abastaguafuera'].map({0: 0, 1: 2})

HPtestN['abastaguafuera'] = HPtestN['abastaguafuera'].map({0: 0, 1: 2})


# In[ ]:


HPtrainN['abastaguano'] = HPtrainN['abastaguano'].map({0: 0, 1: 3})

HPtestN['abastaguano'] = HPtestN['abastaguano'].map({0: 0, 1: 3})


# In[ ]:


HPtrainN['abastaguadentro'] = HPtrainN['abastaguadentro'] + HPtrainN['abastaguafuera'] + HPtrainN['abastaguano']

HPtestN['abastaguadentro'] = HPtestN['abastaguadentro'] + HPtestN['abastaguafuera'] + HPtestN['abastaguano']


# In[ ]:


HPtrainN.drop(['abastaguafuera', 'abastaguano'], inplace=True, axis=1)

HPtestN.drop(['abastaguafuera', 'abastaguano'], inplace=True, axis=1)


# In[ ]:


HPtrainN['abastaguadentro'].value_counts()


# In[ ]:


HPtrainN.shape


# In[ ]:





# In[ ]:


HPtrainN.drop(['v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2'], inplace=True, axis=1)

HPtestN.drop(['v2a1', 'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 'r4m3', 'r4t1', 'r4t2'], inplace=True, axis=1)


# In[ ]:


print(HPtrainN.shape)

print(HPtestN.shape)


# In[ ]:





# In[ ]:


HPtrainN['hacapo'].value_counts()


# In[ ]:


HPtrainN.loc[HPtrainN['hacapo']==1,['idhogar', 'r4t3','rooms', 'bedrooms', 'overcrowding']]


# In[ ]:





# # (e)Converting some variables from 'object' type to numeric
# 
# We examine the data type of the variables and we see that some of them are of object type (mixed str,float,int). We have to convert them to int/float to extract any thing meaningfull. We have three such variables 'dependency', 'edjefe' and 'edjefa'.

# In[ ]:


HPtrainN.dtypes


# dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)

# In[ ]:


HPtrainN['dependency'].value_counts()


# We see that it is really a mixture of many different kind of characters

# In[ ]:


HPtrainN.loc[HPtrainN['dependency']=='no',['idhogar', 'dependency','r4t3', 'hogar_nin', 'hogar_adul', 'hogar_mayor']]


# In[ ]:


HPtrainN.loc[HPtrainN['dependency']=='yes',['idhogar', 'dependency','r4t3', 'hogar_nin', 'hogar_adul', 'hogar_mayor']]


# From the above tables and the formula given for calculating 'dependency' we decide to assine 1 for 'yes' and 0 for 'no'.

# In[ ]:


HPtrainN['dependency'] = HPtrainN['dependency'].replace({'no': 0, 'yes': 1})

HPtestN['dependency'] = HPtestN['dependency'].replace({'no': 0, 'yes': 1})


# In[ ]:


HPtrainN['dependency'].value_counts()


# In[ ]:


HPtrainN['dependency'].dtype


# Its still object type. We convert it to numeric

# In[ ]:


# Finally we convert it to numeric 
HPtrainN['dependency'] = pd.to_numeric(HPtrainN['dependency'])

HPtestN['dependency'] = pd.to_numeric(HPtestN['dependency'])


# In[ ]:





# edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# 
# converting this string categorical variable to the numbers

# In[ ]:


HPtrainN['edjefe'].value_counts()


# In[ ]:


HPtrainN['edjefe'].isnull().sum()


# In[ ]:


HPtrainN['edjefe'].dtype


# In[ ]:


HPtrainN['edjefe'] = HPtrainN['edjefe'].replace({'no': 0, 'yes': 1})

HPtestN['edjefe'] = HPtestN['edjefe'].replace({'no': 0, 'yes': 1})


# In[ ]:


HPtrainN['edjefe'].dtype


# In[ ]:


HPtrainN['edjefe'] = pd.to_numeric(HPtrainN['edjefe'])

HPtestN['edjefe'] = pd.to_numeric(HPtestN['edjefe'])


# In[ ]:


HPtrainN['edjefe'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:





# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0

# In[ ]:


HPtrainN['edjefa'].value_counts()


# In[ ]:


HPtrainN['edjefa'] = HPtrainN['edjefa'].replace({'no': 0, 'yes': 1})

HPtestN['edjefa'] = HPtestN['edjefa'].replace({'no': 0, 'yes': 1})


# In[ ]:


HPtrainN['edjefa'].dtype


# In[ ]:


HPtrainN['edjefa'] = pd.to_numeric(HPtrainN['edjefa'])
HPtestN['edjefa'] = pd.to_numeric(HPtestN['edjefa'])


# In[ ]:


HPtrainN['edjefa'].dtype


# In[ ]:


HPtrainN['edjefa'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:





# In[ ]:


HPtrainN['idhogar']= HPtrainN['idhogar'].astype('category')
HPtrainN['idhogar']= HPtrainN['idhogar'].cat.codes

HPtestN['idhogar']= HPtestN['idhogar'].astype('category')
HPtestN['idhogar']= HPtestN['idhogar'].cat.codes


# In[ ]:


HPtrainN['Target'].value_counts().plot(kind='bar')
sns.despine


# In[ ]:


HPtrainN['Target'].value_counts()


# In[ ]:





# In[ ]:


HPtrainN.dtypes


# # (f) We look at the variable inflation factor to remove variables that are highly correlated.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from sklearn import linear_model


# In[ ]:


X = HPtrainN.drop(['Target','idhogar'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['r4t3']

del HPtestN['r4t3']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['bedrooms']

del HPtestN['bedrooms']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['epared1']

del HPtestN['epared1']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['eviv1']

del HPtestN['eviv1']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['sanitario1']

del HPtestN['sanitario1']


# In[ ]:


X = HPtrainN.drop(['Target','idhogar'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['rooms']

del HPtestN['rooms']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['abastaguadentro']

del HPtestN['abastaguadentro']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['refrig']

del HPtestN['refrig']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['energcocinar1']

del HPtestN['energcocinar1']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['meaneduc']

del HPtestN['meaneduc']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['hogar_adul']

del HPtestN['hogar_adul']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['etecho1']

del HPtestN['etecho1']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['overcrowding']

del HPtestN['overcrowding']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['techozinc']

del HPtestN['techozinc']


# In[ ]:


X = HPtrainN.drop(['Target'],axis=1)
y = HPtrainN['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)


# In[ ]:


del HPtrainN['idhogar']

del HPtestN['idhogar']


# In[ ]:


print(HPtrainN.shape)

print(HPtestN.shape)


# In[ ]:


HPtrainN.columns.values


# In[ ]:


HPtestN.columns.values


# In[ ]:





# # Part3-Model building and making prediction
# 
# We do it in five steps
# 
# (a) As we already know (also shown below) the number of poorest families belong to Target=1 category is a small fraction of total number of families in the train data. It is only 7.2% compared to 65.39% of nonvulnerable families. Also our target will be to build model that can point out extremely poor families. This problem has notable similarity with the credit card fraud detection problem. The data is also highly skewed. For this reason we make a subset from our data which contains all the cases from category 1, 2 and 3 and an equal number representing the combined number of these three cases from category 4. So category 1,2 and 3 contribute 50% to the new dataset and category4 alone 50%.
# 
# (b) We perform a test-train split allocating very low percentage for the test part. Out actual intention is to appy the model on almost all of the training data.
# 
# (c) We try different classification schemes namely K-near neighbour, Support Vector Classifier, Decision Tree Classifier and Random Forest Classifier to pick up the best one based on the training score and cross validation score. We also perform an extensive grid search to pick up the best parameter sets for all these classifiers. We finally pick up Random Forest Classifier as our model. 
# 
# (d) We compute confusion matrix and classification report for the part of the test data we splitted out from the training sample. Since the number of this test sample is a tiny fraction of the whole traning dtaa we will see poor/untrustworthy. This doesn't necessarily mean that our model is poorly trained. The same model when fitted on the whole training data gives excellent precision and most importantly high recall, since we don't want to miss even a single family in extreme need for social aids.
# 
# (e) Finally we apply this model to the test data and make predictions.

# # (a) We see that the percentagle of poorest families (Target1) is only 7.2%

# In[ ]:


HPtrainN['Target'].value_counts()


# WE SEE THAT ALL THE MODELS ARE GOOD AT PREDICTING THE 4TH CLASS WHICH IS NON VULNERABLE BUT BAD AT PREDICTING CLASS 1 WHICH REPRESENTS EXTREMELY POOR GROUP OF PEOPLE. THIS IS AGAIN A BIASED PROBLEM SINCE EVEN IN TTRAIN DATA THEY COMPRICE ONLY 7.2% OF TOTAL DATA.

# In[ ]:


print('Target1', round(HPtrainN['Target'].value_counts()[1]/len(HPtrainN) * 100,2), '% of the dataset')
print('TArget2', round(HPtrainN['Target'].value_counts()[2]/len(HPtrainN) * 100,2), '% of the dataset')
print('Target3', round(HPtrainN['Target'].value_counts()[3]/len(HPtrainN) * 100,2), '% of the dataset')
print('TArget4', round(HPtrainN['Target'].value_counts()[4]/len(HPtrainN) * 100,2), '% of the dataset')


# We make a data frame 'new_df' by following the discussion made above.

# In[ ]:


HPtrainN = HPtrainN.sample(frac=1)

Target1_df = HPtrainN.loc[HPtrainN['Target'] == 1]
Target2_df = HPtrainN.loc[HPtrainN['Target'] == 2]
Target3_df = HPtrainN.loc[HPtrainN['Target'] == 3]
Target4_df = HPtrainN.loc[HPtrainN['Target'] == 4][:1034]

normal_distributed_df = pd.concat([Target1_df,Target2_df,Target3_df,Target4_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

print(new_df.head())
print(new_df.shape)


# # (b) We apply test-train split with negligibly small test_size=0.01.

# In[ ]:


from sklearn.model_selection import train_test_split
X = new_df.drop('Target',axis=1)
y = new_df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)


# In[ ]:


# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# # (c) Transporting and applying different classification schemes from scikit learn libraries.

# In[ ]:


# Classifier Libraries
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# In[ ]:


# Let's implement simple classifiers

classifiers = {
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier()
}


# Applying cross validation technique to the classifiers.

# In[ ]:


from sklearn.model_selection import cross_val_score

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")


# Applying grid search to findout the best parameters for each classifier.

# In[ ]:


# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)


# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# SVC best estimator
svc = grid_svc.best_estimator_


# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree_clf=grid_tree.best_estimator_

# Random Forest Classifier
rfcl = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 
#rfcl = RandomForestClassifier()
param_grid = { 'n_estimators': [600,700,800],  'max_features': ['auto','sqrt','log2']}
rfc_grid = GridSearchCV(estimator=rfcl, param_grid=param_grid)
rfc_grid.fit(X_train, y_train)
rfc=rfc_grid.best_estimator_
print(rfc_grid.best_params_)


# Applying cross validation on these best fitted classifiers.

# In[ ]:


# Overfitting Case

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')

randomforest_score = cross_val_score(rfc, X_train, y_train, cv=5)
print('RandomForest Classifier Cross Validation Score', round(randomforest_score.mean() * 100, 2).astype(str) + '%')


# Based on these scores we pick up RandomForest Classifier as our working model. We predict on X_test and show below the rank of the various features of the training dataset.

# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


from sklearn.feature_selection import RFE
names=list(new_df.columns)
#rank all features, i.e continue the elimination until the last one
rfe = RFE(rfc, n_features_to_select=10, step=1)
rfe.fit(X,y)
print('Features sorted by their rank:')
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

# Plot feature importance
feature_importance = rfc.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# # (d) We define confusion matrix and plot it for two cases as discussed above.

# In[ ]:


#def plot_confusion_matrix(cm,title='Confusion matrix',cmap=plt.cm.Blues):
def plot_confusion_matrix(cm, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


import itertools
from itertools import product
from sklearn.metrics import classification_report,confusion_matrix
cm=confusion_matrix(y_test,rfc_pred)
plot_confusion_matrix(cm, normalize=False, title='Confusion matrix',cmap=plt.cm.Purples)


# In[ ]:


print(classification_report(y_test,rfc_pred))


# We see that the performance of the model on the splitted test dataset of the training data is fairly poor. We apply the same model to the whole training dataset to get an excellent precision, recall and f1-score. Our objective here is to achieve high recall so that we hardly miss any family who is in extreme need of the aids.

# In[ ]:


X_whole = HPtrainN.drop('Target',axis=1)
y_whole = HPtrainN['Target']


# In[ ]:


X_whole.shape


# In[ ]:


rfc_pred_whole = rfc.predict(X_whole)


# In[ ]:


cm_whole=confusion_matrix(y_whole,rfc_pred_whole)
plot_confusion_matrix(cm_whole, normalize=False, title='Confusion matrix',cmap=plt.cm.Purples)


# In[ ]:


print(classification_report(y_whole,rfc_pred_whole))


# # (e) Now we predict on the test data

# In[ ]:


rfc_pred_test = rfc.predict(HPtestN)


# In[ ]:


dftest = pd.DataFrame({'Target': rfc_pred_test})


# In[ ]:


dftest.head()


# In[ ]:


dftest.shape


# In[ ]:


test_predictN=test_predict[['idhogar']].copy()


# In[ ]:


test_predictN.head()


# In[ ]:


test_predictN.drop_duplicates(subset='idhogar', inplace=True)


# In[ ]:


test_predictN.head()


# In[ ]:


test_predictN.shape


# In[ ]:


test_predictN.index = range(len(test_predictN))


# In[ ]:


test_predictN.head()


# In[ ]:


test_pred_concat= pd.concat([test_predictN,dftest], axis=1)


# In[ ]:


test_pred_concat.shape


# In[ ]:


test_pred_concat.head()


# In[ ]:


test_predict= pd.merge(test_predict, test_pred_concat, how='left', on=['idhogar'])


# In[ ]:


test_predict.head()


# In[ ]:


test_predict=test_predict.drop(['idhogar'],axis=1)


# In[ ]:


test_predict.head()


# In[ ]:


test_predict.shape


# In[ ]:


test_predict.to_csv('test_predict.csv',index=False)

