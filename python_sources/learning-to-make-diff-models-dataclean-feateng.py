#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is a step by step kernel for: `Data Cleaning`, `Feature Engineering`, and `Model Making and Prediction`. I will explain as I go what I am doing. 
# 
# ## Index
# 
# + [Imports](#imports)
# + [Some Useful Functions](#someUsefulFunctions)
# + [Data Preprocessing](#dataPreprocessing)
#     - [Taking care of Non-Numerical Features](#numFeatures)
#     - [Taking care of Null Values](#nullValues)
# + [Checking corrupted data](#checkingCorruptedData)
# + [Data Exploration and Visualization](#explore) **\***
# + [Feature Engineering](#featureEng)
#     - [Combine Some Features](#combiningFeatures)
#     - [Remove Highly Correlated Features](#removeHighlyCorrelatedFeatures)
#     - [An Error I got during training (Infinite values)](#anErrorIGot)
#     - [Making new Features](#makingNewFeatures)
#         * [More features using PCA](#pcaFeat)
# + [Handling - few data points for some categories](#handlingSmallData)
# + [Random Forest](#randomForest)
#     - [Check Feature's Importance](#checkFeatureImportance)
#     - [Removing Redundant Features](#removingRedundantFeatures)
# + [Gradient Boosting](#gradBoosting)
# + [Deep Neural Network](#dnn)
# + [Comparing Models](#compModels) **\*\***
# + [Making Submission File](#makingSubmission)

# # Imports  <a id="imports"></a>
# ---

# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torch


# # Some Useful functions <a id="someUsefulFunctions"></a>
# ---

# F1 score (macro) is what we need as a `metrics` to check how good or bad our model is. F1 score is a better quantifier of viability of model than `accuracy`.  

# In[ ]:


from sklearn.metrics import f1_score
def F1score(actuals, preds):
    """
    To get F1 score (macro) for our predictions.
    -----------------------------------------------------------
    Parameters:
        preds: Array of Predicted values
        actuals: Array of Actual labels
    Output:
        Return F1 score (macro)
    """
    return f1_score(actuals, preds, average = 'macro')


# # Data Preprocessing <a id="dataPreprocessing"></a>
# ---

# In[ ]:


PATH = "../input/"

train = pd.read_csv(f'{PATH}train.csv')
test = pd.read_csv(f'{PATH}test.csv')

train.info()


# ### Taking care of non-numerical features:  <a id="numFeatures"></a>

# Here, we will check for features (columns) that are non numerical. We need to take care of them because we cannot send `objects` into a Neural Network.
# So, we will convert non-numerical data to numerical data and then it will work for any model we use.

# In[ ]:


obj_cols = train.columns[train.dtypes == "object"]; obj_cols


# Description of these variables:
# 1.  **Id : ** a unique identifier for each row.
# 1.  **idhogar : ** this is a unique identifier for each household.
# 1.  **dependency : ** Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# 1.  **edjefe : ** years of education of male head of household
# 1. **edjefa : ** years of education of female head of household

# In[ ]:


train[obj_cols].describe()


# First,  we will take care of `Id` and `idhogar`:
# 
# We will use sklearn's `LabelEncoder` to encode these values. We can delete them as they are unique person or household identifier and they are different for every single unit of them, so they can't give any useful information for our output `Target`. For example: every household with`Target` value of `1` will have different `idhogar` and every individual with `Target` value of `1` will have different `Id`.
# But we will keep them for now, as we will use them in further Data Preprocessing.

# In[ ]:


# For saving space and compute time (mostly comparison for these)
from sklearn.preprocessing import LabelEncoder
# We will have to use two different label encoders. One for 'Id' and other for 'idhogar'.
lb1 = LabelEncoder()
lb1.fit(list(train['Id'].values))
lb2 = LabelEncoder()
lb2.fit(list(train['idhogar'].values))
# Now we will replace each unique id's with a unique number.
train['Id'] = lb1.transform(list(train['Id'].values))
train['idhogar'] = lb2.transform(list(train['idhogar'].values))

lb3 = LabelEncoder()
lb3.fit(list(test['Id'].values))
lb4 = LabelEncoder()
lb4.fit(list(test['idhogar'].values))
# Now we will replace each unique id's with a unique number.
test['Id'] = lb3.transform(list(test['Id'].values))
test['idhogar'] = lb4.transform(list(test['idhogar'].values))


# Now let's see the others:
# 
# 1) **Dependency** :

# In[ ]:


train['dependency'].unique()  # rate dependency  (yes:1, no:0)


# In[ ]:


train['dependency'].replace('yes', '1', inplace=True)
train['dependency'].replace('no', '0', inplace=True)
train['dependency'].astype(np.float64);


# In[ ]:


test['dependency'].replace('yes', '1', inplace=True)
test['dependency'].replace('no', '0', inplace=True)
test['dependency'].astype(np.float64);


# 2) **Edjefe** :

# In[ ]:


train['edjefe'].unique()  # years of education of male head of household  (given, yes:1, no:0)


# In[ ]:


train['edjefe'].replace('yes', '1', inplace=True)
train['edjefe'].replace('no', '0', inplace=True)
train['edjefe'].astype(np.float64);


# In[ ]:


test['edjefe'].replace('yes', '1', inplace=True)
test['edjefe'].replace('no', '0', inplace=True)
test['edjefe'].astype(np.float64);


# 3) **Edjefa** :

# In[ ]:


train['edjefa'].unique()  # years of education of female head of household  (given, yes:1, no:0)


# In[ ]:


train['edjefa'].replace('yes', '1', inplace=True)
train['edjefa'].replace('no', '0', inplace=True)
train['edjefa'].astype(np.float64);


# In[ ]:


test['edjefa'].replace('yes', '1', inplace=True)
test['edjefa'].replace('no', '0', inplace=True)
test['edjefa'].astype(np.float64);


# ### Taking care of NULL values: <a id="nullValues"></a>

# Null values in dataset can arise from many factors:
# 1.  Non availability of data as not applicable for that particular row
# 1.  Non availability of data as the Org. was not able to get it for some reason
# 1. Due to some error or misplacement
# 
# Here, we will consider that every feature had some data for every individual and put missing values equal to mean if no other option is available.

# In[ ]:


null_counts = train.isnull().sum()
null_counts[null_counts>0]


# Description of columns with missing values:
# 
# 1. **v2a1 :** Monthly rent payment
# 1. **v18q1 :** number of tablets household owns
# 1. **ez_esc : ** Years behind in school
# 1. **meaneduc :** average years of education for adults (18+)
# 1. **SQBmeaned :** square of the mean years of education of adults (>=18) in the household
# ---

# In[ ]:


test_null_counts = test.isnull().sum()
test_null_counts[test_null_counts>0]


# Same as train.

#  **1) v2a1 **: Monthly rent payment
# 
#         For this lets check these columns:
#         a) v2a1 : Monthly rent payment
#         b) v18q : owns a tablet
#         c) hacapo : Overcrowding by rooms
#         d) rooms : number of all rooms in the house
#         e) r4t3 : Total persons in the household
#         f) hhsize : household size
#         g) escolari : years of schooling
#         h) epared3 : =1 if walls are good
#         i) epared2 : =1 if walls are regular
#         j) tipovivi1 : =1 own and fully paid house
#         k) tipovivi2 : =1 own,  paying in installments
#         l) tipovivi3 : =1 rented
#         m) tipovivi4 : =1 precarious
#         n) tipovivi5 : =1 other(assigned,  borrowed)
#         p) Target : poverty level
# 
#     And check if they own their house or not.

# In[ ]:


cols = ['Id', 'parentesco1', 'v2a1', 'v18q', 'hacapo', 'rooms', 'r4t3', 'hhsize', 'escolari', 'epared2',
        'epared3', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 'Target']


# In[ ]:


v2a1_null = train.query('v2a1 == "NaN"')[cols]; v2a1_null.shape


# In[ ]:


# Let us get the family heads of each household in this
v2a1_null_heads = v2a1_null.query('parentesco1 == 1'); v2a1_null_heads.shape


# In[ ]:


v2a1_null_heads.query('hacapo != 1').shape, v2a1_null_heads.query('hacapo == 1').shape


# So there are *`26`* families who have overcrowding in their home. (where `v2a1` is null)
# 
# It means out of these null values, most of them are not living in poverty or extreme poverty. i.e. having `Target` value of *`1`* (most probably).

# In[ ]:


v2a1_null.query('Target == 1').shape, v2a1_null.query('Target == 2').shape


# Out of *`6860`* people having null `v2a1`, *`1862`* are from `Target` of *`1`* or *`2`*. Not much. 
# 
# So it is a possibility that many of them own their house or data is missing for some other reason.

# In[ ]:


v2a1_null_heads.query('epared2 != 1 & epared3 != 1').shape # Families who don't have regular or good walls


# Not much, again.

# In[ ]:


v2a1_null_heads.query('tipovivi1 != 1').shape # Families who don't own thier own home.


# So out of *`2156`* families, *`300`* don't own their home.

# In[ ]:


v2a1_null_heads.query('tipovivi2 == 1').shape, v2a1_null_heads.query('tipovivi3 == 1').shape, v2a1_null_heads.query('tipovivi4 == 1').shape, v2a1_null_heads.query('tipovivi5 == 1').shape


# Out of people who don't own their home have either `precarious`, or `other (assigned or borrowed)` homes.

# In[ ]:


v2a1_null_heads.query('tipovivi1 != 1 & tipovivi2 != 1 & tipovivi3 != 1 & tipovivi4 != 1 & tipovivi5 != 1') 
# Checking for any wrong data


# In[ ]:


v2a1_null_heads.query('(tipovivi4 == 1 | tipovivi5 == 1) & Target == 1').shape, v2a1_null_heads.query('(tipovivi4 == 1 | tipovivi5 == 1) & Target == 2').shape, v2a1_null_heads.query('(tipovivi4 == 1 | tipovivi5 == 1) & Target == 3').shape


# Out of families with `precarious` or `other` homes, *`114`* have `Target` value <=*`2`*.
# 
# As we see, out of these *`300`* families who don't own their homes, we have a mix of families, with all `Target` values.
# 
# So, we will put `v2a1` value equal to mean of `v2a1` values in set of that `Target` value.

# But firstly lets put `v2a1` values for people who own their homes equal to zero.

# In[ ]:


train.loc[train['v2a1'].isnull() & train['tipovivi1'] == 1, 'v2a1'] = 0


# Doing the same with test set:

# In[ ]:


test.loc[test['v2a1'].isnull() & test['tipovivi1'] == 1, 'v2a1'] = 0


# In[ ]:


train.query('v2a1 == "NaN"').shape, test.query('v2a1 == "NaN"').shape


# And now we will make others equal to their means, taking data only from their category.

# In[ ]:


train.query('v2a1 != "NaN"')['v2a1'].describe()


# In[ ]:


a, b = train.query('Target == 1 & v2a1 != "NaN"')['v2a1'].mean(), train.query('Target == 2 & v2a1 != "NaN"')['v2a1'].mean(); a, b


# In[ ]:


c, d = train.query('Target == 3 & v2a1 != "NaN"')['v2a1'].mean(), train.query('Target == 4 & v2a1 != "NaN"')['v2a1'].mean(); c, d


# In[ ]:


train.loc[train['v2a1'].isnull() & (train['Target']== 1), 'v2a1'] = a
train.loc[train['v2a1'].isnull() & (train['Target']== 2), 'v2a1'] = b
train.loc[train['v2a1'].isnull() & (train['Target']== 3), 'v2a1'] = c
train.loc[train['v2a1'].isnull() & (train['Target']== 4), 'v2a1'] = d
train.loc[train['v2a1'].isnull()]


# In[ ]:


test.loc[test['v2a1'].isnull(), 'v2a1'] = (a+b+c+d)/4  # We cannot check Target value here
test.loc[test['v2a1'].isnull()]


# ** 2) v18q1** : number of tablets household owns
# 
#     For this lets check these columns:
#     a) v18q : owns a tablet  # And no value is null here, we will use this.
# 

# In[ ]:


v18q1_null = train.query('v18q1 == "NaN"'); v18q1_null.shape


# In[ ]:


h_ids = v18q1_null['idhogar'].unique(); h_ids.shape


# In[ ]:


# For every household we will calulate how many of them owns a tablet and put 'v18q1' equal to that sum
for idn in h_ids:
    train.loc[(train['idhogar'] == idn), 'v18q1'] = train.query(f'idhogar == {idn}')['v18q'].sum()


# Doing the same for test set:

# In[ ]:


test_v18q1_null = test.query('v18q1 == "NaN"')
h_ids = test_v18q1_null['idhogar'].unique()


# In[ ]:


for idn in h_ids:
    test.loc[(test['idhogar'] == idn), 'v18q1'] = test.query(f'idhogar == {idn}')['v18q'].sum()


# **3) rez_esc** : Years behind in school
# 
#         For this we will check columns:
#         a) escolari: years of schooling
#         b) estadocivil1: =1 if less than 10 years old
#         c) estadocivil2: =1 if free or coupled union        # We are checking these ones, because they may be old
#         d) estadocivil3: =1 if married                      # and it might be the case that IADB does not have 
#         e) estadocivil4: =1 if divorced                     # this data about them.
#         f) estadocivil5: =1 if separated
#         g) estadocivil6: =1 if widower
#         h) estadocivil7: =1 if single
#         i) instlevel1: =1 no level of education
#         j) age: Age in years

# In[ ]:


cols = ['Id', 'idhogar', 'escolari', 'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4',
        'estadocivil5', 'estadocivil6', 'estadocivil7', 'instlevel1', 'age', 'Target']
cols2 = cols[:-1]


# In[ ]:


rez_esc_null = train.query('rez_esc == "NaN"')[cols]
test_rez_esc_null = test.query('rez_esc == "NaN"')[cols2]; rez_esc_null.shape, test_rez_esc_null.shape


# In[ ]:


rez_esc_null.query('instlevel1 == 1').shape, test_rez_esc_null.query('instlevel1 == 1').shape


# Good, out of these *`1183`* don't have any level of education (for training data). We will put `rez_esc` for them equal to *`0`*.

# In[ ]:


train.loc[(train['rez_esc'].isnull()) & (train['instlevel1'] == 1), 'rez_esc'] = 0
test.loc[(test['rez_esc'].isnull()) & (test['instlevel1'] == 1), 'rez_esc'] = 0


# In[ ]:


rez_esc_null = train.query('rez_esc == "NaN"')[cols]
test_rez_esc_null = test.query('rez_esc == "NaN"')[cols2]; rez_esc_null.shape, test_rez_esc_null.shape


# In[ ]:


# estadocivil1: =1 if less than 10 years old
rez_esc_null.query('estadocivil1 == 1').shape, test_rez_esc_null.query('estadocivil1 == 1').shape


# So, we don't have any child for whom this value is missing. Maybe we were right about thinking that they don't have this value for adults.

# In[ ]:


rez_esc_null.query('estadocivil2 == 1').shape, rez_esc_null.query('estadocivil3 == 1').shape, rez_esc_null.query('estadocivil4 == 1').shape


# So, we have *`1111`* `free or coupled union`, *`2486`* `married` and *`300`* `divorced`.

# In[ ]:


test_rez_esc_null.query('estadocivil2 == 1').shape, test_rez_esc_null.query('estadocivil3 == 1').shape, test_rez_esc_null.query('estadocivil4 == 1').shape


# In[ ]:


rez_esc_null.query('estadocivil5 == 1').shape, rez_esc_null.query('estadocivil6 == 1').shape, rez_esc_null.query('estadocivil7 == 1').shape


# *`564`* `separated`, *`279`* `widower` and *`2005`* `single`.

# In[ ]:


test_rez_esc_null.query('estadocivil5 == 1').shape, test_rez_esc_null.query('estadocivil6 == 1').shape, test_rez_esc_null.query('estadocivil7 == 1').shape


# We will put `rez_esc` equal to average value in their category:

# In[ ]:


a = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil2'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
b = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil3'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
c = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil4'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
d = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil5'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
e = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil6'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
f = train.loc[(~train['rez_esc'].isnull()) & (train['estadocivil7'] == 1) & (train['escolari'] > 0)]['rez_esc'].mean()
a, b, c, d, e, f

train.loc[(train['rez_esc'].isnull()) & (train['estadocivil2'] == 1), 'rez_esc'] = 3
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil3'] == 1), 'rez_esc'] = 0
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil4'] == 1), 'rez_esc'] = 0
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil5'] == 1), 'rez_esc'] = 2
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil6'] == 1), 'rez_esc'] = 0
train.loc[(train['rez_esc'].isnull()) & (train['estadocivil7'] == 1), 'rez_esc'] = 1

test.loc[(test['rez_esc'].isnull()) & (test['estadocivil2'] == 1), 'rez_esc'] = 3
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil3'] == 1), 'rez_esc'] = 0
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil4'] == 1), 'rez_esc'] = 0
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil5'] == 1), 'rez_esc'] = 2
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil6'] == 1), 'rez_esc'] = 0
test.loc[(test['rez_esc'].isnull()) & (test['estadocivil7'] == 1), 'rez_esc'] = 1


# In[ ]:


train['rez_esc'].isnull().sum(), test['rez_esc'].isnull().sum()


#  4) **meaneduc** : average years of education for adults (18+) and 
#  
#  5) **SQBmeaned**: square of the mean years of education of adults (>=18) in the household
# 
#        For this we will check: 
#        a) instlevel1 : =1 no level of education
#        b) instlevel2 : =1 incomplete primary
#        c) instlevel3 : =1 complete primary
#        d) instlevel4 : =1 incomplete academic secondary level
#        e) instlevel5 : =1 complete academic secondary level
#        f) instlevel6 : =1 incomplete technical secondary level
#        g) instlevel7 : =1 complete technical secondary level
#        h) instlevel8 : =1 undergraduate and higher education
#        i) instlevel9 : =1 postgraduate higher education

# In[ ]:


cols = ['Id', 'idhogar', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
       'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']


# In[ ]:


# We will just put this value equal to avg year of education of adults with the help of selected cols
meaneduc_null = train.query('meaneduc == "NaN"')[cols]
test_meaneduc_null = test.query('meaneduc == "NaN"')[cols]
h_ids = meaneduc_null['idhogar'].unique(); h_ids


# *`3`* families whose `meaneduc` is not available.

# In[ ]:


print(train.loc[(train['idhogar'] ==  326), 'meaneduc'].values)
print(train.loc[(train['idhogar'] == 1959), 'meaneduc'].values) 
print(train.loc[(train['idhogar'] == 2908), 'meaneduc'].values)


# So, all of them are actually here. Otherwise we could have put `meaneduc` equal to someone in their family who had that value present.

# In[ ]:


def meaneduc_correction(null_view, df, hids):
    """
    Function to correct null_values in "meaneduc" feature. Will put them equal to mean, after calculating it
    using "instlevel"'s.
    ---------------------------------------------------------------------------------------------------------
    Parameters:
        null_view: View of origianl dataframe with null values of "meaneduc"
        df: Original DataFrame
        hids: Unique Household ids of households with null "meaneduc"
    """
    for idn in hids:
        # Number of people with no education and so on
        a = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel1'] == 1)].shape[0] # No ed
        b = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel2'] == 1)].shape[0] # Inc. prim
        c = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel3'] == 1)].shape[0] # Com. prim
        d = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel4'] == 1)].shape[0] # Inc Acad Sec L.
        e = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel5'] == 1)].shape[0] # Com Acad Sec L.
        f = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel6'] == 1)].shape[0] # Inc Tech Sec L.
        g = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel7'] == 1)].shape[0] # Com Tech Sec L.
        h = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel8'] == 1)].shape[0] # UndGrad n HigEd
        i = null_view.loc[(null_view['idhogar']==idn) & (null_view['instlevel9'] == 1)].shape[0] # Postgrad

        mean_educ = (a*0 + b*4 + c*8 + d*2 + e*4 + f + g*2 + h*4 + i) / (a+b+c+d+e+f+g+h+i)

        df.loc[(df['meaneduc'].isnull()) & (df['idhogar'] == idn), 'meaneduc'] =  mean_educ
        df.loc[(df['SQBmeaned'].isnull()) & (df['idhogar'] == idn), 'SQBmeaned'] =  mean_educ**2


# In[ ]:


meaneduc_correction(meaneduc_null, train, h_ids)


# In[ ]:


null_counts = train.isnull().sum()
null_counts[null_counts>0]


# Now same for test:

# In[ ]:


h_ids = test_meaneduc_null['idhogar'].unique(); h_ids


# In[ ]:


meaneduc_correction(test_meaneduc_null, test, h_ids)


# In[ ]:


test_null_counts = test.isnull().sum()
test_null_counts[test_null_counts>0]


# # Checking for corrupted data <a id="checkingCorruptedData"></a>
# ---

# The only things we can check here are:
#      
#     1. Check if all Id's are unique. (Should have checked first. But all are unique)
#     2. Check if same household has same Target value, meaneduc value, zone value ( urban or rural), region value, house properties (wall type, ceiling type etc), number of persons in houshold, number of adults, number of childern, number of tablets household owns.
# 
# We cannot check the others, because there is no way to check their validity. If for some reson they are wrong, they are wrong. But such cases happen rarely.  So, we don't need to worry about that.

# In[ ]:


train.shape, test.shape


# In[ ]:


train['Id'].unique().size, test['Id'].unique().size  # So, Ids are unique


# In[ ]:


# Now for the second part
cols = ['v2a1', 'hacdor', 'rooms', 'hacapo', 'v14a', 'refrig', 'v18q1',
       'r4h3', 'r4m3', 'r4t3', 'tamhog', 'tamviv', 'hhsize', 'paredblolad',
       'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc',
       'paredfibras', 'paredother', 'pisomoscer', 'pisocemento', 
       'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera', 'techozinc',
       'techoentrepiso', 'techocane', 'techootro', 'cielorazo',
       'abastaguadentro', 'abastaguafuera', 'abastaguano', 'public', 
       'planpri', 'noelec', 'coopele', 'sanitario1', 'sanitario2', 
       'sanitario3', 'sanitario5', 'sanitario6', 
       'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4',
       'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5',
       'elimbasu6', 'epared1', 'epared2', 'epared3', 'etecho1', 'etecho2', 
       'etecho3', 'eviv1', 'eviv2', 'eviv3', 'hogar_nin', 'hogar_adul',
       'hogar_mayor', 'hogar_total', 'dependency', 'meaneduc', 'bedrooms', 
       'overcrowding', 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4',
       'tipovivi5', 'computer', 'television', 'qmobilephone', 'lugar1',
       'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'Target']


# In[ ]:


import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

def check_for_wrong_data(data, columns, labelE, gpby='idhogar'):
    """
    Checks for data mismatches in rows of every group, which we get by gpby (groupby)
    feature, on columns in "columns". If mismatch is there, put it equal to value in
    columns of head of the household and print a message for this.
    ----------------------------------------------------------------------------------
    Input:
        data : Train or Test set or their sliced views
        columns : columns to check for corrupted data
        labelE : Label encoder of "data"'s  "idhogar" column used in inverse_transform 
        gpby : feature to group by to check for diff "cols" in that group
    Output:
        Return four arrays:
        1) Array with ids of households with no head
        2) Array with ids of households with wrong data
        3) Array of arrays with column name with wrong data for each household in (2)nd array
        4) Array of arrays with ids of members with wrong data for each household in (2)nd array
    """
    id_head_zero = [] # Will contain house ids with no head
    idhogarId_f = []
    cols_f = []
    mem_f = []
    grouped = data.groupby(gpby, sort=True)
    for gid in range(len(grouped)):
        members = grouped.get_group(gid)
        h_Head = members.loc[(members['parentesco1'] == 1)]
        if h_Head.shape[0] == 0:
            id_head_zero.append(members['idhogar'].values[0])
            continue
        idhogarId_w = []
        cols_t = []
        mem_t = []
        if members.shape[0] > 1:
            for col in columns:
                for m in members.iterrows():
                    if h_Head[col].values[0] != m[1][col]:
                        if h_Head['idhogar'].values[0] not in idhogarId_w : idhogarId_w.append(h_Head['idhogar'].values[0])
                        if col not in cols_t : cols_t.append(col)
                        if m[1]['Id'] not in mem_t : mem_t.append(m[1]['Id'])
                        # Correct this column
                        data.loc[(train['Id'] == m[1]['Id']), col] = h_Head[col].values[0]
        idhogarId_f.append(idhogarId_w); cols_f.append(cols_t); mem_f.append(mem_t)
        if len(idhogarId_w) > 0:
            for i in range(len(idhogarId_w)):
                print("Household with Id: "
                +str(labelE.inverse_transform([idhogarId_w[i]])[0])
                +" has " + str(len(mem_t)) + " member(s) with diff. value(s) of " + str(len(cols_t)) + " column(s)."
                + " " + str(cols_t) )
    return id_head_zero, idhogarId_f, cols_f, mem_f


# In[ ]:


id_head_zero, *_ = check_for_wrong_data(train, cols, lb2)


# Now all of them are fixed.

# In[ ]:


train.loc[(train['idhogar'] == id_head_zero[11])]   # 4, 6, 7, 8, 11 have more than 1 persons in home but no head


# We will leave it as it is, as we don't know what to change it to.

# Same for `test` dataset:

# In[ ]:


cols = cols[:-1] # Remove "Target"


# In[ ]:


id_head_zero, *_ = check_for_wrong_data(test, cols, lb4)


# # Data Exploration and Visualization: <a id="explore"></a>
# ---

# In[ ]:


import seaborn as sns
columns = train.select_dtypes('number').drop(['Id', 'idhogar', 'Target'], axis=1).columns

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(20, 15))
fig.subplots_adjust(top=1.3)
#train.loc[:,columns[1:22]].boxplot(ax=axes[0])
a = sns.boxplot(x = "variable", y = "value", data = pd.melt(train.loc[:,columns[1:22]]), ax=axes[0])
b = sns.boxplot(x = "variable", y = "value", data = pd.melt(train.loc[:,columns[22:70]]), ax=axes[1])
b.set_xticklabels(rotation=30, labels = columns[22:70]);
c = sns.boxplot(x = "variable", y = "value", data = pd.melt(train.loc[:,columns[70:98]]), ax=axes[2])
c.set_xticklabels(rotation=30, labels = columns[70:120]);
d = sns.boxplot(x = "variable", y = "value", data = pd.melt(train.loc[:,columns[99:120]]), ax=axes[3])


# In[ ]:


possible_outliers = [columns[0]] + [columns[98]]; columns[0], columns[98]


# In[ ]:


sns.boxplot(data = train[possible_outliers[0]])


# Someone or some families have `v2a1` i.e. `Monthly rent Payment` of more than **`2,000,000`**!!

# In[ ]:


train.loc[(train['v2a1'] > 300000), ['idhogar', 'v2a1', 'Target']].query("Target != 4")  # Actually, all above 300,000 are from "Target" of 4


# But this one particular family pays about **`2,000,000`** (i.e. `$3456.20` at current rates) and I checked at a [site](https://www.propertiesincostarica.com) and some bunglows have similar rates...

# In[ ]:


train.loc[(train['v2a1'] > 2000000), ['idhogar', 'v2a1', 'Target']]   # So leave it


# In[ ]:


sns.boxplot(data = train[possible_outliers[1]])


# In[ ]:


train.loc[(train['meaneduc'] > 25), ['idhogar', 'Target', 'meaneduc']].query("Target != 4")


# They are also from `Target` value of **`4`**, so leave them too...
# 
# Now let's check correlation between some of the columns I selected:
# 1. `v2a1` :  Monthly rent payment
# 1. `rooms`: number of all rooms in the house
# 1. `tamhog`: size of the household
# 1. `overcrowding`: # persons per room
# 1. `v18q1`: number of tablets household owns
# 1. `r4t3`: Total persons in the household
# 1. `meaneduc`: average years of education for adults (18+)
# 1. `qmobilephone`: # of mobile phones
# 1. `Target`: the target is an ordinal variable indicating groups of income levels

# In[ ]:


columns = ['v2a1', 'rooms', 'tamhog', 'overcrowding', 'v18q1', 'r4t3', 'meaneduc', 'qmobilephone', 'Target']


# In[ ]:


sns.pairplot(train[columns])


# Some insights from this pair plot:
# 1. `meaneduc` with `v2a1` has a kind of Gaussian Distribution with people with about 20yrs of education paying greater house rent than others,
# 1. `v2a1` decreases as `overcrowing` increases. It may be possible that most of the overcrowded are from `Target` **`4`**,
# 1. `v2a1` also decreases as `r4t3` increases. It may be because of similar reason above, more # of people increase may be indication of a lower `Target` value,
# 1. `r4t3` and `tamhog` showing strong positive linear behaviour. Size of house hold doesn't guarantee quality.

# # Feature Engineering <a id="featureEng"></a>
# ---

# Firstly we can remove all the Square values columns : 
# 
#     SQBescolari     : escolari squared
#     SQBage          : age squared
#     SQBhogar_total  : hogar_total squared
#     SQBedjefe       : edjefe squared
#     SQBhogar_nin    : hogar_nin squared
#     SQBovercrowding : overcrowding squared
#     SQBdependency   : dependency squared
#     SQBmeaned       : square of the mean years of education of adults (>=18) in the household
#     agesq           : Age squared
# 
# because they will be highly correlated with their unit degree counterparts, and we will be using a Neural Network and in Neural Network you don't need higher order terms of your features to check if that degree of feature explains better or not.  We do that in Linear Regression to capture non linear relationship of some features with the output. Here, Neural Network will learn these relations by itself by adjusting the weights.
# 
# In RandomForest and GradientBoosting also, we wont use these highly correlated featues.

# In[ ]:


train.drop(['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding',
            'SQBdependency', 'SQBmeaned', 'agesq'], axis = 1, inplace=True)


# In[ ]:


test.drop(['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding',
            'SQBdependency', 'SQBmeaned', 'agesq'], axis = 1, inplace=True)


# In[ ]:


# Plotting a heat map
import seaborn as sns
plt.subplots(figsize=(20,15))
sns.heatmap(train.corr().abs(), cmap="BuPu")


# ### Combining some of the features: <a id="combiningFeatures"></a>

# We also can combine some ordinal groups, by making one feature from a group of features which give information about the same thing and have a ordinal relationship between them. Combining features will save us some space and compute time.
# 
# Ordinal feature Groups:
# *  Material outside wall:  `[ 'paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother' ]`
# *  Material Floor : `[ 'pisomoscer', 'pisocemento', 'pisoother' , 'pisonatur', 'pisonotiene', 'pisomadera' ]`
# *  Material Roof : `[ 'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo' ]`
# *  Water Provisoin : `[ 'abastaguadentro', 'abastaguafuera', 'abastaguano' ]`
# *  Electricity Provision : `[ 'public', 'planpri', 'noelec', 'coopele' ]`
# *  Sanitary Provision : `[ 'sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6' ]`
# *  Cooking Provision : `[ 'energcocinar1', 'energcocinar4', 'energcocinar3', 'energcocinar2' ]`
# *  Disposal Type : `[ 'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6' ]`
# *  Walls Type : `[ 'epared1', 'epared2', 'epared3' ]`
# *  Roof Type : `[ 'etecho1', 'etecho2', 'etecho3' ]`
# *  Floor Type : `[ 'eviv1', 'eviv2', 'eviv3' ]`
# *  Education Level : `[ 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9' ]`
# *  House Type : `[ 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5' ]`
# 
# We won't combine Material used features because we don't know the relative price of things there. 
# 
# **For water provision** : `no` < `outside` < `inside` **OR** `no` < `inside` < `outside`, we don't really know. So we will leave this group too.
# 
# **For elec Povision** : `no` < `JASEC/ESPH` < `Cooperative` < `private plant` **OR** `no` < `Coop` < `ESPH` < `Private Plant`; Leave can leave this too
# 
# **For Sanit Provision** : `no` < `blackhole` < `septic` < `sever`, `other` (?) **OR**  `no` < `blackhole` < `sever` < `septic`, `other`(?)
# 
# **For Cook'n Prov** : `no` < `wood` < `gas` < `elec`
# 
# **For Disposal type** : `river` >< `burning` >< `throwUnoccu` < `botan` < `Truck`, `other`(?); Leave it.
# 
# **For Wall Type** : `bad` < `reg` < `good`
# 
# **For Roof Type** : `bad` < `reg` < `good`
# 
# **For Floor Type** : `bad` < `reg` < `good`
# 
# **Ed Level** :  This group can be used
# 
# **House Type** : `Precarious` < `other` < `rented` < `installment` < `fullyOwned`
# 

# In[ ]:


DropCols = ['energcocinar1', 'energcocinar4', 'energcocinar3', 'energcocinar2', 'epared1', 'epared2', 'epared3',
        'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', #'instlevel1', 'instlevel2', 'instlevel3', 
        #'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',
        'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']

train['CookingType'] = np.argmax(np.array(train[[ 'energcocinar1', 'energcocinar4', 'energcocinar3', 'energcocinar2' ]]), axis=1)
train['WallType'] = np.argmax(np.array(train[[ 'epared1', 'epared2', 'epared3' ]]), axis=1)
train['RoofType'] = np.argmax(np.array(train[[ 'etecho1', 'etecho2', 'etecho3' ]]), axis=1)
train['FloorType'] = np.argmax(np.array(train[[ 'eviv1', 'eviv2', 'eviv3' ]]), axis=1)
# EdLevel is being removed during deletion of highly correlated features
# train['EdLevel'] = np.argmax(np.array(train[ [ 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9' ]]), axis=1)
train['HouseType'] = np.argmax(np.array(train[[ 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5' ]]), axis=1)

test['CookingType'] = np.argmax(np.array(test[[ 'energcocinar1', 'energcocinar4', 'energcocinar3', 'energcocinar2' ]]), axis=1)
test['WallType'] = np.argmax(np.array(test[[ 'epared1', 'epared2', 'epared3' ]]), axis=1)
test['RoofType'] = np.argmax(np.array(test[[ 'etecho1', 'etecho2', 'etecho3' ]]), axis=1)
test['FloorType'] = np.argmax(np.array(test[[ 'eviv1', 'eviv2', 'eviv3' ]]), axis=1)
# EdLevel is being removed during deletion of highly correlated features
# test['EdLevel'] = np.argmax(np.array(test[[ 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9' ]]), axis=1)
test['HouseType'] = np.argmax(np.array(test[[ 'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5' ]]), axis=1)

train.drop(DropCols, axis=1, inplace=True)
test.drop(DropCols, axis=1, inplace=True)
test.shape, train.shape


# More Domain Knowledge Features: (from [here](https://www.kaggle.com/willkoehrsen/featuretools-for-good))

# In[ ]:


# Per member features
train['phones-per-mem'] = train['qmobilephone'] / train['tamviv']
train['tablets-per-mem'] = train['v18q1'] / train['tamviv']
train['rooms-per-mem'] = train['rooms'] / train['tamviv']
train['rent-per-adult'] = train['v2a1'] / train['hogar_adul']

test['phones-per-mem'] = test['qmobilephone'] / test['tamviv']
test['tablets-per-mem'] = test['v18q1'] / test['tamviv']
test['rooms-per-mem'] = test['rooms'] / test['tamviv']
test['rent-per-adult'] = test['v2a1'] / test['hogar_adul']


# ### Remove Highly Correlated features:  <a id="removeHighlyCorrelatedFeatures"></a>
# 
# All Highly Correlated features are not necessary to kept in the dataset. We can take only one of them, which will be sufficient for getting what they were all telling together. Keeping all of them will be redundant, as they have same trend in dataset, and even one of them can capture that trend.

# In[ ]:


def chk_n_remove_corr(df):
    """
    Checks for highly correlated features and removes them.
    ---------------------------------------------------------------------
    Parameters:
        df: Dataframe to check for correlation
    Output:
        Return list of removed features/columns.
    """
    corr_matrix = train.corr()
    
    # Taking only the upper triangular part of correlation matrix: (We want to remove only one of corr features)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.975
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.975)]
    
    train.drop(to_drop, axis=1, inplace=True)
    return to_drop


# In[ ]:


to_drop = chk_n_remove_corr(train)
to_drop


# In[ ]:


test.drop(to_drop, axis=1, inplace=True)
train.shape, test.shape


# #### Changing the type of columns:
# 
# We will set type of all columns according to their possible values. 

# In[ ]:


hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6',
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1','tamviv','hogar_nin',# 'hhsize', 'tamhog',
              'CookingType', 'WallType', 'RoofType', 'HouseType' , 'FloorType', 
              'hogar_adul','hogar_mayor',  'bedrooms', 'qmobilephone'] # ,'hogar_total']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding',
          'phones-per-mem', 'tablets-per-mem', 'rooms-per-mem', 'rent-per-adult']

ind_bool = ['v18q', 'dis', 'male', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
            'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9',
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'mobilephone']

ind_ordered = ['age', 'escolari', 'rez_esc']#, 'EdLevel']


# In[ ]:


train[hh_bool + ind_bool] = train[hh_bool + ind_bool].astype(bool)
test[hh_bool + ind_bool] = test[hh_bool + ind_bool].astype(bool)


# In[ ]:


train[hh_cont] = train[hh_cont].astype('float64')
test[hh_cont] = test[hh_cont].astype('float64');


# In[ ]:


train[hh_ordered + ind_ordered] = train[hh_ordered + ind_ordered].astype(int)
test[hh_ordered + ind_ordered] = test[hh_ordered + ind_ordered].astype(int);


# In[ ]:


train['Target'] = train['Target'].astype(int);


# ### During prediction, I got an error that some values in validation set are null or infinity. So lets check: <a id="anErrorIGot"></a>

# In[ ]:


train.isnull().sum()[train.isnull().sum() > 0], test.isnull().sum()[test.isnull().sum() > 0]


# In[ ]:


# Only one value in train and 10 in test
train.loc[train['rent-per-adult'].isnull(), 'rent-per-adult'] = train.loc[train['rent-per-adult'].isnull(), 'v2a1'].values[0]
test.loc[test['rent-per-adult'].isnull(), 'rent-per-adult'] = test.loc[test['rent-per-adult'].isnull(), 'v2a1'].values[0]


# In[ ]:


train.isnull().sum()[train.isnull().sum() > 0], test.isnull().sum()[test.isnull().sum() > 0]


# In[ ]:


for c in train.columns:
    if train[c].dtype != 'float64': continue
    s = np.where(train[c].values >= np.finfo(np.float32).max)
    if len(s[0])>0:
        print(c)
        print(s)


# In[ ]:


train[train['rent-per-adult'] > np.finfo(np.float32).max][['rent-per-adult']]


# Woah! how did this happen?

# In[ ]:


train[train['rent-per-adult'] > np.finfo(np.float32).max][['Id', 'idhogar', 'v2a1', 'hogar_adul', 'age']]


# Thats why, because for these `hogar_adul` is zero.

# In[ ]:


train[train['idhogar']==1959][['idhogar', 'v2a1', 'age']]


# In[ ]:


train[train['idhogar']==2908][['idhogar', 'v2a1', 'age']]


# So, they are the only ones in their household.

# In[ ]:


h_ids = train[train['rent-per-adult'] > np.finfo(np.float32).max]['idhogar'].unique()


# In[ ]:


for h_id in h_ids:
    rent_per_adul = train.loc[(train['idhogar']==h_id), 'v2a1'].values[0] / train.loc[(train['idhogar']==h_id)].shape[0]
    # Assuming the rent is being divided among them equally
    train.loc[train['idhogar']==h_id, 'rent-per-adult'] = rent_per_adul
    train.loc[train['idhogar']==h_id, 'rent-per-adult_sum'] = train.loc[(train['idhogar']==h_id), 'v2a1'].values[0]


# #### Now for test set:

# In[ ]:


for c in test.columns:
    if test[c].dtype != 'float64': continue
    s = np.where(test[c].values >= np.finfo(np.float32).max)
    if len(s[0])>0:
        print(c)
        print(s)


# In[ ]:


test[test['rent-per-adult'] > np.finfo(np.float32).max][['rent-per-adult']]


# In[ ]:


test[test['rent-per-adult'] > np.finfo(np.float32).max][['Id', 'idhogar', 'v2a1', 'hogar_adul', 'age']]


# Assuming the rent is being divided among them equally.

# In[ ]:


h_ids = test[test['rent-per-adult'] > np.finfo(np.float32).max]['idhogar'].unique()

for h_id in h_ids:
    rent_per_adul = test.loc[(test['idhogar']==h_id), 'v2a1'].values[0] / test.loc[(test['idhogar']==h_id)].shape[0]
    # Assuming the rent is being divided among them equally
    test.loc[test['idhogar']==h_id, 'rent-per-adult'] = rent_per_adul
    test.loc[test['idhogar']==h_id, 'rent-per-adult_sum'] = test.loc[(test['idhogar']==h_id), 'v2a1'].values[0]


# ## Making new features: <a id="makingNewFeatures"></a>
# 
# Making new features from the existing features can help our model to learn new trends in data which were not given in dataset before. We take groups of data points from the dataset, and calculate a feature which is true for that group, and we do this for all groups.
# 
# Here, we group by `idhogar` (household id) and calculate `count`, `mean`, `max`, `min`, and `sum` of all numeric type features and make new features for all groups, in hopes that now these features will explain more about `Target` value.

# In[ ]:


def make_new_features_grouping(df, dtypes, gpby, customAggFunc=None):
    """
    Make new features aggregating on groups found by "gbpy".
    -----------------------------------------------------------------
    Parameters:
        df: Dataset for which new features are to be made
        dtypes: Data Types of features which will be used to create new features (string, type or array)
                eg: bool, 'number', 'float' etc
        gbpy: Feature on which grouping will be done
        customAggFunc: A custom Aggregation function or a list of such functions
    Output: 
        Returns Original DataFrame with new features
    """
    # Grouping
    if 'Target' in df.columns: numeric_type = df.select_dtypes(dtypes).drop(['Target', 'Id'], axis=1).copy()
    else: numeric_type = df.select_dtypes(dtypes).drop(['Id'], axis=1).copy()
    
    funcs = ['count', 'mean', 'max', 'min', 'sum', 'std', 'var', 'quantile']
    
    if customAggFunc is None: new = numeric_type.groupby(gpby).agg(funcs)
    elif isinstance(customAggFunc, list): new = numeric_type.groupby(gpby).agg(funcs + customAggFunc)
    else: new = numeric_type.groupby(gpby).agg(funcs + [customAggFunc])
    
    # Rename all columns and remove levels
    columns = []
    for old_col in new.columns.levels[0]:
        if old_col != 'idhogar':
            for new_col in new.columns.levels[1]:
                columns.append(old_col + '_' + new_col)
    new.columns = columns
    
    return df.merge(new.reset_index(), on="idhogar", how='left')


# In[ ]:


train.shape, test.shape


# In[ ]:


get_ipython().run_line_magic('time', 'train = make_new_features_grouping(train, [\'number\', bool], "idhogar")')
get_ipython().run_line_magic('time', 'test = make_new_features_grouping(test, ["number", bool], "idhogar")')


# In[ ]:


train.shape, test.shape


# In[ ]:


train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


# **Checking the correlation of all features again: **
# 
# Most probably we have created many correlated features in previous step. Infact some features even have 100% correlation. That is they are exactly same.

# In[ ]:


to_drop = chk_n_remove_corr(train)
len(to_drop), 'Target' in to_drop


# In[ ]:


test.drop(to_drop, axis=1, inplace=True)
train.shape, test.shape


# ## Adding more features using PCA:  <a id="pcaFeat"></a>
# 
# 
# Now we will add more features by **`PCA`** (Principal Component Analysis) method. It uses `SVD` (Singular Value Decomposition) method to reduce dimentionality from `N` to `n`, where `n < N`. This actually gives us direction of vectors in which data has the most variance with top component having the most variance. All `n` components are orthogonal to each other because the next highest variance direction is always orthogonal to the previous ones.
# *And because they are orthogonal they are not linearly correlated*.
# 
# ![Source](http://www.nlpca.org/fig_pca_principal_component_analysis.png)

# And PCA works [better](http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py) if we standardize our data first. So:

# In[ ]:


from sklearn.preprocessing import RobustScaler
scaler1 = RobustScaler()
scaler2 = RobustScaler()

scaled1 = scaler1.fit_transform(train.drop(['Target', 'Id', 'idhogar'], axis=1))
scaled2 = scaler2.fit_transform(test.drop(['Id', 'idhogar'], axis=1))

cols1 = train.drop(['Target', 'Id', 'idhogar'], axis=1).columns
cols2 = test.drop(['Id', 'idhogar'], axis=1).columns

trPCA = pd.DataFrame(scaled1, index=np.arange(train.shape[0]), columns=cols1)
tsPCA = pd.DataFrame(scaled2, index=np.arange(test.shape[0]), columns=cols2)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=5, svd_solver='full')


# In[ ]:


transformed1 = pca.fit_transform(trPCA)
transformed2 = pca.transform(tsPCA)
for i in range(5):
    train[f'PCA{i+1}'] = transformed1[:,i]
    test[f'PCA{i+1}'] = transformed2[:,i]
train.shape, test.shape


# In[ ]:


pca.explained_variance_ratio_


# ## Handling small number of data points for some categories: <a id="handlingSmallData"></a>

# If we have only few data points for some category/categories, our model might not learn about that category that much or anything at all.
# 
# Here to bridge the gap we I have taken two approaches:
# 1.  Cutting down the category with many data points to make it comparable to others.  [Down Sampling]
# 1.  Copying category with small datapoints again and again to make them comparable to others. [Up Sampling] (For more info on such methods see [SMOTE](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/chawla2002.html))
# 1. Using `sample_weights` or `class_weight` hyperparameter. (We will see these during [RandomForest](#randomForest) and GradientBoosting)

# In[ ]:


train['Target'].value_counts().plot.barh()


# #### 1) Downsampling:
# ---

# In[ ]:


train['Target'].value_counts(), 774+1221+1558+1500


# In[ ]:


rows1 = (train['Target'] == 1)
rows2 = (train['Target'] == 2)
rows3 = (train['Target'] == 3)
rows123 = (rows1 | rows2 | rows3)
rows4 = (train['Target'] == 4)
rows123.sum(), rows4.sum()


# In[ ]:


# We will take only first count1+count2 rows. Where count1 will go to train and count2 will go to validation set.
def train_val_split(rows, tvlen=None, vper=None):
    """
    Takes in "row" array which is location matrix for specific category(say) and 
    divides it into "train row" and "validation row" of locations. If you only want
    limited rows from "rows" then specify tvlen, which is a tuple of number of rows
    you want in train and validaion set.
    -----------------------------------------------------------------------------------
    Parameters:
        rows = An array of specific selected rows. (Where ith row is true if selected)
        tvlen = An array or a tuple of number of elements in train and val. set
        vper = perecent of elements you want in Validation set (Use it if you want all rows 
                to be divided into test and val sets from the "rows" Array or pd.Series)
    Output:
        Returns two Arrays or pd.Series of selected rows for train and validation set
        where ith element is True if that row is selected.
    """
    if tvlen is not None and vper is None:
        count1 = tvlen[0]
        count2 = tvlen[1]
    elif tvlen is None and vper is not None:
        c = rows.sum()
        count1 = int((1-vper)*c)
        count2 = int(vper*c)
    else:
        raise Exception('One of "tvlen" or "vper" should be given.')
    
    rowst, rowsv = rows.copy(), rows.copy()
    
    for i in range(len(rows)):
        
        # If we have taken count1 rows in training set, put all values equal to False. (after, count1 == 0)
        if not count1:
            rowst[i] = False
            # If we have got count2 rows in validation set, set all others equal to False.
            if not count2:
                rowsv[i] = False
            # Don't do anything to fisrt count2 rows after first count1 rows of training set,
            # where Target = selected Target and dec. count2
            count2 -= rowsv[i] # As True = 1 and False = 0
            continue
        # Equal to False because they will be in Training set
        rowsv[i] = False
        # Don't do anything to fisrt count2 rows, where Target = selected Target, and dec. count1
        count1 -= rowst[i]
    
    return rowst, rowsv


# We will take only 1500 rows from `Target` of 4 to make a balance between all categories.

# In[ ]:


rows123t, rows123v = train_val_split(rows123, vper=0.1)
rows4t, rows4v = train_val_split(rows4, tvlen=(1300, 200))
rows123t.sum(), rows123v.sum(), rows4t.sum(), rows4v.sum()


# In[ ]:


train.drop(['Id', 'idhogar'], axis=1, inplace=True)


# In[ ]:


xtrain, xvalid = train.loc[rows123t|rows4t].drop('Target', axis=1).copy(), train.loc[rows123v|rows4v].drop('Target', axis=1).copy()
ytrain, yvalid = train['Target'].loc[rows123t|rows4t].copy(), train['Target'].loc[rows123v|rows4v].copy()


# In[ ]:


xtrain.shape, ytrain.shape, xvalid.shape, yvalid.shape


# In[ ]:


xtrain.head()


# In[ ]:


ytrain.value_counts()


# In[ ]:


yvalid.value_counts()


# In[ ]:


ytrain.value_counts().plot.barh()


# #### 2) Upsampling:
# ---

# In[ ]:


train['Target'].value_counts()


# In[ ]:


target1 = train.loc[train['Target']==1].copy()
target2 = train.loc[train['Target']==2].copy()
target3 = train.loc[train['Target']==3].copy()


# In[ ]:


target1 = pd.concat([target1]*8, ignore_index=True).copy(); target1.shape


# In[ ]:


target2 = pd.concat([target2]*4, ignore_index=True).copy(); target2.shape


# In[ ]:


target3 = pd.concat([target3]*5, ignore_index=True).copy(); target3.shape


# In[ ]:


train2 = train.copy()
train2 = pd.concat([train2, target1, target2, target3], ignore_index=True); train2.shape


# We have any rows with same `Target` value in the end. If we don't want same `Target` value in our validation set. And to do that I found a way [here](https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows).

# In[ ]:


train2 = train2.sample(frac=1).reset_index(drop=True)


# In[ ]:


xtrain2, xvalid2 = train2.iloc[:15000].drop(['Target'], axis=1).copy(), train2.iloc[15000:].drop(['Target'], axis=1).copy()
ytrain2, yvalid2 = train2.iloc[:15000]['Target'].copy(), train2.iloc[15000:]['Target'].copy()


# In[ ]:


xtrain2.shape, ytrain2.shape, xvalid2.shape, yvalid2.shape


# In[ ]:


train2['Target'].value_counts().plot.barh()


#  # Random Forest Ensemble: <a id="randomForest"></a>
#  ---

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import math


# In[ ]:


def print_score(m, trn, val):
    """
    Print F1 score for training set and validation set, where m is a
    RandomForestClassifier.
    ----------------------------------------------------------------------
    Parameters:
        m: RandomForestClassifier model
        trn: tuple or array of Input and Output training data points
        val: tuple or array of Input and Output validation data points
    """
    print("Train F1score: ", str(F1score(trn[1], m.predict(trn[0]))),
    ",  Valid. F1score: ", str(F1score(val[1], m.predict(val[0]))))
    print("Train Acc.: ", str(m.score(trn[0], trn[1])),
    ", Valid. Acc.: ", str(m.score(val[0], val[1])))


# In[ ]:


m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.7, n_jobs=-1)
m.fit(xtrain, ytrain)
print_score(m, (xtrain, ytrain), (xvalid, yvalid))


# In[ ]:


# Here I increased min_sample_leaf hyperparameter
m2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=150, max_features=0.5, n_jobs=-1)
m2.fit(xtrain2, ytrain2)
print_score(m2, (xtrain2, ytrain2), (xvalid2, yvalid2))


# We are getting this much validation score here, because we have many repetitions of rows and it has learned many of them. (i.e. it is overfitted, but we can control that by `max_depth`, `max_leaf_nodes` etc. hyperparameters.)
# 
# Now, we will also use the hyperparameter `class_weights` = `'balanced'` which we discussed in [Handling small datapoints](#handlingSmallData) section.

# In[ ]:


from sklearn.model_selection import train_test_split
a, b, c, d = train_test_split(train.drop('Target', axis=1), train['Target'], test_size=0.20,
                                                    stratify=train['Target'])
xtrain3, xvalid3, ytrain3, yvalid3 = a.copy(), b.copy(), c.copy(), d.copy()


# In[ ]:


m3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, class_weight='balanced')
m3.fit(xtrain3, ytrain3)
print_score(m3, (xtrain3, ytrain3), (xvalid3, yvalid3))


# Function to plot stacked bar plot:

# In[ ]:


def plot_bar_stacked(y, preds):
    """
    For plotting predictions, right and wrong. For wrong predictions it will
    plot stacked bars in diff. colors denoting the class to which it was 
    misplaced.
    -------------------------------------------------------------------------
    Parameters:
        y : actual ouput values
        preds : predicted output values
    Output:
        Plot a stacked graph of count of right and wrong
    """
    # Output Categories 
    categories = np.array([1, 2, 3, 4])
    # This will keep count of right predictions and count of wrong prediction in each category
    counts = [[0, 0, 0, 0], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    # Calculating wrong and right predictions for all categories
    for cat in categories:
        index = (y == cat)
        right = (preds[index] == y[index]).sum()
        # For wrong preds:
        p = preds[index]
        w1 = (p[(p != y[index])] == 1).sum()
        w2 = (p[(p != y[index])] == 2).sum()
        w3 = (p[(p != y[index])] == 3).sum()
        w4 = (p[(p != y[index])] == 4).sum()

        counts[1][cat-1] = [w1, w2, w3, w4]
        counts[0][cat-1] = right
        
    # Plotting
    ind = np.arange(4)
    width = 0.15

    fig, ax = plt.subplots(figsize=(15,10), sharey=True)
    
    # Quite a simple way to plot stacked bar plot
    df = pd.DataFrame(counts[1], index=np.arange(1, 5), columns=['W Pred=1', 'W Pred=2', 'W Pred=3', 'W Pred=4'])
    df.plot.bar(ax=ax, width=width, stacked=True, colormap='RdYlBu')

    ax.bar(ind+width, counts[0], width=-width, color='green', label='Right')

    ax.set(xticks=ind + width, xticklabels=categories, xlim=[2*width - 1, 4])
    ax.legend()


# In[ ]:


preds = m.predict(xvalid)
plot_bar_stacked(yvalid, preds)


# In[ ]:


# It is not necessary that it will generalize well for test set too. (Though this is 
# giving me better results on public leaderboard.)
preds = m2.predict(xvalid2)
plot_bar_stacked(yvalid2, preds)


# In[ ]:


preds = m3.predict(xvalid3)
plot_bar_stacked(yvalid3, preds)


# ### Checking the feature inportances: <a id="checkFeatureImportance"></a>

# In[ ]:


fi = pd.DataFrame({'cols':xtrain.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False); fi[0:10]


# So, the most important feature is `escolari_mean`, which is average of years of schooling of members per household. Makes sense.

# In[ ]:


fi.plot('cols', 'imp', figsize=(10, 6), legend=False)


# Wow, only a few features can predict quite accurately. (every point shows its contribution to the model prediction.)

# In[ ]:


to_keep = fi.loc[(fi['imp']>0.005), 'cols']; to_keep.shape


# In[ ]:


xtrain_new, xvalid_new = xtrain[to_keep].copy(), xvalid[to_keep].copy()


# In[ ]:


m = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
m.fit(xtrain_new, ytrain)
print_score(m, (xtrain_new, ytrain), (xvalid_new, yvalid))


# ---
# Now according to the second RandomForestClassifier `m2`:

# In[ ]:


fi = pd.DataFrame({'cols':xtrain2.columns, 'imp':m2.feature_importances_}).sort_values('imp', ascending=False); fi[0:10]


# 
# Here we have somewhat different features' importances, but on the top is still `escolari_mean`.

# In[ ]:


fi.plot('cols', 'imp', figsize=(10, 6), legend=False)


# In[ ]:


to_keep = fi.loc[(fi['imp']>0.005), 'cols']; to_keep.shape


# In[ ]:


xtrain2_new, xvalid2_new = xtrain2[to_keep].copy(), xvalid2[to_keep].copy()
m2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=150, max_features=0.5, n_jobs=-1)
m2.fit(xtrain2_new, ytrain2)
print_score(m2, (xtrain2_new, ytrain2), (xvalid2_new, yvalid2))


# ---
# Now according to the third RandomForestClassifier `m3`:

# In[ ]:


fi = pd.DataFrame({'cols':xtrain3.columns, 'imp':m3.feature_importances_}).sort_values('imp', ascending=False); fi[0:10]


# In[ ]:


to_keep = fi.loc[(fi['imp']>0.005), 'cols']; to_keep.shape


# In[ ]:


xtrain3_new, xvalid3_new = xtrain3[to_keep].copy(), xvalid3[to_keep].copy()
m3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, class_weight="balanced")
m3.fit(xtrain3_new, ytrain3)
print_score(m3, (xtrain3_new, ytrain3), (xvalid3_new, yvalid3))


# ### Removing redundant features: <a id="removingRedundantFeatures"></a>

# For this we will use Dendrogram plot to see closely related features.

# In[ ]:


import scipy
from scipy.cluster import hierarchy as hc


# In[ ]:


corr = np.round(scipy.stats.spearmanr(xtrain_new).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,15))
dendrogram = hc.dendrogram(z, labels=xtrain_new.columns, orientation='left', leaf_font_size=16)
plt.show()


# There are 3 groups which are quite closer to each other than others. Lets remove some features from them from model one by one and lets see what happens to our `F1score`.

# In[ ]:


m = RandomForestClassifier(n_estimators=50, min_samples_leaf=25, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(xtrain_new, ytrain)
print(m.oob_score_)
check_with = m.oob_score_


# In[ ]:


#scores = []
#for col in ['v2a1_sum', 'v2a1']:
#    m = RandomForestClassifier(n_estimators=50, min_samples_leaf=25, max_features=0.6, n_jobs=-1, oob_score=True)
#    m.fit(xtrain_new.drop(col, axis=1), ytrain)
#    scores.append(m.oob_score_)
#    print(m.oob_score_)


# In[ ]:


#to_drop = []
#for i, col in enumerate(['v2a1_sum', 'v2a1']):
#    if scores[i] > check_with: to_drop.append(col)


# In[ ]:


#xtrain_new, xvalid_new = xtrain_new.drop(to_drop, axis=1), xvalid_new.drop(to_drop, axis=1) 


# ### RandomForest with top features:

# In[ ]:


m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1)
m.fit(xtrain_new, ytrain)
print_score(m, (xtrain_new, ytrain), (xvalid_new, yvalid))


# In[ ]:


preds = m.predict(xvalid_new)
plot_bar_stacked(yvalid, preds)


# ### Random Forest with top features from Upsampled Dataset:

# In[ ]:


pass


# # Gradient Boosting: <a id="gradBoosting"></a>

# In[ ]:


# from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


# In[ ]:


# It needs labels from 0 to n-1, where n is number of classes
#ytrain3 = ytrain3-1
#yvalid3 = yvalid3-1


# Best hyperparameter search method and LR reduction callback are taken from [here](https://www.kaggle.com/mlisovyi/lighgbm-hyperoptimisation-with-f1-macro).

# In[ ]:


# For decreasing learning rate of model with time
def learningRateAnnl(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return max(lr, min_learning_rate)

def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True)


# In[ ]:


fit_params={"early_stopping_rounds":300, 
            "eval_metric" : evaluate_macroF1_lgb, 
            "eval_set" : [(xvalid3, yvalid3.copy()-1)],
            'eval_names': ['valid'],
            'callbacks': [lgb.reset_parameter(learning_rate=learningRateAnnl)],
            'verbose': False,
            'categorical_feature': 'auto'}


# In[ ]:


from scipy.stats import randint
from scipy.stats import uniform
param_test ={'num_leaves': randint(12, 20), 
             'min_child_samples': randint(40, 120), 
             #'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': uniform(loc=0.75, scale=0.20), 
             'colsample_bytree': uniform(loc=0.8, scale=0.15),
             #'reg_alpha': [0, 1e-3, 1e-1, 1, 10, 50, 100],
             #'reg_lambda': [0, 1e-3, 1e-1, 1, 10, 50, 100],
             #'boosting': ['dart', 'goss', 'gbdt']
            }


# In[ ]:


maxHPs = 400
classifier = lgb.LGBMClassifier(learning_rate=0.05, n_jobs=-1, n_estimators=500, objective='multiclass')

#rs = RandomizedSearchCV(estimator= classifier, param_distributions=param_test, n_iter=maxHPs,
#                        scoring='f1_macro', cv=5, refit=True, verbose=True)


# In[ ]:


#_ = rs.fit(xtrain3, (ytrain3).copy()-1, **fit_params)


# In[ ]:


#opt_parameters = rs.best_params_; opt_parameters
# op_parameters found by above method (Random Search)
opt_parameters = {'colsample_bytree': 0.8755593602517565,
 'min_child_samples': 51,
 'num_leaves': 19,
 'subsample': 0.9437154452377117}


# In[ ]:


classifier = lgb.LGBMClassifier(**classifier.get_params())
classifier.set_params(**opt_parameters)

fit_params['verbose'] = 200
_ = classifier.fit(xtrain3, (ytrain3).copy() -1, **fit_params)


# ### K-Fold Fitting:

# In[ ]:


kfold = 5
kf = StratifiedKFold(n_splits=kfold, shuffle=True)

for trn_idx, tst_idx in kf.split(train.drop(['Target'], axis=1), train['Target']):
    xtr, xval = train.drop(['Target'], axis=1).iloc[trn_idx], train.drop(['Target'], axis=1).iloc[tst_idx]
    ytr, yval = train['Target'].iloc[trn_idx].copy() -1, train['Target'].iloc[tst_idx].copy() -1
    
    classifier.fit(xtr, ytr, eval_set=[(xval, yval)], 
            early_stopping_rounds=300, verbose=200)


# In[ ]:


preds = classifier.predict(xvalid)


# In[ ]:


((preds+1) == yvalid).sum()/len(preds)


# In[ ]:


plot_bar_stacked(yvalid, preds+1)


# ### Gradient boosting with copied rows in training set:
# 
# We will use the same hyper-parameters that we discovered above:

# In[ ]:


classifier2 = lgb.LGBMClassifier(**classifier.get_params())

kfold = 5
kf = StratifiedKFold(n_splits=kfold, shuffle=True)

for trn_idx, tst_idx in kf.split(train2.drop(['Target'], axis=1), train2['Target']):
    xtr, xval = train2.drop(['Target'], axis=1).iloc[trn_idx].copy(), train2.drop(['Target'], axis=1).iloc[tst_idx].copy()
    ytr, yval = train2['Target'].iloc[trn_idx].copy()-1, train2['Target'].iloc[tst_idx].copy() -1
    
    classifier2.fit(xtr, ytr, eval_set=[(xval, yval)], 
            early_stopping_rounds=300, verbose=200)


# In[ ]:


# Won't necessarily generalize well.
preds = classifier2.predict(xvalid2)
plot_bar_stacked(yvalid2, preds+1)


# ### Gradient Boosting with top features:

# In[ ]:


pass


# # Deep Neural Network: <a id="dnn"></a>
# 
# For introduction on how to make custom Neural Network with PyTorch look at my work: [Training your own CNN using PyTorch](https://www.kaggle.com/puneetgrover/training-your-own-cnn-using-pytorch)

# In[ ]:


from torch.utils.data import TensorDataset, DataLoader
from torch.autograd.variable import Variable


# In[ ]:


Ttrain = TensorDataset(torch.DoubleTensor(np.array(xtrain.values, dtype="float32")), #.cuda
                       torch.LongTensor(np.array(ytrain.values, dtype="float32")-1)) #.cuda


# In[ ]:


trainLoader = DataLoader(Ttrain, batch_size = 20, shuffle=True)


# In[ ]:


class Net(nn.Module):
    def __init__(self, n_cols):
        super(Net, self).__init__()
        
        self.first = nn.Sequential(
            nn.BatchNorm1d(n_cols),
            nn.Linear(n_cols, 10),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #nn.Linear(80, 80),
            #nn.BatchNorm1d(80),
            #nn.ReLU(),
            #nn.Linear(80, 80),
            #nn.BatchNorm1d(80),
            #nn.ReLU(),
            #nn.Dropout(p=0.25),
            #nn.Linear(80, 20),
            #nn.BatchNorm1d(20),
            #nn.ReLU(),
            #nn.Linear(50, 20),
            #nn.ReLU(),
            nn.Linear(10, 4),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.first(x)
net = Net(len(xtrain.columns)).double() #.cuda()


# In[ ]:


loss = nn.CrossEntropyLoss()
metrics = [F1score]
#opt = optim.SGD(net.parameters(), 1e-3, momentum=0.999, weight_decay=1e-3, nesterov=True)
opt = optim.Adam(net.parameters(), weight_decay=1e-3)
#opt = optim.RMSprop(net.parameters(), momentum=0.9, weight_decay=1e-3)


# In[ ]:


def fit(model, lr, xtr, ytr, xvl, yvl, train_dl, n_epochs, loss, opt, metrics, annln=False, mult_dec=True):
    """
    Function to fit the model to training set and print F1 scores for both training set
    and validation set.
    -------------------------------------------------------------------------------------
    Parameters:
        model: Model (Neural Network) to which Training set will fit
        lr: Learning rate (initil learning rate if annln=True)
        xtr: Input train array (for getting F1score on whole array)
        ytr: Output train array (for getting F1score on whole array)
        xvl: Input validation array (for val. F1score)
        yvl: Output validation array (for val. F1score)
        train_dl: Train DataLoader which loads training data in batches (should give Tensors as output)
        n_epochs: number of epochs
        loss: Loss function to calculate and backpropagate loss (eg: CrossEntropy)
        opt: Optimizer, to update weights (eg: RMSprop)
        metrics: Function to calculate score of model (eg: accuracy, F1 score)
        annln: (default=False) If to use LRAnnealing or not
        mult_dec: (default=True) If to dec. max Learning rate on every cosine cycle
    """
    if(annln): annl = lrAnnealing(lr, 40, 449, mult_dec)  # itr_per_epoch = len(xtrain) // batch_size
    for epoch in range(n_epochs):
        tl = iter(train_dl)
        length = len(train_dl)
        
        for t in range(length):
            xt, yt = next(tl)

            #y_pred = model(Variable(xt).cuda())
            #l = loss(y_pred, Variable(yt).cuda())
            y_pred = model(Variable(xt))
            l = loss(y_pred, Variable(yt))
            if(annln): annl(opt)
            opt.zero_grad()
            l.backward()
            opt.step()
        
        val_score = get_f1score(model, 
                                torch.DoubleTensor(np.array(xvl, dtype = "float32")), #.cuda
                                torch.LongTensor(np.array(yvl, dtype = "float32")-1)) #.cuda
        trn_score = get_f1score(model, 
                                torch.DoubleTensor(np.array(xtr, dtype = "float32")), #.cuda
                                torch.LongTensor(np.array(ytr, dtype = "float32")-1)) #.cuda
        
        if (epoch+1)%5 == 0:
            print("Epoch " + str(epoch) + "::"
                + "  trnF1score: " + str(trn_score)
                +", valF1score: " + str(val_score))
            
def get_f1score(model, x, y):
    """
    To get F1score of predictions from Neural Network.
    -----------------------------------------------------------------
    Parameters:
        model: Neural Network Model
        x: Input Values to be sent to model() function to get predictions
        y: Output Values to be checked with predictions
    Output:
        Return F1 score of predictions 
    """
    pred = model(Variable(x).contiguous())
    ypreds = np.argmax(pred.contiguous().data.numpy(), axis=1) #.cpu()
    yactuals = y.contiguous().numpy() #.cpu()
    return F1score(yactuals, ypreds)

def set_lr(opt, lr):
    """
    Function to set lr for optimizer in every layer.
    ------------------------------------------------------------------
    Parameters:
        opt: optimizer used in neural network
        lr: New Learning rate to be set in each layer
    """
    for pg in opt.param_groups: pg['lr'] = lr

class lrAnnealing():
    def __init__(self, ini_lr, epochs, itr_per_epoch, mult_dec):
        """
        Class to Anneal learning rate with warm restarts with time. It decreases 
        learning rate as multiple cosine waves with dec. amplitudes.1e-10 is taken 
        as zero. (The lower point for cosine)
        ---------------------------------------------------------------------------
        Parameters:
            ini_lr: Initial learning rate
            epochs: Number of epochs
            itr_per_epoch: iterations per epoch
            mult_dec: T/F, If to use Annealing with warm restarts or hard
        """
        self.epochs = epochs
        self.ipe = itr_per_epoch
        self.m_dec = mult_dec
        self.ppw = (self.ipe * self.epochs) // 4    # Points per wave of cosine (For 4 waves per fit method)
        self.count = 0
        self.lr = ini_lr
        self.values = np.cos(np.linspace(np.arccos(self.lr), np.arccos(1e-10), self.ppw))
        self.mult = 1
    def __call__(self, opt):
        """
            opt: optimizer of which lr is to set
        """
        self.count += 1
        set_lr(opt, self.values[self.count-1]*self.mult)
        if self.count == len(self.values):
            self.count = 0
            if(self.m_dec): self.mult /= 2


# In[ ]:


from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# In[ ]:


get_ipython().run_line_magic('time', 'fit(net, 1e-2, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)')


# In[ ]:


get_ipython().run_line_magic('time', 'fit(net, 1e-3, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)')


# In[ ]:


# Getting max F1 score of .85 in training and .85 for test
# But not giving good results on public leaderboard (0.303)
get_ipython().run_line_magic('time', 'fit(net, 1e-4, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)')


# In[ ]:


get_ipython().run_line_magic('time', 'fit(net, 1e-5, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)')


# In[ ]:


get_ipython().run_line_magic('time', 'fit(net, 1e-6, xtrain, ytrain, xvalid, yvalid, trainLoader, 40, loss, opt, metrics, annln=False)')


# In[ ]:


# Plotting the result:
ypreds = net(Variable(torch.DoubleTensor(np.array(xvalid, dtype="float32"))).contiguous()).data.numpy().argmax(1)+1 #.cuda, .cpu()
plot_bar_stacked(yvalid, ypreds)


# Though it giving me acceptable score here, but on Public Leaderboard this giving me poor results.
# So, neural network is not giving good results for the optimizers and hyperparameters I tried.
# 
# I am still trying to tweak it a bit. I have made it less deeper now.

# ### Now fitting to Upsampled dataset:

# In[ ]:


#Ttrain = TensorDataset(torch.DoubleTensor(np.array(xtrain2.values, dtype="float32")), #.cuda
#                       torch.LongTensor(np.array(ytrain2.values, dtype="float32")-1)) #.cuda
#trainLoader = DataLoader(Ttrain, batch_size = 20, shuffle=True)
#net2 = Net(len(xtrain2.columns)).double() #.cuda()
#loss = nn.CrossEntropyLoss()
#metrics = [F1score]
#opt = optim.Adam(net.parameters(), weight_decay=1e-3)
#opt = optim.SGD(net.parameters(), 1e-3, momentum=0.999, weight_decay=1e-3, nesterov=True)
#opt = optim.RMSprop(net.parameters(), momentum=0.9, weight_decay=1e-3)


# In[ ]:


#ytrain2.unique(), yvalid2.unique()


# In[ ]:


#%time fit(net2, 1e-2, xtrain2, ytrain2, xvalid2, yvalid2, trainLoader, 40, loss, opt, metrics, annln=False)


# In[ ]:


#%time fit(net2, 1e-3, xtrain2, ytrain2, xvalid2, yvalid2, trainLoader, 40, loss, opt, metrics, annln=False)


# In[ ]:


#%time fit(net, 1e-4, xtrain2, ytrain2, xvalid2, yvalid2, trainLoader, 40, loss, opt, metrics, annln=False)


# In[ ]:


#%time fit(net, 1e-5, xtrain2, ytrain2, xvalid2, yvalid2, trainLoader, 40, loss, opt, metrics, annln=False)


# In[ ]:


# Plotting the result:
#ypreds = net2(Variable(torch.DoubleTensor(np.array(xvalid2, dtype="float32"))).contiguous()).data.numpy().argmax(1)+1 #.cuda, .cuda()
#plot_bar_stacked(yvalid2, ypreds)


# Our Upsampled dataset, is giving worse results than our first neural network. (Opposite of what we saw in case of RandomForeset and GradientBoosting)

# # Comparison of Different models: <a id="compModels"></a>
# 
# ---
# 
# *We had very small dataset here. We had total 9557 rows and 4 categories to predict from. Out of the total 9557 rows about 6000 belonged to one category only. Thats a huge mismatch in quantity. And that was the main challenge. But still F1score of about 0.40 was achievable.* 
# 
# \*\* = Kaggle Takes only 5 submissions per day.
# 
# -- = Not Implemented Yet
# 
# | Models \ Data Type | Downsampled Data | Upsampled Data | FeatEng Data | Original Data | If class_weight = 'balanced' for original data |
# |-|:-:|:-:|:-:|:-:|:-:|
# | Random Forest | 0.99, 0.60, 0.346,  | 0.94, 0.92, **0.420** | 0.99, 0.59, \*\*  |  0.80, 0.74, **0.414**  |  Yes  |
# | LightGBM 5-Fold| --  |  --, 0.99, **0421**  |  --  |  --, 0.96, 0.387 (0.406)  |
# | Neural Network (4 Hidden Layers) |  0.88, 0.40, 0.342 |  0.90, 0.50, 0.295   |  --   |   --   |
# | Neural Network (2 hidden Layers) | 0.90, 0.87, 0.303 | -- | -- | -- |
# 
# .
# 
# Format : TrainF1Score, ValF1Score, PublicF1Score
# 
# ---

# # Make Submission File: <a id="makingSubmission"></a>

# * **Random Forest** with Downsampled data:

# In[ ]:


m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1)
m.fit(xtrain, ytrain)
print_score(m, (xtrain, ytrain), (xvalid, yvalid))

to_pred = test[xtrain.columns]
npArray = np.stack([lb3.inverse_transform(test['Id'].values), m.predict(to_pred)], axis=-1)

res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission1.csv", index=False)


# * **Random Forest** with Upsampled data:

# In[ ]:


m2 = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1)
m2.fit(xtrain2, ytrain2)
to_pred = test[xtrain2.columns]
npArray = np.stack([lb3.inverse_transform(test['Id'].values), m2.predict(to_pred)], axis=-1)
res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission2.csv", index=False)


# * **Random Forest** with original Dataset:

# In[ ]:


# Uncomment it to get output for 3rd RandomForest Classifier (Ready for submission)
m3 = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, class_weight='balanced')
m3.fit(xtrain3, ytrain3)
to_pred = test[xtrain3.columns]
npArray = np.stack([lb3.inverse_transform(test['Id'].values), m3.predict(to_pred)], axis=-1)

res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission3.csv", index=False)


# * **Neural Network** with Downsampled data:

# In[ ]:


# For output of net (DNN-1 with all features)
pred = net(Variable(torch.DoubleTensor(np.array(test.drop(['Id', 'idhogar'], axis=1).values, dtype="float32"))).contiguous()) # .cuda
ypreds = pred.contiguous().data.numpy().argmax(1) + 1 # .cpu()
npArray = np.stack([lb3.inverse_transform(test['Id'].values),ypreds], axis=-1); npArray[0]

res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission4.csv", index=False)


# * **XGBoost** with Downsampled data:

# In[ ]:


ypreds = classifier.predict(test[xtrain.columns]) + 1
npArray = np.stack([lb3.inverse_transform(test['Id'].values),ypreds], axis=-1)

res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission5.csv", index=False)


# * **XGBoost** with Upsampled data:

# In[ ]:


ypreds = classifier2.predict(test[xtrain2.columns]) + 1
npArray = np.stack([lb3.inverse_transform(test['Id'].values),ypreds], axis=-1)
res = pd.DataFrame(npArray, index=np.arange(len(npArray)), columns=['Id', 'Target'])
res.to_csv("submission6.csv", index=False)

