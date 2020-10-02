#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set_palette(palette='icefire', n_colors=10)


# In[ ]:


density = sns.color_palette(palette= 'RdYlGn', n_colors= 10)


# In[ ]:


in_train = pd.read_csv('../input/train.csv')
in_test = pd.read_csv('../input/test.csv')


# In[ ]:


print (in_train.shape)
print (in_test.shape)


# In[ ]:


print ('Unique households in training set:', in_train['idhogar'].nunique())
print ('Unique households in training set:', in_test['idhogar'].nunique())


# In[ ]:


### Check if there are any households both in the train, test set
set(in_train['idhogar'].unique()).intersection(in_test['idhogar'].unique())


# In[ ]:


in_train['Id'].duplicated().sum()
in_test['Id'].duplicated().sum()


# In[ ]:


### Combine the train and test dataframes
df = pd.concat([in_train, in_test], axis = 0, sort=False, ignore_index= True)
df.shape


# In[ ]:


df.head()


# ### Missing Values

# In[ ]:


df.isnull().sum()[df.isnull().sum() > 0]


# **v2a1** Monthly rent  
# **v18q1** Owns a tablet  
# **rez_esc** Years behind in school  
# **meaneduc** average years of education for adults (18+)  
# **SQBmeaned** square of the mean years of education of adults (>=18) in the household  

# #### Monthly Rent Payment
# We can see that there are 6860 instances where the monthly rent amount is `Nan`. We'll first check what the home ownership type is for these records.

# In[ ]:


df[['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']][df['v2a1'].isnull()].sum()


# We can see that `NaN` values occure only when the ownership status is _'owned', 'precarious', 'assigned/borrowed'_.  
# There are 20,844 instances the home is owned by the family, and the rent field is `NaN`, which is understandable as no rent is paid is these istances. So we'll fill the `NaN` values with zero. 
# We need to identify whether rent can expect any values in the rent feild when the ownership is _'precarious' or 'assigned/borrowed'_

# In[ ]:


df[(df['tipovivi4'] == 1)&(df['v2a1'] > 0)]


# In[ ]:


df[(df['tipovivi5'] == 1)&(df['v2a1'] > 0)]


# In[ ]:


df['v2a1'].fillna(0, inplace = True)


# ###  Tablet Ownership
# The `v18q1` field indicates number of tablets the household owns. We also have a seperate field to show whether the individual respondant owns a tablet. We'll check whether the variable `v18q`indicate that the individual has a tablet.

# In[ ]:


df['v18q'][df['v18q1'].isnull()].value_counts(dropna=False)


# So we can confirm that the `NaN` values in `V18q` corresponds to instances the individual does not have tablet

# In[ ]:


df['v18q1'].fillna(0, inplace = True)


# #### Years behind in School

# In[ ]:


sns.distplot(df['age'][df['rez_esc'].isnull()])


# In[ ]:


df['rez_esc'].fillna(0, inplace = True)
df['Schooling_age'] =((df['age'] >= 7)&(df['age'] <= 19))*1


# In[ ]:


##df[df['rez_esc']==99].index
df.loc[13069,'rez_esc'] = 0


# #### Average education for adults

# In[ ]:


df[['age', 'escolari', 'meaneduc', 'idhogar', 'parentesco1']][df['meaneduc'].isnull()].sort_values(by = 'idhogar')


# In[ ]:


### Identify the household id corresponding to the missing report
no_meanedu = df['idhogar'][df['meaneduc'].isnull()].unique()


# In[ ]:


### Check if 'meaneduc' is available for any other records for the same household
for household in no_meanedu:
    print (household, df['meaneduc'][df['idhogar'] == household].values)


# When the `meaneduc` is missing it is missing from all records for the 'household'. We can also try to calculate the `meaneduc` for each household by using the `escolari` value.

# In[ ]:


mean_educ = {}
for household in no_meanedu:
    mean_educ[household] = df['escolari'][(df['idhogar'] == household)&(df['age'] >= 18)].mean()


# In[ ]:


mean_educ


# In[ ]:


## we still get two NaN values..
df[['escolari', 'age']][df['idhogar'] == 'c49af2e64']


# A closer look shows that these are due to households where the members are less than 18 yrs. We'll set the mean value to zero for these two households, for the rest we can use the calculated values.

# In[ ]:


for household in no_meanedu:
    df.loc[df['idhogar'] == household, 'meaneduc'] = mean_educ[household]


# In[ ]:


df['meaneduc'].fillna(0, inplace = True)


# #### Squared Mean Education for household adults

# In[ ]:


## We'll recalculate the squared mean education based on the mean education column
df['SQBmeaned'] = df['meaneduc'] **2


# ### Exploratory 

# In[ ]:


ind_indicators = ['Id', 'escolari', 'rez_esc', 'Schooling_age', 'dis', 'male', 'female',
                         'estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 
                         'estadocivil5', 'estadocivil6', 'estadocivil7', 'parentesco1',
                         'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6',
                         'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11',
                         'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
                         'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', 'age', 'SQBescolari', 'SQBage', 'agesq'] 


# In[ ]:


hh_indicators = [col for col in df.columns if col not in ind_indicators]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### Inconsistant values within household

# In[ ]:


inconsistant_col = {}

### for col in hh_indicators:
for col in ['Target']:
    ## Check if the value is unique across all records for a given household
    consistant_value = df[df[col].notnull()].groupby('idhogar')[col].apply(lambda x: x.nunique() == 1)
    
    # Record variables where inconsistancies exist
    if (consistant_value == False).sum() > 0:
        print ('{} Inconsistant value for {} within household'.format((consistant_value == False).sum(), col))
        inconsistant_col[col] = consistant_value[consistant_value == False]


# So we have 85 instances whether the `Target` is not consistant within the household. For these we will replace the `Target` with the value for `parentesco1`

# In[ ]:


for household in inconsistant_col['Target'].index:
    target_val_for_parentesco1 = df['Target'][(df['idhogar'] == household)&(df['parentesco1'] == 1)]
    if target_val_for_parentesco1.notnull().any():
        df.loc[df['idhogar'] == household, 'Target'] = target_val_for_parentesco1.values


# #### Households without head of household

# In[ ]:


no_head = set(df['idhogar'].unique()).difference(df['idhogar'][df['parentesco1'] == 1].unique())


# In[ ]:


len(no_head)


# #### Distribution of The Target variable

# In[ ]:


sns.countplot('Target', data = df[df['parentesco1'] == 1])
plt.xticks((0,1,2,3), ('Extreme Poverty', 'Moderate Poverty', 'Vulnerable', 'Non-Vulnerable'), rotation = 90)


# #### Monthly Rent Payment
# We identified that the rent payment is only applicable for households where he 

# In[ ]:


sns.distplot(df['v2a1'][(df['Target'] == 1)&(df['v2a1'] > 0 )], hist=False, label= 'Extreme Poverty')
sns.distplot(df['v2a1'][(df['Target'] == 2)&(df['v2a1'] > 0 )], hist=False, label= 'Moderate Poverty')
sns.distplot(df['v2a1'][(df['Target'] == 3)&(df['v2a1'] > 0 )], hist=False, label= 'Vulnerable Households')
sns.distplot(df['v2a1'][(df['Target'] == 4)&(df['v2a1'] > 0 )], hist=False, label= 'Non-vulnerable households')

### plt.gcf()
plt.legend()


# #### Overcrowding by bedroom

# In[ ]:


df['hacdor'][df['Target'].notnull()].value_counts()


# In[ ]:


sns.countplot('Target', data = df[df['hacdor'] == 1])
plt.xticks((0,1,2,3), ('Extreme Poverty', 'Moderate Poverty', 'Vulnerable', 'Non-Vulnerable'), rotation = 90)


# In[ ]:


df['Target'][df['hacdor'] == 1].value_counts(normalize = True)


# #### Number of rooms in the house

# In[ ]:


sns.countplot('rooms', data = df[df['Target'].notnull()], hue= 'Target')


# #### Overcrowding by rooms

# In[ ]:


df['hacapo'][df['Target'].notnull()].value_counts()


# In[ ]:


###sns.countplot('Target', data = df[df['hacapo'] == 1])
norm_val = df['Target'][df['hacapo'] == 1].value_counts(normalize = True)
sns.barplot(norm_val.index, norm_val)
plt.xticks((0,1,2,3), ('Extreme Poverty', 'Moderate Poverty', 'Vulnerable', 'Non-Vulnerable'), rotation = 90)


# ### Availability of Bathroom in the Household

# In[ ]:


norm_val = df['Target'][df['v14a'] == 1].value_counts(normalize = True)

sns.barplot( norm_val.index, norm_val)
plt.xticks((0, 1, 2, 3), ('Extreme Poverty', 'Moderate Poverty', 'Vulnerable', 'Non-Vulnerable'), rotation = 90)


# #### Has Refrigerator

# In[ ]:


norm_val = df['Target'][df['refrig'] == 1].value_counts(normalize = True)

sns.barplot( norm_val.index, norm_val)
plt.xticks((0, 1, 2, 3), ('Extreme Poverty', 'Moderate Poverty', 'Vulnerable', 'Non-Vulnerable'), rotation = 90)


# #### Owns a Tablet

# In[ ]:


norm_val = df['Target'][df['v18q'] == 1].value_counts(normalize = True)

sns.barplot( norm_val.index, norm_val)
plt.xticks((0, 1, 2, 3), ('Extreme Poverty', 'Moderate Poverty', 'Vulnerable', 'Non-Vulnerable'), rotation = 90)


# In[ ]:


## How many individuals in each poverty level own a tablet.
df['Target'][(df['Target'].notnull())&(df['v18q'] == 1)].value_counts()/df['Target'].value_counts()


# In[ ]:


## Number of tablets in household
sns.countplot('v18q1', data = df[df['parentesco1'] == 1], hue= 'Target' )
plt.legend(loc = 1)


# In[ ]:


df[df['parentesco1'] == 1].pivot_table(values = 'idhogar', columns = 'v18q1', index = 'Target', aggfunc = 'count')


# #### Ratio of Children vs Adults in Household

# In[ ]:


### Average number of children, adults in household
pivot = df[df['parentesco1'] == 1].pivot_table(values= ['r4t2', 'r4t1'], index = 'Target', aggfunc='mean')
sns.heatmap(pivot, annot= True, cmap = density, vmin = 0, vmax = 3)


# #### Size of Household

# In[ ]:


norm_val = df['tamhog'][df['parentesco1'] == 1].value_counts(normalize = True)

sns.countplot('tamhog', data = df[df['parentesco1'] == 1], hue= 'Target',)
plt.legend(loc = 1)


# In[ ]:


df[df['parentesco1'] == 1].pivot_table(index = 'Target', values = 'tamhog' )


# #### Number of persons living in household

# In[ ]:


norm_val = df['tamviv'][df['parentesco1'] == 1].value_counts(normalize = True)

sns.countplot('tamviv', data = df[df['parentesco1'] == 1], hue= 'Target',)
plt.legend(loc = 1)


# In[ ]:





# In[ ]:





# In[ ]:





# #### Years of Schooling

# In[ ]:


sns.distplot(df['escolari'][(df['Target'] == 1)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Extreme Poverty')
sns.distplot(df['escolari'][(df['Target'] == 2)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Moderate Poverty')
sns.distplot(df['escolari'][(df['Target'] == 3)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Vulnerable Households')
sns.distplot(df['escolari'][(df['Target'] == 4)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Non-vulnerable households')

### plt.gcf()
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# #### Building Material for Walls

# In[ ]:


wall_material = {}
material_types = ['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 'paredzinc', 'paredfibras', 'paredother' ]
for material in material_types:
    pov_level = df['Target'][(df['Target'].notnull())&(df[material]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[material]> 0)).sum()
    
    wall_material[material] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(wall_material)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')

for ax in plt.gcf().get_axes():
    ax.set_xticklabels(['Brick', 'Socket', 'Cement', 'WasteMat', 'Wood', 'Zink', 'NaturalFiber', 'Other'])


# In[ ]:


floor_material = {}
for material in ['pisomoscer', 'pisocemento', 'pisoother', 'pisonatur', 'pisonotiene', 'pisomadera']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[material]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[material]> 0)).sum()
    
    floor_material[material] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(floor_material)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


roof_material = {}
for material in ['techozinc', 'techoentrepiso', 'techocane', 'techootro']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[material]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[material]> 0)).sum()
    
    roof_material[material] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(roof_material)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


water_provision = {}
for level in ['abastaguadentro', 'abastaguafuera', 'abastaguano']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    water_provision[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(water_provision)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


electricity = {}
for level in ['public', 'planpri', 'noelec', 'coopele']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    electricity[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(electricity)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


toilet = {}
for level in ['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5', 'sanitario6']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    toilet[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(toilet)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


cooking = {}
for level in ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    cooking[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(cooking)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


cooking = {}
for level in ['energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = df['Target'].value_counts()
    
    cooking[level] = (pov_level/in_level)
    
temp = pd.DataFrame(cooking)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


(pov_level/in_level).sum()


# In[ ]:


wasteDisposal = {}
for level in ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    wasteDisposal[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(wasteDisposal)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


wasteDisposal = {}
for level in ['elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 'elimbasu5', 'elimbasu6']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = df['Target'].value_counts()
    
    wasteDisposal[level] = (pov_level/in_level)
    
temp = pd.DataFrame(wasteDisposal)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


WallQual = {}
for level in ['epared1', 'epared2', 'epared3']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    WallQual[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(WallQual)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


WallQual = {}
for level in ['epared1', 'epared2', 'epared3']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = df['Target'].value_counts()
    
    WallQual[level] = (pov_level/in_level)
    
temp = pd.DataFrame(WallQual)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


RoofQual = {}
for level in ['etecho1', 'etecho2', 'etecho3']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    RoofQual[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(RoofQual)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


RoofQual = {}
for level in ['etecho1', 'etecho2', 'etecho3']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = df['Target'].value_counts()
    
    RoofQual[level] = (pov_level/in_level)
    
temp = pd.DataFrame(RoofQual)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


FloorQual = {}
for level in ['eviv1', 'eviv2', 'eviv3']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    FloorQual[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(FloorQual)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


indGroup = {}
for level in ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    indGroup[level] = (pov_level/in_level).to_dict()
    
    
temp = pd.DataFrame(indGroup)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


indGroup = {}
for level in ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = df['Target'].value_counts()
    
    indGroup[level] = (pov_level/in_level)
    
temp = pd.DataFrame(indGroup)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# #### Relationships Within Household

# In[ ]:


relationship = {}
for level in ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    relationship[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(relationship)

plt.figure(figsize= (10,4))
sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


relationship = {}
for level in ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', 'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', 'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = df['Target'].value_counts()
    
    relationship[level] = (pov_level/in_level)
    
temp = pd.DataFrame(relationship)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# #### Household Composition by Age Group

# In[ ]:





# In[ ]:


df[['hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total']].sum()


# `hogar_adul` variable also includes individuals over 65 years. So we'll create a seperate column to capture period within 19 and 65

# In[ ]:


df['hogar_mid'] = df['hogar_adul'] -df['hogar_mayor']


# #### Education Level

# In[ ]:


education = {} 
for level in ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    education[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(education)

plt.figure(figsize=(10,4))
sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


education = {} 
for level in ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = df['Target'].value_counts()
    
    education[level] = (pov_level/in_level)
    
temp = pd.DataFrame(education)

plt.figure(figsize=(10,4))
sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# #### House Ownership

# In[ ]:


homeOwnership = {}
for level in ['tipovivi1', 'tipovivi2', 'tipovivi3', 'instlevel4', 'tipovivi4', 'tipovivi5']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    homeOwnership[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(homeOwnership)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


df['Target'][df['computer'] == 1].value_counts()/df['Target'].value_counts()


# In[ ]:


df['Target'][df['television'] == 1].value_counts()/df['Target'].value_counts()


# In[ ]:


df['Target'][df['mobilephone'] == 1].value_counts()/df['Target'].value_counts()


# In[ ]:


sns.countplot(x = 'qmobilephone', data = df, hue = 'Target')


# In[ ]:


Region = {}
for level in ['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    Region[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(Region)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


Region = {}
for level in ['area1', 'area2']:
    pov_level = df['Target'][(df['Target'].notnull())&(df[level]>0)].value_counts()
    in_level = ((df['Target'].notnull())&(df[level]> 0)).sum()
    
    Region[level] = (pov_level/in_level).to_dict()
    
temp = pd.DataFrame(Region)

sns.heatmap(temp, cmap = density, vmin = 0 ,vmax = 1, annot=True, linewidths=.1, linecolor='white')


# In[ ]:


sns.distplot(df['age'][(df['Target'] == 1)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Extreme Poverty')
sns.distplot(df['age'][(df['Target'] == 2)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Moderate Poverty')
sns.distplot(df['age'][(df['Target'] == 3)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Vulnerable Households')
sns.distplot(df['age'][(df['Target'] == 4)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Non-vulnerable households')

### plt.gcf()
plt.legend()


# #### Age by Income Level

# In[ ]:


sns.distplot(df['age'][df['Target'] == 1], hist=False, rug=False, label= 'Extreme Poverty')
sns.distplot(df['age'][df['Target'] == 2], hist=False, rug=False, label= 'Moderate Poverty')
sns.distplot(df['age'][df['Target'] == 3], hist=False, rug=False, label= 'Vulnerable Households')
sns.distplot(df['age'][df['Target'] == 4], hist=False, rug=False, label= 'Non-vulnerable households')

### plt.gcf()
plt.legend()


# #### Mean Education of Head of Household

# In[ ]:


sns.distplot(df['meaneduc'][(df['Target'] == 1)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Extreme Poverty')
sns.distplot(df['meaneduc'][(df['Target'] == 2)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Moderate Poverty')
sns.distplot(df['meaneduc'][(df['Target'] == 3)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Vulnerable Households')
sns.distplot(df['meaneduc'][(df['Target'] == 4)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Non-vulnerable households')

### plt.gcf()
plt.legend()


# In[ ]:


df['dependencyR'] = (df['hogar_nin'] + df['hogar_mayor']) / df['hogar_mid']


# In[ ]:


df['dependencyR'].value_counts()


# In[ ]:


df['dependencyR'].fillna(0, inplace = True)
df['dependencyR'].replace(np.inf, 9, inplace = True)


# In[ ]:


#### df[['edjefa','edjefe', 'escolari']]
df['edjefa'].replace(to_replace= 'no' , value= np.nan, inplace= True)
df['edjefe'].replace(to_replace= 'no' , value= np.nan, inplace= True)


# In[ ]:


sns.distplot(df['meaneduc'][(df['male'] == 1)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Male')
sns.distplot(df['meaneduc'][(df['female'] == 1)&(df['parentesco1'] == 1 )], hist=False, rug=False, label= 'Female')


# #### Feature Engineering
# **Per Capita Features**

# In[ ]:


df['phonespp'] = df['qmobilephone'] / df['tamviv']
df['tabletspp'] = df['v18q1'] / df['tamviv']
df['roomspp'] = df['rooms'] / df['tamviv']
df['rentpp'] = df['v2a1'] / df['tamviv']


# **Maximum Education for Family**

# In[ ]:


df['median_schooling'] = df['escolari'].groupby(df['idhogar']).transform('median')
df['max_schooling'] = df['escolari'].groupby(df['idhogar']).transform('max')


# In[ ]:


## Education of the Household Head
df['eduForHeadofHH'] = 0
df.loc[(df['parentesco1']== 1), 'eduForHeadofHH'] = df['escolari']


# In[ ]:


df['eduForHeadofHH'] = df['eduForHeadofHH'].groupby(df['idhogar']).transform('max')


# In[ ]:


df['SecondaryEduLess'] = ((df[['instlevel1','instlevel2', 'instlevel3', 'instlevel4']] == 1).any(axis = 1)&(df['age'] > 19))*1
df['SecondaryEduMore'] = ((df[['instlevel5','instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']] == 1).any(axis = 1)&(df['age'] > 19))*1


# In[ ]:


df['SecondaryEduMore'].value_counts()


# In[ ]:


df['MembersWithSecEdu']  = df['SecondaryEduMore'].groupby(df['idhogar']).transform('sum')
df['MembersWithPrimEdu']  = df['SecondaryEduLess'].groupby(df['idhogar']).transform('sum')


# In[ ]:


df['Educated_Ratio'] = (df['MembersWithSecEdu']/df['MembersWithPrimEdu'])
df['Educated_Ratio'].replace(np.inf, -1, inplace = True)
df['Educated_Ratio'].fillna(value = 0, inplace = True)


# In[ ]:


sns.distplot(df['Educated_Ratio'][df['Target'] ==1])
sns.distplot(df['Educated_Ratio'][df['Target'] ==2])
sns.distplot(df['Educated_Ratio'][df['Target'] ==3])
sns.distplot(df['Educated_Ratio'][df['Target'] ==4])


# **Access to technology**

# In[ ]:


df['access_to_tech'] = ((df['v18q1'] >= 1)&(df['qmobilephone'] >= 1)&(df['computer'] >= 0)&(df['television'] >= 0))*1


# **Marital Status of Head of Household**

# In[ ]:


df['marital_status'] = (((df['estadocivil3'] ==1)|(df['estadocivil4'] == 1))&(df['parentesco1'] == 1))*1

df['marital_status'] = df['marital_status'].groupby(df['idhogar']).transform('max')


# In[ ]:


df['FemaleHousehold'] = ((df['male'] == 0)&(df['parentesco1'] == 1))*1
df['FemaleHousehold'] = df['FemaleHousehold'].groupby(df['idhogar']).transform('max')


# In[ ]:


df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
df['rent_to_rooms'] = df['v2a1']/df['rooms']
df['rooms_to_tamviv'] = df['rooms']/df['tamviv'] 
df['v2a1_to_r4t3'] = df['v2a1']/df['r4t3']


# In[ ]:


df['female_to_males'] = df['r4m2']/df['r4h2']
df['less12_to_adult'] = df['r4t1']/df['hogar_mid']

df['less12_to_older'] = df['r4t1']/df['r4t2']


# In[ ]:


(df== np.inf).sum()[(df== np.inf).sum() >0]


# In[ ]:


df['less12_to_adult'].fillna(0, inplace = True)
df['less12_to_adult'].replace(np.inf, -1, inplace = True)

df['female_to_males'].fillna(0, inplace = True)
df['female_to_males'].replace(np.inf, -1, inplace = True)


# In[ ]:


df['NoMalesInHH'] = (df['r4h2'] == 0)*1
df['NoFemalesInHH'] = (df['r4m2'] == 0)*1


# #### Modeling

# In[ ]:


from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.metrics import f1_score, confusion_matrix


# In[ ]:


(df == np.inf).sum()[(df == np.inf).sum() > 0]


# In[ ]:


def get_train_test(df):
        
    train = df[df['Target'].notnull()].copy()
    test = df[df['Target'].isnull()].copy()
    
    test.drop(columns = 'Target', inplace = True)
    
    return train, test


# In[ ]:


train, test = get_train_test(df)


# In[ ]:


columnstoDrop = ['idhogar','Id', 'edjefe', 'edjefa','dependency']
                 # 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin',
                 # 'SQBovercrowding', 'SQBdependency', 'SQBmeaned']

train.drop(columns= columnstoDrop, inplace = True)
test.drop(columns= columnstoDrop, inplace = True)

X = train.drop(columns= 'Target')
y = train['Target']

X_col = X.columns

std_scl = StandardScaler()
X= std_scl.fit_transform(X)

test= std_scl.transform(test)


# In[ ]:


kfold = KFold(n_splits= 5, shuffle= True)


# In[ ]:


X_train, X_test, y_train, y_test= train_test_split(X,y)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf_clf = RandomForestClassifier(n_estimators= 10, class_weight= 'balanced')
rf_clf.fit(X,y)


# In[ ]:


featureImp = pd.Series(data= rf_clf.feature_importances_, index = X_col)
featureImp


# In[ ]:


select =rf_clf.feature_importances_ > 0.001
sum(select)


# ### Logistic Regression

# In[ ]:


clf = LogisticRegression(class_weight = 'balanced', solver= 'liblinear', multi_class= 'ovr')


# In[ ]:


cross_val_score(clf,  , y, scoring = 'f1_macro', cv= kfold)


# In[ ]:


### cross_val_score(clf, X, y, scoring = 'f1_macro', cv= kfold)

### array([ 0.4278837 ,  0.45065472,  0.48420681,  0.42884422,  0.44957925])


# In[ ]:


for reg in [0.001, 0.01, 0.1, 1, 10, 20, 30]:
    clf = LogisticRegression(class_weight = 'balanced', C= reg, solver= 'liblinear', multi_class= 'ovr')
    f1score = cross_val_score(clf, X[:,select], y, scoring = 'f1_macro', cv=kfold).mean()
    print (f1score)

### 0.47250164494877483
### 0.4900965155141634
### 0.49374600294003257
### 0.49692298999701945
### 0.49017075339444416
### 0.4972529682883858
### 0.49290073775881293


# In[ ]:


clf = LogisticRegression(class_weight = 'balanced', solver= 'liblinear', multi_class= 'ovr')

pred = cross_val_predict(clf, X[:,select], y, cv=kfold).astype(int)


# In[ ]:


## confusion_matrix(y, pred)

## array([[ 392,  188,   85,  109],
##        [ 275,  656,  242,  385],
##        [ 160,  253,  344,  464],
##        [ 274,  443,  395, 4892]])


# In[ ]:


clf = LogisticRegression(class_weight = 'balanced', solver= 'liblinear', multi_class= 'ovr', C= 20)
clf.fit( X[:,select], y)
prediction = clf.predict(test[:,select]).astype(int)


# In[ ]:


submission = pd.DataFrame({'Id': in_test['Id'], 'idhogar': in_test['idhogar'], 'isHead': in_test['parentesco1'] , 'Target':prediction})
submission['Target'].value_counts(normalize = True)


# In[ ]:


submission.to_csv('./Submission_log_clf.csv', columns=['Id', 'Target'], index= False)


# ### Gradient Boost

# In[ ]:


gb_clf = GradientBoostingClassifier()


# In[ ]:


##cross_val_score(gb_clf, X[:, select], y, scoring = 'f1_macro', cv= kfold).mean()
## 0.6303527489770167


# In[ ]:


## prediction = cross_val_predict(gb_clf, X[:, select], y, cv=kfold)


# In[ ]:


##  f1_score(y, prediction, average='macro')
##  0.6287221065578112


# In[ ]:


## confusion_matrix(y, prediction)

## array([[ 400,  108,   22,  244],
##       [  33,  855,   23,  647],
##       [  27,  176,  350,  668],
##       [  36,  176,   31, 5761]])


# In[ ]:


gb_clf.fit(X[:,select],y)
prediction = gb_clf.predict(test[:, select]).astype(int)


# In[ ]:


submission = pd.DataFrame({'Id': in_test['Id'], 'idhogar': in_test['idhogar'], 'isHead': in_test['parentesco1'] , 'Target':prediction})
submission['Target'].value_counts(normalize = True)


# In[ ]:


submission.to_csv('./Submission_GradientBoosting_clf.csv', columns=['Id', 'Target'], index= False)


# ### Tree

# In[ ]:


tree_clf = DecisionTreeClassifier()


# In[ ]:


## cross_val_score(tree_clf, X[:,select], y, scoring = 'f1_macro', cv= kfold).mean()
## 0.9298023676497221


# In[ ]:


## tree_clf.fit(X[:,select],y)
## prediction = tree_clf.predict(test[:, select]).astype(int)


# In[ ]:


## submission = pd.DataFrame({'Id': in_test['Id'], 'idhogar': in_test['idhogar'], 'isHead': in_test['parentesco1'] , 'Target':prediction})
## submission['Target'].value_counts(normalize = True)


# In[ ]:


## submission.to_csv('./Submission_tree_clf.csv', columns=['Id', 'Target'], index= False)


# ### Voting Classifier

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


voting_clf = VotingClassifier(estimators= [('tree', tree_clf), ('gradiantBoost', gb_clf), ('logR', clf)], voting= 'hard')


# In[ ]:


cross_val_score(voting_clf, X[:, select], y, scoring = 'f1_macro', cv= kfold)


# In[ ]:


prediction = cross_val_predict(voting_clf,  X[:, select], y, cv=kfold)
confusion_matrix(y, prediction)


# In[ ]:


voting_clf.fit(X[:, select],y)
prediction = voting_clf.predict(test[:, select]).astype(int)


# In[ ]:


submission = pd.DataFrame({'Id': in_test['Id'], 'idhogar': in_test['idhogar'], 'isHead': in_test['parentesco1'] , 'Target':prediction})
submission['Target'].value_counts(normalize = True)


# In[ ]:


submission.to_csv('./Submission_voting_clfHardVote.csv', columns=['Id', 'Target'], index= False)


# In[ ]:


voting_clf = VotingClassifier(estimators= [('tree', tree_clf), ('gradiantBoost', gb_clf), ('logR', clf)], voting= 'soft')


# In[ ]:


## cross_val_score(voting_clf, X, y, scoring = 'f1_macro', cv= kfold)


# In[ ]:


## prediction = cross_val_predict(voting_clf, X, y, cv=kfold)
## confusion_matrix(y, prediction)


# In[ ]:


voting_clf.fit(X[:, select],y)
prediction = voting_clf.predict(test[:, select]).astype(int)


# In[ ]:


submission = pd.DataFrame({'Id': in_test['Id'], 'idhogar': in_test['idhogar'], 'isHead': in_test['parentesco1'] , 'Target':prediction})
submission['Target'].value_counts(normalize = True)


# In[ ]:


submission.to_csv('./Submission_voting_clfSoftVote.csv', columns=['Id', 'Target'], index= False)


# In[ ]:





# In[ ]:





# In[ ]:




