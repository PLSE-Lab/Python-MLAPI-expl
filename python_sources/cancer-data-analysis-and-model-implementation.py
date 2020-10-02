#!/usr/bin/env python
# coding: utf-8

# # **Index**
# 
# ## 1.     Handle NaN's
# ## 2.     Feature Correlation
# ## 3.     Feature Extraction
# ## 4.     Feature Correlation and Feature Importance After Extraction
# ## 5.     Model (Coming Soon)
# ## 6.     Model Selection (Coming Soon)
#  ## 7.     Conclusion (Coming Soon)

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../input/kag_risk_factors_cervical_cancer.csv")

df_nan = df.replace("?", np.nan)


df1 = df_nan.convert_objects(convert_numeric=True)


df1.isnull().sum()


# 
# # 1) Handle NaN's

# To handle Nan we have to see correlation between variables. But according to data, size higher than 100 NaN values can be effect data. We can fill features which have lower than 100 Nan values with median values.

# In[58]:


df1['First sexual intercourse'].fillna(df1['First sexual intercourse'].median(), inplace = True)
df1['Num of pregnancies'].fillna(df1['Num of pregnancies'].median(), inplace = True)
df1['First sexual intercourse'].fillna(df1['First sexual intercourse'].median(), inplace = True)
df1['Smokes'].fillna(0,inplace = True)
df1['Number of sexual partners'].fillna(df1['Number of sexual partners'].median(), inplace = True)
l = (df1['Smokes']==1)
df1.loc[l,'Smokes (years)'] = df1.loc[l,'Smokes (years)'].fillna(df1.loc[l,'Smokes (years)'].median())
l = (df1['Smokes']==0)
df1.loc[l,'Smokes (years)'] = df1.loc[l,'Smokes (years)'].fillna(0)
l = (df1['Smokes']==1)
df1.loc[l,'Smokes (packs/year)'] = df1.loc[l,'Smokes (packs/year)'].fillna(df1.loc[l,'Smokes (packs/year)'].median())
l = (df1['Smokes']==0)
df1.loc[l,'Smokes (packs/year)'] = df1.loc[l,'Smokes (packs/year)'].fillna(0)
df2 = df1.drop(['Hinselmann','Schiller','Citology','Biopsy'], axis = 1)


# ##  1.1)Hormonal Contraceptives (HC)

# In the data diagnosis data have too much Nan values. Because of that we cannot determine effect of this data. We have to drop them. Then using pearson correlation we can determine which feature is effect 'Hormonal Contraceptives'.

# In[59]:


corrmat = df2.corr()
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Hormonal Contraceptives')['Hormonal Contraceptives'].index

cols =cols.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])

cm = df2[cols].corr()

plt.figure(figsize=(9,9))

sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, cmap='Set1' ,annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# According to heatmap, we have to fill Nan values with corralated features.

# In[60]:


# If patient is older than sample mean or number of pregnancies is lower than mean then patient may take Hormonal Contraceptives
l = (df2['Age']>df2['Age'].mean())
df2.loc[l,'Hormonal Contraceptives'] = df2.loc[l,'Hormonal Contraceptives'].fillna(1)
l = (df2['Num of pregnancies']<df2['Num of pregnancies'].mean())
df2.loc[l,'Hormonal Contraceptives'] = df2.loc[l,'Hormonal Contraceptives'].fillna(1)
df2['Hormonal Contraceptives'].fillna(0,inplace = True)

df2['Hormonal Contraceptives'].isnull().sum()


# For HC(years) NaN values we can fill with median values by using HC feature.

# In[61]:


l = (df2['Hormonal Contraceptives'] == 1)
df2.loc[l,'Hormonal Contraceptives (years)'] = df2.loc[l,'Hormonal Contraceptives (years)'].fillna(df2['Hormonal Contraceptives (years)'].median())
l = (df2['Hormonal Contraceptives'] == 0)
df2.loc[l,'Hormonal Contraceptives (years)'] = df2.loc[l,'Hormonal Contraceptives (years)'].fillna(0)


# Also we need to check relationship between HC and HC (years)

# In[62]:


len(df2[(df2['Hormonal Contraceptives'] == 1) & (df2['Hormonal Contraceptives (years)'] == 0) ])


# In[63]:


len(df2[(df2['Hormonal Contraceptives'] == 0) & (df2['Hormonal Contraceptives (years)'] != 0) ])


# ## 1.2) IUD

# Using pearson correlation we can determine which feature is effect 'IUD'.

# In[64]:


corrmat = df2.corr()
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'IUD')['IUD'].index

cols =cols.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])

cm = df2[cols].corr()

plt.figure(figsize=(9,9))

sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cmap = 'rainbow', cbar=True, annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# This figure show that Age and Number of pregnancies features have effect on IUD feature.

# In[65]:


len(df2[(df2['Age']>df2['Age'].mean())&(df2['IUD']==1)])


# In[66]:


len(df2[df2['IUD']==1])


# This show that %80 of patients who take IUD are older than age mean. We can fillna values according to this discovery.

# In[67]:


l = (df2['Age']>df2['Age'].mean())
df2.loc[l,'IUD'] = df2.loc[l,'IUD'].fillna(1)

len(df2[(df2['Num of pregnancies']<df2['Num of pregnancies'].mean())&(df2['IUD']==0)])


# In[68]:


len(df2[df2['IUD']==0])


# %70 of patients who do not take IUD have lower number of pregnancies than mean of number of pregnancies. We can fill remaining NaN with 0 values.

# In[69]:


df2['IUD'].fillna(0, inplace = True)


# For IUD (years) feature we can fill NaN values with IUD feature.

# In[70]:


l = (df2['IUD'] == 1)
df2.loc[l,'IUD (years)'] = df2.loc[l,'IUD (years)'].fillna(df2['IUD (years)'].median())
l = (df2['IUD'] == 0)
df2.loc[l,'IUD (years)'] = df2.loc[l,'IUD (years)'].fillna(0)


# Also we need to check relationship between IUD and IUD (years)

# In[71]:


len(df2[(df2['IUD'] == 1) & (df2['IUD (years)'] == 0) ])


# If patient take IUD then UID (years) have to be non zero values then we need to change it to mean values.

# In[72]:


l = (df2['IUD'] == 1) & (df2['IUD (years)'] == 0)
df2.loc[l,'IUD (years)'] = df2['IUD (years)'].mean()

len(df2[(df2['IUD'] == 1) & (df2['IUD (years)'] == 0) ])


# In[73]:


len(df2[(df2['IUD'] == 0) & (df2['IUD (years)'] != 0) ])


# ## 1.3)STDs

# In[74]:


corrmat = df2.corr()
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'STDs')['STDs'].index

#cols =cols.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])

cm = df2[cols].corr()

plt.figure(figsize=(9,9))

sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cmap = 'hot', cbar=True, annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# According to heatmap we can easily say that 'STDs:condylomatosis' and STDs:vulvo-perineal condylomatosis' features effect 'STDs'. We cannot take 'STD (number)' and 'STDs: Number of diagnosis' because they are same features as 'STDs'. According to our knowledge about STDs we can easily fill Nan values with 1 or zero because if patient have any of STDs diseases then patient STDs feature must be 1 others must be zero.

# In[75]:


df3 = df2.copy()

l = (df3['STDs:condylomatosis']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:cervical condylomatosis']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:vaginal condylomatosis']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:vulvo-perineal condylomatosis']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:syphilis']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:pelvic inflammatory disease']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:genital herpes']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:molluscum contagiosum']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:AIDS']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:HIV']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:Hepatitis B']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
l = (df3['STDs:HPV']==1)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)


# It seems that STDs and STD diseases features are paralel. According to that we have to look other features. Smokes and IUD features can be used.

# In[76]:


len(df[(df3['STDs'] == 1)])


# In[77]:


len(df3[(df3['Smokes'] == 0) & (df3['STDs'] == 1)])


# According to data %73 of patients who are not smoking have STD. But not have Nan values

# In[78]:


l = (df3['Smokes']==0)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)

len(df3[(df3['IUD'] == 0) & (df3['STDs'] == 1)])


# In[79]:


len(df3[df3['STDs']==1])


# %75 of patients who does not take IUD are also have STDs

# In[80]:


l = (df3['IUD']==0)
df3.loc[l,'STDs'] = df3.loc[l,'STDs'].fillna(1)
df3['STDs'].fillna(0, inplace = True)

df3['STDs'].isnull().sum()


# Also STDs (number) is same feature as STDs.

# In[81]:


df3['STDs (number)'].median()


# This case median is not useful so we can use mean values.

# In[82]:


l = (df3['STDs']==1)
df3.loc[l,'STDs (number)'] = df3.loc[l,'STDs (number)'].fillna(df3['STDs (number)'].mean())
df3['STDs (number)'].fillna(0, inplace = True)

df3['STDs (number)'].isnull().sum()


# ## 1.4)STDs Diseases

# In[83]:


corrmat = df3.corr()

plt.figure(figsize=(20,20))

sns.set(font_scale=2)
hm = sns.heatmap(corrmat,cmap = 'tab20c', cbar=True, annot=True,vmin=0,vmax =1,center=True, square=True, fmt='.2f', annot_kws={'size': 10},
             yticklabels = df3.columns, xticklabels = df3.columns)
plt.show()


# In[84]:


df4= df3.copy()


# According to heatmap and also our knowledge about field all STDs diseases depend on STDs feature and also STD (number).

# In[85]:


l = (df4['STDs']==0)
df4.loc[l,'STDs:condylomatosis'] = df4.loc[l,'STDs:condylomatosis'].fillna(0)
l = (df4['STDs']==0)
df4.loc[l,'STDs:cervical condylomatosis'] = df4.loc[l,'STDs:cervical condylomatosis'].fillna(0)
l = (df4['STDs']==0)
df4.loc[l,'STDs:vaginal condylomatosis'] = df4.loc[l,'STDs:vaginal condylomatosis'].fillna(0)
l = (df4['STDs']==0)
df4.loc[l,'STDs:vulvo-perineal condylomatosis'] = df4.loc[l,'STDs:vulvo-perineal condylomatosis'].fillna(0)
l = (df4['STDs']==0)
df4.loc[l,'STDs:syphilis'] = df4.loc[l,'STDs:syphilis'].fillna(0)
l = (df4['STDs']==0)
df4.loc[l,'STDs:pelvic inflammatory disease'] = df4.loc[l,'STDs:pelvic inflammatory disease'].fillna(0)
l = (df4['STDs']==0)
df4.loc[l,'STDs:genital herpes'] = df4.loc[l,'STDs:genital herpes'].fillna(0)
l = (df4['STDs']==0)
df4.loc[l,'STDs:molluscum contagiosum'] = df4.loc[l,'STDs:molluscum contagiosum'].fillna(0)

df4['STDs:molluscum contagiosum'].isnull().sum()


# For other values we have to fill Nan values with median because all STDs depends on other STDs and also we cannot sure about person disease.

# In[86]:


df4['STDs:condylomatosis'].fillna(df4['STDs:condylomatosis'].median(),inplace = True)
df4['STDs:cervical condylomatosis'].fillna(df4['STDs:cervical condylomatosis'].median(),inplace = True)
df4['STDs:vaginal condylomatosis'].fillna(df4['STDs:vaginal condylomatosis'].median(),inplace = True)
df4['STDs:vulvo-perineal condylomatosis'].fillna(df4['STDs:vulvo-perineal condylomatosis'].median(),inplace = True)
df4['STDs:syphilis'].fillna(df4['STDs:syphilis'].median(),inplace = True)
df4['STDs:pelvic inflammatory disease'].fillna(df4['STDs:pelvic inflammatory disease'].median(),inplace = True)
df4['STDs:genital herpes'].fillna(df4['STDs:genital herpes'].median(),inplace = True)
df4['STDs:molluscum contagiosum'].fillna(df4['STDs:molluscum contagiosum'].median(),inplace = True)

df4['STDs:condylomatosis'].isnull().sum()


# ## 1.5)AIDS

# This feature correlation not show us any clue. But we know that AIDS also a STDs disease. Then we can fill NaN values as;

# In[87]:


l = (df4['STDs']==0)
df4.loc[l,'STDs:AIDS'] = df4.loc[l,'STDs:AIDS'].fillna(0)
df4['STDs:AIDS'].fillna(df4['STDs:AIDS'].median(),inplace = True)

df4['STDs:AIDS'].isnull().sum()


# ## 1.6)STDs:HIV

# This feature effect by STD feature.

# In[88]:


len(df4[df4['STDs:HIV']==1])


# In[89]:


len(df4[(df4['STDs:HIV']==1) & (df4['STDs']==1)])


# %100 of patient who have STDs then they have STDs:HIV

# In[90]:


l = (df4['STDs']==1)
df4.loc[l,'STDs:HIV'] = df4.loc[l,'STDs:HIV'].fillna(1)
df4['STDs:HIV'].fillna(0,inplace = True)


# Cheking contradiction values;

# In[91]:


len(df4[(df4['STDs']==0) & (df4['STDs:HIV'] == 1)])


# In[92]:


df4['STDs:HIV'].isnull().sum()


# ## 1.7)STDs:Hepatitis B

# This feature effect by STDs:HIV feature.

# In[93]:


len(df4[df4['STDs:Hepatitis B']==1])


# There is a one person who have disease. According to population this value is so minimal. We can fill Nan values with 0.

# In[94]:


df4['STDs:Hepatitis B'].fillna(0, inplace = True)

df4['STDs:Hepatitis B'].isnull().sum()


# ## 1.8)STDs:HPV

# But HPV's positive values are not enough. Null values cannot define by using important features. We can fill NaN values with 0.

# In[95]:


df4['STDs:HPV'].fillna(0, inplace = True)

df4['STDs:HPV'].isnull().sum()


# ## 1.9)STDs: Time since first diagnosis and STDs: Time since last diagnosis 

# If patient STDs is zero then first diagnosis and last diagnosis cannot be a value. We can fill Nan values with this knowledge.

# In[96]:


l = (df4['STDs']==1)
df4.loc[l,'STDs: Time since first diagnosis'] = df4.loc[l,'STDs: Time since first diagnosis'].fillna(df4['STDs: Time since first diagnosis'].median())
l = (df4['STDs']==1)
df4.loc[l,'STDs: Time since last diagnosis'] = df4.loc[l,'STDs: Time since last diagnosis'].fillna(df4['STDs: Time since last diagnosis'].median())
df4['STDs: Time since last diagnosis'].fillna(0, inplace = True)
df4['STDs: Time since first diagnosis'].fillna(0, inplace = True)

df4['STDs: Time since last diagnosis'].isnull().sum()


# # 2)Feature correlation

# All features pearson correlations between them after handling NaN values;

# In[97]:


corrmat = df4.corr()

plt.figure(figsize=(20,20))

sns.set(font_scale=2)
hm = sns.heatmap(corrmat,cmap = 'Set1', cbar=True, annot=True,vmin=0,vmax =1,center=True, square=True, fmt='.2f', annot_kws={'size': 10},
             yticklabels = df4.columns, xticklabels = df4.columns)
plt.show()


# In[98]:


df4['Biopsy'] = df1['Biopsy']

corrmat = df4.corr()
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Biopsy')['Biopsy'].index

#cols =cols.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])

cm = df4[cols].corr()

plt.figure(figsize=(9,9))

sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cmap = 'Set2', cbar=True, annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# ## 2.1)RandomForest for features

# In[99]:


from sklearn.ensemble import RandomForestClassifier

X = df4.drop('Biopsy', axis =1)
Y = df4["Biopsy"]
names = X.columns
rf = RandomForestClassifier()
rf.fit(X, Y)

result_rf = pd.DataFrame()
result_rf['Features'] = X.columns
result_rf ['Values'] = rf.feature_importances_
result_rf.sort_values('Values', inplace = True, ascending = False)

plt.figure(figsize=(11,11))
sns.set_color_codes("pastel")
sns.barplot(x = 'Values',y = 'Features', data=result_rf, color="Blue")
plt.show()


# ## 2.2)ExtraTreesClassifier

# In[100]:


from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, Y)

result_et = pd.DataFrame()
result_et['Features'] = X.columns
result_et ['Values'] = model.feature_importances_
result_et.sort_values('Values', inplace=True, ascending =False)

plt.figure(figsize=(11,11))
sns.set_color_codes("pastel")
sns.barplot(x = 'Values',y = 'Features', data=result_et, color="red")
plt.show()


# ## 2.3)RFE

# In[101]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 10)
rfe = rfe.fit(X, Y)

result_lg = pd.DataFrame()
result_lg['Features'] = X.columns
result_lg ['Ranking'] = rfe.ranking_
result_lg.sort_values('Ranking', inplace=True , ascending = False)

plt.figure(figsize=(11,11))
sns.set_color_codes("pastel")
sns.barplot(x = 'Ranking',y = 'Features', data=result_lg, color="orange")
plt.show()


# In[102]:


df5 = df4.copy()


# # 3)Feature Extraction

# Created features and their meanings are followings;
#     - YAFSI : How many years pass after patient had first sexual intercourse
#     - SSY : How many years patient did not smoke
#     - SPYP : After first sexual intercourse how many partners patients had percentage.
#     - SP: Smoking percentage over age.
#     - HCP : Hormonal Contraceptices percentage over age
#     - STDP: STDs percentage over age
#     - IUDP: IUD percantage over age
#     - TSP : Total pack of cigarettes of patient smoked
#     - NPP : Number of pregnancies percantage over age
#     - NSPP: Number of sexual partners percentage over age
#     - NDP : Number of STDs diagnosis percentage over age
#     - TBD : Time betweem diagnosis
#     - YAHC : How many years patient dont take Hormonal Contraceptives
#     - YAIUD: How many years patient dont take IUD
#     - NPSP : Average pregnancy over one sexual partner
#     - IUDSY: How many years patient take IUD after first sexual intercourse percentage
#     - HCSY : How many years patient take Hormonal Contraceptives after first sexual intercourse percentage

# In[103]:


df5['YAFSI'] = df5['Age'] - df5['First sexual intercourse']
df5['SSY'] = df5['Age'] - df5['Smokes (years)']
df5['SPYP'] = df5['Number of sexual partners'] / df5['YAFSI']
df5['SP'] = df5['Smokes (years)'] / df5['Age']
df5['HCP'] = df5['Hormonal Contraceptives (years)'] / df5['Age']
df5['STDP'] = df5['STDs (number)'] / df5['Age']
df5['IUDP'] = df5['IUD (years)'] / df5['Age']
df5['TSP'] = df5['Smokes (packs/year)'] * df5['Smokes (years)']
df5['NPP'] = df5['Num of pregnancies'] / df5['Age']
df5['NSPP'] = df5['Number of sexual partners'] / df5['Age']
df5['NDP'] = df5['STDs: Number of diagnosis'] / df5['Age']
df5['TBD'] = (df5['STDs: Time since first diagnosis'] - df5['STDs: Time since last diagnosis']) / df5['STDs: Number of diagnosis']
df5['YAHC'] = df5['Age'] - df5['Hormonal Contraceptives (years)']
df5['YAIUD'] = df5['Age'] - df5['IUD (years)']
df5['NPSP'] = df5['Num of pregnancies'] / df5['Number of sexual partners']
df5['IUDSY'] = df5['IUD (years)'] / df5['YAFSI']
df5['HCSY'] = df5['Hormonal Contraceptives (years)'] / df5['YAFSI']


# After feature creation we probably have infinite and null values because some divisions are (0/0=Null) and (number/0=infinite). We have to check them and fill them with zeros.

# In[104]:


df5.replace([np.inf, -np.inf], np.nan, inplace = True)
df5.fillna(0,inplace=True)


# # 4)Feature Correlation and Feature Importance After Extraction

# ## 4.1)Feature Correlation

# In[105]:


corrmat = df5.corr()
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Biopsy')['Biopsy'].index

#cols =cols.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])

cm = df5[cols].corr()

plt.figure(figsize=(9,9))

sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cmap = 'Set3', cbar=True, annot=True,vmin=0,vmax =1, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels = cols.values, xticklabels = cols.values)
plt.show()


# Our created features are not coralated with biopsy let see their importance over Biopsy feature.

# ## 4.2)Random Forest

# In[106]:


from sklearn.ensemble import RandomForestClassifier

X_p = df5.drop('Biopsy', axis =1)
Y_p = df5["Biopsy"]
names = X_p.columns
rf = RandomForestClassifier()
rf.fit(X_p, Y_p)

result_rf = pd.DataFrame()
result_rf['Features'] = X_p.columns
result_rf ['Values'] = rf.feature_importances_
result_rf.sort_values('Values',inplace=True, ascending = False)

plt.figure(figsize=(11,11))
sns.set_color_codes("pastel")
sns.barplot(x = 'Values',y = 'Features', data=result_rf, color="Yellow")
plt.show()


# ## 4.3)ExtraTreesClassifier

# In[107]:


from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_p, Y_p)

result_et = pd.DataFrame()
result_et['Features'] = X_p.columns
result_et ['Values'] = model.feature_importances_
result_et.sort_values('Values',inplace =True,ascending=False)

plt.figure(figsize=(11,11))
sns.set_color_codes("pastel")
sns.barplot(x = 'Values',y = 'Features', data=result_et, color="black")
plt.show()


# ## 4.4)RFE

# In[108]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 10)
rfe = rfe.fit(X_p, Y_p)

result_lg = pd.DataFrame()
result_lg['Features'] = X_p.columns
result_lg ['Ranking'] = rfe.ranking_
result_lg.sort_values('Ranking', inplace=True ,ascending = False)

plt.figure(figsize=(11,11))
sns.set_color_codes("pastel")
sns.barplot(x = 'Ranking',y = 'Features', data=result_lg, color="green")
plt.show()


# After feature importances clearly seen that we did great job!. Our created features in first 5 features all of feature selection models.
