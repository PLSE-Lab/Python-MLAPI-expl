#!/usr/bin/env python
# coding: utf-8

# <h1>Is the data suffer from confounding bias? and what factors are dominating the odds of heart disese diagnotic?</h1>
# The study on the data will be focus on disriminating factors that affect the probability of being diagnotic to have heart disease after admission to hospital. Along the journey on factor analysis, we will also study on if any bias occurs, especically on confounding as it may lead to spectacularly disatrous result on analysis.</br>
# The study will utilize odds ratio, case-control study(contingency table) from medical statistics together with logit regression to answer the title questions.

# In[2]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[4]:


heart_disease=pd.read_csv('../input/heart.csv')


# # Exploring Data
# 
# The data source contains both category and numical exploratory variable as follow : 
# 
# |  Category | Numerical   |
# |---------- |-------------|
# | 1.sex     | 9.age       |
# | 2.cp      | 10.trestbps |
# | 3.fbs     | 11.chol     |
# | 4.restecg | 12. thalach |
# | 5.exang   | 13. oldpeak |
# | 6.slope   |             |
# | 7.ca      |             |
# | 8.thal    |             |
# 

# In[5]:


#separate the column by numerical and category in list for future use
cat_column=['sex','cp','fbs','restecg','exang','slope','ca','thal']
num_column=['age','trestbps','chol','thalach','oldpeak']
#checking if the data's value matching the description
heart_disease.cp.unique()
heart_disease.fbs.unique()
heart_disease.restecg.unique()
heart_disease.exang.unique()
heart_disease.slope.unique()
heart_disease.ca.unique()
heart_disease.thal.unique()


# It is found that some of the category variable's value is not match with the definition defined on the web. For example , the variable cp is defined to have value 1,2,3,4 , but the actual value is left shiftted by 1, so we will follow the left shiftted pattern to align with the description if any mis-match is found.

# ## Individual attribute study
# 
# First, let's try to get some feeling on how each variable is distributed with the target variable, we will simply use count plot on those category variables and violin for those numerical data.

# ### Target
# Target Distrubtion :
# >From the diagram below, it shows that the data set contains roughly the same amount record of getting heart disease or not.

# In[6]:


sns.catplot('target',data=heart_disease,kind='count',height=5,aspect=.8)


# ### Age
# Age vs Target : 
# >The diagram shows that for those getting heart disease, it got a flat and wide range of distribution which center around 51. The distribution on not having heart discese is more concentrated when compare to target=1. However, it is see that the box plot on target=0 is well included in the range with heart disease.

# In[7]:


sns.catplot(x='age',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)
sns.catplot(x='age',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)


# ### Sex
# Sex vs Target :
# >The number of data on male or female suffer from heart disease is approximately the same, however, it is easy to shows that the ratio between suffer from heart disease or not by sex got a significant different. The data set on female which diagnotic to have heart disease is 3 times higher than male. Furthermore,it is find that the records on male is 2 times more than the female in total.
# 
# >Finding :
# >>-Sex may be a significant factor on heart disease, but further examination is need.</br>
# >>-Selection bisa ? the number of record is in huge different by sex.</br>
# >>-Confounding bias? The heart disease raito is triple for female, examination is need.</br>

# In[8]:


sns.catplot('sex',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)


# In[9]:


heart_disease.groupby(['sex','target']).describe()


# ### CP
# CP vs Target :
# > The number of record on chest pain type that are not cause by heart disease is heavily concentrated on typical angia. On the other hand, those who are identified to be cause by heart disease are much more evenly distributed in comparison.
# 
# >Finding:
# 
# >>-In comparison on having heart disease or not, the number of record on chest pin type other than typical angia case by heart disease may got a significant different with those are not in heart disease, especially on atypical angina and non-angial pain

# In[10]:


sns.catplot('cp',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)


# ### Thalach
# thalach vs Target :
# > The diagram below shows the target group are left skew(negative skew) distributed and with little skewness on the control groupm but the distribution on no heart disease is more flatten than those with heart disease. The mode of those suffer from heart disease is located at around 160 and 140 for those who are not, however both distribution are heavily overlapped. No further insight is found for this variable.

# In[11]:


sns.catplot(x='thalach',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)
sns.catplot(x='thalach',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)


# ### exang
# exang vs Target :
# > From the diagram below, it shows that patient with heart disease suffer from angina are mostly not induce by exercise, in comparison with those are not with heart disease which is more evenly distributed.

# In[12]:


sns.catplot('exang',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)


# ### oldpeak
# oldpeak vs Target :
# > The distribution on oldpeak shows a quite different shape on patient with or with out heart disease. Although both are tend to be right skew, but the data of oldpeak for those with heart disease are concerntrated arount 0 with a relatively sharp peak in contrast to those without disease which is widely spread with flat sloping

# In[13]:


sns.catplot(x='oldpeak',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)


# ###  slope
# slope vs Target :
# > The record shows that patient with heart disease are tend to be with slope equal to 2 or 1. On the other hand, those who are without heart disease are mostly like to have slope equal to 1

# In[14]:


sns.catplot('slope',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)


# ### ca
# ca vs Target :
# > The record shows that patient with heart disease are tend to be with ca equal to 0. On the other hand, those who are without heart disease are more evently distributed over level 0,1,2.

# In[15]:


sns.catplot('ca',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)


# ### thal
# thal vs Target :
# > The record shows that patient with heart disease are tend to be with thal equal to 2. On the other hand, those who are without heart disease are mostly like to have thal equal to 3

# In[16]:


sns.catplot('thal',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)


# ### Other's attributes
# Other's attributes vs Target - from those plot below on trestbps ,chol ,fbs and restecg, those plot shows approximately the same distribution  over suffer or not on heart disease. So, in general, it is believe that those variables are less significance than those above.

# In[34]:


sns.catplot(x='trestbps',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)
sns.catplot(x='chol',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)
sns.catplot(x='chol',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)
sns.catplot('fbs',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
sns.catplot('restecg',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)


# # Data Transformation
# We will focus on those numerical variables - age,trestbps,chol,thalach,oldpeak for transformation purpose. The table below shows a summary statistic of the list.

# ## Skewness
# - Variable age, trestbps and chol are observed to be approximately normal, so transformation is not essential.
# - Variable thalach and oldpeak are both right and left skew accordingly.
# - log1p transforamtion is applied for oldpeak, but the effect is not good, may be it is due to there are 99 records(~32%) in total which is exactly 0.
# 
# In result, only normalization will be applied due to the difference in amplitude of those variable, it is done by sklearn fit_transform

# In[18]:


heart_disease[['age','trestbps','chol','thalach','oldpeak']].describe()


# ## Normalization

# In[19]:


#Normalize Data 
scaler=StandardScaler()
numerical=pd.DataFrame(scaler.fit_transform(heart_disease[num_column]),columns=num_column,index=heart_disease.index)
heart_disease_transform=heart_disease.copy(deep=True)
heart_disease_transform[num_column]=numerical[num_column]
heart_disease_transform.head()


# In[20]:


# adding intercept term for logit regression
heart_disease_transform['intercept']=1.0
#rearranging the column of data set , put target and intercept in front will make lifes better 
heart_disease_transform.head()
cols=num_column.copy()
cols.insert(0,'target')
cols.insert(1,'intercept')
cols.extend(cat_column)
heart_disease_transform=heart_disease_transform.reindex(columns=cols)
#one hot encoding for cat variable ,with no drop first 
heart_disease_transform_nodrop=pd.get_dummies(heart_disease_transform,prefix_sep='_',columns=cat_column)
#one hot encoding for cat variable ,with drop first 
heart_disease_transform=pd.get_dummies(heart_disease_transform,prefix_sep='_',columns=cat_column,drop_first=True)


# ## Correlation Heapmap

# In[21]:


heart_disease_transform.head()
# Create correlation map on the transformed data set( one hot encode drop first = true)  
corr=heart_disease_transform.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr,mask=mask, cmap=cmap,linewidths=.5 ,center=0,square=True, cbar_kws={"shrink": 1})
# Create correlation map on the transformed data set( one hot encode drop first = false)


# In[22]:


corr2=heart_disease_transform_nodrop.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(corr2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corr2,mask=mask, cmap=cmap,linewidths=.5 ,center=0,square=True, cbar_kws={"shrink": 1})


# # Model Fitting 

# ## Full Model 

# In[23]:


#prepare training and testing dataset for sklearn logit
X_train, X_test, y_train, y_test = train_test_split(heart_disease_transform.drop('target',axis=1), 
                                                    heart_disease_transform['target'], test_size=0.30, random_state=101)
#sklearn Logistic instance
sklogit=LogisticRegression(solver='lbfgs')
sklogitv2=LogisticRegression(solver='lbfgs')
sklogitv3=LogisticRegression(solver='lbfgs')
sklogitv4=LogisticRegression(solver='lbfgs')
sklogitv5=LogisticRegression(solver='lbfgs')


# In[24]:


#statsmodels for summary report on Full Model
logitv1=sm.Logit(heart_disease_transform['target'],heart_disease_transform[heart_disease_transform.columns[1:]])
result=logitv1.fit()


# In[25]:


result.summary2()


# In[26]:


# Full Model on sklearn
sklogit.fit(X_train,y_train)
sk_predict=sklogit.predict(X_test)
sklogit.score(X_test,y_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, sk_predict))


# ### Summary on Full Model:
# From the summary of the result, those variables with coefficient not equal to zero with confidence level set to 90% (P<=0.05) are list below : 
# 1. trestbps
# 2. sex_1
# 3. cp_2
# 4. cp_3
# 5. ca_1
# 6. ca_2
# 7. ca_3
# 
# ** We will include cp_1 , ca_4 in the model for easy interpretion.
# 
# What susprising from the result is that the variable trestbps got a relativly strong effect on the model while it shows no significance during data inspection on the plot of trestbps vs target. 
# Also, when comparing with the correlation heapmap and thoe "VS" plot, variables thalach , exang , oldpeak , slope and thal shows a relatively strong correlation and significance to the target, however, it is reverse from the logit model.
# 
# Why ? Let defer the question a bit and see what's the result from the reduced model

# ## Reduce Model 1

# In[27]:


#To prepare the dataset of reduced model,drop variable for those P-value is >0.05 from the summary report of statsmodels 
# W
heart_disease_transform_v2=pd.DataFrame(heart_disease_transform[['target','intercept','trestbps','sex_1','cp_1','cp_2','cp_3','ca_1','ca_2','ca_3','ca_4']],
                                        columns=['target','intercept','trestbps','sex_1','cp_1','cp_2','cp_3','ca_1','ca_2','ca_3','ca_4'],index=heart_disease_transform.index)


# In[28]:


#statsmodels for summary report on Reduced Model
logitv2=sm.Logit(heart_disease_transform_v2['target'],heart_disease_transform_v2[heart_disease_transform_v2.columns[1:]])
resultv2=logitv2.fit()


# In[29]:


resultv2.summary2()


# In[30]:


#prepare training and testing dataset for sklearn on reduced model
X_trainv2, X_testv2, y_trainv2, y_testv2 = train_test_split(heart_disease_transform_v2.drop('target',axis=1),
                                                            heart_disease_transform_v2['target'], test_size=0.30, random_state=101)
# Reduced Model on sklearn
sklogitv2.fit(X_trainv2,y_trainv2)
sk_predictv2=sklogitv2.predict(X_testv2)
sklogitv2.score(X_testv2,y_testv2)
from sklearn.metrics import classification_report
print(classification_report(y_testv2, sk_predictv2))
# exp(coefficient of the model) 
np.exp(resultv2.params)


# ### Analysis on Reduce Model
# Let try to interpret the coefficient from the resulting model by taking exponentail on those. As the model is in exponentical function, those variable is in multiplicative relation, so if the coefficient is smaller than 1, it has a weakening effect while the effect will be strength if it is greater than 1.
# 
# ### Intercept (Baseline)
# >The intercept in this model is representing the odd ratio between being diagnostic with heart disease or not for record that is a female with cp,ca and trestbps $=$ 0,it is $\sim$ 2.93 times more likely to have heart disease with the condition unchanged.
# 
# ### cp_1, cp_2 , cp_3 (coefficient >1)
# >Both cofficients are greater than 1, so the present of either variables will increase the odd ratio base on the intercept. 
# 
# ### sex_1 , ca_1 , ca_2 , ca_3  (coefficient < 1)
# >All those coefficient got a weakening effect on the odd ratio, for example, if the model is present with a new record which is the same with the intercept setting except it is come with ca=3 , the odd ratio will become 2.93*0.062941 $\simeq$ 0.1844, which favor on not having heart disease.
# 
# From the above result, it is shows that the probability of having heart disease for female is much more higher than male when holding the same condition,  it is consistent with the finding in the first section of Exploring Data on target vs sex. However it is suspicious that if sex is a major factor on heart disease diagnotic, with further study on the correlation heapmap above, it is found that female and thal=2 is positively correlated while it is reversed for male. Let's plot some graph and number to see if any clue can be found. </br>

# In[31]:


sns.catplot('thal',col='target',row='sex',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
sns.catplot('thal',col='sex',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
heart_disease.groupby(['target','sex','thal']).describe()


# # Cofounding - Thal and with others ?

# The above plot shows the distribution of thal group by target and sex, it is find that almost all female data that diagnotic with heart disease happen to have variable thal equal to 2 (69/72). On the other hand, the study on thal from the correlation map has also suggest a positive correlation with the target variable as well, so it might indicate that the association beteewn sex and target may be distorted by thal. 
# 
# 
# > #### Odds Raito of association between Sex and target (Table 1)
# 
# | Exposure Category  |  Case | Control   | Heart Disease Rate|
# |-------------------:|:-----:|:---------:|:-----------------:|
# |            Female  | 72    |  24       | 0.75|
# |              Male  | 93    |  114      | 0.45|
# |             Total  |  165  |   138     | 0.54|
# 
# >> $\widehat{OR}$=$\frac{72*114}{24*93} = \frac{8208}{2232} = 3.6774$ </br>
# >>It is shows that the heart disease rate for female is higher than male on this sample, and the OR>1 indicate a positive association for sex and target. 
# 
# > #### Odds Raito of association between Sex and target group by 3 level of thal
# 
# >>Thal = 1  (Table 2.1)
# 
# | Exposure category | Case    | Control    |Total | Heart Disease Rate| 
# |------------------:|:-------:|:----------:|:----:|:-----------------:|
# |            Female |   1     |    1       |  2   |  0.5              |
# |              Male |   6     |   12       | 18   |  0.33             |
# | Total             |   7     |   13       | 20   |  0.35             |
# 
# >> $\widehat{OR}$=$\frac{1*12}{1*6} = \frac{12}{6} = 2$ </br>
# 
# >>Thal = 2 (Table 2.2)
# 
# 
# | Exposure category | Case    | Control    |Total | Heart Disease Rate|   
# |------------------:|:-------:|:----------:|:----:|:-----------------:|
# |            Female |  69     |   10       |  79  |  0.87             |
# |              Male |  61     |   26       |  87  |  0.70             |
# | Total             | 130     |   36       | 166  |  0.78             | 
# 
# >> $\widehat{OR}$=$\frac{69*26}{10*610} = \frac{1794}{610} = 2.94$ </br>
# 
# >>Thal =3 (Table 2.3)
# 
# | Exposure category | Case    | Control    |Total | Heart Disease Rate|
# |------------------:|:-------:|:----------:|:----:|:-----------------:|
# |            Female |   2     |   13       |  16  |  0.13             |
# |              Male |  26     |   76       | 102  |  0.25             |
# | Total             |  28     |   89       | 118  |  0.23             |
# 
# >> $\widehat{OR}$=$\frac{2*76}{13*26} = \frac{152}{338} = 0.45$ </br>
# 
# From the above case-contrl table, it is shows that both female and male have the same property on the first 2 level of thal, which is higher heart disease rate for female(not more than 23%) and are associated with the target variable, but the situation has changed at level 3, the odds ratio suggest no interdependence between the exposure and target, as well as the male heart disease rate has rise over the female's. 
# On the other hand, it is to see that the distribution of record is relatively uneven in view of sex as discuss in the opening, female's data is heavily concentrated at level2 while for male, it got more at level 3 in compare to female.    
# 
# 
# Furthermore , we can combine the row total of each table(2.1-2.3) to form another case-control for variable thal as below : 
# 
# >> Case-Control Table by Thal (Table 3.0)
# 
# | Exposure category | Case    | Control    |Total | Heart Disease Rate|
# |------------------:|:-------:|:----------:|:----:|:-----------------:|
# |            Thal=1 |   7     |   13       |  20  |  0.35             |
# |            Thal=2 | 130     |   36       | 166  |  0.78             |
# |            Thal=3 |  28     |   89       | 118  |  0.24             |
# |            Total  | 165     |  138       | 303  |  0.54             |
# 
# From the table above, a shape peak is located at level 2 for the heart disease rate(0.78) on factor thal, it is almost double or triple in compare to level 1 and 3 accordingly. It is easy to show the disease rate is highly assosicate with the level of thal : level 2 is more likely to be diagnotic with heart disease.
# 
# Although, you may found that from table 2.2, both male and female record are roughtly the same, but in combining with the founding from the above tables(2.1-3 and 3.0) on association with heart disease with thal level =2 and the data distribution by sex on thal, there is a greater proportion of positive diagnotic for female than male as those data for male(102/207~50%) is found to be having thal equal to 3 which has a less disease rate. 
# 
# In repsonse to the suspicious cofounding between sex and thal, it is consider to employ the Mantel-Haenszel odd ratio to adjust the odd ratio on sex with thal as strata.</br>
#  $\widehat{OR_{MH}}$=$\frac{12/20+1794/166+152/118}{6/20+610/166+338/118} = \frac{12.69534}{6.8391} = 1.8563$
# 
# The adjusted odd ratio for sex suggest the association with the taraget still exists, but in a smaller magnitude. The result bring's out curiosity on if it is the same with other's attribute that has a confounding relation with thal.
# 
# For the variable thal, there is no much information from the data description section but only value definition, however, the value definition is not consistent with the actual value as well. After some research on the internet, the variable may be referring to a blood disorder disease - Thalassemia, but it is not know what's the actual meaning of the value defined.
# 
# Let assume the attribute thal is referring to Thalassemia, it is possible that those attributes collected is distorted due to mixture of two  diseases? Let's examine those varaibles selected by models (ca ,cp ) above group by thal level by graphical plot below.

# In[32]:


sns.catplot('ca',col='target',row='thal',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
sns.catplot('cp',col='target',row='thal',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)


# Referring to the plot above, it is see that ca=0 has a strong correlation with thal=2 while cp is correlated to thal on 2 and 3. We further analysis on those attributes that is having correlation with thal=2 by referring to the heap map, which include : slope, restecg, oldpeak and thalach, it is found the same correlation pattern appear for slope and restecg on thal=2, however, it is not easy to demonstrate for those numerical variables.
# 
# Let have a brief summary on what have found so far and decide what to do next :
# 1. Attribute Thal may be refer to blood disorder disease, but the value defined don't match with the record 
# 2. The target is correlated to Thal = 2 and have the highest heart disease rate among all levels
# 3. sex, ca , cp ,slope, restecg, oldpeak and thalach are correlated to thal and have highest disease rate at thal =2 
# 
# In conclusion with the above summary, it can be further deduce :
# 1. Thal = 2 is one of the factor affecting diagnotic on heart disease or not
# 2. The Data may be distorted due to mixture of disease (Thalassemia and Heart disease), it is not sure if the attriubes selected from  model is due to Thalassemia(=2 ?) or heart disease 
# 
# In response to the above summary, we may have the following action on the data : 
# 1. remove those records with thal=2 
# 
# The above action is taken on the data set, however, the record of diagnotic to have heart disease has been heavy reduce to 35 which reduce the significance in terms of statistic ,furthermore, logit model is fail to converage in this case as well.

# # Conclusion
# Althought, it is shows from the model that sex, ca and cp are significance and with high probability on positive heart disease diagnotic,but it is not sure the effect is due to thalassemia or not. In result, the data may be suffer from confounding bias in between heart disease with  Thalassemia (assuming it is true for thal=thalassemia). So, it is hard to conduct factor exploration on the diagnotic of heart disease. It is suggested to collect another data set which is not distorted by any kindy of disease which may have correlation to heart disease in order to analyze major factors on heart diagnotic.

# #  Learson Learn
# 1. Category variable is much more harder to interrpet than numerical in EDA, usually numerical attribute can be easily exploer by correlation matrix to identify relationship initially while it is need to explore individually by plot for category variable 
# 2. EDA is critial process and time consuming but it provide much information gain/insight to the problem 
# 3. Try more model, in terms on dropping,adding variable although it doesn't has strong statistical edvience to support. While studying in uni with stable example, you can find great difference in significance during model/variable selection, however, it is not true in the real world, sometime, you have to try by intuitive with no support from figure, it may turn you back to origin (most likely), but you may get interesting finding during the process if time is not a concern.
# 4. Normalization, it happened behind the scenes while i were still struggle with dropping sex attribute, after sex is drop from data set and fit with logit, oldpeak is comming in into the model but the intercept was with large p-value on not being zero.During the process of model fitting, I have accidentially fitted the logit model with original set instead of the one which is normalized, surprisingly, the intercept of original data got a signifiance result on not being 0. The reason behine is that the different meaning on the value 0 for the two data set. First of all, from the plot on oldpeak vs target, it is shows that most of the observations with heart disease are having 0 in oldpeak, while the normalized data set is referring to the means of oldpeak - 1.0396, so, it is certain that the coefficient of intercept is more significant on the original dataset.

# In[ ]:




