#!/usr/bin/env python
# coding: utf-8

# 1.no feature selction done 
# 2.no outlier  treatment
# 

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (10,5)


# In[ ]:


df=pd.read_excel('../input/credit-card-customers-data/final_case_study_credit.xlsx')


# In[ ]:


df.head()


# In[ ]:


imp=['ownfax', 'ownvcr', 'active', 'voice', 'internet', 'owntv', 'tollmon', 'equipten', 'card2spent', 'equip', 'tollfree', 'cardmon', 'wireten', 'ebill', 'cardtenurecat', 'equipmon', 'wireless', 'card2benefit', 'pager', 'response_03', 'response_01', 'card2tenurecat', 'ownpc', 'owngame', 'card2type', 'card2fee', 'wiremon', 'tollten', 'cardspent', 'owncd', 'churn', 'callcard', 'multline', 'forward', 'owndvd', 'response_02', 'tenure', 'cardten', 'ownpda', 'ownipod', 'card2', 'news']


# In[ ]:


ndf=df.loc[:,imp]


# In[ ]:


a=['address', 'age', 'bfast', 'birthmonth', 'callid', 'callwait', 'card2tenure', 'cardtenure', 'cartype', 'commutebike', 'commutebus', 'commutecar', 'commutecarpool', 'commutecat', 'commutemotorcycle', 'commutenonmotor', 'commutepublic', 'commuterail', 'commutetime', 'commutewalk', 'confer','custid', 'ed', 'employ', 'hourstv', 'lncardmon', 'lncardten', 'lncreddebt', 'lnequipmon', 'lnequipten', 'lninc', 'lnlongmon', 'lnlongten', 'lnothdebt', 'lntollmon', 'lntollten', 'lnwiremon', 'lnwireten', 'longmon', 'longten', 'pets_birds', 'pets_cats', 'pets_dogs', 'pets_freshfish', 'pets_reptiles', 'pets_saltfish', 'pets_small', 'spoused']


# In[ ]:


a=set(a)


# In[ ]:


b=set(df.columns)


# In[ ]:


ndf=df.loc[:,b-a]


# In[ ]:


ndf['total_spend']=ndf['cardspent']+ndf['card2spent']


# In[ ]:


del ndf['card2spent']


# In[ ]:


del ndf['cardspent']


# In[ ]:


import seaborn as sns
import  matplotlib


# In[ ]:


ndf.describe()


# In[ ]:


ndf=ndf.replace('#NULL!',np.NaN)


# In[ ]:


ndf.townsize=ndf.townsize.astype('float')


# In[ ]:


ndf.dtypes.value_counts()


# In[ ]:


ndf.townsize.dtype


# In[ ]:


del ndf['cardten']


# In[ ]:


ndf.card2fee.nunique(dropna=True)


# In[ ]:


df['pager']


# In[ ]:


c=list(ndf.columns)


# In[ ]:


a=[]
b=[]
for e in c:
    if ndf[e].nunique(dropna=True)>5:
        a.append(e)
    else:
        b.append(e)
        


# In[ ]:


df_dis=ndf.loc[:,a]


# In[ ]:


df_con=ndf.loc[:,b]


# In[ ]:


sns.distplot(np.sqrt(np.sqrt(df_dis.total_spend)))


# In[ ]:


sns.regplot(x='total_spend',y='debtinc',data=df_dis)


# In[ ]:


df_con


# In[ ]:


ndf.equipten.value_counts()[0]/ndf.shape[0]*100
    


# In[ ]:


ndf.tollten.value_counts()[0]/ndf.shape[0]*100


# In[ ]:


#more than 70 % values in wireten are null so it is wise to drop columns
ndf=ndf.drop(['equipten','tollten'],axis='columns')


# In[ ]:


df_dis=df_dis.drop(['wireten','equipten','tollten'],axis='columns')


# In[ ]:


sns.distplot(df_dis.total_spend)


# In[ ]:


df_dis.total_spend.describe()


# In[ ]:


ndf['trans_spend']=np.sqrt(np.sqrt(ndf.total_spend))


# In[ ]:


del ndf['total_spend']


# In[ ]:


equ='trans_spend~card2fee+card2benefit+internet+owndvd+wireless+tollfree+active+ownipod+pager+ownpda+tenure+churn+multline+voice+owngame+ownpc+wiremon+callcard+equipmon+equip+forward+card2+ownvcr+owncd+card2type+response_03+tollmon+ebill+news+response_02+ownfax+cardmon+cardtenurecat+owntv+response_01+card2tenurecat'


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x,y=train_test_split(ndf,test_size=0.3,random_state=1234)


# In[ ]:


ndf=ndf.fillna(0)


# In[ ]:


x=ndf[ndf.columns.difference(['trans_spend'])]
y=ndf['trans_spend']


# In[ ]:


x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.33, random_state=42)


# ## model with whole data set

# In[ ]:


import statsmodels.formula.api as smf
model = smf.ols(equ, data=ndf).fit()
print(model.summary())


# ### feature engineeering

# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


Lasso(alpha=0.1).fit(x,y).coef_


# In[ ]:


ls=pd.DataFrame(data=Lasso(alpha=0.1).fit(x,y).coef_,index=x.columns,columns=['a'])


# In[ ]:


list(ls[ls['a']<0].index)


# In[ ]:


list(ls[ls['a']>0].index)


# In[ ]:


lasso_imp=[list(ls[ls['a']>0].index),list(ls[ls['a']<0].index)]


# In[ ]:


lasso_imp


# In[ ]:


from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
rfe = RFE(lm, n_features_to_select=15).fit(x,y)
bool_RFE = rfe.get_support()


# In[ ]:


x.loc[:,bool_RFE]


# In[ ]:


from sklearn.feature_selection import f_regression
F_values, p_values  = f_regression(x,y)
F_values = pd.Series(F_values, name='F_values')
p_values = pd.Series(p_values, name='p_values')
col_names = pd.Series(x.columns,name='col_names')
f_result = pd.concat([col_names,F_values,p_values],axis=1)
f1=list(f_result.sort_values(by=["F_values"], ascending=False).head(15).col_names)


# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif, chi2
selector = SelectKBest(f_classif, k=15)
selector.fit(x,y)
K_Best = x.columns[selector.get_support()]
f=list(K_Best)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
VIF = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]
VIF = pd.Series(VIF, name='VIF')
cols = pd.Series(x.columns, name='Column_Name')
VIF_res = pd.concat([cols,VIF],axis=1)
vif=list(VIF_res.sort_values(by=['VIF']).head(15).Column_Name)


# In[ ]:


[f1,f,vif]


# In[ ]:


eqn='trans_spend~active+carbuy+carcatvalue+card+card2benefit+card2fee+cardfee+carown+carvalue+churn+creddebt+debtinc+default+edcat+empcat+gender+inccat+internet+othdebt+owncd+owndvd+ownpda+owntv+ownvcr+pets+polcontrib+polparty+reason+response_01+response_02+response_03+retire+telecommute+tenure+union+wiremon'


# In[ ]:


n=['active','carbuy','carcatvalue','card','card2benefit','card2fee','cardfee','carown','carvalue','churn','creddebt','debtinc','default','edcat','empcat','gender','inccat','income','internet','othdebt','owncd','owndvd','ownpda','owntv','ownvcr','pets','polcontrib','polparty','reason','response_01','response_02','response_03','retire','telecommute','tenure','union','wiremon']


# In[ ]:


len(n)


# In[ ]:


model2 = smf.ols(eqn, data=ndf).fit()
print(model2.summary())


# In[ ]:


lasso_imp


# In[ ]:


lasso_eqn='trans_spend~carvalue+equipmon+income+card+cardmon'


# In[ ]:


lasso = smf.ols(lasso_eqn, data=ndf).fit()
print(lasso.summary())


# ### from th above model it's evident that these variables  'carvalue,equipmon,income,card,cardmon' were able to explain almost 20% of variance in the target variable, so we regard them as top 4 key drivers for the  customers spend

# In[ ]:


df_dis


# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


sns.boxplot(x=df_dis['debtinc'],data=df_dis)

1.finding right way for outlier treatment 
2.dealing with null values
3.separate way for extreme outliers
# In[ ]:


def outlier_miss_treat(x):
    x = x.clip(lower=x.quantile(0.05),upper=x.quantile(0.95))
    return x


# In[ ]:


df_dis=df_dis.apply(outlier_miss_treat)


# In[ ]:


df_dis['total_spend']=ndf['trans_spend']


# In[ ]:


cdf=pd.concat([df_dis,df_con],axis=1)


# In[ ]:


cdf=cdf.apply(outlier_miss_treat)


# In[ ]:


eqn='total_spend~active+carbuy+carcatvalue+card+card2benefit+card2fee+cardfee+carown+carvalue+churn+creddebt+debtinc+default+edcat+empcat+gender+inccat+internet+othdebt+owncd+owndvd+ownpda+owntv+ownvcr+pets+polcontrib+polparty+reason+response_01+response_02+response_03+retire+telecommute+tenure+union+wiremon'


# ### model using lasso

# In[ ]:


lasso_eqn='total_spend~income+card+cardmon'


# In[ ]:


clean= smf.ols(lasso_eqn, data=cdf).fit()
print(clean.summary())


# In[ ]:


#after standard data cleaning income, card,cardmon were able to explain 21% of variance in data


# ## model using f1

# In[ ]:


print(f1)


# In[ ]:


f1='total_spend~inccat+income+card+retire'


# In[ ]:


clean= smf.ols(f1, data=cdf).fit()
print(clean.summary())


# ## model using f1 and lasso

# In[ ]:


u='total_spend~card+inccat+income+retire'


# In[ ]:


clean= smf.ols(u, data=cdf).fit()
print(clean.summary())


# ### accuracy increased by 1% with two new varibles inccat,retire

# ## model using f regression variables

# In[ ]:


print(f)


# In[ ]:


w='total_spend~carcatvalue+card2benefit+carown+carvalue+churn+creddebt+debtinc+default+edcat+internet+othdebt+ownpda+reason+response_02+tenure'


# In[ ]:


clean= smf.ols(w, data=cdf).fit()
print(clean.summary())


# ### f regression is giving very bad accuracy

# In[ ]:


print(vif)


# In[ ]:


v1='total_spend~response_01+response_03+response_02+union+card2fee+cardfee+telecommute+polcontrib+polparty+carbuy+churn+pets+default+active+gender'


# In[ ]:


clean= smf.ols(v1, data=cdf).fit()
print(clean.summary())


# #### vif is also giving very bad accuracy

# total_spend~card+inccat+income+retire'

# In[ ]:


imp=cdf.loc[:,['card','inccat','income','retire']]


# In[ ]:


sns.distplot(imp.income)


# In[ ]:


sns.boxplot(x=imp.income,data=imp)


# In[ ]:


sns.boxplot(x=imp.income.clip(upper=imp.income.quantile(0.90)),data=imp)


# In[ ]:


imp['t_income']=imp.income.clip(upper=imp.income.quantile(0.90))


# In[ ]:


cdf['t_income']=imp.income.clip(upper=imp.income.quantile(0.90))


# In[ ]:


f3='trans_spend~card+inccat+t_income+retire+creddebt+gender+internet+owndvd+owntv+response_03'


# In[ ]:


f2='total_spend~card+t_income+retire+creddebt+gender+internet+owndvd+response_03'


# In[ ]:


clean= smf.ols(f2, data=cdf).fit()
print(clean.summary())


# In[ ]:




