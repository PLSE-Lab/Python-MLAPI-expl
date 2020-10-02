#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Toy Data

# In[ ]:


df=pd.DataFrame({'Age':[8,12,15,17,18,10,6],'AK':[0,0,1,1,1,0,0],'WT':[20,25,35,40,38,18,24]})


# In[ ]:


plt.plot(df['AK'],df['WT'],'*')


# In[ ]:


import statsmodels.formula.api as smf


# In[ ]:


statmoddel=smf.ols('WT~AK',df).fit()


# In[ ]:


statmoddel.params


# In[ ]:


pred=statmoddel.predict(df['AK'])


# In[ ]:


plt.plot(pred,df['WT'],"*")
plt.plot(df["AK"],df['WT'],"+")


# In[ ]:


df=pd.DataFrame({'exp':[2.5,3.0,5.7,4.8,7,12.0,8],'g':[1,0,1,0,0,1,0],'sal':[6500,8000,4000,12000,10000,15000,6000]})


# In[ ]:


df


# In[ ]:


import statsmodels.formula.api as smf


# In[ ]:


model=smf.ols('sal~exp+g',df).fit()


# In[ ]:


model.params


# introducing 'intraction term'

# In[ ]:


new_col=df['exp']*df['g']
new_col
A=pd.concat([df,new_col],axis=1)


# In[ ]:


A.columns=['exp', 'g', 'sal', 'gXexp']


# In[ ]:


A


# In[ ]:


model=smf.ols('sal~exp+g+gXexp',A).fit()


# In[ ]:


r=model.params
r


# In[ ]:


exp1=10
g1=0 #ale=0 female=1
gXexp1=(exp1*g1)
gXexp1


# In[ ]:


(11116.710875-371.352785*(exp1)-9439.866818*(g1)+1384.692777*(gXexp1))


# In[ ]:


# male with no exp=11,116 with exp=7,403


# In[ ]:


#female with no exp=1,676  with exp=1,1810


# In[ ]:


Data=pd.read_csv('../input/advertising-data/Advertising.csv',index_col=0)


# In[ ]:


Data.head()


# In[ ]:


corr=Data.corr()
corr


# In[ ]:


sns.heatmap(corr,annot=True)


# In[ ]:


import statsmodels.formula.api as smf


# In[ ]:


model=smf.ols('Sales~TV+Radio+Newspaper',Data).fit()


# In[ ]:


model.params


# In[ ]:


model.summary()


# ###### here Newspaper not passing the statistical test so we are removing it from the model

# In[ ]:


model=smf.ols('Sales~TV+Radio',Data).fit()


# In[ ]:


model.summary()


# ###### so it is a bi-variant model

# ### sales = 2.94+0.0458(TV)+0.1880(Radio)

# In[ ]:


TV=0
Radio=100000


# In[ ]:


sales = 2.94+0.0458*(TV)+0.1880*(Radio)
sales

if we invest onelac for TV and Radio sales will be 23,382 lack
if we invest onelac only TV sales willbe=4,582 lack
if we invest onelac only Radio sales willbe=18,802 lack

# ###### introducing the third intraction term to find intraction between them

# In[ ]:


TVxRadio=Data['TV']*Data['Radio']


# In[ ]:


A=pd.concat([Data,TVxRadio],axis=1)


# In[ ]:


A.columns=['TV', 'Radio', 'Newspaper', 'Sales', 'TVxRadio']


# In[ ]:


A.head()


# In[ ]:


model=smf.ols('Sales~TV+Radio+TVxRadio',A).fit()


# In[ ]:


model.summary()


# In[ ]:


TV=312500.0
Radio=187500.0
TVxRadio=TV*Radio


# In[ ]:


6.7502+0.0191*(TV)+0.0289*(Radio)+0.0011*(TVxRadio)


# In[ ]:


0.0011+0.0289


# In[ ]:


0.0191+0.0289


# In[ ]:


(0.03/0.048)*500000


# In[ ]:


500000-312500.0


# In[ ]:


312500/187500.0


# In[ ]:


64464519.25020001


# In[ ]:


###################################
Data.columns


# In[ ]:


x=Data.drop(['Sales','Newspaper'],axis=1)


# In[ ]:


y=Data[['Sales']]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=2)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model=LinearRegression()
model.fit(X_train,Y_train)


# In[ ]:


pred=model.predict(X_test)


# In[ ]:


rmse=np.sqrt(np.mean((Y_test-pred)**2))


# In[ ]:


rmse


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


X_train.shape


# In[ ]:


from sklearn import metrics


# In[ ]:


np.sqrt(metrics.mean_squared_error(Y_test,pred))


# In[ ]:


plt.plot(Y_test,pred,'*')

