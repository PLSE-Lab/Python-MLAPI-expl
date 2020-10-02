#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd  

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import scipy.stats as stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score,cross_val_predict
from numpy import log, log1p
from scipy.stats import boxcox
import pylab
from sklearn.linear_model import LinearRegression
#! pip install yellowbrick
from yellowbrick.regressor import residuals_plot
from scipy.stats import shapiro,boxcox,yeojohnson
from yellowbrick.regressor import prediction_error
get_ipython().system('pip install dython')
from dython import nominal


# In[ ]:


data=pd.read_csv("/kaggle/input/insurance/insurance.csv")
df=data.copy()

df.head()


# In[ ]:


print("row :",df.shape[0]," ","column :",df.shape[1])


# In[ ]:


df.describe().T


# In[ ]:


df.describe(include=["object"]).T


# In[ ]:


print("Sum of missing values :",df.isnull().sum().sum())


# In[ ]:


df.eq(0).sum()


# In[ ]:


nominal.associations(df,figsize=(20,10),mark_columns=True);


# In[ ]:


plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10,5))
corr=df.corr()
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask);


# In[ ]:


plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(df.charges,color="b");
plt.subplot(122)
sns.distplot(log(df.charges),color="b");


# In[ ]:


sns.pairplot(df,kind="reg",hue="smoker",aspect=2);


# In[ ]:


sns.pairplot(df,kind="reg",hue="sex",aspect=2);


# In[ ]:


sns.relplot(x="bmi",y="charges",hue="smoker",data=df,kind="scatter",aspect=2);


# In[ ]:


sns.relplot(x="bmi",y="charges",hue="children",data=df,kind="scatter",aspect=2,palette='coolwarm');


# In[ ]:


sns.lmplot(x="age", y="charges", hue="smoker", data=df,aspect=2);


# In[ ]:


sns.lmplot(x="bmi", y="charges", hue="smoker", data=df,aspect=2);


# In[ ]:


sns.distplot(df.age);


# In[ ]:


stats.probplot(df.charges, dist="norm", plot=pylab) ;


# In[ ]:


df.groupby("smoker")["charges"].mean().plot.bar(color="r");


# In[ ]:


df.groupby("children")["charges"].mean().plot.bar(color="g");


# In[ ]:


print(sns.FacetGrid(df,hue="sex",height=5,aspect=2).map(sns.kdeplot,"charges",shade=True).add_legend());


# In[ ]:


print(sns.FacetGrid(df,hue="region",height=5,aspect=2).map(sns.kdeplot,"charges",shade=False).add_legend());


# In[ ]:


print(sns.catplot(x="sex",y="charges",hue="smoker",data=df,kind="bar",aspect=2));


# In[ ]:


print(sns.catplot(x="sex",y="charges",hue="region",data=df,kind="bar",aspect=2));


# In[ ]:


sns.catplot(x="smoker",y="charges",data=df,kind="box",aspect=2);


# In[ ]:


sns.catplot(x="sex",y="charges",data=df,kind="box",aspect=2);


# In[ ]:


sns.catplot(x="sex",y="charges",hue="smoker",data=df,kind="box",aspect=2);


# In[ ]:


sns.catplot(x="region",y="charges",data=df,kind="box",aspect=2);


# In[ ]:


sns.catplot(x="children",y="charges",data=df,kind="box",aspect=2);


# In[ ]:


labels=["too_weak","normal","heavy","too_heavy"]
ranges=[0,18.5,24.9,29.9,np.inf]
df["bmi"]=pd.cut(df["bmi"],bins=ranges,labels=labels)


# In[ ]:


print(sns.FacetGrid(df,hue="bmi",height=5,aspect=2).map(sns.kdeplot,"charges",shade=False).add_legend());


# In[ ]:


print(sns.catplot(x="bmi",y="charges",kind="bar",data=df,aspect=2));


# In[ ]:


print(sns.catplot(x="bmi",y="charges",hue="children",kind="bar",data=df,aspect=3));


# In[ ]:


print(sns.catplot(x="bmi",y="charges",hue="smoker",data=df,kind="bar",aspect=2));


# In[ ]:


plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10,5))
corr=df.corr()
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask);


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(121)
sns.boxplot(df["charges"],color="y");
plt.subplot(122)
sns.boxplot(df["age"],color="y");


# In[ ]:


pd.crosstab(df.age,df.children)[:10]


# In[ ]:


df[(df["age"]==18)&(df["sex"]=="female")&(df["children"]>0)]


# In[ ]:


df[(df["age"]==18)&(df["sex"]=="male")&(df["children"]>0)]


# In[ ]:


clf=LocalOutlierFactor(n_neighbors=50)
clf.fit_predict(df[["age","children"]])


# In[ ]:


clf_scores=clf.negative_outlier_factor_


# In[ ]:


np.sort(clf_scores)[0:20]


# In[ ]:


treshold=np.sort(clf_scores)[20]


# In[ ]:


df[clf_scores<treshold]


# In[ ]:


df[(df["age"]==18)&(df["children"]>1)]


# In[ ]:


df.drop(df[(df["age"]==18)&(df["children"]>0)].index,inplace=True)


# In[ ]:


df.corr()


# In[ ]:


print(sns.catplot(x="children",y="charges",hue="smoker",data=df,kind="bar",aspect=3));


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df_new=df.copy()
df_new=pd.get_dummies(data=df,columns=["sex","smoker"],drop_first=True)


# In[ ]:


df_new.head()


# In[ ]:


df_new=pd.get_dummies(data=df_new,columns=["region","bmi"])


# In[ ]:


df_new.head()


# In[ ]:


df_new.charges=log(df_new.charges)

sc=StandardScaler()
df_scaled=pd.DataFrame(sc.fit_transform(df_new),columns=df_new.columns,index=df_new.index)

df_scaled.head()


# In[ ]:


X=df_scaled.drop("charges",axis=1)
y=df_scaled["charges"] 


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


lm=sm.OLS(y_train,X_train)
model=lm.fit()
model.summary()


# In[ ]:


X=df_scaled.drop(["charges","region_northwest"],axis=1)
y=df_scaled["charges"] 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lm=sm.OLS(y_train,X_train)
model=lm.fit()
model.summary()


# In[ ]:


X=df_scaled.drop(["charges","region_northwest","bmi_heavy"],axis=1)
y=df_scaled["charges"] 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lm=sm.OLS(y_train,X_train)
model=lm.fit()
model.summary()


# In[ ]:


X=df_scaled.drop(["charges","region_northwest","bmi_heavy","bmi_too_weak"],axis=1)
y=df_scaled["charges"] 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lm=sm.OLS(y_train,X_train)
model=lm.fit()
model.summary()


# In[ ]:


X=df_scaled.drop(["charges","region_northwest","bmi_heavy","bmi_too_weak","bmi_normal"],axis=1)
y=df_scaled["charges"] 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lm=sm.OLS(y_train,X_train)
model=lm.fit()
model.summary()


# In[ ]:


X=df_scaled.drop(["charges","region_northwest","bmi_heavy","bmi_too_weak","bmi_normal","region_northeast"],axis=1)
y=df_scaled["charges"] 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lm=sm.OLS(y_train,X_train)
model=lm.fit()
model.summary()


# In[ ]:


model.params


# In[ ]:


model=LinearRegression()
lin_mo=model.fit(X_train,y_train)
y_pred=lin_mo.predict(X_test)


# In[ ]:


lin_mo.score(X_train,y_train)


# In[ ]:


lin_mo.score(X_test,y_test)


# In[ ]:


r2_score(y_test,y_pred)


# In[ ]:


ax1=sns.distplot(y_test,hist=False)
sns.distplot(y_pred,ax=ax1,hist=False);


# In[ ]:


residuals_plot(model, X_train, y_train, X_test, y_test,line_color="red");


# In[ ]:


prediction_error(model, X_train, y_train, X_test, y_test);


# In[ ]:




