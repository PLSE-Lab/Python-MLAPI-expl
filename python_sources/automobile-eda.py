#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis and Regression Notebook

# Import the necessary libraries.

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy import stats
import pandas_profiling as pp
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv("/kaggle/input/automobile-dataset/Automobile_data.csv")
df.head()


# In[ ]:


# numbers of columns and rows in dataset.
df.shape


# In[ ]:


# Name of the columns of the dataset.
df.columns


# In[ ]:


# Datatypes of every column for the dataset.
df.dtypes


# In[ ]:


# Information(no of rows and columns, datatypes for the columns , null values in dataframe memory usage) about the dataset.
df.info()


# **Data cleaning**

# In[ ]:


#replace ? with the nan
df.replace("?",np.nan,inplace =True)
df.head()


# In[ ]:


df.info()


# In[ ]:


# sum of null values in every columns.
df.isnull().sum()


# In[ ]:


# no of duplicated rows in data frames
df.duplicated().value_counts()


# In[ ]:


#Count the unoque values in num-of-doors
Counter(df["num-of-doors"])


# Replace null values with the mean of the feature for the numeric variables and for categorical variables replace null value with the maximum count.

# In[ ]:


miss_col= ["normalized-losses","bore","stroke","horsepower","peak-rpm","price"]
for col in miss_col:
    df[col].replace(np.nan,df[col].astype("float").mean(axis=0),inplace=True)
    
df["num-of-doors"].replace(np.nan,df["num-of-doors"].value_counts().idxmax(),inplace=True)
df.head().T
    


# In[ ]:


print("Data Types of Variables \n",df.dtypes)


# change the appropriate data type.

# In[ ]:


df[["normalized-losses","bore","stroke","horsepower","peak-rpm","price"]]=df[["normalized-losses","bore","stroke","horsepower","peak-rpm","price"]].astype("float")
df.dtypes


# In[ ]:


pp.ProfileReport(df)


# **Statistical Data discription**

# In[ ]:


#Statistical discription of the data for numerical features.
df.describe()


# In[ ]:


#Statistical discription of the data for categorical features.
df.describe(include='object')


# **Standardisation**

# In[ ]:


df["city-L/100km"]=235/df["city-mpg"]
df["highway-L/100km"]=235/df["highway-mpg"]


# In[ ]:


df.drop(["city-mpg","highway-mpg"],axis=1)


# **Normalization**

# In[ ]:


for col in ["length","width","height"]:
    df[col]=df[col]/df[col].max()
    
df[["length","width","height"]].head()


# In[ ]:



df[["horsepower"]].hist()
plt.show()


# **Binning**

# In[ ]:


df["horsepower_binned"]=pd.cut(df["horsepower"],bins=np.linspace(min(df["horsepower"]),max(df["horsepower"]),4),
                               labels=["low","medeium","high"],include_lowest=True)
df[["horsepower","horsepower_binned"]].head()


# In[ ]:


plt.bar(["low","medeium","high"],df["horsepower_binned"].value_counts())
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("Horsepower Bins")
plt.show()


# In[ ]:


df.hist(bins=3,figsize=(15,12))
plt.tight_layout()


# In[ ]:


df.columns


# **Dummy Variable**

# In[ ]:


dummy_var=pd.get_dummies(df[["fuel-type","aspiration"]])


# In[ ]:


df=pd.concat([df,dummy_var],axis=1)


# In[ ]:


data=df.drop(df[["fuel-type","aspiration"]],axis=1)
data.head()


# In[ ]:


dt=data.corr()
dt[dt["price"]>0.5]


# In[ ]:


dt[dt["price"]<-0.5].T


# In[ ]:


data.describe()


# In[ ]:


data.describe(include='object')


# In[ ]:


data["drive-wheels"].value_counts().to_frame().rename(columns={"drive-wheels":"value_counts"})


# **Groupby Function**

# In[ ]:


group_data=df[["drive-wheels","body-style","price"]].groupby(by=["drive-wheels","body-style"],as_index=False).mean()
group_data


# **Pivot Table**

# In[ ]:


group_data.pivot(index="drive-wheels",columns="body-style")


# ### Correlation and p value

# In[ ]:



cols=["wheel-base","bore","horsepower","length","width","height","curb-weight","engine-size","city-mpg","highway-mpg"]
for i in cols:
    pearson_coef,p_value=stats.pearsonr(df[i],df["price"])
    print("For {} :  pearson coefficient= {} and p value={} ".format(i,pearson_coef,p_value))


# In[ ]:


df_gptest = data[['drive-wheels','body-style','price']]
grouped_test=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test.head()


# In[ ]:


grouped_test.get_group('4wd')['price']


# ### Anova

# In[ ]:


f_val,p_val=stats.f_oneway(grouped_test.get_group('4wd')['price'],grouped_test.get_group('rwd')['price'],grouped_test.get_group('fwd')['price'])
print("F-value is = ",f_val," P-Values is = ",p_val)


# In[ ]:


f_val,p_val=stats.f_oneway(grouped_test.get_group('4wd')['price'],grouped_test.get_group('rwd')['price'])
print("F-value is = ",f_val," P-Values is = ",p_val)


# In[ ]:


f_val,p_val=stats.f_oneway(grouped_test.get_group('4wd')['price'],grouped_test.get_group('fwd')['price'])
print("F-value is = ",f_val," P-Values is = ",p_val)


# In[ ]:


f_val,p_val=stats.f_oneway(grouped_test.get_group('fwd')['price'],grouped_test.get_group('rwd')['price'])
print("F-value is = ",f_val," P-Values is = ",p_val)


#  ## Machine Learning
# 
# ****MODEL DEVELOPMENT****

# ### Linear Regression

# In[ ]:


lm=LinearRegression()
X=df[['highway-mpg']]
Y=df['price']
lm.fit(X,Y)
y_pred=lm.predict(X)
y_pred[0:5]


# In[ ]:


print("intercept:",lm.intercept_)
print("coefficient:",lm.coef_)


# In[ ]:


plt.scatter(Y,y_pred)
plt.title("Predicted price by Linear Regression with one variable (Highway-mpg)",fontsize=15)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()


# In[ ]:


#regression plot
plt.figure(figsize=(8,8))
sns.regplot(x='highway-mpg',y='price',data=df)
plt.title("Regression Plot",fontsize=20)
#plt.ylim(0)
plt.show()

# Residual plot
plt.figure(figsize=(8,8))
sns.residplot(x='highway-mpg',y='price',data=df)
plt.title("Residual Plot",fontsize=20)
plt.show()


# In[ ]:


plt.scatter(Y,y_pred)
plt.title("Relation between Actual and Predicted values",fontsize=15)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()


# In[ ]:


r2_score(Y,y_pred)


# ### Multiple Linear Regression

# In[ ]:


x = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y= df["price"]
mlr=LinearRegression()
mlr.fit(x,y)
pred_mlr=mlr.predict(x)
pred_mlr[0:5]


# In[ ]:


print("intercept:",mlr.intercept_)
print("coefficient:",mlr.coef_)


# In[ ]:


plt.scatter(y,pred_mlr)
plt.title("Predicted price by Multiple Linear Regression with more than one variables",fontsize=15)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()


# In[ ]:


# function to plot the data.
def PollyPlot(model,x,y,name):
    x_new=np.linspace(15,55,100)
    y_new=model(x_new)
    plt.plot(x,y,'.',x_new,y_new,'-')
    plt.xlabel(name)
    plt.ylabel('price of cars')
    plt.show()


# In[ ]:


x=df["highway-mpg"]
y=df['price']
f=np.polyfit(x,y,3)
p=np.poly1d(f)
print(p)


# In[ ]:


PollyPlot(p,x,y,'highway-mpg')


# ### Polynomial Linear Regression
# 
# We can perform a polynomial transform on multiple features. 

# In[ ]:


z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
pf=PolynomialFeatures(degree=2)
pf


# In[ ]:


z.shape


# In[ ]:


z_pf=pf.fit_transform(z)
z_pf


# In[ ]:


plr=LinearRegression()
plr.fit(z_pf,y)
pred_plr=plr.predict(z_pf)


# In[ ]:


plt.scatter(y,pred_plr)
plt.title("Predicted price by Polynomial Linear Regression with more than one variables",fontsize=15)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()


# In[ ]:


plt.scatter(Y,y_pred)
plt.title("Predicted price by Linear Regression with one variable (Highway-mpg)",fontsize=15)
plt.xlabel("Actual_price")
plt.ylabel("Predicted Price")
plt.show()
plt.scatter(y,pred_mlr)
plt.title("Predicted price by Multiple Linear Regression with more than one variables",fontsize=15)
plt.xlabel("Actual_price")
plt.ylabel("Predicted Price")
plt.show()
plt.scatter(y,pred_plr)
plt.title("Predicted price by Polynomial Linear Regression with more than one variables",fontsize=15)
plt.xlabel("Actual_price")
plt.ylabel("Predicted Price")
plt.show()


# In[ ]:


# comparison of  r2 score  for MLR and PLR for same data .
print("Multiple Linear Regression R2 Score:\n",r2_score(y,pred_mlr))
print("Polynomial Regression R2 Score:\n",r2_score(y,pred_plr))

