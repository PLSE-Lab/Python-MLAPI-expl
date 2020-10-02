#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries and Reading the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os


# In[ ]:


df=pd.read_csv('../input/data-house-price/train.csv')
print(df.head())


# In[ ]:


print(df.info())


# In[ ]:


print(df.columns)


# In[ ]:


#descriptive statistics summary
df['SalePrice'].describe()


# In[ ]:


#histogram
sns.distplot(df['SalePrice']);


# In[ ]:


fig,ax=plt.subplots(figsize=(30,30))
sns.heatmap(df.corr(),ax=ax,annot=True,linewidths=0.05,fmt='.2f',cmap="magma")
plt.show()


# In[ ]:


features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
df= df[features]
df.head()


# In[ ]:


print("There are",len(df.columns),"columns:")
for x in df.columns:
    sys.stdout.write(str(x)+",")


# In[ ]:


df.info()


# In[ ]:


fig,ax=plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(),ax=ax,annot=True,linewidths=0.05,fmt='.2f',cmap="magma")
plt.show()


# In[ ]:


df["YearBuilt"].plot(kind='hist',bins=200,figsize=(6,6))
plt.title("YearBuilt")
plt.xlabel("YearBuilt")
plt.ylabel("LotArea")
plt.show()


# In[ ]:


plt.scatter(df["YearBuilt"],df["1stFlrSF"])

plt.title("YearBuilt")
plt.xlabel("YearBuilt")
plt.ylabel("1stFlrSF")
plt.show()


# In[ ]:


plt.scatter(df["YearBuilt"],df["2ndFlrSF"])
plt.title("YearBuilt")
plt.xlabel("YearBuilt")
plt.ylabel("2ndFlrSF")
plt.show()


# In[ ]:


df[df['FullBath'] >0].plot(kind='scatter',x='FullBath',y='2ndFlrSF',color='red')
plt.xlabel("FullBath")
plt.ylabel("2ndflrSF")
plt.title("FullBath >0")
plt.grid(True)
plt.show()


# In[ ]:


s = df[df["FullBath"] >1]["BedroomAbvGr"].value_counts().head(5)
plt.title("BedroomAbvGr")
s.plot(kind='bar',figsize=(10,10))
plt.xlabel("BedroomAbvGr")
plt.ylabel("FullBatch")
plt.show()


# In[ ]:


plt.scatter(df["LotArea"],df.TotRmsAbvGrd)
plt.xlabel("Lotarea")
plt.ylabel("TotRmsAbvGrd")
plt.title("Lotarea for TotRmsAbvGrd")
plt.show()


# In[ ]:


plt.scatter(df["LotArea"],df.BedroomAbvGr)
plt.xlabel("Lotarea")
plt.ylabel("BedroomAbvGr")
plt.title("Lotarea for BedroomAbvGrd")
plt.show()


# ## <a id='regression'> REGRESSION ALGORITHMS </a>

# ### <a id='prepareForRegression'>Preparing Data for Regression</a>

# In[ ]:


import pandas as pd
#reading the dataset
df=pd.read_csv("../input/data-house-price/train.csv",sep = ",")
# Create target object and call it y
y = df.SalePrice.values
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df[features]


# In[ ]:


#separating train (80%) and test (20%) sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[ ]:


#normalization
from sklearn.preprocessing import MinMaxScaler
scalerX=MinMaxScaler(feature_range=(0,1))
x_train[x_train.columns]=scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns]=scalerX.transform(x_test[x_test.columns])


# ### <a id=' linearRegression'>Linear Regression</a>

# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_head_lr=lr.predict(x_test)

print("real value of y_test[1]:"+str(y_test[1]) + "-> the predict:" +str(lr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(x_test.iloc[[2],:])))

from sklearn.metrics import r2_score
print("r_square score:",r2_score(y_test,y_head_lr))

y_head_lr_train=lr.predict(x_train)
print("r_square score (train dataset):",r2_score(y_train,y_head_lr_train))


# ### <a id ="randomForestRegression">Random Forest Regresssion</a>

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=100,random_state=42)
rfr.fit(x_train,y_train)
y_head_rfr=rfr.predict(x_test)

from sklearn.metrics import r2_score
print("r_square score:",r2_score(y_test,y_head_rfr))
print("real value of y_test[1]:" +str(y_test[1])+"-> the predict:"+str(rfr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]:" +str(y_test[2])+"-> the predict:"+str(rfr.predict(x_test.iloc[[2],:])))

y_head_rf_train=rfr.predict(x_train)
print("r_square score (train dataset):",r2_score(y_train,y_head_rf_train))


# ### <a id="DecisionTreeRegression">Decision Tree Regression </a>

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=42)
dtr.fit(x_train,y_train)
y_head_dtr=dtr.predict(x_test)

from sklearn.metrics import r2_score
print("r_square score:",r2_score(y_test,y_head_dtr))
print("real value of y_test[1]:" +str(y_test[1])+ "-> the predict" +str(dtr.predict(x_test.iloc[[1],:])))
print("real value of y_test[2]:" +str(y_test[2])+ "-> the predict"+ str(dtr.predict(x_test.iloc[[2],:])))

y_head_dtr_train=dtr.predict(x_train)
print("r_square score (train dataset):",r2_score(y_train,y_head_dtr_train))


# ### <a id='comparisonOfRegression'>Comparison of Regression Algorithms</a>
# 
# * Linear regression and random forest regression algorithms were better than decision tree regression algorithm.

# In[ ]:


y = np.array([r2_score(y_test,y_head_lr),r2_score(y_test,y_head_rfr),r2_score(y_test,y_head_dtr)])
x=["LinearRegression","RandomForestReg","DecisionTreeReg."]
plt.bar(x,y)
plt.title("Comparision of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()


# # Creating a Model For the Competition
# 
# Build a Random Forest model and train it on all of **X** and **y**.  

# In[ ]:


import pandas as pd
#reading the dataset
df=pd.read_csv("../input/data-house-price/train.csv",sep = ",")
# Create target object and call it y
y = df.SalePrice.values
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = df[features]


# In[ ]:


# To improve accuracy, create a new Random Forest model which you will train on all training data
rfr_full_data =RandomForestRegressor(random_state=0)

# fit RandomForest_model_on_full_data on all data from the training data
rfr_full_data.fit(X,y)


# ## <a id="makepred">Make Predictions</a>
# Read the file of "test" data. And apply the model to make predictions

# In[ ]:


test_data=pd.read_csv("../input/data-house-price/test.csv",sep = ",")


# In[ ]:


features=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']


# In[ ]:


test_X = test_data[features]


# In[ ]:


# make predictions which we will submit. 
test_preds=rfr_full_data.predict(test_X)
test_preds


# In[ ]:


output = pd.DataFrame({'Id': test_data.Id,
                     'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
#print(output.shape)

