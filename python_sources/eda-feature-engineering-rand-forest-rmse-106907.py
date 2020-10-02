#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls ../input')
get_ipython().system('ls ../input/*')


# In[ ]:


#importing libraries
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


image = Image.open("../input/kingcountyimage/Screen Shot 2019-10-18 at 1.36.04 PM.png")
#Just a screenshot from Google maps
image = np.array(image)


# In[ ]:


df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
print("Dataset length: " + str(len(df)))
df.head()


# In[ ]:


df.describe()


# In[ ]:


print(df.columns.values)
low = df["price"] < 500000
high = df["price"] > 500000
long = df["long"][high].mean()
lat = df["lat"][high].mean()


# In[ ]:


ids = df["id"]
df.drop(columns=["id"],inplace=True)


# In[ ]:


f,axs = plt.subplots(1,2,figsize=(24,7))
tomato = "#FF6347"
axs[0].scatter(df["long"],df["lat"],alpha=0.05,c=low,cmap="flag")
axs[0].set_title("Dataset Distribution")
axs[0].scatter([[long]],[lat],c="y",s=200)
axs[1].imshow(image)
axs[1].set_title("Map")
plt.show()
#red is high value (above averge) houses


# <h5>Most high value housing are centered around the Redmond region (centered around the yellow point)</h5>

# In[ ]:


get_ipython().run_line_magic('timeit', 'df["dis"] = np.sqrt((df["long"] - long) ** 2 + (df["lat"] - lat) ** 2)')
#cheaper houses are further from the center
df[high]["dis"].mean(), df[low]["dis"].mean()


# In[ ]:


get_ipython().run_line_magic('timeit', 'df["date"] = pd.to_datetime(df["date"],format="%Y%m%dT000000")')
print("Min Date: {}\nMax Date: {}".format(df["date"].min(),df["date"].max()))


# In[ ]:


plt.scatter(df["dis"],df["price"],alpha=0.3,color=tomato)
plt.show()


# In[ ]:


df[["sqft_lot","sqft_living","sqft_lot15","sqft_above","sqft_basement"]].describe()


# In[ ]:


f,axs = plt.subplots(1,2,figsize=(20,8))
axs[0].hist(df["sqft_lot"],color=tomato,bins=50)
axs[0].set_title("sqft_lot")
axs[1].hist(df["sqft_lot15"],color=tomato,bins=50)
axs[1].set_title("sqft_lot15")
plt.show() #(they are definitely not the same thing though)


# In[ ]:


corr_mat = df.corr()
abs(corr_mat["price"]).sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(20,8))
sns.boxplot(df["condition"],df["price"])
plt.title("Condition")
plt.figure(figsize=(20,8))
sns.boxplot(df["floors"],df["price"])
plt.title("Floors")
plt.show()


# In[ ]:


time = df.sort_values(by="date")
plt.figure(figsize=(30,12))
plt.plot(time["date"],time["price"],color=tomato)
plt.show()


# <h5>There doesn't seem to be any direct correlation</h5>

# In[ ]:


#the "dis" feature has a stronger correlation with "price" than longitude or latitude
df = df.sort_values(by="date")
df_c = df.copy()
df.drop(columns=["long","lat","zipcode","yr_built","sqft_lot","sqft_lot15","date"],inplace=True)


# In[ ]:


df.nunique().sort_values(ascending=False)


# In[ ]:


y = df["price"].copy()
df.drop(columns=["price"],inplace=True)
X = df.copy()


# In[ ]:


scaler = RobustScaler()
X = scaler.fit_transform(X)


# In[ ]:


train_X,test_X = X[:-2000],X[-2000:]
train_y,test_y = y[:-2000],y[-2000:]
print("Train size: {}\nTest size : {}".format(train_X.shape,test_X.shape))


# In[ ]:


forest = RandomForestRegressor(min_samples_leaf=3,random_state=42)
forest.fit(train_X,train_y)


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
preds = forest.predict(test_X)
preds_train = forest.predict(train_X)
#This is not a bad result, since the other kernels I looked at only achieved 150000 RMSE
#I did not fine tune the model but I imagine the score will be better after fine tuning
print("Test:\nMSE : {:.5f}\nRMSE: {:.5f}\nMAE : {:.5f}\n".format(mean_squared_error(test_y,preds),mean_absolute_error(test_y,preds),np.sqrt(mean_squared_error(test_y,preds))))
print("Train:\nMSE : {:.5f}\nRMSE: {:.5f}\nMAE : {:.5f}".format(mean_squared_error(train_y,preds_train),mean_absolute_error(train_y,preds_train),np.sqrt(mean_squared_error(train_y,preds_train))))


# <h5>Final Score: 106907 (RMSE)</h5>
