#!/usr/bin/env python
# coding: utf-8

# # PUBG Exploratory Data Analysis
# 
# ![pubg](https://user-images.githubusercontent.com/13174586/48706353-01eaac00-ec22-11e8-9800-5bd0f645ad9f.jpg)

# ### Import required libraries

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import random


# ### Import data

# In[ ]:


data= pd.read_csv("../input/train_V2.csv")


# In[ ]:


#EXPLORATORY DATA ANALYSIS

data.head()


# In[ ]:


#Details of the data type
data.info()


# In[ ]:


data["groupId"].value_counts()
data["matchId"].value_counts()


# In[ ]:


#Dimension of the data
data.shape


# ##### Check if the matchID and Group ID has any correlation with the Win Place % as people might get into an easy lobby or might be grouped with a ***good*** Squad or in Duo

# In[ ]:


df=pd.DataFrame()
df[["groupId","matchId", "winPlacePerc"]]= pd.DataFrame(data[["groupId","matchId", "winPlacePerc"]].copy())

#Converting strings to categorical codes
df["groupId"]= df["groupId"].astype('category').cat.codes
df["matchId"]= df["matchId"].astype('category').cat.codes
df["winPlacePerc"]= df["winPlacePerc"].astype('category').cat.codes

#Check Correlation
df.corr()

#The correlation matrix does not show any relation


# In[ ]:


list(data.columns.values)


# #### Dropping "Id", "groupId", "matchId" from the dataframe

# In[ ]:


data=data.drop(data[["Id", "groupId", "matchId"]], axis=1)
data.columns.values


# In[ ]:


data.head()


# ### Check the correlation matrix

# In[ ]:


cor= data.corr()
cor.style.background_gradient().set_precision(2)


# ### Creating Buckets for the players based on **WinPlacePerc** variable
# #### Quartile 1 - Top 25% of the Players - 75-100 %ile
# #### Quartile 2 - 50-75 %ile
# #### Quartile 3 - 25-50 %ile
# #### Quartile 4 - 0-25 %ile - Noobs :-P

# In[ ]:


data['WinPlaceBucket'] = np.where(data.winPlacePerc >0.75, 'Quartile1', 
                          np.where(data.winPlacePerc>.5, 'Quartile2', 
                                   np.where(data.winPlacePerc>.25, 'Quartile3', 'Quartile4')))
data[["winPlacePerc","WinPlaceBucket"]]


# ### Scatter Plot of all the variables to see their relationships

# In[ ]:


sns.pairplot(data.sample(100), size = 2.5) #hue="WinPlaceBucket")


# In[ ]:


data.columns.values


# ### We checked the correlation matrix above
# ### Using scatter plot for the variables correlated with Winning Percentage to find if there is any linear relationship
# ### Note: Correlation is not causation

# In[ ]:


col=["boosts", "damageDealt", "kills", "killStreaks", "walkDistance", "weaponsAcquired","winPlacePerc"]
sns.pairplot(data[col].sample(100000), size = 2.5, kind="reg", diag_kind="kde")


# ### Check Counts of the palyers belonging to each quartile
# 
# ![flat 550x550 075 f](https://user-images.githubusercontent.com/13174586/48707380-462b7b80-ec25-11e8-96f0-41083b3d7319.jpg)

# In[ ]:


sns.catplot(x="WinPlaceBucket", kind="count", palette="ch:.25", data=data.sort_values("WinPlaceBucket"))


# ### Check the relationship between WalkDistance, SwimDistance, and RideDistance with Kills based on WinPlace Quartile
# ![download](https://user-images.githubusercontent.com/13174586/48707564-e5e90980-ec25-11e8-82e3-e3d36c85cbf0.jpg)
# ![download 1](https://user-images.githubusercontent.com/13174586/48707565-e5e90980-ec25-11e8-8135-ee1bf7e994c0.jpg)
# ![images](https://user-images.githubusercontent.com/13174586/48707568-e681a000-ec25-11e8-8a35-ed3db2a3816e.jpg)
# ![images 1](https://user-images.githubusercontent.com/13174586/48707563-e5507300-ec25-11e8-824e-c439f89f482e.jpg)
# 

# In[ ]:


sns.relplot(x="walkDistance", y="kills", hue="WinPlaceBucket", data=data)
sns.relplot(x="swimDistance", y="kills", hue="WinPlaceBucket", data=data)
sns.relplot(x="rideDistance", y="kills", hue="WinPlaceBucket", data=data)


# #### Plot1: Mostly the top Quartile players prefer camping and kill. They may travel only to move to the safe zone
# #### Plot2: Most of the players avoid swimming as it makes them the most vulnerable to attacks
# #### Plot3:  Similar to plot 1, the players avoid riding to much as they get easily exposed to enemy fires and even their vecicles can explode. They prefer to travel only when they are far away from the safe zone after being deployed or if they are far away from the safe zone and it starts shrinking
# 
# 
# ### Check the relationship between WalkDistance, and RideDistance with Weapons Acquired based on WinPlace Quartile

# In[ ]:


sns.relplot(x="walkDistance", y="weaponsAcquired", hue="WinPlaceBucket", data=data)
sns.relplot(x="rideDistance", y="weaponsAcquired", hue="WinPlaceBucket", data=data)


# #### Generally Top Quartile Players does not focus much on scavenging too weapons and attachments. They require basic components like, good automatic guns, scopes, grenades, smoke. They mostly travel for changing camping locations and move to safe zones. It's all skills that matter
# 
# ### Check the relationship between DBNOs with Weapons Acquired based on WinPlace Quartile and kills with WinPlace Quartile

# In[ ]:


sns.relplot(x="DBNOs", y="weaponsAcquired", hue="WinPlaceBucket", data=data)


# In[ ]:


sns.catplot(x="WinPlaceBucket", y="kills", kind="boxen",
            data=data.sort_values("WinPlaceBucket"))


# In[ ]:


#data.columns.values
data["matchType"].value_counts()


# ### Create bucket for Kills:
# ![unsunggrizzledhornedviper-size_restricted](https://user-images.githubusercontent.com/13174586/48753002-0bbbf000-ecb2-11e8-9194-05b64565682d.gif)
# #### Kill: 0
# #### Kill: 1
# #### Kills: 2
# #### Kills: 3 to 5
# #### Kills: 6 to 10
# #### Kills: 10+

# In[ ]:


data['KillsBucket'] = np.where(data.kills >10, 'Kills:10+', 
                          np.where(data.kills>5, 'Kills:6 to 10', 
                                   np.where(data.kills>=3, 'Kills:3 to 5', 
                                            np.where(data.kills==2, 'Kills:2', 
                                                     np.where(data.kills==1, 'Kill:1', 'Kill:0')))))


# In[ ]:


data["KillsBucket"].value_counts()


# #### The players with 10+ Kilss are Pro they most of the time finish with Win Place Percentage of almost 100%. Similarly players with 6 to 10 Kills gives good fight and also mostly end up in the to quartile with median win place as asound 92%. Players with 0 or 1 kills are the one who mostly fight to survive. These are the kind of players who avoids gunfights and keeps of hiding until they are spotted by enemies or end up in  a very small safe zone  

# In[ ]:


sns.catplot(x="winPlacePerc", y="KillsBucket", kind="boxen",
            data=data.sort_values("KillsBucket"))


# In[ ]:


sns.relplot(x="walkDistance", y="KillsBucket", hue="WinPlaceBucket", data=data)
sns.relplot(x="swimDistance", y="KillsBucket", hue="WinPlaceBucket", data=data)
sns.relplot(x="rideDistance", y="KillsBucket", hue="WinPlaceBucket", data=data)


# ### Relation between Heals and Boosts
# ![maxresdefault](https://user-images.githubusercontent.com/13174586/48753435-f5af2f00-ecb3-11e8-9c7b-45cd1a9ff9e6.jpg)
# #### Heals
# ![download 2](https://user-images.githubusercontent.com/13174586/48753424-ecbe5d80-ecb3-11e8-80f7-1c24b712d60b.jpg)
# #### Boosts
# ![download 3](https://user-images.githubusercontent.com/13174586/48753452-0c558600-ecb4-11e8-8af4-ff2c1811a522.jpg)
# #### Boosts are always required for inflicting more damage to opponents and run faster. It mostly helps in attacks
# #### Heals are required to treat the palyer himself when dealt with wounds or too much damage. We can see drop in win percentage with high heal usage. This typically happens when palyers enter in combat or being spotted while entering safe zone at the near end moments

# In[ ]:


random.seed(120)
sns.pointplot(x="boosts", y="winPlacePerc", data=data.sample(1000), color="maroon")
sns.pointplot(x="heals", y="winPlacePerc", data=data.sample(1000), color="purple")
plt.text(14,0.5,"heals", color="purple")
plt.text(14,0.4,"boosts", color="maroon")
plt.xlabel("Heals/Boosts")
plt.ylabel("Win Place %")


# ### SOLO v DUO v SQUAD
# ![download 4](https://user-images.githubusercontent.com/13174586/48754523-f053e380-ecb7-11e8-9004-c18abe13794f.jpg)
# ![download 5](https://user-images.githubusercontent.com/13174586/48754525-f21da700-ecb7-11e8-8155-6b348b613d49.jpg)
# ![download 6](https://user-images.githubusercontent.com/13174586/48754526-f3e76a80-ecb7-11e8-8104-c5edd52f5ec9.jpg)
# 

# In[ ]:


data['GroupBucket'] = np.where(data.numGroups >50, 'Solo', 
                          np.where(data.numGroups>25 , 'Duo', 'Squad'))
data["GroupBucket"].value_counts()


# In[ ]:


sns.relplot(x="kills", y="winPlacePerc", hue="GroupBucket", kind="line", data=data)


# ### Relation between Vehicle Destroys, Kills and Win Place
# ![images 2](https://user-images.githubusercontent.com/13174586/48754611-4aed3f80-ecb8-11e8-9499-3b782106f5f6.jpg)

# In[ ]:


sns.relplot(x="vehicleDestroys", y="winPlacePerc", kind="line",ci="sd", data=data)
sns.relplot(x="vehicleDestroys", y="kills", kind="line",ci="sd", data=data)
sns.relplot(x="vehicleDestroys", y="kills", kind="line",ci=None, hue="WinPlaceBucket", data=data)


# In[ ]:


data.columns.values


# In[ ]:


data_v2= data.drop(data[["matchDuration", "matchType", "maxPlace", "rankPoints", "WinPlaceBucket",
                        "KillsBucket", "GroupBucket"]], axis=1)


# In[ ]:


data_v2.columns.values


# In[ ]:


data_v2.info()


# ## Building Predictive Models

# ### Regression

# In[ ]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as split


# In[ ]:


train, test= split(data_v2, test_size=0.2, random_state=123)


# In[ ]:


train_y= train["winPlacePerc"].copy()
test_y= test["winPlacePerc"].copy()


# In[ ]:


train_x=train.drop(train[["winPlacePerc"]],axis=1)
test_x=test.drop(test[["winPlacePerc"]],axis=1)


# #### Check for NAN

# In[ ]:


print("train_x: ", np.isnan(train_x).any())
print("train_y: ",np.isnan(train_y).any())
print("test_x: ",np.isnan(test_x).any())
print("test_y: ",np.isnan(test_y).any())


# #### Missing Values in: 
# ### test_Y
# #### Will impute with mean

# In[ ]:


test_y=test_y.fillna(test_y.mean())


# In[ ]:


#Check again
print("test_y: ",np.isnan(test_y).any())


# #### Create Regression model object

# In[ ]:


reg= linear_model.LinearRegression()


# In[ ]:


reg.fit(train_x, train_y)


# In[ ]:


#Predicting the test set
test_y_pred= reg.predict(test_x)


# #### Print Regression Coefficients

# In[ ]:


list(zip(train.columns.values,reg.coef_))


# #### Mean Squared Error

# In[ ]:


mse= mean_squared_error(test_y_pred,test_y)
print("Mean Squared Error: ",mse)


# #### Explained Variance- R-Squared

# In[ ]:


r2_score(test_y_pred,test_y)


# In[ ]:


test_actual= pd.read_csv("../input/test_V2.csv")


# In[ ]:


test_actual.head()


# In[ ]:


test_model= test_actual.drop(test_actual[["Id", "groupId", "matchId", "matchDuration", "matchType", "maxPlace", "rankPoints"]], axis=1)


# In[ ]:


test_model_predict= reg.predict(test_model)


# In[ ]:


test_model_predict


# In[ ]:


op= pd.DataFrame(list(zip(test_actual["Id"], test_model_predict)))


# In[ ]:


op= op.rename(columns={0: 'Id', 1: 'winPlacePerc'})


# In[ ]:


op.head()


# In[ ]:


op.to_csv("sample_submission.csv", encoding='utf-8', index=False)


# In[ ]:




