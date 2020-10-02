#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("fivethirtyeight")
import warnings
warnings.filterwarnings("ignore")
import folium
import webbrowser
from folium.plugins import HeatMap


# In[ ]:



#load the house data
housedata=pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv")
housedata.head()


# In[ ]:


housedata.describe().transpose()


# In[ ]:


housedata.isnull().sum()


# In[ ]:


sns.heatmap(housedata.isnull(),yticklabels=False,cbar=False,cmap="viridis")
#we dont have any null values


# In[ ]:


housedata.corr()


# In[ ]:


sns.heatmap(housedata.corr(),cmap='rocket',cbar=True,yticklabels=True)


# In[ ]:


housedata.corr()["price"].sort_values()


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(x="price",y="sqft_living",data=housedata,color="g",palette='viridis')


# In[ ]:


fig=plt.figure(figsize=(12,5))
axis=fig.add_subplot(121)
sns.distplot(housedata['price'],color='g')
plt.ylim(0,None)
plt.xlim(0,2000000)
axis.set_title('distribution of prices')

axis=fig.add_subplot(122)
sns.distplot(housedata['sqft_living'],color='b')
plt.ylim(0,None)
plt.xlim(0,6000)
axis.set_title('distribution of sqft_living')


# In[ ]:


plt.figure(figsize=(10,6))
sns.jointplot(x='sqft_living',y='price',kind='hex',data=housedata)
plt.ylim(0,3500000)
plt.xlim(0,None)


# In[ ]:


plt.figure(figsize=(10,6))
sns.lmplot(x='sqft_living',y='price',palette='viridis',height=7,data=housedata)
plt.title('sqft_living vs price')


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(housedata["bedrooms"])


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(housedata["grade"])


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x='bedrooms',y='price',palette='viridis',data=housedata)
plt.title("bedrooms vs price")


# In[ ]:


fig=plt.figure(figsize=(19,12.5))
ax=fig.add_subplot(2,2,1,projection='3d')
ax.scatter(housedata['floors'],housedata['bedrooms'],housedata['bathrooms'],c="blue")
ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nBathrooms')
ax.set(ylim=[0,12])

ax=fig.add_subplot(2,2,2,projection='3d')
ax.scatter(housedata['price'],housedata['sqft_living'],housedata['bedrooms'],c="green")
ax.set(xlabel='\nprice',ylabel='\nsqt_living',zlabel='\nBedrooms')
ax.set(zlim=[0,12])

ax=fig.add_subplot(2,2,3,projection='3d')
ax.scatter(housedata['floors'],housedata['bedrooms'],housedata['sqft_living'],c="red")
ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nsqft_living')
ax.set(ylim=[0,12])

ax=fig.add_subplot(2,2,4,projection='3d')
ax.scatter(housedata['grade'],housedata['price'],housedata['sqft_living'],c="violet")
ax.set(xlabel='\ngrade',ylabel='\nprice',zlabel='\nsqft_living')
ax.set(xlim=[2,12])


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(x="price",y="long",data=housedata,color="r")


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(x="price",y="lat",data=housedata)


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(x="long",y="lat",data=housedata,hue="price")


# In[ ]:


new_data=housedata.sort_values("price",ascending=False).iloc[200:]
#we are removing outliers


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(x="long",y="lat",data=new_data,hue="price",palette="RdYlGn",alpha=0.2,edgecolor=None)


# In[ ]:


latitude=47.6
longitude=-122.3
dup=housedata.copy()
def worldmap(location=[latitude,longitude],zoom=9):
    map=folium.Map(location=location,control_state=True,zoom_start=zoom)
    return map
fmap=worldmap()
folium.TileLayer("cartodbpositron").add_to(fmap)
HeatMap(data=dup[["lat","long"]].groupby(["lat","long"]).sum().reset_index().values.tolist(),
       radius=8,max_zoom=13,name='Heat Map').add_to(fmap)
folium.LayerControl(collapsed=False).add_to(fmap)
fmap


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x="waterfront",y="price",data=housedata)


# In[ ]:


housedata.drop(['id','zipcode'],axis=1,inplace=True)


# In[ ]:


housedata["date"].head()


# In[ ]:


housedata["date"]=pd.to_datetime(housedata["date"])
housedata["date"].head()


# In[ ]:


housedata["year"]=housedata["date"].apply(lambda date: date.year)
housedata["month"]=housedata["date"].apply(lambda date: date.month)


# In[ ]:


housedata.head()


# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(x="month",y="price",data=housedata)


# In[ ]:


housedata.groupby("month").mean()["price"]


# In[ ]:


plt.figure(figsize=(10,6))
housedata.groupby("month").mean()["price"].plot()


# In[ ]:


plt.figure(figsize=(10,6))
housedata.groupby("year").mean()["price"].plot()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


x=housedata.drop(["price","date"],axis=1).values
y=housedata["price"].values


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=23)


# In[ ]:


scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)


# In[ ]:


model=Sequential()


# 
# RELU :- Stands for Rectified linear unit. It is the most widely used activation function. Chiefly implemented in hidden layers of Neural network.
# 
# Equation :- A(x) = max(0,x). It gives an output x if x is positive and 0 otherwise.

# In[ ]:


def model_creating():
    model=Sequential()
    model.add(Dense(19,activation="relu"))
    model.add(Dense(19,activation="relu"))
    model.add(Dense(19,activation="relu"))
    model.add(Dense(19,activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam",loss="mse")
    return model


# In[ ]:


model=model_creating()


# In[ ]:


model.fit(x=x_train,y=y_train,
          validation_data=(x_test,y_test),
         batch_size=130,epochs=550,verbose=1)


# In[ ]:


model.summary()


# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[ ]:


loss=pd.DataFrame(model.history.history)
loss.head()


# In[ ]:


loss.plot()
#if both lines are coincide then our model is not overfitting
#if we get spikes in our plot then our model is overfitting


# In[ ]:


y_pred=model.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score


# In[ ]:


error=pd.DataFrame([[mean_squared_error(y_test,y_pred),
                     np.sqrt(mean_squared_error(y_test,y_pred)),
                    mean_absolute_error(y_test,y_pred),
                    explained_variance_score(y_test,y_pred)]],
                   columns=["mean_squared_error","mean_squared_root_error",
                                 "mean_absolute_error","explained_variance_score"])
error


# In[ ]:


print(error["mean_absolute_error"],housedata.describe()["price"]["mean"])


# In[ ]:


sample_house=housedata.drop(["price","date"],axis=1).iloc[0].values
sample_house=sample_house.reshape(-1,19)


# In[ ]:


sample_house=scalar.transform(sample_house)


# In[ ]:


sample_predict=model.predict(sample_house)
print(sample_predict,housedata.iloc[0:1,1:2].values)


# In[ ]:


plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred,color="blue",marker="o")
plt.plot(y_pred,y_pred,marker='o',
         color='green',markerfacecolor='red',
         markersize=7,linestyle='dashed')

