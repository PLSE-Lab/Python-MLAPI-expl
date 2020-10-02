#!/usr/bin/env python
# coding: utf-8

# **Intro:**
# It is a kernel calculate the weather with K- Nearest Neighbour algorithm
# 

# **1)**
# In first step we will import libaries and our dataset and we will see first 100 rows of our data also we will drop nan and infinity values from our dataframe
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("../input/did-it-rain-in-seattle-19482017/seattleWeather_1948-2017.csv")
data =data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
print(data.head())


# **2)**
# Second step we will drop DATE values 

# In[ ]:


data.drop(["DATE"],axis = 1,inplace = True)

print(data.head())


# **3)**
# Now we will make plotting TMAX column will be our x label and TMIN column will be our y label and of course Trues and Falses in RAIN column our values
# 

# In[ ]:


t = data[data.RAIN == True]
f = data[data.RAIN == False]

plt.figure(1)
plt.scatter(t.TMAX,t.TMIN,color="red",label="TRUE",alpha= 0.3)
plt.scatter(f.TMAX,f.TMIN,color="green",label="FALSE",alpha= 0.3)

plt.xlabel("TMAX")
plt.ylabel("TMIN")
plt.legend()
plt.show()


# **4)**
# we will change True values to 1 and false values to 0 for make easier predictions and define our features(x_data) and results(y) 

# In[ ]:


data.RAIN = [1 if each == True else 0 for each in data.RAIN]
y = data.RAIN.values
x_data = data.drop(["RAIN"], axis = 1)


# **5)**
# It is the time make normalization(convert to values between 0-1 to protect weight coefficients) and split our datas to tests and trains
# 

# In[ ]:


x = (x_data -np.min(x_data))/ (np.max(x_data) - np.min(x_data))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)


# **6)**
# now we create our model with sklearn library

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 2) 
knn.fit(x_train,y_train)

prediction = knn.predict(x_test)


# **7)**
# Lets print our score

# In[ ]:


print("{} nn score {}".format(2,knn.score(x_test,y_test)))


# **8)**
# Finally lets find the best k value for our model and draw figure y label is reliability scores x values are k values between (1-15)

# In[ ]:


score_list=[]
for i in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.figure(2)    
plt.plot(range(1,15),score_list)
plt.ylabel(" reliability scores")
plt.xlabel("k values")
plt.show()


# **Conclusion:**
# We will see on our graph that the best k values for our model very close to 2 and we use 2 it means our model will look 2 nearest neighbours for predict it did rain or not 
