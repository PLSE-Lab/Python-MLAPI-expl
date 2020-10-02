#!/usr/bin/env python
# coding: utf-8

# # Summary
# **The aim of this project is training a model which is used for car price prediction based on the performance and structure of cars and doing some further analysis about the features which are important or confusing for me.**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Case 1: Car Price Prediction

# ## 1 Loding the Data

# In[ ]:


car = pd.read_csv("/kaggle/input/car-price/CarPrice_Assignment.csv")


# In[ ]:


car.shape


# In[ ]:


car


# ## 2 EDA

# ### a. Brief check

# In[ ]:


car.info()


# In[ ]:


car["price"] = car["price"].astype(int)


# ### b. Check the missing value

# In[ ]:


car.isnull().sum()


# There is no missing value in this dataset

# ### c. Handle the object data

# I use both LabelEncoder and map to change the string to number.  
# 1. When the labels have no relationship to the number, I use LabelEncoder 
# 2. When the meaning of the labels is exactly related to the number, I use map.  
# Take "doorsnumber" as an example.

# In[ ]:


car["fueltype"].value_counts().plot.bar()
plt.show()


# In[ ]:


from sklearn import preprocessing
encoder = preprocessing.LabelEncoder().fit(car["fueltype"])
car["fueltype"] = encoder.transform(car["fueltype"])


# In[ ]:


car["fueltype"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["aspiration"].value_counts().plot.bar()
plt.show()


# In[ ]:


encoder = preprocessing.LabelEncoder().fit(car["aspiration"])
car["aspiration"] = encoder.transform(car["aspiration"])


# In[ ]:


car["fueltype"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["doornumber"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["doornumber"] = car["doornumber"].map({"four":4,"two":2})
car["doornumber"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["carbody"].value_counts().plot.bar()
plt.show()


# In[ ]:


encoder = preprocessing.LabelEncoder().fit(car["carbody"])
car["carbody"] = encoder.transform(car["carbody"])
car["carbody"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["drivewheel"].value_counts().plot.bar()
plt.show()


# In[ ]:


encoder = preprocessing.LabelEncoder().fit(car["drivewheel"])
car["drivewheel"] = encoder.transform(car["drivewheel"])
car["drivewheel"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["enginelocation"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["enginelocation"] = car["enginelocation"].map({"front":1,"rear":2})
car["enginelocation"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["enginetype"].value_counts().plot.bar()
plt.show()


# In[ ]:


encoder = preprocessing.LabelEncoder().fit(car["enginetype"])
car["enginetype"] = encoder.transform(car["enginetype"])
car["enginetype"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["cylindernumber"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["cylindernumber"] = car["cylindernumber"].map({"four":4,"two":2,"six":6,"five":5,"eight":8,"three":3,"twelve":12})
car["cylindernumber"].value_counts().plot.bar()
plt.show()


# In[ ]:


car["fuelsystem"].value_counts().plot.bar()
plt.show()


# In[ ]:


encoder = preprocessing.LabelEncoder().fit(car["fuelsystem"])
car["fuelsystem"] = encoder.transform(car["fuelsystem"])
car["fuelsystem"].value_counts().plot.bar()
plt.show()


# In[ ]:


car.info()


# **Becaue I want to analyze the relationship between car price and their performance and structure, I don't need "CarName".**

# In[ ]:


car_noname = car.drop("CarName",axis=1)
car_noname


# ### d. Correlation 

# In[ ]:


plt.subplots(figsize=(25,25))
ax = plt.axes()
corr = car_noname.corr()
sns.heatmap(corr)


# In[ ]:


pd.set_option('display.max_columns',None)


# In[ ]:


corr


# **curbweight and engine size are strongest correlated with price**

# use scatter matrix to show the relationship

# In[ ]:


plt.scatter(car_noname["curbweight"],car_noname["price"])
plt.xlabel("curbweight")
plt.ylabel("price")


# In[ ]:


plt.scatter(car_noname["enginesize"],car_noname["price"])
plt.xlabel("enginesize")
plt.ylabel("price")


# * **"enginesize" and "horsepower" both have high correlation with price**  
# * **the ratio between "horsepower" and "enginesize" shows the horsepower every unit enginesize provides,   
# which is related to the quality of engine**  
# 
# **if this ratio is highly related to the price**
Build a new feature "hp_es"(= horsepower/enginesize)  
Analyse the relationship between hp_es and price.
# In[ ]:


import statsmodels.formula.api as smf
car_noname.eval('hp_es = horsepower / enginesize',inplace = True)
results = smf.ols('price ~hp_es',data=car_noname).fit()
results.summary()


# In[ ]:


corr = car_noname.corr()
corr


# **The relationship is weak, it seems that car price is more dependant on the performance of overall engine**

# ## 3 Modeling and Prediction

# ### a. Choose the features which have the correlation more than 0.30

# In[ ]:


car1 = car_noname[["price","drivewheel","enginelocation","wheelbase","carlength","carwidth","curbweight","cylindernumber",
                   "enginesize","fuelsystem","boreratio","horsepower","citympg","highwaympg"]]


# In[ ]:


car1.shape


# In[ ]:


car1.head(5)


# In[ ]:


features1 = car1.columns.drop('price')


# In[ ]:


train1 = car1
train1.shape


# In[ ]:


train1_features = train1.drop("price",axis=1)
train1_target = train1["price"]
print(train1_features.shape,train1_target.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
import eli5
X_train1,X_test1,Y_train1,Y_test1 = train_test_split(train1_features,train1_target,
                                                 test_size=0.2,shuffle=True,random_state = 133)
print(X_train1.shape,Y_train1.shape,X_test1.shape,Y_test1.shape)


# Pick the data of 10 cars randomly for prediction display

# In[ ]:


testfeatures1 = X_test1.sample(n=10)
testdata1 = pd.merge(testfeatures1,Y_test1,left_index=True,right_index=True)
testdata1


# Use different model  
# 1. **RandomForestRegressor**  
# 2. **SVR**  
# 3. **KNeighborsRegressor**

# In[ ]:


#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
RFR1 = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=133).fit(X_train1, Y_train1)
score1 = RFR1.score(X_test1,Y_test1)
score1


# In[ ]:


#SVR
from sklearn.svm import SVR
l_svr = SVR(kernel='linear')
l_svr.fit(X_train1,Y_train1)
l_svr.score(X_test1,Y_test1)


# In[ ]:


#KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(weights="uniform")
knn.fit(X_train1,Y_train1)
knn.score(X_test1,Y_test1)


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=111)
cv_results1 = cross_val_score(estimator=RFR1, X=train1_features, y=train1_target, cv=kf, scoring='r2', n_jobs=-1).mean()
cv_results2 = cross_val_score(estimator=l_svr, X=train1_features, y=train1_target, cv=kf, scoring='r2', n_jobs=-1).mean()
cv_results3 = cross_val_score(estimator=knn, X=train1_features, y=train1_target, cv=kf, scoring='r2', n_jobs=-1).mean()
print(cv_results1,cv_results2,cv_results3)


# According to the performance of each model in Cross-validation, I choose RFR.

# Do the prediction

# In[ ]:


prediction = RFR1.predict(testfeatures1)
output = pd.DataFrame({"price":testdata1["price"],"prediction":prediction})
output


# I wanna see how does the prediction model perform concretely.  
# Use mean_absolute_error to show the accuracy of the model and take it as a factor of evaluation.

# In[ ]:


from sklearn.metrics import mean_absolute_error
predicts1 = RFR1.predict(X_test1)
mae1 = mean_absolute_error(Y_test1,predicts1)
mae1


# Feature importance

# In[ ]:


from eli5.sklearn import PermutationImportance
perm = PermutationImportance(RFR1, random_state=123).fit(X_train1, Y_train1)
eli5.show_weights(perm, feature_names = features1.tolist(), top=30)


# ### b. Choose the features which have the correlation more than 0.1

# In[ ]:


car2 = car_noname[["price","aspiration","drivewheel","enginelocation","wheelbase","carlength","carwidth","carheight","curbweight",
                   "cylindernumber","enginesize","fuelsystem","boreratio","horsepower","fueltype","citympg","highwaympg","hp_es"]]
features2 = car2.columns.drop("price")
train2_features = car2.drop("price",axis=1)
train2_target = car2["price"]
print(train2_features.shape,train2_target.shape)


# In[ ]:


X_train2,X_test2,Y_train2,Y_test2 = train_test_split(train2_features,train2_target,
                                                 test_size=0.2,shuffle=True,random_state = 133)
print(X_train2.shape,Y_train2.shape,X_test2.shape,Y_test2.shape)


# In[ ]:


RFR2 = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=133).fit(X_train2, Y_train2)
score2 = RFR2.score(X_test2,Y_test2)
score2


# In[ ]:


l_svr = SVR(kernel='linear')
l_svr.fit(X_train2,Y_train2)
l_svr.score(X_test2,Y_test2)


# In[ ]:


knn = KNeighborsRegressor(weights="uniform")
knn.fit(X_train2,Y_train2)
knn.score(X_test2,Y_test2)


# In[ ]:


predicts2 = RFR2.predict(X_test2)
mae2 = mean_absolute_error(Y_test2,predicts2)
mae2


# In[ ]:


from eli5.sklearn import PermutationImportance
perm = PermutationImportance(RFR2, random_state=123).fit(X_train2, Y_train2)
eli5.show_weights(perm, feature_names = features2.tolist(), top=30)


# ### c. Choose all features

# In[ ]:


car3 = car_noname
features3 = car3.columns.drop("price")
train3_features = car3.drop("price",axis=1)
train3_target = car3["price"]
X_train3,X_test3,Y_train3,Y_test3 = train_test_split(train3_features,train3_target,
                                                 test_size=0.2,shuffle=True,random_state = 133)
print(X_train3.shape,Y_train3.shape,X_test3.shape,Y_test3.shape)


# In[ ]:


RFR3 = RandomForestRegressor(n_estimators=200, max_depth=3, random_state=133).fit(X_train3, Y_train3)
score3 = RFR3.score(X_test3,Y_test3)
score3


# In[ ]:


predicts3 = RFR3.predict(X_test3)
mae3 = mean_absolute_error(Y_test3,predicts3)
mae3


# In[ ]:


from eli5.sklearn import PermutationImportance
perm = PermutationImportance(RFR3, random_state=123).fit(X_train3, Y_train3)
eli5.show_weights(perm, feature_names = features3.tolist(), top=30)


# ## 4 Evaluation

# ### Cross-validation

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=111)
cv_results1 = cross_val_score(estimator=RFR1, X=train1_features, y=train1_target, cv=kf, scoring='r2', n_jobs=-1).mean()
cv_results2 = cross_val_score(estimator=RFR2, X=train2_features, y=train2_target, cv=kf, scoring='r2', n_jobs=-1).mean()
cv_results3 = cross_val_score(estimator=RFR3, X=train3_features, y=train3_target, cv=kf, scoring='r2', n_jobs=-1).mean()
print(cv_results1,cv_results2,cv_results3)


# ### Show the score, MAE, CV_Results of 3 models 

# In[ ]:


model = {'name':['RFR1','RFR2','RFR3'],'score':[score1,score2,score3],"MAE":[mae1,mae2,mae3],"CV_Results":[cv_results1,cv_results2,
                                                                                                           cv_results3]}
model_df = pd.DataFrame(model)
model_df


# ### 5 Conclution
# **1. Accoring to the chart, RFR3 has the best score and performance in Cross-validation.Although the MAE of RFR3 is a little bit bigger than others, I think RFR3 is the most suitable model.**   
# 
# **2. Enginesize is one of the most important factors of car price.**

# # Case 2: Further Analysis

# Use KMeans to do the clustering and show the mean value of each features in each cluster.

# In[ ]:


from sklearn.cluster import KMeans
car1_noprice = car1.drop("price",axis=1)
km = KMeans(n_clusters=5).fit(car1_noprice)
car1_noprice['cluster'] = km.labels_
car1_noprice.sort_values('cluster')
cluster_centers = km.cluster_centers_
car1_noprice.groupby("cluster").mean()


# ## 1 Why curbweight, enginesize, horsepower are strong correlated with price

# **"curbweight","enginesize","horsepower" are strong and positive correlated with price(over 0.80). And according to the cluster, curbweight, cylindernumber, enginesize, horsepower have the more clear difference among the clusters.**

# In[ ]:


from pandas.plotting import scatter_matrix 
centers = car1_noprice.groupby("cluster").mean().reset_index
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
colors = np.array(['red','green','blue','yellow','purple'])


# Use scatter matrix to show the relationship between features.

# In[ ]:


scatter_matrix(car1_noprice[["curbweight","cylindernumber","enginesize","horsepower"]],
               s=50,alpha=1,c=colors[car1_noprice["cluster"]],figsize=(10,10))


# In[ ]:


corr = car1_noprice[["curbweight","cylindernumber","enginesize","horsepower"]].corr()
corr


# ### Conclusion

# * **It makes sense that the higher horsepower a car have, the higher price it will cost.**
# * **It's obvious that enginesize is highly correlated with the performance of a car.Besides, the correlations between cylindernumber and enginesize, horsepower and enginesize are strong and positive, which indicates that enginesize is a overall indicator of a engine.**
# * **We all know that bigger cars are more likely to be heavier. Actually, according to the scatter matrix, curbweight also highly depends on the enginesize. These two factors are the main reasons why curbweight has the strong correlation with price.**  
# * **More powerful engine means more cost the automakers pay for it, which leads to the high price**
# * **If we have a dataset including more features of cars, the features related to the engine should be highly paid attention to.Maybe they are the key to improve the car price prediction model**

# We can use 3D scatter matrix to show a more clear correlation among these three features.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
xs = car1_noprice["curbweight"]
ys = car1_noprice["horsepower"]
zs = car1_noprice["enginesize"]
ax.scatter(xs,ys,zs,c=colors[car1_noprice["cluster"]],s=20)
ax.set_xlabel('curbweight')
ax.set_ylabel('horsepower')
ax.set_zlabel('enginesize')
plt.show()


# ## 2 Why highwaympg and citympg are negative correlated with price

# MPG:  MILE PER GALLON

# **The highwaympg and citympg are negative correlated with price.**  
# **When I know this fact, I'm confused. Because I think everyone hopes the fuel consumption of the car they buy is low in order to pay less for the fuel.Actually automakers will not raise the price because of less fuel consumption.**

# In[ ]:


scatter_matrix(car1_noprice[["curbweight","enginesize","horsepower","highwaympg","citympg"]],
               s=50,alpha=1,c=colors[car1_noprice["cluster"]],figsize=(10,10))


# ### Conclusion

# **According to the scatter matrix, high mpg -> less powerful engine -> lower price.**  
# **The automakers need to pay for the cost of engine but not the cost of fuel**
# 
# **It also reflect an intersting social phnomenon that the rich like buying premium car which has high fuel comsumption, but they don't need to worry about the payment for the fuel. The situation is quite the opposite for the poor.**
# 

# ## 3 About hp_es

# **hp_es shows weak relationship with the price. But I wonder if it can be the factor of the price for the cars in the same cluster because it shows the efficiency of the engine.**

# In[ ]:


car11 = car_noname[["price","drivewheel","enginelocation","wheelbase","carlength","carwidth","curbweight","cylindernumber",
                   "enginesize","fuelsystem","boreratio","horsepower","citympg","highwaympg","hp_es"]]
km = KMeans(n_clusters=5).fit(car11)
car11['cluster'] = km.labels_
car11.sort_values('cluster')
cluster_centers = km.cluster_centers_
car11.groupby("cluster").mean()


# In[ ]:


from pandas.plotting import scatter_matrix 
centers = car1_noprice.groupby("cluster").mean().reset_index
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10
colors = np.array(['red','green','blue','yellow','purple'])
plt.scatter(car11["hp_es"],car11["price"],c=colors[car11["cluster"]])
plt.xlabel("hp_es")
plt.ylabel("price")


# ### Conclusion

# * **Actually, hp_es still cannot become an important features of the car price even in the same cluster. That's to say, even though some expensive cars have the powerful engine, the efficiency of their engines may be lower than that of cheaper car. **  
# * **Automakers are not willing to invest much time and money in improving the effciency of engine.**
# * **If I'm a motor traders, I will pay more attention to overall engine performance and the cost than the efficiency of the engine while considering which type of cars to lay in.**
# 

# ## 4 Class Division

# In[ ]:


car11.groupby("cluster").mean()


# **We can make a brief class division of cars according to this chart.**
