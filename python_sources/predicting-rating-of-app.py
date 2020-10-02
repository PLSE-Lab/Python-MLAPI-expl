#!/usr/bin/env python
# coding: utf-8

# ****Importing the Necesary Libraries****

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[ ]:


data=pd.read_csv("googleplaystore.csv")
data.head(5)


# In[ ]:


data.shape


# There are 10841 rows and ****13**** columns in the dataset.

# **Summary Statistics**

# In[ ]:


data.describe()


# **Exploratory Data Analysis**

# In[ ]:


data.boxplot()
plt.show()


# In[ ]:


data.hist()
plt.show()


# In[ ]:


data.info()


# > ****Data Cleaning****

# In[ ]:


data.isnull().sum()


# > There are Missing values in the attributes(Rating, Type, Content Rating, Current Ver and Android Ver.

# ** How many ratings are more than 5- (Outliers)**

# In[ ]:


data[data.Rating>5]


# In[ ]:


data.drop([10472],inplace=True)
data[10470:10475]


# In[ ]:


data.boxplot()
plt.show()


# In[ ]:


data.hist()
plt.show()


# > ** Remove columns that are >80% empty**

# In[ ]:


threshold= len(data)*0.1
threshold


# In[ ]:


data.dropna(thresh=threshold,axis=1,inplace=True)
print(data.isnull().sum())


# > **Data Imputation and Manipulation**

# In[ ]:


def impute_median(series):
    return series.fillna(series.median())


# In[ ]:


data["Rating"]= data["Rating"].transform(impute_median)


# In[ ]:


data.isnull().sum()


# In[ ]:


# Now imputing the categorical values
data["Type"].fillna(str(data["Type"].mode().values[0]),inplace=True)
data["Current Ver"].fillna(str(data["Current Ver"].mode().values[0]), inplace=True)
data["Android Ver"].fillna(str(data["Android Ver"].mode().values[0]),inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


# Let's convert Price, Reviews and Installs into numerical values
data["Price"]= data["Price"].apply(lambda x:str(x).replace("$","") if "$" in str(x) else str(x))
data["Price"]= data["Price"].apply(lambda x: float(x))
data["Reviews"]= pd.to_numeric(data["Reviews"], errors="coerce")

data["Installs"]= data["Installs"].apply(lambda x: str(x).replace("+","") if "+" in str(x) else str(x))
data["Installs"]= data["Installs"].apply(lambda x: str(x).replace(",","") if "," in str(x) else str(x))
data["Installs"]=data["Installs"].apply(lambda x: float(x))


# > **Data Visualization**

# ****Distribution of ratings of apps****

# In[ ]:


data["Rating"]= pd.to_numeric(data["Rating"],errors="coerce")
sns.distplot(data["Rating"],kde=True);


# * The distribution is highly skewed towards the right which implies that there are lesser apps that have low rating.
# * The distribution has high kurtosis indicating many of the apps are having a rating around 4.1-4.5

# # App with Maximum number of reviews

# In[ ]:


data.loc[data["Reviews"].idxmax()]


# # **Find out the Highest rated App**

# In[ ]:


data.loc[data["Rating"].idxmax()]


# # **Analysing the pricing on apps based on content rating**

# In[ ]:


data.loc[data["Installs"].idxmax()]


# # **App which hasn't been updated**

# In[ ]:


data.iloc[0]['App']


# # **Most Popular Category**

# In[ ]:



data["Category"].value_counts()[:10].sort_values(ascending=True).plot(kind="barh")
plt.show()


# In[ ]:


sns.boxplot(x="Content Rating",y="Rating",hue="Type",data=data)
plt.show()


# All paid apps have a better rating in the content rating.

# In[ ]:


sns.lmplot("Reviews","Rating",data=data,hue="Type",fit_reg=False,palette="Paired",scatter_kws={"marker":"D","s":100})
plt.show()


# Free apps have more reviews as compare to paid apps.

# # **Paid Vs Free App**

# In[ ]:


labels=["Free","Paid"]
d=[data["Type"].value_counts()[0],data["Type"].value_counts()[1]]
fig1,ax1=plt.subplots()
ax1.pie(d,labels=labels,shadow=True)
ax1.axis("equal")
plt.show()


# > Predict rating of app using Linear Regression

# Preparing the data

# In[ ]:


X=data[["Reviews","Price"]]
y=data.Rating

X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=42)


# In[ ]:


# Lets bring the dataset features into same scale
scaler=StandardScaler()
X= scaler.fit_transform(X)


# In[ ]:


from sklearn.linear_model import LinearRegression
lin_r= LinearRegression()
model= lin_r.fit(X_train,y_train)


# In[ ]:


print(model.intercept_)
print(model.coef_)


# In[ ]:


rating= model.predict(np.array([[1000,3]]))
print("Predicted rating is:",rating)


# In[ ]:


y=model.intercept_ +(1000*model.coef_[0]+2*model.coef_[1])
print("Rating is:",y)


# In[ ]:


pred= model.predict(X_test)
pred


# Test Evaluation

# In[ ]:


from sklearn.metrics import mean_squared_error
print("RMSE: {}".format(np.sqrt(mean_squared_error(y_test,pred))))

