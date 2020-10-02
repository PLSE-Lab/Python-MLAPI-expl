#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing the Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[ ]:


#Loading the dataset
data = pd.read_csv("../input/insurance.csv")
data.head()


# In[ ]:


#Checking the summary of the dataset
data.describe()


# In[ ]:


#Checking for Null values in the dataset
data.isnull().sum()


# We can see that there is no **Null value** in our dataset.

# In[ ]:


# Making copy of the dataset
df = data.copy()


# There are 3 categorical features : **"sex","smoker","region"** in our dataset. So we will encode these categorical features.

# In[ ]:


#Encoding the features
from sklearn.preprocessing import LabelEncoder
#smoker
labelencoder_smoker = LabelEncoder()
df.smoker = labelencoder_smoker.fit_transform(df.smoker)
#sex
labelencoder_sex = LabelEncoder()
df.sex = labelencoder_sex.fit_transform(df.sex)
#region
labelencoder_region = LabelEncoder()
df.region = labelencoder_region.fit_transform(df.region)


# After encoding our dataset looks like this.

# In[ ]:


df.head()


# Finding the correlation between **charges** and  other features and arranging them in increasing order.

# In[ ]:


df.corr()['charges'].sort_values()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(df.charges,bins = 10,alpha=0.5,histtype='bar',ec='black')
plt.title("Frequency Distribution of the charges")
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()


# The above plot represents the ditribution of the medical charges which tells us
# how many patients spend how much money on treatment on average.

# In[ ]:


sns.boxplot(x=data.region,y=data.charges,data=data)
plt.title("Medical charges per region")
plt.show()


# The above boxplot shows money spent by people on their** treatment** in different **regions**. From the plot we can say that the region doesn't have much impact on medical charges.

# In[ ]:


sns.boxplot(x=data.smoker,y=data.charges,data=data)
plt.title("Medical charges of Smokers and Non-Smokers")
plt.show()


# The boxplot between the **medical charges** of **smokers** and **non-smokers** conveys that those who smoke spend around 4 times more on medicines or treatment as comparerd to those who don't smoke.

# In[ ]:


f = plt.figure(figsize=(12,5))
ax = f.add_subplot(121)
sns.distplot(df[df.smoker==1]['charges'],color='c',ax=ax)
ax.set_title('Medical charges for the smokers')

ax = f.add_subplot(122)
sns.distplot(df[df.smoker==0]['charges'],color='b',ax=ax)
ax.set_title('Medical charges for non-smokers')
plt.show()


# These plots represents the distribution of **medical charges** for the **smokers** and **non-smokers**.

# In[ ]:


sns.boxplot(x=data.sex,y=data.charges,data=data)
plt.title("Charges by Gender")
plt.show()


# The boxplot between **sex** and **charges** shows that there is no **gender biasing** with the medical charges . It doesn't matter whether you are a male or a female the charges remains same for all.

# In[ ]:


plt.subplot(1,2,1)
sns.distplot(df[df.smoker==1]['age'],color='red')
plt.title("Distribution of Smokers")

plt.subplot(1,2,2)
sns.distplot(df[df.smoker==0]['age'],color='green')
plt.title("Distribution of Non-Smokers")
plt.show()


# These plots represents the distribution of **smokers** and **non-smokers** with in accordance with their **age**.

# In[ ]:


sns.lmplot(x="bmi",y='charges',hue='smoker',data=data)


# Medical charge increases in case of **smoker** with the increasing **bmi**. But in case of **non-smokers** the increase in **bmi** doesn't have large impact on the medical charges.

# **Visualizing the regression model **

# In[ ]:


sns.lmplot(x='age',y='charges',hue='smoker',data=data,palette='inferno_r')


# In the case of non-smokers, the cost of treatment increases with age, while in case of the  smokers, there is not do such dependence. We can also see that age has positive correlation and has larger affect after the smoker that leads to increase in the charges. And it is also a common phenomenon that with increasing age medical expenses will increase whether you are a smokeror not.

# Now we will predict the cost of treatment on the basis of the given features.
# We will start with **regression** , in the beginning we are considering **all the features**.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Considering all the features

X = df.iloc[:,:6].values
Y = df.iloc[:,6].values

#Splitting the dataset into train and test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

# Training
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

print(regressor.score(X_test,Y_test))


# In the second model we are using **regression** but not considering **region** due to its **negative correlation** with medical** charges**.

# In[ ]:


#Not considering region
X1 = df.iloc[:,[0,1,2,3,4]].values
Y1 = df.iloc[:,6].values

X_train,X_test,Y_train,Y_test = train_test_split(X1,Y1,test_size=0.25)

#Training
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

print(regressor.score(X_test,Y_test))


# Now, we will check the result using **Polynomial regression**.

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

poly_reg  = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_poly,Y,test_size=0.25)

lin_reg = LinearRegression()
lin_reg  = lin_reg.fit(X_train,Y_train)

print(lin_reg.score(X_test,Y_test))


# This is a good result as compared to the previous models.
# 

# In[ ]:


pred_train = lin_reg.predict(X_train)
pred_test = lin_reg.predict(X_test)

plt.scatter(pred_train,pred_train - Y_train,label='Train data',color='mediumseagreen')
plt.scatter(pred_test,pred_test-Y_test,label="Test data",color='darkslateblue')
plt.legend(loc = 'upper right')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.show()


# Our model is able to predict good results for the medical charges of the people.
# 

# **Thank you**

# In[ ]:




