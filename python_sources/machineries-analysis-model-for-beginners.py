#!/usr/bin/env python
# coding: utf-8

# ## Data Science Structure:-
# #### 10 step logical performance for any datascience project
# 1. Data Exploration,Domain understanding and Data collection
# 2. Feature Engineering
#    a. Feature Extraction
#    b. Feature Selection
# 4. Preprocessing
# 5. Apply Machine learning algorithm 
# 6. Performance Analysis
# 7. Optimisation and Tuning
# 8. Export optimised model
# 9. Deployment to Production
# 10. Monitoring Perfomance in production

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#load data
df = pd.read_excel(r"../input/analysis-of-breaking-of-machineries/Combined_Cycle_powerplant.xlsx")
df.shape


# ## 1. Data Exploration:-

# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.info()


# All of the data are as float64 that means no missing value exist in the data .

# In[ ]:


df.describe()


# The percentage value are not that far in the data so it don't have any outlier .

# According to Chevbvey's theorem 68% of data lies between mean-std to mean+std

# ## 2. Data Cleaning:-

# In[ ]:


#Check for missing values
df.duplicated().sum()


# In[ ]:


#Drop duplicates
df.drop_duplicates(inplace=True)


# In[ ]:


#check for missing values
df.isnull().sum()


# ## 3. Feature Engineerring:-
# - Feature Extraction : Extracting relavant data
# - Feature Selection : Selecting relavant data

# In[ ]:


#it comapres all column with all the columns and shows the graph
sns.pairplot(df)
plt.show()


# From the figure it it clear that all the attributes are having some specific relation with each of the attribute so 
# here all the attribute are equally important . We do not need to rremove any column for further processing

# In[ ]:


cor=df.corr()
plt.figure(figsize=(9,7))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()


# Linear REgression can be used when the attributes are having some linear relationship between each other,
# so first we have to check the correlation values  . If the attributes are having a good correlation value,
# then applying llinear regression algorithm it will give us an effective result .

# In[ ]:


#Separate features and label
x = df[['AT','V','AP','RH']]
y = df[['PE']]


# ## 4. Preprocessing:-

# In[ ]:


# split data into train and test
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.2)
# we have to split the data into 80% as train and 20% as test so we have specified test_size as 0.2
print(x.shape)
print(xtr.shape)
print(xts.shape)
print(y.shape)
print(ytr.shape)
print(yts.shape)


# ## 5.Applying ML Algorithm:-
# - Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[ ]:


#train the model with the training data
model.fit(xtr,ytr)


# ## 6.Performance Analysis:-

# In[ ]:


new_data=np.array([[13.97,39.16,1016.05,84.6]])
model.predict(new_data)


# In[ ]:


#get prediction of xts
ypred = model.predict(xts)


# In[ ]:


#calculating r2score
from sklearn.metrics import r2_score
r2_score(yts,ypred)


# In[ ]:


#To find the error
from sklearn.metrics import mean_squared_error
mean_squared_error(yts,ypred)


# We got to know that our model is giving 92% of accurate value . So we are happy with our model so no potimisation
# tuning is required in this case .

# ## 7. Export model as a portable file - pickle file

# In[ ]:


import joblib
#from sklearn.externals import joblib
joblib.dump(model,r"C:\Users\HP\Desktop\p.practice\ML AI COURSE\ccpp_model.pkl")


# In[ ]:




