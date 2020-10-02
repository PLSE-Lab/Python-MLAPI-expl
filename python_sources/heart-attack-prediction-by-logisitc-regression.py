#!/usr/bin/env python
# coding: utf-8

# # UPLOADING THE DATASET

# **I have implemented Logistic Regression on the dataset. This is my first ever written kernel and I hope that it will be valuable to others. The tasks I have done will be described as the code goes on.**

# Libaries are imported for the Code
# 

# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv("../input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv")
df.head()


# **The total missing values from the dataset are calculated and shown on the screen.**

# In[ ]:


df.isnull().sum()


# **This shows the CigsPerDay column and displays all the Null values in the column.**

# # CLEANING OF THE DATASET

# In[ ]:


series = pd.isnull(df['cigsPerDay'])
df[series]


# The advantage of above table shown is that the 'currentSmoker' column has value of 1 for all 'cigsPerDay' null values. This helps in conveying that all missing values of cigsPerDay would not be zero.

# * **As Education does not play a factor in Heart Attack, I have dropped that column**
# * **Moreover, I dropped currentSmoker column too as cigsPerDay column already denotes that the person is a Smoker, that's just extra information which is not useful**

# In[ ]:


data = df.drop(['currentSmoker','education'], axis = 'columns')
data.head()


# In[ ]:


cigs = data['cigsPerDay']
cigs.head()


# **I have calculated the mean of people smoking a cigaratte per day**

# In[ ]:


cig = cigs.mean()


# I rounded that value to the neartest integer

# In[ ]:


import math
integer_value = math.floor(cig)
integer_value


# **I filled the null values in 'cigsPerDay' with the mean of the Cigarattes smoked by a person in a day.**

# In[ ]:


cigs.fillna(integer_value, inplace = True)


# In[ ]:


data.isnull().sum()


# **I dropped the rest of the rows of the null values**

# In[ ]:


data.dropna( axis = 0, inplace = True)


# **This shows that there are no Null value anymore in the DataFrame**

# In[ ]:


data.isnull().sum()


# # ANALYZING THE DATASET

# In[ ]:


data.shape


# **Created a separate DataFrame for the people having chances of heart attack**

# In[ ]:


Heart_Attack = data[data.TenYearCHD == 1]
Heart_Attack.head()


# **Created another separate DataFrame for people having low chances of heart attack**

# In[ ]:


No_Heart_Attack = data[data.TenYearCHD == 0]
No_Heart_Attack.head()


# **This groups the data on the basis of 'TenYearCHD' indicating the overall dependence of columns on 'TenYearCHD'**

# In[ ]:


data.groupby('TenYearCHD').mean()


# **As 'diaBP', 'BMI', 'heartRate' have values in the similar zone and almost equal to one another for 1 or 0 values for 'TenYearCHD', they are dropped.**

# In[ ]:


final = data.drop(['diaBP','BMI','heartRate'], axis = 'columns')


# In[ ]:


No_Heart_Attack = final[final.TenYearCHD == 0]
No_Heart_Attack.head()


# In[ ]:


Heart_Attack = final[final.TenYearCHD == 1]
Heart_Attack.head()


# In[ ]:


final.groupby('TenYearCHD').mean()


# # PREPARING THE MODEL

# In[ ]:


X = final[['male','age','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','glucose']]


# In[ ]:


y = final['TenYearCHD']


# In[ ]:


X


# In[ ]:


y


# **I have split the model in 20-80 test and train size with a given random state so that the output doesnt differ everytime I execute the program**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 99)


# **Logistic Regression is implemented**

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.score(X_test,y_test)


# **The model is 85.5% accurate. If I change the random state, it still gives the output in the range of 83-86%.**

# In[ ]:





# In[ ]:





# In[ ]:




