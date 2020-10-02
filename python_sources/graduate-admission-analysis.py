#!/usr/bin/env python
# coding: utf-8

# # Overview of Data

# ###  Importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Importing the dataset

# In[ ]:


df = pd.read_csv('../input/Admission_Predict.csv')
df.columns = df.columns.to_series().apply(lambda x: x.strip())


# ### A quick glance at the data

# In[ ]:


df.head()


#  Finding number of rows and columns in the dataset

# In[ ]:


df.shape


# This means there are 400 rows and 9 columns in the dataset. Now let's drop the null values from the dataset

# In[ ]:


df = df.dropna()
df.shape


# This means there are no null values in the dataset. Let's see some basic information about the data features

# In[ ]:


df.describe()


# Analysing the corelation between the different features

# In[ ]:


plt.figure(figsize=(10, 5))
p = sns.heatmap(df.corr(), annot=True)


# # Visualisation of Data

# Now by visualisation of data, let's try to find out how these features affect the chances of admission

# In[ ]:


p = sns.pairplot(df)


# Now let's find out the number of applicants with research papers

# In[ ]:


p = sns.countplot(x="Research", data=df)


# Now the plot below shows that most of the applicants are from universities with rating 3

# In[ ]:


p = sns.countplot(x="University Rating", data=df)


# Here we are analysing how different features affect chance of admission

# In[ ]:


p = sns.lineplot(x="GRE Score", y="Chance of Admit", data=df)
_ = plt.title("GRE Score vs Chance of Admit")


# In[ ]:


p = sns.lineplot(x="TOEFL Score", y="Chance of Admit", data=df)
_ = plt.title("TOEFL Score vs Chance of Admit")


# In[ ]:


p = sns.lineplot(x="University Rating", y="Chance of Admit", data=df)
_ = plt.title("University Rating vs Chance of Admit")


# In[ ]:


p = sns.lineplot(x="CGPA", y="Chance of Admit", data=df)
_ = plt.title("CGPA vs Chance of Admit")


# # Predictions

# Now using sklearn let's make our predictions

# ### Importing the libraries

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 


# ### Preparing the dataset

# In[ ]:


df = pd.read_csv("../input/Admission_Predict.csv")
df.columns = df.columns.to_series().apply(lambda x: x.strip())
serialNo = df["Serial No."].values
df.drop(["Serial No."], axis=1, inplace=True)

df = df.rename(columns = {'Chance of Admit': 'Chance of Admit'})

y = df["Chance of Admit"].values
x_data = df.drop(["Chance of Admit"], axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=123)


# Using Linear regression we will predict the chances of admission

# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


predictions = model.predict(x_test)


# In[ ]:


mean_squared_error(y_test, predictions)


# ## Conclusion

# We can conclude that GRE , TOEFL Scores and CGPA have a significant effect on the chances of getting admission
