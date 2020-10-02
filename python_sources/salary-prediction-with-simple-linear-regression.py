#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regresssion model to predict Salary on the basis Experience(in Years)

# ### Let us first import all the required libraries

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from ipywidgets import interact
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Now let us read/load the dataset and display it .

# In[ ]:


df = pd.read_csv('../input/salary-dataset/Data_salary.csv') #read_csv() is used to read the csv file
df.head() #head() in pandas displays first five rows of the dataset by default


# In[ ]:


df.shape #By shape attribute we can check the shape of the dataset


# In[ ]:


df.isnull().sum() #Let us check for any NaN values in the dataset


# In[ ]:


df = df.rename(columns={'YearsExperience' : 'Experience'}) #Rename the column names in the dataset
df.head()


# In[ ]:


#Initialize the independent and dependent variables

inp = df.iloc[:, :1]  
outp = df.Salary.values


# In[ ]:


#Train the model

trainer = LinearRegression()
trainer.fit(inp, outp)


# In[ ]:


# 'm' and 'c' are the 'slope' and 'intercept' of the best fit line

m = trainer.coef_
c = trainer.intercept_
m,c


# In[ ]:


# Let us now do the prediction and store the predicted values in an array.

prediction = trainer.predict(inp)


# In[ ]:


prediction


# In[ ]:


# Plot the best fit line on the graph

plt.scatter(inp, outp) #plots the scatter plot 
plt.plot(inp, prediction, 'g') #plots the best fit line 


# In[ ]:


#Let us now see the accuracy of the model we just trained.

print(r2_score(outp, prediction))


# In[ ]:


def salary_predict(Experience):
    salary = trainer.predict([[Experience]])
    print("Salary should be : ",salary[0])


# In[ ]:


#Lets create an user-friendly scroller to predict the salary
#Note : If the scroller doesn't appear, try running the code in jupyter notebook

interact(salary_predict, Experience=(0,50))


# In[ ]:




