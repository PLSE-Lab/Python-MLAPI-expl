#!/usr/bin/env python
# coding: utf-8

# # I am using a dataset which contains information about cars, using this dataset I will try to find if the no of kilometers driven by a car affects its Present price.

# In[ ]:


#Import Relevent Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()


# In[ ]:


#Import dataset 
data=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
x1,y=data['Kms_Driven'],data['Present_Price']
data.describe()


# In[ ]:


#Analyze the dataset
plt.scatter(x1,y)
plt.xlabel('Kms driven')
plt.ylabel('Present Price')
plt.show()


# In[ ]:


#Perform Regression
x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()  #Ordinary Least Squares tries to find the line of best fit
results.summary() #Print summary of your results 


# #### From the above summary we found the line of best fit, which is y=5.9559+x1*452700
# std err=0.675 It shows the accuracy of prediction(Smaller is better)
# p=0.000( p value <0.05 means the intercept is far away from zero)
# R-squared=0.041 (It means how the regression explains variability of the data Ranges from 0 to 1,
#     0 means regression explains none of the variability and
#     1 means regression explains the entire variability 
#     Since the value is very small which means the regression doesn't strongly explain the variability of data
