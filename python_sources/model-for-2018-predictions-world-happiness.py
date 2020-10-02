#!/usr/bin/env python
# coding: utf-8

# # Introductory Analysis of World Happiness Dataset (For Beginner readers)
# ##           What i am Analyzing and What i am trying to predict ?
# 
# 1. I am Analyzing the Correlation between the GDP and Happiness Score     
# 2. I am Training Linear Regression Model with the dataset of 2016 HAPPINESS Records
# 
# --------->Characteristics or features Used are::- GDP,Corruption,Life Expectancy  
# ---------->Label:- Happiness Score
# 
# 3. Then using this model to predict the Happiness Score of 2017
# 4. Then cross checking the accuracy of Model as i already have 2017 Happiness Records with the predicted values of the Happiness Score of 2017 by the Regression Model
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import sqrt
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/2017.csv')
df.head()


# In[ ]:


#List of columns in the dataset
df.columns


# * ## *Check for Null Values

# In[ ]:


print(df.isnull().any())


# ## *PLOT For the relation between the Happiness.Score and Economy..GDP.per.Capita*

# In[ ]:


#First checking the correlation between the two values
print('Correlation between Happiness Score and GDP:',df['Happiness.Score'].corr(df['Economy..GDP.per.Capita.']))


# ##This score shows that they are highly correlated 
# ### *Using Scatterplot to visualize the correlation*
# 

# In[ ]:


happiness_score = df['Happiness.Score']
Economy_GDP = df['Economy..GDP.per.Capita.']

plt.scatter(happiness_score,Economy_GDP)
plt.title('Correlation between Happiness and GDP')
plt.xlabel('Happiness Score')
plt.ylabel('Economy GDP')
plt.show()


# # Highly Positively Correlated :
# ### **RESULT**:-Represent that the country having High Economy GDP have more Happiness Score

# # Now Analyze FREEDOM and LIFE EXPECTANCY   

# In[ ]:


freedom = df['Freedom']
Life_expectacny = df['Health..Life.Expectancy.']
print('Correlation between Freedom and Life Expectancy:',freedom.corr(Life_expectacny))


# ## This show freedom is not significantly relates with Life Expectancy

# ![](http://)## Plot for the Corruption VS Happiness in a country

# In[ ]:


corruption_data = df['Trust..Government.Corruption.']
happiness_score = df['Happiness.Score']


# In[ ]:


ax1 = plt.subplot2grid((1,1),(0,0))
ax1.scatter(corruption_data,happiness_score,label='Corruption VS Happiness')
ax1.plot(corruption_data,happiness_score,label='Corruption VS Happiness')
plt.xlabel('Corruption')
plt.ylabel('HAppiness')
plt.plot()
plt.show()


# ## Result:- Initially Graph shows they are correlated and then in the end the correlation decrease 

# # Generating a Regression Model:

# ## Details of the ML Algorithm Working
# <i>Happiness Data from Year 2016 is used to train the model and then we use dataset from year 2017 to test and see the predictive results ;)</i>

# In[ ]:


df_2016 = pd.read_csv('../input/2016.csv')
#df_2016.head()
df_2016.columns


# In[ ]:



# USING GDP,Life Expectancy,Corruption as the features and Happiness SCore as Label we train linear Regression Model
features = ['Trust (Government Corruption)','Health (Life Expectancy)','Economy (GDP per Capita)','Country']
temp_dataframe = df_2016[features]
temp_dataframe = temp_dataframe.sort_values('Country')
#temp_dataframe

del temp_dataframe['Country']
X = temp_dataframe
X


# In[ ]:


# Label for Regression MOdel TRaining
label = 'Happiness Score'
y = df_2016[label]


# # Lets Train the Linear Regression Model

# In[ ]:


regressor = LinearRegression()
model = regressor.fit(X,y)
print(model) #this shows model is an object of type LinearRegression 


# # Lets get the Test Data

# In[ ]:


#DataFrame of 2017 doesn't have Region Column so we drop it from DataFrame of 2016 also
del df_2016['Region']
df.columns = df_2016.columns
df


# In[ ]:


temp_dFrame = df.sort_values('Country')
temp_dFrame


# In[ ]:


new_features = ['Trust (Government Corruption)','Health (Life Expectancy)','Economy (GDP per Capita)']
train_x = temp_dFrame[new_features]
train_y = temp_dFrame[label]
prediction = regressor.predict(train_x)


# In[ ]:


temp_df = pd.DataFrame()
temp_df['Original Happiness Score'] = train_y
temp_df['Predicted Happiness Score'] = prediction 
temp_df


# In[ ]:


temp_df['Country'] = df_2016['Country']


# ## Lets Check the Accuracy of the Trained Linear Regression Model

# In[ ]:


mean_squared_error_value = sqrt(mean_squared_error(y_true=train_y,y_pred=prediction))
mean_squared_error_value


# ##  *Result:-Error is quite Big* 
# ## *Conclusion* :- Model require more improvement 
# ## *Comments can help Thanks !*

# In[ ]:




