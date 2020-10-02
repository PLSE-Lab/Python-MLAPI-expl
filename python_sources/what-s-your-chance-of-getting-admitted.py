#!/usr/bin/env python
# coding: utf-8

# In this kernel I have performed Exploratory Data Analysis on the Graduate Admissions dataset and tried to identify relationship between a student's admission chance  and various other features. After EDA data pre-processing is done I have applied Linear Regression Algorithm  to make the predictions. I will use various other algorithms for predictions in future and add them in this kernel.

# I hope you find this kernel helpful and some **<font color='red'>UPVOTES</font>** would be very much appreciated

# In[ ]:


import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')


# ### **Importing Required Libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt     #data visualization
import seaborn as sns               #data visualization

get_ipython().run_line_magic('matplotlib', 'inline')


# ### **Reading the Data**

# In[ ]:


df = pd.read_csv('../input/Admission_Predict.csv')
df.head()


# Since the **Serial No.** column is not required, removing it from the data frame

# In[ ]:


df.drop('Serial No.', axis = 1, inplace = True)


# ### **Data Set Summary**

# In[ ]:


print(df.info())


# In[ ]:


df.describe()


# The features described in the above data set are:
# 
# **1. Count** tells us the number of NoN-empty rows in a feature.
# 
# **2. Mean** tells us the mean value of that feature.
# 
# **3. Std** tells us the Standard Deviation Value of that feature.
# 
# **4. Min** tells us the minimum value of that feature.
# 
# **5. 25%, 50%**, and **75%** are the percentile/quartile of each features.
# 
# **6. Max** tells us the maximum value of that feature.

# ### **Exploratory Data Analysis**

# #### **1.Heatmap**

# In[ ]:


plt.figure(figsize = (10,6))
sns.heatmap(df.corr(), annot = True, linewidths=0.5, cmap = 'coolwarm')
plt.show()


# Chances of admit highly depend upon a student's GRE Score, TOEFL Score and CGPA.
# University Ranking and LOR (Letter of Recommendation) are also important factors in one's admission.

# #### **2.Pair Plot**

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(df)
plt.show()


# ### **3. Number of universities on the basis of  University Rating**

# In[ ]:


sns.countplot(df['University Rating'])
plt.show()


# Most of the Universities in the dataset have a rating of 3 in the dataset

# ### **4. Scatter Plot for GRE Score vs. CGPA**

# In[ ]:


sns.lmplot('CGPA','GRE Score', data = df,palette='hls',hue = 'University Rating',fit_reg=False)
plt.show()


# Students who are more focused towards studies and have higher CGPA tend to have a higher GRE Score.<br>
# Also, most of the students from top tier universities(rating 4 & 5) who applied have a CGPA of 8.5 or gre

# ### **5. Scatter Plot for TOEFL Score vs. CGPA**

# In[ ]:


sns.lmplot('CGPA','TOEFL Score', data = df,palette='hls',hue = 'University Rating',fit_reg=False)
plt.show()


# ### **5. Scatter Plot for Chance of Admit vs. CGPA**

# In[ ]:


sns.lmplot('CGPA','Chance of Admit ', data = df,palette='hls',hue = 'University Rating',fit_reg=False)
plt.show()


# ### **6.Plotting TOEFL Score distribution**

# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df['TOEFL Score'], kde = False, bins = 30, color = 'blue')
plt.title('TOEFL Score Distribution')
plt.show()


# ### **7. Plotting GRE Score distribution**

# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df['GRE Score'], kde = False, bins = 30, color = 'red')
plt.title('GRE Score Distribution')
plt.show()


# ### **8. Plotting CGPA Distribution of Students**

# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df['CGPA'], kde = False, bins = 30, color = 'purple')
plt.title('CGPA Distribution')
plt.show()


# ## **Applying Linear Regression Model**

# ## Splitting the dataset into training and testing dataset
#  Let's split the data into training and testing sets. Set a variable X equal to the numerical features of the customers and a variable y equal to the "Chance of Admit" column.

# In[ ]:


X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research']]
y = df['Chance of Admit ']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)


# ### **Training the Model**<br><br>

# In[ ]:


from sklearn.linear_model import LinearRegression
linearmodel = LinearRegression()


# ### **Fit linearmodel on the training data.**

# In[ ]:


linearmodel.fit(X_train,y_train)


# ### **Print out the coefficients of the model**

# In[ ]:


cdf = pd.DataFrame(linearmodel.coef_, X.columns, columns=['Coefficient'])
cdf


# ## Predicting Using Test Data
# 

# In[ ]:


prediction = linearmodel.predict(X_test)


# ### **Create a scatterplot of the real test values versus the predicted values. **

# In[ ]:


plt.figure(figsize=(9,6))
plt.scatter(y_test,prediction)
plt.xlabel('Test Values for y')
plt.ylabel('Predicted Values for y')
plt.title('Scatter Plot of Real Test Values vs. Predicted Values ')
plt.show()


# ## Evaluating the Model
# 

# In[ ]:


from sklearn import metrics
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,prediction))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# **Suggestions are welcome**

# In[ ]:




