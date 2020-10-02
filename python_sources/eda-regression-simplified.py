#!/usr/bin/env python
# coding: utf-8

# Hi all !<br> This is my THIRD Kernel but for me the most experience gaining one....
# I hope, it will be able to make the **Beginners** of **DATA SCIENCE** comfortable with small dataset and they can learn to apply different **Machine Learning** algorithms on **Regression** problem.

# **<font size="4">Importing Libraries</font>**

# In[ ]:


import numpy as np
# Scientific computing library
import pandas as pd 
# DataFrame dealing Library
import seaborn as sns
# Graphical/Plotting Library
import matplotlib.pyplot as plt
# Graphical/Plotting Library
get_ipython().run_line_magic('matplotlib', 'inline')


# **<font size="4">Reading Dataset</font>**

# In[ ]:


data=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
data.head()
# First 5 rows of the dataset


# <font size="4">From the above table, we can derive one easy thing......Think what will be that ????<br></font>
# **<font size="4">Answer</font>**<font size="4"> - No need of 1st Column, i.e, Serial No.</font>

# **<font size="4">Dropping Column/s</font>**

# In[ ]:


new_data=data.drop('Serial No.',axis=1)
# above one is 1st way
# new_data=data.iloc[:,1:9] <-- Skip 1st column, this is 2nd way
new_data.head()


# **<font size="4">Describe</font>**<font size="4"> is used to see the </font>*<font size="4"> STATISTICS</font>*<font size="4"> of the data </font>

# In[ ]:


new_data.describe()
# You can see "mean" of everything and that will be shown in countplot afterwards too!


# **<font size="4">Missing Values</font>** <font size="4">can interrupt our Analysis, Lets look if there exist NA values or not....</font>

# In[ ]:


new_data.isnull().sum(axis=0)


# **<font size="4">Correlation</font>** <font size="4">between different features in the Data should be taken into account now....</font>

# In[ ]:


f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(new_data.corr(),annot=True)


# <font size="4">We can see that </font>**<font size="4">GRE Score,TOEFL Score,CGPA</font>** <font size="4">are highly correlated with </font>**<font size="4">Chance of Admit</font>**<br>
# <font size="4">But we cannot ignore other variables as they are also having High Correlation(>0.5) too.</font>

# **<font size="4">Barplots</font>**

# In[ ]:


plt.subplots(figsize=(20,4))
sns.barplot(x="GRE Score",y="Chance of Admit ",data=data)
plt.subplots(figsize=(25,5))
sns.barplot(x="TOEFL Score",y="Chance of Admit ",data=data)
plt.subplots(figsize=(20,4))
sns.barplot(x="University Rating",y="Chance of Admit ",data=data)
plt.subplots(figsize=(15,5))
sns.barplot(x="SOP",y="Chance of Admit ",data=data)


# <font size="4">Why </font>**<font size="4">PIE</font>** <font size="4">chart for only "Research" feature....????<br>
#     **<font size="4">Answer</font>**<font size="4">- Pie chart, as explained by number of Data Visualizations Experts are confusing for features having more than three categories.</font>

# In[ ]:


temp_series = new_data.Research.value_counts()
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html
labels = (np.array(temp_series.index))
# https://docs.scipy.org/doc/numpy-1.15.0/user/basics.creation.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.index.html
sizes = (np.array((temp_series/temp_series.sum())*100))
# calculating %ages
colors = ['Pink','SkyBlue']
plt.pie(sizes,labels = labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90)
# https://www.commonlounge.com/discussion/9d6aac569e274dacbf90ed61534c076b#pie-chart
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html
plt.title("Research Percentage")
plt.show()


# **<font size="4">Countplots</font>**

# In[ ]:


plt.subplots(figsize=(20,4))
sns.countplot(x="GRE Score",data=data)
plt.subplots(figsize=(25,5))
sns.countplot(x="TOEFL Score",data=data)
plt.subplots(figsize=(20,4))
sns.countplot(x="University Rating",data=data)
plt.subplots(figsize=(15,5))
sns.countplot(x="SOP",data=data)


# **<font size="4">TABLEAU VIZ</font>**

# <img src="https://imgur.com/7e9mpS7.jpg" width="1000px"/>

# <font size="4">Now, this </font>*<font size="4">VIZ</font>* <font size="4">easily makes us understand that the problem is of </font>**<font size="4">LINEAR REGRESSION</font>**

# **<font size="4">Seperating Independent and Dependent Variables</font>**

# In[ ]:


X=new_data.iloc[:,:7]
y=new_data["Chance of Admit "]


# In[ ]:


print(X.shape)
print(y.shape)
X.head()


# **<font size="4">Splitting into Test and Training Sets</font>**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)


# **<font size="4">Importing and Fitting Linear Regression Model</font>**

# In[ ]:


from sklearn.linear_model import LinearRegression
#Linear Regression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred


# **<font size="4">Checking Prediction Values</font>**

# In[ ]:


from sklearn.metrics import mean_absolute_error,r2_score
print("R2 score ",r2_score(y_pred,y_test))
print("mean_absolute_error ",mean_absolute_error(y_pred,y_test))


# **<font size="4">This Notebook will be updated soon, as we will walk through different algorithms of Regression on it.</font>**<br>
#     <font size="4">Also one can FORK this notebook for further algorithms implementation.<br>
#     Happy Learning :)
# </font>
