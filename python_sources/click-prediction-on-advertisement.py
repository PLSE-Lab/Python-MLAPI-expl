#!/usr/bin/env python
# coding: utf-8

# 
# ### Model that will predict whether or not they will click on an ad based off the features of that user
# This data set contains the following features:
# 
# dd
# 1. 'Daily Time Spent on Site': consumer time on site in minutes
# 2. 'Age': cutomer age in years
# 3. 'Area Income': Avg. Income of geographical area of consumer
# 4. 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# 5. 'Ad Topic Line': Headline of the advertisement
# 6. 'City': City of consumer
# 7. 'Male': Whether or not consumer was male
# 8. 'Country': Country of consumer
# 9. 'Timestamp': Time at which consumer clicked on Ad or closed window
# 9. 'Clicked on Ad': 0 or 1 indicated clicking on Ad

# In[ ]:


# Importing the libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing the dataset

# In[ ]:


ad_data = pd.read_csv('../input/advertising.csv')
ad_data.head(3)


# # Discription of the dataset

# In[ ]:


ad_data.info()


# In[ ]:


ad_data.describe()


# # Exploratory Data Analysis

# In[ ]:


sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(ad_data, hue="Clicked on Ad", size=6)    .map(plt.scatter, "Age", "Area Income")    .add_legend()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(ad_data, hue="Clicked on Ad", size=6)    .map(plt.scatter, "Age", "Daily Time Spent on Site")    .add_legend()


# people who spend less time daily clicked on AD

# In[ ]:


#sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data)
sns.set_style("whitegrid");
sns.FacetGrid(ad_data, hue="Clicked on Ad", size=6)    .map(plt.scatter, "Daily Internet Usage", "Daily Time Spent on Site")    .add_legend()


# people with less daily internet usage mostly click on the ad

# In[ ]:


sns.pairplot(ad_data,hue='Clicked on Ad')


# # Splitting the data

# In[ ]:


from sklearn.model_selection import train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=455)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# # Predictions and Evaluations

# In[ ]:


y_pred = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# In[ ]:




