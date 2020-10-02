#!/usr/bin/env python
# coding: utf-8

# 
# # Logistic Regression Project - Solutions
# 
# this project contains a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad
# 
# ## Import Libraries
# 
# **Import a few libraries you think you'll need (Or just import them as you go along!)**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# In[ ]:


ad_data = pd.read_csv('../input/advertising/advertising.csv')


# **Check the head of ad_data**

# In[ ]:


ad_data.head()


# ** Use info and describe() on ad_data**

# In[ ]:


ad_data.columns # displays column names


# In[ ]:


ad_data.info()


# In[ ]:


ad_data.describe()


# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# In[ ]:


sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


# In[ ]:


pd.crosstab(ad_data['Country'], ad_data['Clicked on Ad']).sort_values( 1,ascending = False).tail(10)


# In[ ]:


ad_data[ad_data['Clicked on Ad']==1]['Country'].value_counts().head(10)


# In[ ]:


ad_data['Country'].value_counts().head(10)


# In[ ]:


pd.crosstab(index=ad_data['Country'],columns='count').sort_values(['count'], ascending=False).head(10)


# It seems that users are from all over the world with maximum from france and czech republic with a count of 9 each.
# 
# 

# ## Check for Missing Values

# In[ ]:


ad_data.isnull().sum()


# **Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column?**

# In[ ]:


type(ad_data['Timestamp'][1])


# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[ ]:


# Extract datetime variables using timestamp column
ad_data['Timestamp'] = pd.to_datetime(ad_data['Timestamp']) 


# In[ ]:


# Converting timestamp column into datatime object in order to extract new features
ad_data['Month'] = ad_data['Timestamp'].dt.month 


# In[ ]:


# Creates a new column called Month
ad_data['Day'] = ad_data['Timestamp'].dt.day     


# In[ ]:


# Creates a new column called Day
ad_data['Hour'] = ad_data['Timestamp'].dt.hour   


# In[ ]:


# Creates a new column called Hour
ad_data["Weekday"] = ad_data['Timestamp'].dt.dayofweek 


# In[ ]:


# Dropping timestamp column to avoid redundancy
ad_data = ad_data.drop(['Timestamp'], axis=1) # deleting timestamp


# we can also  use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column

# In[ ]:


# ad_data['Hour'] = ad_data['timeStamp'].apply(lambda time: time.hour)
# ad_data['Month'] = ad_data['timeStamp'].apply(lambda time: time.month)
# ad_data['Day of Week'] = ad_data['timeStamp'].apply(lambda time: time.dayofweek)


# In[ ]:


ad_data.head()


# ## Visualize Target Variable

# In[ ]:


sns.countplot(x = 'Clicked on Ad', data = ad_data)


# In[ ]:


# Jointplot of daily time spent on site and age 
sns.jointplot(x = "Age", y= "Daily Time Spent on Site", data = ad_data) 


# We can see that more people aged between 30 to 40 are spending more time on site daily.

# In[ ]:


# scatterplot of daily time spent on site and age with clicking ads as hue
sns.scatterplot(x = "Age", y= "Daily Time Spent on Site",hue='Clicked on Ad', data = ad_data) 


# We can see that more people aged between 20 to 40 are spending more time on site daily but less chances of them to click on the ads.

# In[ ]:


# Jointplot of daily time spent on site and age clicking ads as hue
sns.lmplot(x = "Age", y= "Daily Time Spent on Site",hue='Clicked on Ad', data = ad_data) 


# We can see that people that are younger and spends more time on site  click on the ads less and people who are in between 25-55 and spends less time click on the ads more.

# ## Distribution and Relationship Between Variables

# In[ ]:


# Creating a pairplot with hue defined by Clicked on Ad column
sns.pairplot(ad_data, hue = 'Clicked on Ad', vars = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage'],palette = 'rocket')


# Pairplot represents the relationship between our target feature/variable and explanatory variables. It provides the possible direction of the relationship between the variables. We can see that people who spend less time on site and have less income and are aged more relatively are tend to click on ad.

# In[ ]:


plots = ['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage']
for i in plots:
    plt.figure(figsize = (12, 6))
    
    plt.subplot(2,3,1)
    sns.boxplot(data= ad_data, y=ad_data[i],x='Clicked on Ad')
    plt.subplot(2,3,2)
    sns.boxplot(data= ad_data, y=ad_data[i])
    plt.subplot(2,3,3)
    sns.distplot(ad_data[i],bins= 20,)       
    plt.tight_layout()
    plt.title(i)    
    plt.show()
    


# In[ ]:


print('oldest person didn\'t clicked on the ad was of was of:', ad_data['Age'].max(), 'Years')
print('oldest person who clicked on the ad was of:', ad_data[ad_data['Clicked on Ad']==0]['Age'].max(), 'Years')


# In[ ]:


print('Youngest person was of:', ad_data['Age'].min(), 'Years')
print('Youngest person who clicked on the ad was of:', ad_data[ad_data['Clicked on Ad']==0]['Age'].min(), 'Years')


# In[ ]:


print('Average age was of:', ad_data['Age'].mean(), 'Years')


# In[ ]:


fig = plt.figure(figsize = (12,10))
sns.heatmap(ad_data.corr(), cmap='viridis', annot = True) 
# Degree of relationship i.e correlation using heatmap


# >Heatmap gives us better understanding of relationship between each feature. Correlation is measured between -1 and 1. Higher the absolute value, higher is the degree of correlation between the variables. We expect daily internet usage and daily time spent on site to be more correlated with our target variable. Also, none of our explantory variables seems to correlate with each other which indicates there is no collinearity in our data.

# # Data visualization with Time, Days and Month

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
ad_data['Month'][ad_data['Clicked on Ad']==1].value_counts().sort_index().plot(ax=ax[0])
ax[0].set_ylabel('Count of Clicks')
pd.crosstab(ad_data["Clicked on Ad"], ad_data["Month"]).T.plot(kind = 'Bar',ax=ax[1])
#ad_data.groupby(['Month'])['Clicked on Ad'].sum() 
plt.tight_layout()
plt.suptitle('Months Vs Clicks',y=0,size=20)
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
pd.crosstab(ad_data["Clicked on Ad"], ad_data["Hour"]).T.plot(style = [], ax = ax[0])
pd.pivot_table(ad_data, index = ['Weekday'], values = ['Clicked on Ad'],aggfunc= np.sum).plot(kind = 'Bar', ax=ax[1]) # 0 - Monday
plt.tight_layout()
plt.show()


# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!
# 
# You'll have the freedom here to choose columns that you want to train on!

# ** Split the data into training set and testing set using train_test_split**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[ ]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# ** Train and fit a logistic regression model on the training set.**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression(solver='lbfgs')
logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations
# ** Now predict values for the testing data.**

# In[ ]:


predictions = logmodel.predict(X_test)


# ** Create a classification report for the model.**

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


# Importing a pure confusion matrix from sklearn.metrics family
from sklearn.metrics import confusion_matrix

# Printing the confusion_matrix
print(confusion_matrix(y_test, predictions))


# The results from evaluation are as follows:
# 
# Confusion Matrix:
# 
# The users that are predicted to click on commercials and the actually clicked users were 154, the people who were predicted not to click on the commercials and actually did not click on them were 170.
# 
# The people who were predicted to click on commercial and actually did not click on them are 1, and the users who were not predicted to click on the commercials and actually clicked on them are 5.
# 
# We have only a few mislabelled points which is not bad from the given size of the dataset.
# 
# Classification Report:
# 
# From the report obtained, the precision & recall are 0.98 which depicts the predicted values are 98% accurate. Hence the probability that the user can click on the commercial is 0.98 which is a great precision value to get a good model.
# 
# 

# In[ ]:


logmodel.coef_


# In[ ]:




