#!/usr/bin/env python
# coding: utf-8

# # Advertising data
# I will be using mock advertising data in order to demonstrate data cleaning, exploration and application. In order to complete the data wrangling stage, I will use the libraries numPy and pandas. For the data visualisations I will use the libraries seaborn, matplotlib and plotly. 
# During the exploration stage, I will explore the relationships between features of each customer and infer meaning from the trends. I will then use this analylsis to build a clear picture and understanding of the context of the data.
# During the application stage, I will implement a logistic regression model to predict whether a customer clicked on an Ad or not based on the features of the customer.

# #### Import the required libraries 

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Read the data into a pandas dataframe

# In[ ]:


data = pd.read_csv('../input/online_ads_dummy_data.csv')


# In[ ]:


data.head()


# 
# # Data Cleaning 
#  

# #### The data types of each column seem to be in the correct format except for the Timestamp column.
#  - We must change the data type of the Timestamp column to DataTime in order to extract information efficiently.
#  - We must change the Male column to data type 'int' in order to use it as binary in our Logistic Regression Model.

# In[ ]:


data.info()


# In[ ]:


#Converting Timestamp column to 'datetime' Dtype
import datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
#Converting Male column to 'int' Dtype
data['Male'] = data['Male'].apply(lambda x: int(x))
data.info()


# #### We can also see that there are no empty values in this dataset.

# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# #### Create columns for to add extra features for the time

# In[ ]:


def get_hour(x):
    return x.hour

days = dict(enumerate(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))

def get_day(x):
    return days[x.weekday()]

months = dict(enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']))

def get_month(x):
    return months[x.month-1]


# In[ ]:


data['Month'] = data['Timestamp'].apply(lambda x:get_month(x))
data['Day'] = data['Timestamp'].apply(lambda x:get_day(x))
data['Hour'] = data['Timestamp'].apply(lambda x:get_hour(x))


# In[ ]:


data.head(4)


# # Exploratory Data Analysis

# #### Identifying initial trends.

# In[ ]:


sns.pairplot(data, hue='Clicked on Ad')


# In[ ]:


data.corr()[data.corr()>0.3]


# In[ ]:


sns.heatmap(data.corr(),cmap=sns.color_palette("BuGn_r"))


# #### Overall, we can see that there is quite a weak correlation between variables, but the most correlated variables are:
#     - 'Daily Internet Usage' and 'Daily Time Spent on Site'
#     - 'Area Income' and 'Daily Time Spent on Site'
#     - 'Area Income' and 'Daily Internet Usage'
#     - 'Age' and 'Clicked on Ad'

# #### Average characteristics of those who click on the Ads
#  - On average, we can see that for a person who clicks on an Ad:
#      - Clicks on the Ad at around midday.
#      - They are likely to be a middle-aged female.
#      - They are earning around $50,000.
#      - Daily, they spend 2-3 hours online and 53 minutes on the site.

# In[ ]:


data.groupby('Clicked on Ad').mean()


# ## Graphical representation of relationships between variables

# #### Looking at the relationship between 'Daily Internet Usage' and 'Daily Time Spent on Site':
# We can see that there is a positive correlation between someone's internet usage and the amount of time that they spend on the site. This makes sense because the two variables are likely dependent on each other.
# Additionally, we can see that on average, those who spent less time on the site clicked the Ad.
# This implies that the people who are clicking on the Ad are those that may already know what they want to buy and are not spending as much time browsing the site as others.
# 
# The regression line implies that, in a single day, for every extra minute spent on the internet results in approximately 10 more minutes spent on the site.
#     

# In[ ]:


import plotly.express as px

fig = px.scatter(data, 
                y="Daily Time Spent on Site", 
                x="Daily Internet Usage", 
                title='How daily internet usage affects daily time spent on site.',
                color='Clicked on Ad', 
                trendline='ols'
                )
fig.show()


# #### Looking at the relationship between 'Area Income' and 'Daily Time Spent on Site':
# We can see that there is a positive correlation between the average person's income and the amount of time that they spend on the site.
# This makes sense because it is likely that a person in an area with a higher average income, will have more money to spend on the site.
# Additionally, we can see that on average the areas that earned less on average were more likely to have clicked the Ad.
# This implies that the product in the Ad may not have been appealing to the higher end of the market.
# 
# The regression line implies that an increase of $1000 in the average area income results in an extra 20 seconds spent on the site.

# In[ ]:


px.scatter(data, 
              x = 'Area Income',
              y = 'Daily Time Spent on Site',
              color='Clicked on Ad', 
              trendline='ols',
              title = 'How Area income effects the daily amount of time that someone spends on the site.'
             )


# #### Looking at the relationship between 'Area Income' and 'Daily Internet Usage':
# We can see that there is a positive correlation between the average person's income and the amount of time that they spend on the internet. This makes sense because, like outlined above, a person of higher income is likely to spend more time shopping online. 
# 
# The regression line implies that an increase in average area income of $1000 results in an extra minute spent online.
#     

# In[ ]:


px.scatter(data, 
              x = 'Area Income',
              y = 'Daily Internet Usage',
              color='Clicked on Ad', 
              trendline='ols',
              title = 'How Area income effects the daily amount of time that someone spends on the site.'
             )


# #### Looking at the relationship between 'Hour' and 'Age':
# We can see that as the day progresses, the average age stays approximately constant with a few decreases. Most of the decreases occur between the times 12:00 and 21:00. This may be because the visitors are at leisure points of the day such as: lunchtime, after-work, dinner, late-evening etc.
# Additionally, we can see that on average, those that clicked on the Ad were older than those that didn't.
# This implies that the Ad may have targeted the older age group and does not appeal to a younger audience.
# 
# The regression line indicates that on the average age of people using the site decreases very gradually throughout the day however this change is almost negligable.

# In[ ]:


by_hour = data[['Age','Hour']].groupby('Hour').mean()
sns.barplot(data = by_hour,
            x = by_hour.index,
            y='Age',
            color='green',
           ).set_title('Average age of site visitor per hour')


# In[ ]:


px.scatter(data, x='Hour',
           y='Age', 
           color='Clicked on Ad',
           size_max=5, 
           trendline='lowess', 
           title = 'Age of visitors to the site during each hour of the day.')


# # Logistic Regression Analysis

# ### Correctly formatting the categorical data

# #### Dealing with categorical columns

# In[ ]:


month_dummies = pd.get_dummies(data['Month'],drop_first=True)


# In[ ]:


day_dummies = pd.get_dummies(data['Day'],drop_first=True)


# In[ ]:


hour_dummies = pd.get_dummies(data['Hour'],drop_first=True)


# In[ ]:


age_dummies = pd.get_dummies(data['Age']>30,drop_first=True,prefix='>30')


# In[ ]:


data.head(3)


# In[ ]:


categories = pd.concat([month_dummies,day_dummies,hour_dummies,age_dummies, data[['Male','Clicked on Ad']]],axis=1)
categories


# ### Instantiating, training and using the model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = categories.drop('Clicked on Ad',axis=1)
y = data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train,y_train)


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:


error = lm.predict(X_test)-y_test


# In[ ]:


false_negative = 0
correct = 1
false_positive = 0

for i in error:
    if i == -1:
        false_negative+=1
    elif i==0:
        correct+=1
    else:
        false_positive+=1
        
print(f'false negative predictions:\t {false_negative} out of {len(error)} predictions.\n'
     +f'\nfalse negative prediction percentage:  {"{:.2f}".format((100*false_negative)/len(error))}%\n')
print(f'\ncorrect predictions:\t {correct} out of {len(error)} predictions.\n'
     +f'\ncorrect prediction percentage:  {"{:.2f}".format((100*correct)/len(error))}%\n')
print(f'\nfalse positive predictions:\t {false_positive} out of {len(error)} predictions.\n'
     +f'\nfalse positive predictions:  {"{:.2f}".format((100*false_positive)/len(error))}%')
      

