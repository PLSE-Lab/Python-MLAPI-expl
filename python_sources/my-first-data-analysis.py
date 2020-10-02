#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In this kernel, I will complete data analysis upon the FIFA 19 player dataset. I am an avid soccer fan, so for my first-ever mini-data project, I was immediately pulled to this dataset. Please have a look through my work and thought processes and comment on improvements I can make as I start my personal data science journey.

# In[ ]:


# To begin, I load in the csv-data into a pandas dataframe denoted as'df'
df = pd.read_csv('../input/data.csv')
# To take a peek into our data set
df.head()


# Upon first glance, it can be seen that not all columns are shown, so I desire to see all the features the dataset makes available to us.

# In[ ]:


list(df.columns)


# It is immediately obvious that there are a multitude of characteristics each player is rated on, and I want my data analysis to be focused upon the ones that are seem important to a soccer fan like myself. Some of the key features I want to focus on are Age, Nationality, Overall, Potential, Value, and Wage.

# In[ ]:


print(type(df['Age'][0]))
print(type(df['Nationality'][0]))
print(type(df['Overall'][0]))
print(type(df['Potential'][0]))
print(type(df['Value'][0]))
print(type(df['Wage'][0]))


# When looking at these features, I see that some of them represent numerical values, but don't always represent them in a float or int format. For example, the Wage and Value features are written with the Euro, Thousands, and Millions symbols so instead of being written in a numerical form, they are strings. For making them easier to work with, I devise a function that will clean the data and transform it to a numerical form.

# In[ ]:


def clean_d(string) :
    last_char = string[-1]
    if last_char == "0":
        return 0
    string = string[1:-1]
    num = float(string)
    if last_char == 'K':
        num = num * 1000
    elif last_char == 'M': 
        num = num * 1000000
    return num


# The general logic behind the function: First I look at what the last character is. Some strings represent 0 value, so there is no 'K' or 'M' that follows it. In this case, we return 0. If it doesn't end in '0', then we cut off the Euro value sign in the beginning and the last character from the end of the string to get the numerical value alone (which could have a decimal point in it), and then we transform it from a string data type to a float data type. Finally, we examine the last character and multiply by the appropriate amount, returning the number after.

# In[ ]:


df['Wage_Num'] = df.apply(lambda row: clean_d(row['Wage']), axis=1)
df.head()


# When I wrote my first versions of this data cleaning and transformation, I ran into significant problems that really confused me. All I wanted to do was iterate over the rows of the new column and change its value and data type. 
# 
# The first time I tried it, I ran into a "SettingWithCopy" warning, which indicated that my code could be editing the copy of the value rather than the original value. 
# 
# When I changed my code and used the pandas loc function, the code worked without a warning but took an extremely long time. So I did some research and came upon this article: https://engineering.upside.com/a-beginners-guide-to-optimizing-pandas-code-for-speed-c09ef2. Here, I saw that normal iteration (e.g. using a for loop as I was employing) was highly inefficient despite being highly logical due to the inner workings of pandas. 
# 
# Finally, I saw a better way to approach my data transformation would be to use the apply function, which was much more efficient and helped in speed. For using an apply function, I saw it would be cleaner to make a helper function to use to work with the data as I pleased, and this was my reason for writing clean_d above. 

# In[ ]:


df['Value_Num'] = df.apply(lambda row: clean_d(row['Value']), axis=1)
df.head()


# In[ ]:


df.shape


# Now that I have completed some data transformations to make my dataset easier to work with, I begin to wonder if all the rows have full entries or if there is data missing.

# In[ ]:


df.isna().sum()


# When looking through this series of counts for missing values, the thing that stands out the most is the line of "48" counts. It is highly likely that these are all for the same players, and since 48 is such a small number in the grand scheme of the 18,207 players in this dataset, I think it is best to simply drop those rows entirely.

# In[ ]:


df = df.dropna(axis=0, subset=['Preferred Foot'])
df.isna().sum()


# Now our counts are much cleaner, and it is nice to see that the features I identified above as interesting to me all have 0 missing values. Another feature I was contemplating working with was 'Release Clause' but after observing that more than a thousand of its values are missing, it doesn't seem best to work with it. I could drop it, but I know that players even at the top of the game sometimes don't have a release clause, so I shouldn't go down that route. Imputation of the mean is not logical either, so I choose to not focus on this feature for my analysis.

# In[ ]:


import seaborn as sns
sns.set()


# At this point, my data cleaning process is complete and I want to spend time conducting Exploratory Data Analysis, or EDA. For this I import the aesthetically-pleasing visualization library - seaborn.

# In[ ]:


# See the counts of right-footed players vs. left-footed players
foot_plots = df['Preferred Foot'].value_counts()
foot_plots.plot(kind='bar')


# In[ ]:


sns.lineplot(x='Overall',y='Wage_Num',data=df)


# In[ ]:


sns.lineplot(x='Overall',y='Value_Num',data=df)


# In[ ]:


#compare age with difference in potential to overall
df['Growth_Left'] = df['Potential'] - df['Overall']
sns.lineplot(x='Age',y='Growth_Left',data=df)


# In[ ]:


sns.lineplot(x='Growth_Left',y='Wage_Num',data=df)


# In[ ]:


sns.lineplot(x='Age',y='Wage_Num',data=df)
#Observation: line plot might not be best way to visualize this because of outliers


# In[ ]:


sns.lineplot(x='Age',y='Value_Num',data=df)


# In[ ]:


sns.lineplot(x='Growth_Left',y='Value_Num',data=df)


# In[ ]:


sns.lineplot(x='Age',y='Overall',data=df)
#Observation: outliers are significantly influencing overall trend


# The main problem I ran into when creating and observing the patterns present in these line graphs were outliers. These outliers completely changed some parts of the graph to be misrepresentative of the overall trend. I know that it is best practice to drop outliers from data, but I hesitate to do so for this dataset because that would mean getting rid of some of the best, best players like Lionel Messi and Cristiano Ronaldo. In practical data science, outliers mean possible incorrect data collection or impossible entries, but players like LM and CR are just so legendary that they become outliers despite being truly that great. I don't have the heart to remove them from the data because the reality is they are the outliers of the sport, two great sportsmen of their time. In this case, what do I do for my data analysis?

# In[ ]:


top_100 = df[:100]
top_100.shape


# In[ ]:


nationality_100_plots = top_100['Nationality'].value_counts()
nationality_100_plots.plot(kind='bar')
#Observation: European and South American nations dominate in the Top 100 players


# In[ ]:


age_100_plots = top_100['Age'].value_counts()
age_100_plots.plot(kind='bar')
#Observation: The late 20's are where the majority of Top-100 players are aged


# In[ ]:


club_100_plots = top_100['Club'].value_counts()
club_100_plots.plot(kind='bar')
#Observation: Almost ALL of the Top-100 players play in Europe, and if not, then in the generous salary-giving Chinese and Japanese nations


# In[ ]:


#This seaborn function allows you to summarize all basic trends and correlations
sns.pairplot(df, vars=['Age', 'Overall', 'Wage_Num', 'Value_Num', 'Potential', 'Growth_Left'])


# For the final part of this data analysis, I want to apply some machine learning technique to take the features I identified as important to predict the most important one - 'Overall'. However, I am just starting in Machine Learning and the only ML experience I have is from the 'Introduction to Machine Learning' course on Kaggle, in which I was taught Decision Trees and Random Forests. So now I attempt to apply both to predict the overall rating for a player in the FIFA game. My performance metric will be Mean Absolute Error.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learntools.core import *

# Create target object and call it y
y = df.Overall
# Create X
features = ['Age', 'Value_Num', 'Wage_Num', 'Potential']
X = df[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2, random_state=1)

# Specify Model
ml_model = DecisionTreeRegressor(random_state=1)
# Fit Model
ml_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = ml_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation MAE when using a Decision Tree: {:,.0f}".format(val_mae))
print(train_X)
print(train_y)
print(val_X)
print(val_predictions)
print(val_y)


# In[ ]:


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
print(train_X)
print(train_y)
print(val_X)
print(val_predictions)
print(val_y)


# Looking at the outputs, I am very alarmed to see that both of the machine learning techniques employed had almost perfect accuracy in predicting the Overall feature. I am fairly certain that my Mean Absolute Error should be more than 0. This is what I really need help with, so please comment below and let me know what is wrong and what I should fix.

# This concludes my first mini-project on my journey in Data Science. Please provide feedback on what to improve on and what to spend more focus in. How can I take this analysis to the next level?
