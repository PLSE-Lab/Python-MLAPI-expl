#!/usr/bin/env python
# coding: utf-8

# **Machine Learning: A New Approach** 
# 
# This notebook is a complement to the keynote *"Machine Learning: A New Approach"*. If you are visiting from the internet, visit [insert video link here] to view the keynote and follow along. 
# 
# This keynote will focus on looking at implementing machine learning in any software/project from a case study perspective. The focus here is **NOT** to educate you on the types of models and their uses. The goal is to educate you on how to approach machine learning in any project through a structured manner. 
# 
# **What You Need**
#   * Notebook
#   * Basic high school math knowledge
#   * A willingness to learn
# 

# In[ ]:


#This cell is to import the required libraries needed to follow along
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split


# **Scenario**
# 
# Bob's Real Estate Holdings are a real estate investment company based in Australia that has long specialized in commercial properties. With recent market conditions, upper management has decided that they want to expand the company to move into residential properties and rent those units out. However since not every property is investment grade, the company has employed you to create an algorithm to determine whether the property is at the right price to purchase. They are decently skilled to work with technology and are fine with a command line tool. 
# 
# 
# **The Case - Based Approach **
# 
# When given any problem to solve the best way to tackle it, is using the case based approach. This particular approach displays this from a statistical analysis/machine learning point of view. The key steps is not always developing the model, many pre-requisites must be done to ensure that a relevant and accurate model is created. 
# 
# The case - based approach looks at solving any machine - learning problem using these 4 steps:
#     1. Understanding your problem.
#     2. Developing a project structure.
#     3. Data exploration and model development.
#     4. Model evaluation and deployment.
#     
#  
#     

# **Step One: *Understanding your problem.***
# Anything you aim to do with machine learning, will require you to understand your problem. This is a very simple, yet important step in a machine learning project. This step is there to help you create a meaningful product that is **relevant** to your case. 
# 
# To effectively tackle this step, answer the 4 W's. 
#     1. What is your problem?
#     2. Why are you solving this problem?
#     3. Where will this product be used?
#     4. Who will use this product?
#     
# Let's tie it back to Bob's Real Estate Holdings. They are a real estate holding company, and they want to use machine learning to determine whether a prospective investment of theirs is investment worthy and is it overvalued or undervalued. 
# 
#             1. The problem we solving is determining if a property is overvalued or undervalued   
#             according to current market data. 
#             
#             2. We are solving this problem because as cheap as a property may seem based on certain market criteria, it may not be a 
#             good investment. Taking out the impulse emotion factor and creating another data point to be used in the final investment 
#             decision will help generate sustainable returns. 
#             
#             3. There are two things to consider in where it will be used. The first part is the physical location of the product. 
#             This as the case states will be in Australia. The second part is where the product will be used on the computer (website, 
#             command line). As the case states, the company is filled with young professionals with a decent tech background. 
#             Therefore, we can build it to be used on the command line. 
#             
#             4. The people using this product will be Bob's Real Estate Holdings. They are tech-savvy and investors. 
# 
# 

# **Step Two: *Creating a project structure.***
# After you have understood the problem you are solving, determine the structure of your project. The structure is the technical aspect of the planning phase. You will determine the parts that require the machine learning in your project, what datasets you will leverage for your model, and what language will you be using. 
# 
# 
# For our project, we will be needing machine learning to predict if the property is investment grade. Now to see if the house is overvalued or undervalued we must understand how this will done. We can use machine learning to do this, or we can use machine learning to determine the price of the house and implement a logic algorithm to find if the market price is above or below. Computing wise the second option is a better option. So the parts we need machine learning for is determining if a property is investment grade and if the house price is over or understated.
# 
# Lucklily for us we have a dataset on Kaggle that has all the housing prices for us in the city of Melbourne. We will be using this. For the simplicity of the tutorial we will be using Python to develop our models.

# In[ ]:


#importing our dataset and getting it ready
melb_data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")


# **Step Three (A): *Exploratory Data Analysis.***
# Now that we are finally done with the planning phase of the project; let's move into variable exploration and analysis. There are many ways to do this including matplotlib graphs and data visualizations; however that is very time consuming. Kaggle fortunately has data visuals with every single of their datasets. [Visit this link to see them for our dataset](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot).
# 
# The other thing we have to do is explore the relationships in the data. We will focus on the nature of each variable and the values they contain. Let us take note of what each variable represents and how it relates with the target. (A notebook is best for this)

# In[ ]:


melb_data.head()


# In[ ]:


melb_corr = melb_data.corr()
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(melb_corr, cmap=cmap, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


melb_data = melb_data.drop(['Date', 'Method', 'SellerG', 'CouncilArea', 'Address', 'Bedroom2'], axis=1)


# After exploring each variable we notice that there are some we have to drop. These variables do not add value to the data and when we go to input the data back in our model for predictions, they won't change the outcome for better or worse.
# 
# These variables (columns) are: date, address, method, sellerg, latitutde, and longitude. 
# 
# **Step Three (B): *Developing the model.***
# Now let's dive into the fun part of machine learning; model development (YAY!). 
# 
# To develop the model, we have to first process the dataset into numbers. The model can't take in strings as a datapoint unfortunately. To overcome this problem we have two options: 
# 1. Delete columns with text data.
# 2. Turn the text data into numbers.
# 
# No option is better, it will require trial and error to determine which option is more suitable to your problem. Let's tackle both options and check their accuracy score. Also just note for simplicity purposes, we will be using the linear regression model.
# 

# In[ ]:


#Dropping columns with text data.
optionUno = melb_data.drop(['Suburb', 'Type', 'Regionname'], axis = 1)
optionUno = optionUno.dropna(axis=0)
y = optionUno.Price
X = optionUno.drop('Price', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
modelOne = RandomForestRegressor(random_state=1)
modelOne.fit(X_train, y_train)
y_pred = modelOne.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred, y_test))


# In[ ]:


#Changing text data to numerical
optionDeux = melb_data
optionDeux = optionDeux.dropna(axis=0)
label_encode = LabelEncoder()
optionDeux['Regionname'] = label_encode.fit_transform(optionDeux['Regionname'])
optionDeux['Type'] = label_encode.fit_transform(optionDeux['Type'])
optionDeux['Suburb'] = label_encode.fit_transform(optionDeux['Suburb'])
y2 = optionDeux.Price
X2 = optionDeux.drop(['Price'], axis = 1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=1)
modelTwo = RandomForestRegressor(random_state=1)
modelTwo.fit(X_train2, y_train2)
y_pred2 = modelTwo.predict(X_test2)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred2, y_test2))


# In[ ]:


thirdOp = melb_data
thirdOp = thirdOp.dropna(axis=0)
features = ['Rooms', 'Landsize', 'Lattitude', 'Longtitude', 'Bathroom']
modelThree = RandomForestRegressor(random_state=5)
y3 = thirdOp.Price
X3 = thirdOp[features]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, random_state=1)
print(X_test3)
modelThree.fit(X_train3, y_train3)
y_pred3 = modelThree.predict(X_test3)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred3, y_test3))


# With the three scores, we can that using text data is slightly better. Now is the time to deploy it on the command line. 
# 
# 
# **Congratulations! You have now learned the case based approach to machine learning. Visit these external resources to understand machine learning and its technalities in more detail: **
# 
# * https://www.coursera.org/learn/machine-learning
# * https://developers.google.com/machine-learning/crash-course/ml-intro
# * https://www.coursera.org/learn/ml-foundations
# 
