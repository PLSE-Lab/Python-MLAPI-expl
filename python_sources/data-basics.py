#!/usr/bin/env python
# coding: utf-8

# # Welcome to "Data Basics" in Python
# 
# This tutorial is a basic intro to:
# - Enough basic Python to work with lists and some data
# - Dataframes in the pandas library
# - Some introductory operations on pandas dataframes

# ## Getting Started
# 1. Create your own account on Kaggle.com
# 2. Click the blue "Copy and Edit" in the upper-right part of this document to create your own copy to your own Kaggle account.
# 3. As you complete exercises, be sure to click the blue "Save" button to create save points for your work.
# 
# ### Orientation:
# - This notebook is composed of cells. Each cell will contain text either or Python code.
# - To run the Python code cells, click the "play" button next to the cell or click your cursor inside the cell and do "Shift + Enter" on your keyboard. 
# - Run the code cells in order from top to bottom, because order matters in programming and code.
# 
# ### Troubleshooting
# - If the notebook appears to not be working correctly, then restart this environment by going up to **Run** then select **Restart Session**. 
# - If the notebook is running correctly, but you need a fresh copy of this original notebook, go to https://www.kaggle.com/ryanorsinger/data-basics and click "Copy and Edit" to make yourself a new copy.
# - Save frequently and save often, so you have access to all of your exercise solutions!

# # Data Science Pipeline
# - Plan - this is where we form initial hypotheses and seek to understand stakeholder goals from the data
# - Acquire - we have to get our hands on the raw data (from databases, loading spreadsheets, scraping websites, etc...)
# - Prepare - data must be squeeky clean for analysis
# - Explore - Statistical testing and Visualizing relationships in the data and how the data helps us understand what we're trying to predict or discover
# - Model  - Build Machine Learning Models
# - Present Findings & report
# 
# Each stage dovetails into the other. In addition to basic Python and pandas, this notebook example will cover, at a *very high level*, the acquire, prepare, and explore stage.

# In[ ]:


# Variables point to values. 
# All variables have a value and a data type
# Strings can hold text, letters, and other characters
# Single = is the assigment operator 
message = "Howdy, Everybody!"
print(message) # The print function prints whatever you put in the parentheses


# In[ ]:


# Here the type function checks the data type of the variable "message"
# Then the print function prints the result of the type function
print(message)
print(type(message)) # 'str' means string


# In[ ]:


# There are different kinds of numbers
print(5)
print(type(5)) # int means integer (whole numbers either positive or negative)
print(type(5.0)) # float means a number with a "floating point" precision decimal decimal


# In[ ]:


# Comparison operators in Python, like == return True or False. Other math operators like < or > return True or False, too.
print(1 == 1)


# In[ ]:


print(type(True))
print(type(False))
print(True)
print(False)


# In[ ]:


# Lists in Python are created by square brackets and can hold any value.
beatles = ["John", "Paul", "George"]

print(type(beatles))
print(beatles)


# In[ ]:


# .append on a list adds new values onto the end of the list.
beatles.append(Ringo")

# In Python notebooks, the last line of a cell can print a value automatically. (but only the last line)
beatles


# In[ ]:


# Exercise 1
# First, Create a new variable called "numbers" and assign it the numbers 1 through 9 as a list
# Print your list of numbers.


# In[ ]:


# Exercise 2
# Add the number 10 onto your numbers variable. Be sure to use the Python method to add to the end of the list.
# Then print the numbers list


# In[ ]:


# Exercise 3
# In this one cell, print out your new "numbers" variable, then the "beatles" variable, and also the "message" variable, on their own line.


# ## Python is:
# - Super powerful
# - A beginner friendly language and one of the easiest programming languages to learn.
# - A top shelf programming language used everywhere from AI and Data Science to robotics and web application development.
# 
# ### Also: 
# - We are going to skip a whole bunch of basic Python here and go straight to some powerful Python libraries
# - Code libraries allow us to stand on the shoulders of giants and avoid re-inventing the wheel.
# - Some of the libraries we'll start with:
#     - numpy for linear algebra
#     - pandas for data manipulation and file input/output. pandas is our "data wrangling" workhorse.
#     - matplotlib and seaborn are visualization and charting libraries that work well together

# In[ ]:


# Run this cell
import numpy as np    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# This cell creates an empty dataframe
# Dataframes are like spreadsheets or database tables, and can hold TONS of data.
fruits = pd.DataFrame()


# In[ ]:


# Using square-brackets and a string, we can create or reference a column in the dataframe
# We can also assign lists and other data to each column.
fruits["name"] = ['apple', 'banana', 'crab apple', 'dewberry', 'elderberry', 'fig', 'guava', 'huckleberry', 'java apple', 'kiwi']
fruits


# In[ ]:


# Exercise 4
# Create a new column named "quantity" and assign it a list of 10 different quantities. 
# It's OK if there's 1 apple, 2 bananas, 3 crab apples, etc...
# print out your dataframe


# In[ ]:


# Let's assign some prices to each fruit
fruits["price"] = [1, .75, 0.10, 0.55, 0.65, 1.50, 2.00, 0.99, 1.99, 3.25]
fruits


# In[ ]:


# Let's do this together
# Delete the hashtag on the last line to uncomment them. 
# Then run this cell. 
# fruits["subtotal"] = fruits.price * fruits.quantity


# In[ ]:


# Let's print the dataframe to make sure we have:
# Subtotal, price, quantity, and fruit name.
print(fruits)


# In[ ]:


# Run this cell to create a new column where the entire tax column is 0.08
fruits["tax_rate"] = .08
fruits


# In[ ]:


# Uncomment the last two lines of code and run this cell to produce a tax_amount in dollars for each item
# Example of creating a new column and setting it to be the result of multiplying two columns together
# fruits["tax_amount"] = fruits.tax_rate * fruits.subtotal
# fruits


# In[ ]:


# Exercise 5
# Create a new column named "total" then assign it the result of adding the "subtotal" and "tax_amount" column.
# Then print the dataframe


# In[ ]:


# Let's check to see which of our fruits contains the string "berry"
fruits.name.str.contains("apple")


# In[ ]:


# If we use an array of booleans as a filter, we can "turn on" and "turn off" certain rows, and filter our results
fruits[fruits.name.str.contains("apple")]


# In[ ]:


# Exercise 6
# Use the syntax and operations introduced from the above example
# Show all of the rows that contain "berry" in the name.


# ## We'll shift gears here from made up data to acquiring data
# - This is a squeaky clean dataset to simplify the acquire and prep stages of the data science pipeline
# - We'll go directly into exploring the data

# In[ ]:


import seaborn as sns
df = sns.load_dataset("iris")
df


# In[ ]:


# Since we have width and length, let's try adding area as a "derived feature"
# Data scientists will often use the existing datatpoints to synthesize or derive new data that may add additional insight.
df["sepal_area"] = df.sepal_length * df.sepal_width
df


# In[ ]:


# Exercise 7 
# Create a new measurement called "petal_area" that contains the result of multiplying the petal_length by the petal_width values.


# In[ ]:


# Let's visualize all of the measurement pairs and color by species
sns.pairplot(df, hue="species", corner=True)


# ## Our takeaways so far:
# - Does it look like there's a measureable difference between the species? 
# - If so, it's likely we can build a classification algorithm to predict the species based only on the sepal and petal measurements!
# - Let's take this to the next level and build a machine learning model that will predict the species of iris based on the measurements!
# - https://www.kaggle.com/ryanorsinger/classification-intro/

# In[ ]:




