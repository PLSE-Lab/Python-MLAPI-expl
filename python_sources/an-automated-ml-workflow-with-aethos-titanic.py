#!/usr/bin/env python
# coding: utf-8

# # Automated ML Workflow with Aethos

# One of the most time consuming things I have run into in data science, or at least in the early stages of developing a kernel, is the copy and pasting of code from previous projects into a new one. I've often desired a common ML workflow that would automate some of this tedious work so I don't need to create the same for-loop to one-hot encode columns with the sklearn library. Or so that I can get a fast statistics readout on each feature to speed up Exploratory Data Analysis.
# 
# ![](http://)![image.png](attachment:image.png)
# 
# Then I came across this [article](https://towardsdatascience.com/aethos-a-data-science-library-to-automate-workflow-17cd76b073a4) written by Ashton Sidhu where he describes an Automated ML workflow that he has created, called **Aethos**. I decided to share it with the kaggle community in the hopes that it will improve some of your own processes. Link to the github repo [here](https://github.com/Ashton-Sidhu/aethos).
# 
# I hope you find this kernel helpful and some **UPVOTES** would be very much appreciated.
# 
# **Last Updated:** 1/19/2020

# ## 1. Getting Started

# First things first, we need to install the Aethos python package as it is not included as a default kaggle package

# In[ ]:


get_ipython().system('pip install aethos')


# Now, we can import aethos, along with pandas and numpy packages

# In[ ]:


import numpy as np
import pandas as pd
import aethos as at


# There are several options that can be toggle for Aethos. I won't go into the word_report option at this point, but the interactive_table is helpful as it basically allows us more freedom to explore the dataframe.

# In[ ]:


at.options.word_report = True
at.options.interactive_table = True


# Read up the train and test data as normal. For this most of this kernel, I will be working with the train data

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# This is where the magic of Aethos begins to come into play. Just pass the dataframe into the Aethos Data object. When doing this, we are able to specify the variable we are trying to predict in the **target_field** parameter (for the titanic dataset, it's 'Survived'). We are also able to split the dataset into two different sets, one for training and one for validation. This is done via the **split** parameter. The split percentage can also be changed by setting test_split_percentage to a float between 0 and 1 (by default it is 0.2)

# In[ ]:


df = at.Data(train, target_field='Survived', report_name='titanic', split = True, test_split_percentage = 0.2)
df


# We now have two different subsets within the dataframe, x_train and x_test. All of the analysis we do from now on will be applied to both sets. Pretty neat!

# In[ ]:


df.x_train
df.x_test


# ## 2. Analysis

# Now that we have created our dataframe with Aethos, we can treat it in very much the same way that we would treat a pandas dataframe.

# In[ ]:


# For specified column, return Pandas series
df['Age']


# In[ ]:


# For specified conditional operation in the square brackets, return a Pandas DataFrame
df[df['Age'] > 25]


# In[ ]:


# You can even run pandas functions on the Data object, however they only operate on your training data.
df.nunique() 


# Aethos also provides more information than pandas with their describe function in order to get a good first statistical look at your data.

# In[ ]:


df.describe()


# This can be done for individual columns as well. A distribution of the data is also plotted.

# In[ ]:


df.describe_column('Age')


# Specific stats can also be returned from the describe column

# In[ ]:


df.describe_column('Age')['std']


# If you would like to get all of this info in one clean report, the **data_report** method is an amazing feature. This will provide you with all the sinformation you will need about the dataset:
# - Types of variables represented in dataset (numerical, catagorical, etc)
# - Missing values and zeros per column
# - Distribution statistics of each column
# - Correlation heatmap 

# In[ ]:


df.data_report()


# For plotting, joint plots, pair plots and histograms can be easily created. The **output_file** can be specified if you would like to save the plots.

# In[ ]:


df.jointplot('Age', 'Fare', kind='hex', output_file='age_fare_joint.png')
df.pairplot(diag_kind='hist', output_file='pairplot.png')
df.histogram('Age', output_file='age_hist.png')


# ## 3. Clean Data

# We have covered how to start using the package as well as how to visualize data and perform EDA with Aethos. Now it is time to utilize its data cleaning capabilities.

# Simply use the **missing_values** method to print a dataframe breakdown of missing values for each column. If you split your data into train and test, it print out the distribution for BOTH!

# In[ ]:


df.missing_values


# To fill missing data for catagorical data, simply call the **replace_missing_mostcommon** method on the desired column. It will fill all blank cells with the mode of the column.

# In[ ]:


df.replace_missing_mostcommon('Embarked')


# For missing data in numerical columns, you can call the **replace_missing_median** method to fill all blank cells with the median of the column.

# In[ ]:


df.replace_missing_median('Age')


# We can drop columns easily as well. We are dropping **Cabin** since it has a high percentage of missing values

# In[ ]:


df.drop('Cabin')


# Calling missing_values again, we see that our DataFrame now contains no missing values!

# In[ ]:


df.missing_values


# We can use the **apply** method just like we would on a DataFrame, which just minor change in syntax. In this example, we have a function that will return 'Child' if Ages is less than 16. Otherwise it will return the sex of the individual. Using the apply method with the **get_person** method, we can specify that we want to create a new column 'Person' with the aggregation specified in get_person. Then we will drop column 'Sex'. Easy as that!

# In[ ]:


def get_person(data):
    age, sex = data['Age'], data['Sex']
    return 'child' if age < 16 else sex
df.apply(get_person, 'Person')
df.drop('Sex')


# We are also capable of one hot encoding with a simple one-line. We will do this with the **Embarked** and **Person** features. Simply specify the columns to be encoded, and indicate whether you would like to keep the original columns prior to the ohe via **keep_col**.

# In[ ]:


df.onehot_encode('Embarked', 'Person', keep_col=False, drop=None)


# ## Stay tuned for more...

# Next up, we will begin unpacking Aethos' capabilities with **feature engineering**, **model generation** and **hyperparameter tuning**...
