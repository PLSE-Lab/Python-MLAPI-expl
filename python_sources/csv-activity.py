#!/usr/bin/env python
# coding: utf-8

# # Working with Data: Visualizing CSVs
# ## Introduction
# So far, we've done some work on reading data into python and visualizing it. We know that Comma-Separated Value (CSV) formatted text is something that we can read into python readily using modules like `pandas`. (If you're really looking to create a CSV out of some messy text, the `csv` module in python can also be a good choice). Here we'll review some of the important basics for the `pandas` module.
# # `pandas`: The Python Data Analysis Library
# The `pandas` comes with a bunch of super useful functions for loading, subsetting, and visualizing datas. It is documented [here](https://pandas.pydata.org).
# To get more comfortable working in `pandas` we're going to practice loading in CSVs from different places. Let's start with the following CSV where I've stored some details about a recent grocery run:
# 
#       food,color,grams  
#       apple,red,250  
#       chicken,pink,500  
#       kale,green,200  
#       bread,brown,300  
# 
# Here, I've given my grocery list a header labelling the value placed in each column. Data won't always have headers, and it's important to keep in mind that `pandas` assumes that a CSV has a header by default.
# 
# One of the major features of pandas is the DataFrame object, which is a nice way of representing tabular data (mixed data with a column for each variable). We can load our CSV shopping list into a DataFrame using the `pd.read_csv` function:

# In[ ]:


import pandas as pd

data = """
food,color,grams
apple,red,250
chicken,pink,500
kale,green,200
bread,brown,300
"""

df = pd.read_csv(pd.compat.StringIO(data))
print(df)


# We can index DataFrames by row and column names (or boolean arrays) by accessing the object's `.loc` field, and we can index by integers by using the `.iloc` field. But under the hood, DataFrames work a little bit like fancy dictionaries, with column heads as keys:
# 

# In[41]:


print("The food column:\n", df[['food']]) # get the "food" column
print("The food column part 2:\n",df.loc[:,'food']) # also get the food column
print("The food column using numbers:\n",df.iloc[:,0] ) # once again, food
print("Heavy foods:\n",df.loc[df['grams'] >= 300,:])# rows where grams >= 300


# `pandas` lets use the `read_csv` function to read in data in csv format in a number of ways. Above we used the `compat.StringIO` function to convert a CSV string into a format that pandas can read directly. 
# 
# In previous sessions, and in the next section, we see `read_csv` to read in a file stored locally (when we're using Kaggle, this means we're using the 'input' folder you see on the right. If you were running the notebook on your own computer, you could access folders and files stored on your hard drive).

# In[ ]:


pokemon_df = pd.read_csv('../input/competitive-pokemon-dataset/pokemon-data.csv', sep = ';')
pokemon_df.head()


# Note that back in our activity, we broke down a few ways to clean up this pokemon data. One of them we have to include: the `sep` argument, which tells `read_csv` what separates each column in our table. The remaining arguments we had forced the values in each column to be interpreted as we wanted them. Some of our columns contain lists (Types, Abilities, Next Evolution(s), and Moves), but `pandas` is interpreting these as strings:

# In[42]:


print(pokemon_df.loc[pokemon_df['Name'] == 'Charmander','Next Evolution(s)'].values[0])
print(type(pokemon_df.loc[pokemon_df['Name'] == 'Charmander','Next Evolution(s)'].values[0]))


# For now we're not going to worry about that, since we're not going to use these columns!
# 
# You can look back at [week 2's main activity](https://mybinder.org/v2/gh/claire-bomkamp/UBC-LTS-Code/master?filepath=Pokemon%20Dataset.ipynb) to review how to make the types of each column explicit.
# 
# # Exploring Black Friday Sales Data.
# 
# The dataset we'll use to practice is a sample of sales data for a store on Black Friday. It has plenty of information to help us break down our data into groups, which allows to ask more targeted questions.
# 
# You can see what other people have done with this data on [kaggle](https://www.kaggle.com/mehdidag/black-friday/kernels). Each 'kernel' is a Jupyter notebook filled with someone's analysis. One popular one similar to this activity can be seen [here](https://www.kaggle.com/shamalip/black-friday-data-exploration).
# 
# ## Reading in the Data
# We can use the `pd.read_csv` function to read our dataset (a CSV file) into a DataFrame! Let's peak at the values with `head`:

# In[ ]:


black_friday = pd.read_csv("../input/black-friday/BlackFriday.csv")

black_friday.head()


# We see that for each unique user (denoted by `User_ID`), we have some general information defining who they are. In marketing, we might be interested in the amount of money a certain age group is willing to spend in one trip. We can summarize variables for each unique `User_ID` by using the `groupby` function. Let's look at the total amount spent by each user:
# 

# In[ ]:


grouped_data = black_friday.groupby(["User_ID"])
grouped_data[["Purchase"]].sum().head()


# ## Practing Plotting with `matplotlib` and `seaborn`
# Visualization is an important step in learning about any new dataset. At first you might think that you can get every thing from summary data alone. For example, for two measurments we have on our data, we might want to know each measurment's average value, the amount of spread around the average values for each measurment, and how similar the two measurements are to one another. We call these summary statistics mean (average), standard deviation (spread around the average), and correlation (similarity between measures). These statistics tell us some important properties of our data. However, data can be sneaky! Just look at the [Datasaurus Dozen](https://www.autodeskresearch.com/publications/samestats) below:

# In[ ]:


from IPython.display import Image
Image("../input/datasaurus/DataSaurus.gif")


# All of the plots above have the same mean (average), standard deviation (spread) and correlation value (similarity) for x and y! As you can see here, it's not until we actually see the different plots that we realize high-level, interesting patterns in our data. We can start asking these types of questions with our Black Friday shopping data too.
# 
# Let's try to see how much each age group spent, split by their gender:
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# First let's get the total amount spent by each customer
black_friday["Total_Purchase"] = grouped_data['Purchase'].transform('sum')
plot_data = black_friday[
    ["User_ID",
     "Gender",
     "Age",
     "Occupation",
     "Total_Purchase"]]
plot_data = plot_data.drop_duplicates().reindex()
plot_data.head()
sns.set()

sns.swarmplot('Age', 'Total_Purchase', hue='Gender', dodge=True, data=plot_data.sample(frac = 0.25))
plt.show()

sns.swarmplot('Occupation', 'Total_Purchase', hue = 'Gender', dodge=True, data=plot_data.sample(frac = 0.25))
plt.show()


# Here we're sampling just a random 25% of the rows in our dataframe (using the `.sample()` command). Swarmplot tends to go slow when it has lots of points to deal with, so this will speed things up a lot!
# 
# Just from the plots above, we can already start to see that occupation 8 seems to have the lowest spenders, and that the 0-17 age group spent a similar amount to the 55+ Age range. Try to see if you can get any of the plots [here](https://seaborn.pydata.org/api.html#categorical-plots) to give you a better picture of the data!
# We can make our second plot a little more readable by assigning names to each occupation. The dataset we're working with doesn't publish what these occupations are, so for practice let's create a job for each number and put these in a column called 'Job':
# 

# In[ ]:


job_numbers = sorted(plot_data['Occupation'].unique()) #unique sorted job number
print(job_numbers)
job_names = [
    "Boxer", "Doctor", "Business-person", "Astronaut", "Youtuber", "Biologist",
    "Rocket scientist", "Teacher", "Baker", "Janitor", "Magician", "Author",
    "Statistician", "Engineer", "Diplomat", "Coach", "Jeweler", "Roboticist",
    "Spy", "Programmer", "Chemist"]

# create a mapping from numbers to jobs
mapping = {key: value for (key, value) in zip(job_numbers, job_names)}
print(mapping)

# create a new DataFrame with a Job column for our names
named_plot_data = plot_data.assign(
    Job = [mapping[plot_data['Occupation'].iloc[i]] for i in range(len(plot_data))])

named_plot_data.head()


# Above we did a few things. We first got all the unique job numbers in our DataFrame. Then, we listed out a job name for each number. The third command here is a dictionary comprehension, which we use to create a python dictionary mapping each job number to a name. Lastly, we use `assign` to create a new DataFrame with a 'Job' column, and use list comprehension to map each occupation number to a job name.
# 
# Using this new DataFrame with a 'Job' column, we can now add these names to our second plot:

# In[ ]:


sns.swarmplot('Job', 'Total_Purchase', hue = 'Gender', dodge=True, data=named_plot_data.sample(frac = 0.25))
plt.xticks(rotation=90)
plt.show()


# The `plt.xticks` command is used here to rotate our Job labels so they don't overlap.
# ## Saving our Modified Data
# Now that we're done plotting, we can take our modified DataFrame `named_plot_data` and save it away for later using the `to_csv` method:
# 

# In[43]:


named_plot_data.to_csv("../black_friday_plotting.csv")


# There is now a new files called `black_friday_plotting.csv`, which you can see by asking Python for a list of your files:

# In[ ]:


import os
os.listdir('../')


# In[ ]:




