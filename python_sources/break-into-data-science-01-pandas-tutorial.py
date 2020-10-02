#!/usr/bin/env python
# coding: utf-8

# **Break into Data Science**
# 
# Hello everyone! Welcome to the first chapter of the "Break into Data Science" Tutorial Series. In this Series, I will cover essential topics, starting from the basics and scaling towards advanced Machine Learning/Deep Learning techniques. Please provide me feedback, as it would help me in creating better tutorials going forward. 
# 
# I have decided to break this series based on the following three components involved in Data Science:
# 1. Organizing - This is the phase where you organize your data in a way that will help you make better analysis
# 2. Packaging - This is where you analyze, prototype, perform statistics, and create visualizations
# 3. Delivering -  This is where you tell the story and add value based on your findings
# 
# Please set your mindset to have the continuous awarness of the "What, How, Who and Why".
# - What is being created?
# - How will it be created?
# - Who will be involved in creating it/Who are the stakeholders?
# - Why are we creating this? Why something is happening the way it is happening?
# 
# As you would guess, the first tutorial is going to be on the first component - Organizing. Please keep in mind that the way you organize your data  will impact the effort required to obtain valuable insights (Database Theory).
# 
# Before we dive-in the tutorial, it is important to know that the facts and desires of everyone, including each and all of us, come from that data that is presented to us. 

# **Topics Covered:**
# 
# *Organizing the Data with Pandas - the phase where you organize your data in a way that will help you make better analysis*
# 1. Data Input/Output
# 2. Inspecting the Data
# 3. Cleaning the Data
# 4. Data Selection / Groupby / Filtering / Sorting
# 5. BONUS - Creating Test Objects/ Join&Combine

# In[63]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# **Pandas**
# 
# 
# Pandas is an open source library built on top of NumPy that allows for fast analysis and data organization (cleaning/preparation). You can install it through your terminal using either:
# 
# conda install pandas
# 
# pip onstall pandas

# **1. Data Input/Output**
# 
# Pandas can read a variety of file types. As the file type of current Dataset is CSV we will focus on that but before that, below I have included some examples for different file types for your reference:
# 
# *Excel Input*
# > pd.read_excel('Excel_Sample.xlsx',sheetname='Sheet1')
# 
# *Excel Output*
# > df.to_excel('Excel_Sample.xlsx',sheet_name='Sheet1')
# 
# *HTML*
# 
# For this, you may need to install htmllib5,lxml, and BeautifulSoup4. For this, In your terminal/command prompt run:
# > conda install lxml
# 
# > conda install html5lib
# 
# > conda install BeautifulSoup4
# 
# *HTML Input*
# 
# The Pandas "read_html" function will read tables off of a webpage and return a list of DataFrame objects:
# 
# > df = pd.read_html('https://www.fdic.gov/bank/individual/failed/banklist.html')
# 
# *SQL*
# 
# > pd.read_sql(query, connection_object) #This will read from a SQL table/database
# 
# *JSON*
# 
# > pd.read_json(json_string) #This will reads from a JSON formatted string, URL or file
# 
# *Clipboard*
# 
# > pd.read_clipboard() # This will take the contents of your clipboard and passes it to read_table()
# 
# *Dictionary*
# 
# > pd.DataFrame(dict) #From a dict, keys for columns names, values for data as lists
# 
# *From Table*
# 
# > pd.read_table(filename) #From a delimited text file (like TSV)
# 

# OK! So let's start! Importing Pandas and NumPy

# In[ ]:


import numpy as np 
import pandas as pd


# Loading the data:

# In[ ]:


data = pd.read_csv('../input/acs2015_census_tract_data.csv')


# **2. Inspecting the Data**
# 
# Now that we have imported the data, we can go ahead and inspect it. This step is essential as we need to be familiar with the data that we are working with. This will allow us to build an efficient model that porivdes valuable information.

# The below Info command will provide us Index, Datatype and Memory information of our data

# In[ ]:


data.info()


# Now let's look at our data:
# > data.head(n) - First n rows of the DataFrame. If you don't provide n it is defaulted to be n=5
# 
# This will allow you to visually see how your data is structured.

# In[ ]:


data.head(10)


# Similarly:
# > data.tail(n) - Last n rows of the DataFrame. similar to head but will show you the last n rows.

# In[ ]:


data.tail(10)


# Now, let's look at the shape of our data in form of (rows,columns)

# In[ ]:


data.shape


# For a quick way to see statistical metrics on each column of your data you can use the following:
# > data.describe() - Summary statistics for numerical columns. 

# In[ ]:


data.describe()


# You can see how many unique values are in a specified columns with the below command: As expected this will give 52 States

# In[ ]:


data['State'].nunique()


# If you want to see the list of all unique values, you can simply with the below code:
# This will return an array listing all unique values

# In[ ]:


data['State'].unique()


# You can check to see if you have null data. This will return Boolean array. I will add the "tail()" for visual purposes

# In[ ]:


pd.isnull(data).tail(10)


# pd.notnull(data) : is the opposite of the above

# **3. Cleaning the Data**
# 
# Cleaning the data is an essential part of your data organization. Once inspecting your data, you will notice that there will be some data that you don't necessarily need or data that could influenze your numbers and impact your analysis.  

# For purposes of this tutorial, I will drop all rows that contain null values

# In[ ]:


data = data.dropna()
data.shape


# You can see that we have dropped ~1.5K rows.
# 
# Please note that we have updated the data by specifying "data = "
# 
# Please note that this is not always the best thing to do. Use your judgement.

# If you would like to drop columns that contain null values you can do so with the below code:
# 
# > data.dropna(axis=1)
# 
# There's no need for such action. Below I have included additional Data Cleaning Functions for your reference. 
# 
# 
# **Few other useful Data Cleaning functions:**
# 
# > s.astype(float) # This will convert the datatype of the series to float *Please note that "s" here is a Pandas Series
#  
# > s.replace(1,'one') # This will replace all values equal to 1 with 'one' 
#  
# > s.replace([1,3],['one','three']) #This will replace all 1 with 'one' and 3 with 'three' 
#  
# > data.rename(columns=lambda x: x + 1) #Mass renaming of columns 
#  
# > data.rename(columns={'old_name': 'new_ name'}) #Selective renaming 
#  
# > data.set_index('column_one') # This will change the index 
#  
# > data.rename(index=lambda x: x + 1) #Mass renaming of index
#  
#  > data.dropna(axis=1,thresh=n)  #This will drop all rows have have less than n non null values 
# 
# > data.fillna(x) # This will replaces all null values with x 
# 
# > s.fillna(s.mean()) #This will replace all null values with the mean (mean can be replaced with almost any function from the below section) : 
# 
# > > data.corr() #This will return the correlation between columns in a DataFrame 
# > 
# > > data.count() #This will return the number of non-null values in each DataFrame column 
# > 
# > > data.max() #This will return the highest value in each column 
# > 
# > > data.min() #This will return the lowest value in each column 
# > 
# > > data.median() #This will return the median of each column 
# > 
# > > data.std() #This will returns the standard deviation of each column
# 

# Let's remember how our data looks.

# In[ ]:


data.head()


# We don't need the CensusTract column, so we will drop it:

# In[ ]:


data = data.drop(['CensusTract', 'County'],axis=1) #data = data.drop('CensusTract',axis=1)  if dropping only one column
data.head()


# We can add new columns and make operations between different columns. Now let's create a column with the Men/Women ratio as following:

# In[ ]:


data['M/W_Ratio'] = data['Men'] / data['Women']
data.head()


# You can see that the new column has been added to our data.
# 
# Now we are going to drop the new column that we craeted as it is not essential for our analysis

# In[ ]:


data = data.drop('M/W_Ratio',axis=1)
data.head()


# The new column has been dropped. Let's look at our columns:

# In[ ]:


data.columns


# You can make mass calculations on the data. For this instance, let's convert the percentages to actual population numbers. For this we will use a "For Loop"

# In[ ]:


percentages = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific','Poverty', 'ChildPoverty', 'Professional', 'Service', 'Office','Construction','Production','Drive','Carpool','Transit','Walk','OtherTransp','WorkAtHome', 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']
for i in percentages:
    data[i] = round(data['TotalPop'] * data[i] / 100)   
#We won't be doing further data cleaning on our data in this tutorial.


# In[ ]:


data.head(20)
# you can see that we can make multiple calculations and update our table with the power of loops


# **4. Data Selection / Groupby / Filtering / Sorting**
# 
# Data Selection is essential as you should be able to navigate through your data. But first, let's "Grouby" the State Column to 

# **Groupby**

# Now we are going to use "Groupby" to consolidate the data into the States. When grouping by you could also use mean(), etc instead of sum()

# In[ ]:


data = data.groupby('State', as_index=False).sum()


# In[ ]:


data.head(10)


# Let's check the Population of all 52 States.

# In[ ]:


pop = data['TotalPop'].sum()
print('Total Population is: ', pop)


# In[ ]:


data['TotalPop'].max()


# **Sorting**

# Let's sort values in ascending order 
# 
# #data.sort_values(col2,ascending=False) - Sorts values by col2 in descending order

# In[ ]:


data = data.sort_values('TotalPop')
data.head()


# Just curious on how the data looks like...

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

fig, ax = plt.subplots(figsize=(14,4))
fig = sns.barplot(x=data['State'], y=data['TotalPop'], data=data)
fig.axis(ymin=0, ymax=40000000)
plt.xticks(rotation=90)


# Last visualization - promise :) - Data visualization (Packaging of data) will be covered in future tutorials
# 
# This is to see the distribution of the population data

# In[ ]:


sns.distplot(data['TotalPop'])


# **Filter**

# Ok, back to our data.. Let's say that you want to filter based on a given condition...
# 
# I'm curions to see only states that the have over 1 Million of Hispanic residents

# In[ ]:


data_hisp = data[data['Hispanic'] >= 1000000] 
data_hisp


# Just remembering our columns:

# In[ ]:


data.columns


# We can add more that one condition. Let's say that you want to filter: 
# 
# 
# States that have Women to Men Ratio over 1 AND have more Natives than Asians OR has population > 35M
# 
# You'll see that only 3 States match with our first condition and only California has over 35M residents

# In[ ]:


data[(data['Women']/data['Men'] > 1) & (data['Native'] > data['Asian']) | (data['TotalPop']>35000000)]


# **Some additional functions for your reference**

# You can do advanced sorting such as the following:
# 
# > data.sort_values([col1,col2],ascending=[True,False]) # This would sort values by col1 in ascending order then col2 in descending order
# 
# Similartly you can do more advanced Groupby functions such as the following:
# 
# > data.groupby([col1,col2]) # This will return a groupby object values from multiple columns
# 
# > data.groupby(col1)[col2].mean() # This will return the mean of the values in col2, grouped by the values in col1 (mean can be replaced with almost any function discussed above (corr, count, max, sum, etc)
# 
# If you want to create a Pivot Table:
# > data.pivot_table(index=col1,values=[col2,col3],aggfunc=mean) # This will create a pivot table that groups by col1 and calculates the mean of col2 and col3
# 
# Additional useful functions:
# > data.groupby(col1).agg(np.mean) # This will find the average across all columns for every unique column 1 group
# 
# > data.apply(np.mean) # This will apply a function across each column
# 
# > data.apply(np.max, axis=1) # This will apply a function across each row

# **Data Selection**

# Now, let's make soem data selection. Let's say I want to return the column with label 'State'. Please note that this will return a Series
# 
# Please note that I am adding ".head()" to make it visually better for this purpose. Otherwise you will get the whole list

# In[ ]:


data['State'].head() 


# If we add couple of columns: this would return those Columns as a new DataFrame
# 
# Then, you can assign it to a new Dataframe by simply Dataframe2 = data[[col1, col2]] 
# 
# Please note that, again, I am adding ".head()" to make it visually better for this purpose

# In[ ]:


data[['State', 'TotalPop']].head() 


# You can also do "Selection by position" (On a Series)
# 
# The below example will return "Wyoming" as that is our first State (based on the sorted values)
# 
# Please note that in Python the first index is 0 - not 1

# In[ ]:


data['State'].iloc[0] 


#  Lt's say we want to bring the last (most populous) State, California:

# In[ ]:


data['State'].iloc[51]


# Selection by index:
# 
# Please note that if you look at the above table the Index number of the States did not change.
# 
# Wyoming is 51, Vermont is 46, Alaska is 1, and so on (as original data is alphabetical).
# 
# This method will let you select data based on the index number.. so s.loc[0] will be Alabama (first State, alphabetically).

# In[ ]:


data['State'].loc[0] 


# You can select the first row as the following:
# 
# Please note that ":" means to include all. So basically we are saying first row and all columns

# In[ ]:


data.iloc[0,:] 


# In[ ]:


data.iloc[:,1].head() # Second column


# In[ ]:


data.iloc[0,0]# First element of first column, and so on...


# Being able to access exactly to the data that you want and being able to slice is is essential- as this would help you in being able to slice the data in ways that will make it more efficient for you allowing you to potentially provide more value

# **BONUS - Creating Test Objects/ Join&Combine**
# 
# This is not related to our dataset, but thought would be good to cover this topic as it would allow you to practice your data preparation skills.

# **Creating Test Objects**

# Let's create  5 columns and 15 rows of random floats

# In[ ]:


df = pd.DataFrame(np.random.rand(15,5)) 
df


# Let's create a series from an iterable my_list

# In[ ]:


my_list = [1,2,3,4,5,6,7,8,9,10]
pd.Series(my_list) 


# Now, let's  add a date index
# 
# You will see that now we have added a Date index to our df (with random values)

# In[ ]:


df.index = pd.date_range('2018/1/1',periods=df.shape[0]) 
df 


# **Join/Combine**

# Let's create a Data Frame with 2 rows and 5 columns

# In[ ]:


df2 = pd.DataFrame(np.random.rand(2,5)) 
df2


#  Now, let's add the rows in df to the end of df2 (columns should be identical)

# In[ ]:


dfnew = df.append(df2)
dfnew


# As you can see, we have merged our tables.

# In[ ]:


df3 = pd.DataFrame(np.random.rand(16,3))
df3.index = pd.date_range('2018/1/1',periods=df3.shape[0]) #This will add a date index
df3


# What if you want to add them as new columns.. Let'sl add the columns in df3 to the end of dfnew (rows should be identical)
# 
# Please notice how it concatenates based on the Index...

# In[ ]:


dfnew2=pd.concat([dfnew, df3],axis=1)
dfnew2


# **Congrats and thanks for checking this tutorial!**
# 
#  Thank you very much for your time and feedback. 
# 
# In the next tutorial we will cover Data Packaging - which is the phase where we will create visualizations of our data that would allow us to visually understand what is happening, recognize some patterns, get creative and find useful/surprising insights- Stay tuned!
# 
# **Until next time, enjoy Data Science!**

# 
