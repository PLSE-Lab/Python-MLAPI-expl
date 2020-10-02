#!/usr/bin/env python
# coding: utf-8

# # **Quiz 4:** An Analysis of Colleges, Majors and Regions in Relation to Salary Utilizing Pandas

# ![image](https://i.pinimg.com/originals/8c/bf/ed/8cbfed1bf8496e9e20716320bc7b5e49.jpg)

# Hi! Thank you for taking the time to peruse my Kaggle notebook. This notebook serves a dual purpose - it is both an **exploratory data analysis (EDA)** and an **introductory tutorial to Pandas**. Thus, there will be significant explanation interwoven into our analysis of this data, which was originally posted on Kaggle by the Wall Street Journal. This tutorial is best for those who are beginners and are just learning the Pandas tool. The dataset concerns the salaries college graduates receive throughout their careers, and presents this data in terms of college type, major, or region.
# 
# I was inspired to base my tutorial and analysis on this dataset because high school seniors as well as transfer students have recently received their admissions letters (congratulations!), and they may be making the difficult, yet exhilirating, decision of choosing where to attend college and what major to pursue in their time there. Although how lucrative a major or a college is should not be the biggest driver in your decision-making process, it is definitely something to consider. I hope you find this analysis and tutorial useful, whether it be in learning about Pandas or in choosing we're you're headed next.

# # Introduction: Preparing the Dataset for Analysis

# To begin, let's run the cell below in order to import our needed classes as well as input files:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# We will create ***three*** different DataFrames utilizing the provided csv files:
#     1. **sbm**, which will represent salaries in relation to major (sbm serves as shorthand for salaries by major)
#     2. **sbc**, which will represent salaries in relation to college type (sbc serves as shorthand for salaries by college type)
#     3. **sbr**, which will represent salaries in relation to region (sbr serves as shorthand for salaries by region)
# Run the next code cell to store the DataFrames in their respective variables:

# In[ ]:


# The index for salaries based on major is set to the 'Undergraduate Major'
sbm = pd.read_csv('/kaggle/input/college-salaries/degrees-that-pay-back.csv').set_index('Undergraduate Major')
# The index for salaries based on college type and region is set to the 'School Name'
sbc = pd.read_csv('/kaggle/input/college-salaries/salaries-by-college-type.csv').set_index('School Name')
sbr = pd.read_csv('/kaggle/input/college-salaries/salaries-by-region.csv').set_index('School Name')

#set_index() uses an existing column in the DataFrame and sets it as the DataFrame index


# The code in its original form stores the salaries as strings in the format "$XX,XXX.XX". Although this may be very user-friendly aesthetically, it is not what we need for our data analysis. We need to access these values as floats in order to be able to quantify them numerically as we need.
# 
# Run the next code cell to convert the string representations of the salaries from the string type into the float type:

# In[ ]:


dataFrameList = [sbm,sbc,sbr] # dataFrameList stores our three DataFrames
for df in dataFrameList: # For each DataFrame in our list
    for col in df.columns: # For each column in that list
        i = 0
        while i < df.shape[0]: # i corresponds to the row
            if isinstance(df[col].iloc[i],str) and df[col].iloc[i][0] == "$": # Check to see that the value we are altering is a string and a dollar amounr
                df[col].iloc[i] = df[col].iloc[i].replace(',','') # Replace the comma with an empty string
                df[col].iloc[i] = df[col].iloc[i].replace('$','') # Replace the dollar sign with an empty string
                df[col].iloc[i] = float(df[col].iloc[i]) # Cast the conversion-friendly value into a float
            i = i + 1
# Note: do not feel uneasy if the above code is difficult for you to understand. 
# Methods will begin to make more sense as we go through our tutorial, the code was simply necessary at this point for us to proceed properly.


# # 1. Display

# The first DataFrame we will analyze is sbm (salaries in relation to major), which is the most broadly applicable of our data as it can be relevant regardless of one's college or region. 
# 
# Let's preview our data. Using the **head()** function will allow you to see a preview of the top of the DataFrame

# In[ ]:


sbm.head()
# head() returns the first n rows of the DataFrame
# If no integer is provided, head() will default to returning the first 5 rows


# Nice!
# Now, lets utilize the **tail()** function in order to see the last ten rows of our DataFrame.

# In[ ]:


sbm.tail(10)
# tail() returns the last n rows of the DataFrame, defaults to 5 as head() did


# It appears that this DataFrame is currently being sorted in alphabetical order. Although this may have been a user-friendly way to allow individuals to easily locate data for a major they were curious about, this is not necessarily the most intuitive way for us to view the data. We will revisit this topic when we learn about *sorting* in Pandas.

# # 2. Data Attributes

# In our section on **data attributes**, we will explore and identify the various attributes of sbc, which is our DataFrame that relates salaries to colleges and college types.
# 
# Run the following cells to preview the data and get a feel for it:

# In[ ]:


sbc.head()


# In[ ]:


sbc.tail()


# This DataFrame offers us more categorical data than the one we used in section one. Lets utilize the **data attributes **of the DataFrame to learn more about it!

# It is important to know what the indices of a DataFrame entail while you analyze it. Even if there are too many indices to keep track of, having easy access to the information could still prove useful.
# 
# Let's access the **index** data attribute of our sbc DataFrame:

# In[ ]:


sbc.index
# The index data attribute is an immutable ndarray, that implements an ordered and sliceable array - it stores the axis labels for any panda objects


# Similar to accessing the DataFrame's indices, it may also prove useful to have access to the **columns** of a DataFrame

# In[ ]:


sbc.columns
# The columns data attribute contains labels for the DataFrames' columns


# To find out how many different colleges are represented in our DataFrame, we will utilize the **shape** data attribute:

# In[ ]:


sbc.shape[0]
# the shape data attribute contains the DataFrame's dimensions in the format shape[rows,columns]
# shape[0] allows us to access the number of rows


# We can also find out how many columns we have in our DataFrame utilizing shape as well:

# In[ ]:


sbc.shape[1]
# shape[1] allows us to access the number of columns


# # 3. Select

# We can also utilize Pandas to **select** specific subsets of our DataFrame.
# 
# Let's continue with the same DataFrame as above, run the next cell to display a preview of the DataFrame:

# In[ ]:


sbc.head()


# There are very many colleges in our DataFrame (269 to be exact), and although it is great to have a lot of data, it may also make it harder to find the specific data we need. Harvey Mudd College was a college that I was very interested in when I was a high school senior. Let's utilize **.****loc[]** to select this college and gather some more information on it:

# In[ ]:


sbc.loc['Harvey Mudd College']
# .loc[] accesses a group of rows and columns by utilizing the label of a specific index


# If I wanted to traverse the DataFrame to access information on more than one college, we can do that as well by utilizing **.loc[]** with a list of the names of the colleges we would like data for:

# In[ ]:


sbc.loc[['Cooper Union', 'Harvey Mudd College','Amherst College','Auburn University']]
# Note: Make sure to remember the double brackets when locating more than one item! Remember, you are locating a list of colleges.


# Nice! We can utilize **.loc[]** to compare multiple colleges directly and efficiently. 
# 
# Specific data can also be accessed via their indices, we can use **.iloc[]** to access the relative middle of our DataFrame:

# In[ ]:


sbc.iloc[100:-100]
# iloc[] allows for integer-based indexing
# Can be used in the form iloc[x:y] in order to find a subset of data containing the entire row
# Can also be used in the form iloc[x,y] in order to access specific rows and columns, and the two implementations 
# can be combined for even more specificity.


# Another form of selection is **conditional selection**, where we select rows based on specific conditions. Many individuals aspire to be making a "six figure salary" in their careers. Let's take a look at which majors could potentially provide for their aspirations by mid-career:

# In[ ]:


sbm[sbm['Mid-Career Median Salary'] > 100000]
# We do this by designating a condition, which goes inside the outer pair of brackets
# This will return only those rows that satisfy the condition
# NOTE: To combine filtering conditions in Pandas, use bitwise operators ('&' and '|') not pure Python ones ('and' and 'or')


# Looks like the adage proves true: Engineering seems to be a lucrative degree!

# To dig a bit deeper, we can also see what types of schools can potentially provide their alumni with a six-figure salary after graduation:

# In[ ]:


sbc[sbc['Mid-Career Median Salary'] > 100000]


# Interestingly, this data seems to show some deviance from what we would have assumed if we had only looked at the data that relied on major. There are many state schools and schools that are focused on the liberal arts that could lead one to a hefty salary!

# # 4. Summarize

# Beyond analyzing data for specific cases, it is also important that we have the ability to summarize our data in order to see commonalities and aggregations of our data.

# Consider the following scenario: Beyond being unsure of their major or which college they wish to attend, a student is not sure whether they should attend college at all. How could they utilize these DataFrames to analyze whether or not college could be worth it to them financially?
# 
# According to SmartAsset.com, the average yearly income, mid-career, for an individual whose highest educational attainment is high-school is about $35,256.
# 
# In order to give this scenario the proper care it needs, we will calculate both definitions of "average," the **median** and the **mean**, and we will calculate these values for both of the DataFrames we have been using thus far to ensure we can make the most of our data.

# First, lets calculate the two values for our sbm DataFrame:

# In[ ]:


round(sbm['Mid-Career Median Salary'].mean(),2)
# First, we select only the subset of data we need, by designating the 'Mid-Career Median Salary' column, and then summarize utilizing mean()
# Round the value to two decimal places, for readability and since we are comparing dollar values


# In[ ]:


round(sbm['Mid-Career Median Salary'].median(),2)
# Same process, as above, but we instead summarize by utilizing the median (50th percentile)
# Calculating the median of a median salary may seem redundant, but the median we are provided is the median for the particular degree,
# while the new median we are calculating is the median of all of the degrees' mid-career median salary.


# Now, let's also run the same summaries on our sbc DataFrame:

# In[ ]:


round(sbc['Mid-Career Median Salary'].mean(),2)


# In[ ]:


round(sbc['Mid-Career Median Salary'].median(),2)


# In the worst-case scenario, which is the median value of $72,000 annually mid-career across all majors, it seems that on average attending college could lead to great fiscal benefits, netting individuals nearly double what they would without attending college. 
# 
# Of course, it is important to think about the costs (as well as opportunity costs) associated with going to college, but nevertheless this data should provoke careful thought and consideration of attending college.

# However, averages do not tell the full story of how someone may fare once they reach the middle of their career. Lets gather more info by using **.std()** in a similar fashion to how we used **.mean()** and **.median()** in order to analyze the spread of the values.

# In[ ]:


round(sbm['Mid-Career Median Salary'].std(),2)
# Utilizing the std() function allows us to obtain the standard deviation
# The standard deviation allows us to see the spread of values from the average, which can be used to gauge variability depending on our choices


# There is a **standard deviation** of $16,088.40 in mid-career salaries, which means that there is significant spread. This means we can't take the average at face value, and still have to consider our major more specifically and in a larger context.

# In[ ]:


round(sbc['Mid-Career Median Salary'].std(),2)


# Again, salaries between colleges seems to have a large spread as well. Choosing one college over another may lead to a significant variation in salary as one reaches the middle of their career.

# It is important that we look at all the data we can in order to get viewpoints from many different angles. Thankfully, using Pandas makes it a lot easier to do so!

# # 5. Sort

# Although the data has proven very useful to us thus far, the manner in which the data is being sorted may not be the most intuitive for how we want to look at the data. 
# 
# 
# Let's utilize the **sort_values()** function in order to display the DataFrame in order of mid-career median salaries of the various college majors in descending order.
# 
# Then, we'll call **head()** to display the top 5 most lucrative majors.

# In[ ]:


sbm = sbm.sort_values('Mid-Career Median Salary',ascending=False) #ascending is a boolean, 
                                                #when false DataFrame will be sorted in descending order
# sort_values allows us to sort our DataFrame by axis items
sbm.head()


# Utilizing **sort_values()** allows us to quickly see what the best-paying college majors are, and we can now easily compare one major to another by seeing where it is placed in our sorted DataFrame!

# In regards to the DataFrame that relates salaries to specific colleges (sbc), the data is presented alphabetically by the column 'college type', it might be more helpful to see the colleges themselves listed alphabetically so one could easily locate their college of interest on the list.
# 
# Let's use the **sort_value()** function on sbc, this time make sure that the values are ascending alphabetically, not descending.

# In[ ]:


sbc = sbc.sort_values('School Name')
sbc.head()


# What is the highest *starting* salary indicated on the dataframe? to find out lets utilize the **.max()** function, which will give us the maximum value of the requested axis:

# In[ ]:


sbm['Starting Median Salary'].max()
#First, we will select the 'Starting Median Salary' column, and then use .max() in order to find its maximum value


# Knowing the highest starting salary is great! But you're probably wondering "I want that salary, what degree do I need to obtain that?" Thankfully, we can utilize **.idxmax(**) to identify the indice of the maximum value in 'Starting Median Salary'

# In[ ]:


sbm['Starting Median Salary'].idxmax()
# Similar set up to the previous code cell
# .idxmax() returns the row label of the maximum value, rather than the maximum value itself


# Also keep in mind that Pandas incorporates .min() and .idxmin(), which function in the same way but can be used to find minimums.

# # 5. Split-Apply-Combine

# For this section, we will also take a look at sbr, which is very similar to sbc but designates schools by their regions rather than the type of college.
# 
# Run the following cell to preview the DataFrame:

# In[ ]:


sbr.head()


# To enable us to do some more complex analysis of our data, we will be practicing **split-apply-combine**, which works in the following way:
#     1. We **split**the dataset depending on a certain chararcteristic, or column
#     2. We **apply** a function which allows us to aggregate information for each of the split groups
#     3. We **combine** this new group-wise information and create a new dataset

# The **.groupby()** function will allow us to split our dataset based on a grouping. In this DataFrame, a natural way to approach it seems to be grouping by Region, let's try it!

# In[ ]:


sbr.groupby('Region')['Starting Median Salary'].mean()

# We facilitate split-apply-combine functions through .groupby(), which groups a DataFrame by a Series of columns
# We group our colleges together by their region, then we apply the .mean() method to the 'Starting Median Salary'
# and finally, that data is returned to us as a new dataset

# Although we could find the mean first and then select the 'Starting Median Salary' column, this is less efficient
# Because it means we will have to take the mean of every column and then select from that data, 
# rather than only compute for the specific data we are interested in


# According to this data, it seems that attending a Californian or Northeastern schools may lead to the biggest payout.

# # 6. Frequency Counts

# Frequency counts are a useful analytical tool that allows us to see how many unique elements there are in our DataFrame, along with how many times each occurs. 
# 
# There are two ways to go about doing this:
#     1. Using the **.size()** method on a GroupBy object
#     2. Using **.value_counts()** on the column you would like to split your dataset by

# The first method, using .size(), is similar in its implementation to our previous example:

# In[ ]:


sbr.groupby('Region')['Starting Median Salary'].size()
# Split by the Region
# Apply the .size() function to the 'Starting Median Salary' column
# Combine the data into a new dataset


# We can use .value_counts() to obtain similar results, albeit with a different underlying implementation:

# In[ ]:


sbr['Region'].value_counts()
# We select the 'Region' column, and then apply .value_counts(), which is a function that returns a Series containing counts of unique values
# Notice that .groupby() above returned a Series with the same name as the column, while
# .value_counts() on its own returned a Series named after the Region, .value_counts() is also in descending order of values,
# while the .groupby() implementation is ascending in alphabetical order


# # 7. Plots

# A very important part of analyzing data is being able to quickly spot trends. Graphs are a perfect way to do that, and thankfully Pandas has some great ways to view your data.

# To visualize the salaries that different types of colleges may lead to, let's visualize the mean 'median starting salary' of each type of college:

# In[ ]:


sbc.groupby('School Type')['Starting Median Salary'].mean().plot.bar()
# To begin, we split the schools up into their respective types
# Then, we apply the mean() to the 'Starting Median Salary' column, which gives us a new dataset
# Then, we plot() the dataset, and use a .bar, which present us with a bar plot to easily visualize the data


# It may also be interesting to see the breakdown of the counts of the different types of schools, and visualize this information in an easy pie chart:

# In[ ]:


sbc['School Type'].value_counts().plot.pie()
# We first create a new dataset, which utilizes .valuecounts() in order to obtain a Series containing the counts of unique elements
# Then we use .plot(), specifying pie, in order to create a pie chart


# Even in cases where one requires very specific data, rather than big picture data, they can utilize .plot() in Pandas to clearly assess trends. 
# 
# For example, if you are a student debating which Ivy League college you would like to attend (lucky you!), you may have a very specific criteria: you want to go wherever could potentially net you the greatest salary right after graduation. Pandas will allow us to visualize that!

# In[ ]:


sbc[sbc['School Type'] == 'Ivy League'].sort_values('Starting Median Salary')['Starting Median Salary'].plot.barh()
# First, we use conditional selection to only select Ivy League Schools
# Then, we want to sort our values by 'Starting Median Salary', and then select only that column for our dataset as well using ['Starting Median Salary']
# We then simply proceed as before, and utilize plot(), this time specifying barh to indicate we want a 


# # Conclusion

# This data, and its analysis thus far, has many implications. A common theme in the highest paying salaries, across all levels, was their STEM basis. Additionally, it seems that the Ivy League and technical-engineering schools have alumni with the highest salaries. This was not especially surprising. However, analysis of this data also shows that many students from liberal arts school as well as state schools are doing well for themselves. Now that you know how to use Pandas to analyze this data, you can also fork the kernel to do conduct your own analyses to really be informed! Overall, this data serves as another source of information for anyone making exciting decisions regarding college in their life. I hope you learned a thing or two about Pandas or college from my tutorial, and wish you the best of luck in all your future endeavors! (:
