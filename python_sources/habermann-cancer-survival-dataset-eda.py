#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir('../input'))


# In[ ]:


import pandas as pd
df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', header=None, names=['age', 'year', 'nodes', 'status'])
print(df.head())


# <b> DATASET </b>
# 1. Dataset of Patients who underwent breast cancer operations

# In[ ]:


df.columns


# <b> Explanation of Each and Every Field </b>
# <li>Age : Age of Patient at the time of operation</li>
# <li>Year : Year of Operation 1900 onwards</li>
# <li>Nodes : Count of positive auxillary Nodes</li>
# <li>Status : 1 if Patient Survived 5 years or longer 
#          2 if Patient Died within next 5 years</li>
# 

#  <b> Objective : </b> Given This Information for Each Patient Predict Whether given Patient Would Survive After Undergoing the operation

# <b> 1. Higher Level Statistical Analysis </b>

# In[ ]:


## What are the Number of Data Points ? 
df.shape[0]


# In[ ]:


## What are Number of Columns ?
df.shape[1]


# Lets Just Replace Values in Our Status Fields With Readable Text So that Data Analysis Becomes Simple

# In[ ]:


#https://stackoverflow.com/questions/23307301/replacing-column-values-in-a-pandas-dataframe
df['status'] = df['status'].map({1 : 'Survived' , 2 : 'Not Survived'})


# In[ ]:


df['status'].value_counts()


# We Have Imbalanced Data Set 
# There are 225 Patients Who Survived after the Operation and We Have Exactly 81 Patients Who Died After the Operation

# ## 1. Bivariate Analysis 

# Bivariate Analysis is analysis involving two variables (Or Feature in our Case ) Bivariate Analysis can be useful to predict response of the target dependent variable if we know value of independent features or variable

# ### 1.1 Scatter Plot

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Following Two Features 
# <li>1 .  Age</li> 
# <li>2 . Axillary Nodes Count</li>
# <n>Seems to Be most important feature Intiutively Lets Plot a Scatter Plot for This Features and See how Separated Datapoints are</n>

# A scatter plot (also called a scatterplot, scatter graph, scatter chart, scattergram, or scatter diagram)[3] is a type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data. If the points are coded (color/shape/size), one additional variable can be displayed. The data are displayed as a collection of points, each having the value of one variable determining the position on the horizontal axis and the value of the other variable determining the position on the vertical axis.[4]
# 
# Ref : https://en.wikipedia.org/wiki/Scatter_plot

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df, hue="status",size=5)     .map(plt.scatter , "nodes" , "age")     .add_legend();
plt.show();


# <b> Observations </b>
# <li> 1. We See That large Amount of People Having Axillary Nodes Count Less Than 10 are Survivours </li>
# <li> 2. There are Small Amount of People Having Axillary Nodes Count Less than 10 but They Died </li>
# <li> 3. People Having More Than 10 Axillary Active Nodes Tend to Non Survivors </li>
# <li> 4. People Having Age Less Than 60 are Usually Survivors </li>

# Let Us be more Specific by getting actual Numbers and Lets Quantify Our Results , 
# We can use .query() method of Pandas To Query Our Data Frame  , 
# You can read about .query() method here https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html

# In[ ]:


## How many Patients are Survivors Who have less than 10 nodes ?
df.query('status =="Survived" & nodes <= 10')['status'].value_counts()


# <b> We can See Clearly that there are 208 that is nearly 68 %  people having less than 10 nodes and are tend to be survivor </b>
# <b> Lets Query for Non Survivours To Confirm whether Our Numbers are Correct or not </b>

# In[ ]:


## How many Patients Are Non Survivors Who have Less than 10 nodes ?
df.query('status =="Not Survived" & nodes <= 10')['status'].value_counts()


# <b> Nearly 32 % of Non Survivors Have less than 10 nodes </b>

# Lets Query for Age also 

# In[ ]:


df.query('status =="Survived" & age <= 60')['status'].value_counts()


# In[ ]:


df.query('status =="Not Survived" & age <= 60')['status'].value_counts()


# Clearly We cant use Age Alone as Feature as We Can't Select a Suitable Threshold For Age from this Plot

# ### 1.2 Pair Plots

# A Pair Plot is collection of Scatter Plots that helps us to understand How Features are related to our Target Class 

# In[ ]:


plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="status", size=3);
plt.show()


# **Observations**
# 1. We can see that Age and Axil Nodes Count Give Us Most Separation in the Data Points of the Two classes although there is Some Overlap.
# 2. Age and Year seems to be not so useful features as we can't come to any correlation between them and Status of The Patients 
# 3. We can Build Simple If Else Based Model on Axil Nodes Count alone and selecting Threshold

# ## 2. Univariate Analysis

# In Univariate Analysis We Just Analyse Only One Variable at a Time to predict the response variable

# One Way to do Univariate Analysis is by plotting Histograms and Probability Distribution Functions

# #### Histogram for Axil Nodes Count

# In[ ]:


sns.FacetGrid(df , hue = 'status' , size = 5)     .map(sns.distplot , 'nodes')     .add_legend();
plt.show()


# **Observations**
# 1. If We Select 10 as Threshold for Axil Nodes For Survivors , There are chances That We can Make Mistake on Non Survivours as from histogram it is clear that at 10 the Height of Histogram for Non Survivors is More than at Survivors for Height 10 
# 2. Better Choice Would be To use a Number Like 5 or 6 , That is it is better to declare Person Having Axil Nodes Count Less than or equal to 5 as Survivor as We can See from Histogram 

# #### Histogram for Age

# In[ ]:


sns.FacetGrid(df , hue = 'status' , size = 5)     .map(sns.distplot , 'age')     .add_legend();
plt.show()


# **Observations**
# 1. Histograms for Both Classes are Overlapping Significantly , Hence it is not a Good Idea to solely use Age as Feature.
# 2. It was clear from Bivariate analysis also.

# #### Histogram for Year of Operation

# In[ ]:


sns.FacetGrid(df , hue = 'status' , size = 5)     .map(sns.distplot , 'year')     .add_legend();
plt.show()


# **Observations**
# 1. Almost Overlapping Histograms for Both the classes , Hence it is year of operation alone is not useful feature towards classification Task

# ### 3. Basic Statistics for Features for Survivor and Non Survivor Classes

# In[ ]:


survived = df.loc[df['status'] == 'Survived']
not_survived = df.loc[df['status'] == 'Not Survived']


# In[ ]:


## What is Mean Age of Survivor ?
print("Mean Age of Survivor is :")
print(np.mean(survived['age']))


# In[ ]:


## What is Mean Age of Non Survivor ?
print("Mean Age of Non Survivor is :")
print(np.mean(not_survived['age']))


# In[ ]:


## What is Mean Axil Nodes Count of Survivor ?
print("Mean Axil Nodes Count of Survivor is :")
print(np.mean(survived['nodes']))


# In[ ]:


## What is Mean Axil Nodes Count of Non Survivor ?
print("Mean Axil Nodes Count of Non Survivor is :")
print(np.mean(not_survived['nodes']))


# In[ ]:


## Lets Compute Median Also 
print("Median Age of Survivor is :")
print(np.median(survived['age']))
print("Median Age of Non Survivor is :")
print(np.median(not_survived['age']))
print("Median Axil Nodes Count of Survivor is :")
print(np.median(survived['nodes']))
print("Median Axil Nodes Count of Non Survivor is :")
print(np.median(not_survived['nodes']))


# Lets Compute 90th Percentile of Axil Nodes 

# In[ ]:


print("\n90th Percentiles:")
print("For Survivors")
print(np.percentile(survived["nodes"],90))
print("For Non Survivors")
print(np.percentile(not_survived["nodes"],90))


# The 90th Percentile for Survivors Just Show That 90% of Survivors Have Nodes 8 or less than that , For Non Survivors 90% of Non Survivors Have Nodes count as 20 or less than That 

# ## 4. Box Plot for Nodes

# In[ ]:


sns.boxplot(x='status',y='nodes', data=df )
plt.show()


# **Observations**
# 1. The 75th Percentile of Suvivors is almost 4 .
# 2. If we select 4 as Threshold we would be right for survivors 75 % of time and for non survivors we would be incorrect 25 % of time

# ## 5. Conclusions & Learnings

# <p> 1 .Count of active axillary nodes is necessary feature for classification of cancer survivours and cancer non survivors. </p>
# <p> 2 . Count of active axillary nodes along with age can provide reasonable accuracy however sometimes exception may still be there.
# <p> 3. If Active Nodes count is less than or equal to 5 it must be controlled immediately , Nodes count of 20 is mostly lethal and person having count more than that will most probably die within next 5 years.</p>
# <p> 4 . There are some exceptions typically 2 - 3 that don't follow above conditions example : (See Pair Plots)
#     a) A Person was having axil nodes count = 0 and age less than 40 and still didnt survive.
#     b) A Person was having axil nodes count = 45 and age nearly 55 and surprisingly survived 

# In[ ]:




