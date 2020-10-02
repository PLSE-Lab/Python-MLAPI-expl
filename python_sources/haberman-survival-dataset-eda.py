#!/usr/bin/env python
# coding: utf-8

# ### About the dataset
# 
# Name : Haberman's Survival Data
# 
# About : The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# #### Dataset Description:
# 
# 1. Number of data points - 306
# 2. Number of variable - 4 (including the class label)
# 
# #### Variables 
# 
# 1. age - Age of the patient at the time of operation
# 2. year - Year of the operation
# 3. nodes - Number of positive auxillary nodes
# 4. status - Survival Status of the patient 
# 
# Reference : Kaggle.com
# 1. https://en.wikipedia.org/wiki/Axillary_lymph_nodes
# 2.https://ww5.komen.org/BreastCancer/LymphNodeStatusandStaging.html

# ### Objective
# 
# To predict/classify whether the patient survived or not after 5 years of treatment based on the given independent variables such as age of patient , year of operation and the number of lymph nodes

# In[ ]:


# importing all the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# reading the dataset as pandas dataframe

colnames = ['age', 'year', 'nodes', 'status']
df = pd.read_csv('../input/haberman.csv',header= None , names= colnames)


# In[ ]:


# Q1) How many rows and columns in the dataset

print('Number of rows = {}'.format(df.shape[0]))
print('Number of columns = {}'.format(df.shape[1]))


# In[ ]:


# Q2) Name of the columns of the dataset

print('Columns: '+', '.join(df.columns))


# In[ ]:


# Q3) Numerical statics of the features 

df.describe().transpose()


# ## Obervations 
# - Dataset doesn't have any missing values
# - Minimum age of the patients - 30
# - Maximum age of the patients - 83
# - Around 50% of the patients aged between 44 to 61
# - Around 50% of the patients operated between the years 1960 - 1965
# - 75% of the patients have 4 positive nodes 
# - 25% of the patients have 0 positive nodes

# In[ ]:


# Q4) Find if the dataset is balanced or imbalanced
# status is the class variable/ dependent variable
# We will change the status variable numerical values to categorical

df['status'] = df['status'].map({1:'Yes', 2:'No'})
print(df.status.value_counts())
print('*' * 50)
print(df.status.value_counts(normalize=True))


# #### Observation
# 
# Over 73% of the patients survived more than 5 years and hence the dataset is imbalanced

# As we can see there is huge difference between the class label count.Hence the dataset is imbalanced

# ### Univariate Analysis

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Univariate analysis of the age feature

sns.FacetGrid(df, hue ='status', size= 4)     .map(sns.distplot, 'age')       .add_legend();
plt.show();


# ### Observation
# 
# Overlapping is high. Hence classification of survival status based on age is difficult

# In[ ]:


# Univariate analysis of the year feature

sns.FacetGrid(df, hue = 'status', size= 4)     .map(sns.distplot, 'year')         .add_legend();
plt.show();


# ### Observation
# 
# Overlapping of the points is high. Hence classification of the status is difficult

# In[ ]:


# Univariate analysis of the nodes feature against the survived status

sns.FacetGrid(df, hue= 'status', size= 4)     .map(sns.distplot , 'nodes')        .add_legend();
plt.show();


# ### Observation
# 
# Persons having less number of nodes survived 5 or longer years after the treatment. Also 'nodes' variable very helpful in classifying the survival status than the other two variables. i.e : age and year

# In[ ]:


#Box Plots and Voilin plots
sns.boxplot(x = 'status' , y = 'age' , data= df)
plt.show();


# In[ ]:


sns.boxplot(x= 'status' , y= 'year', data= df)
plt.show();


# In[ ]:


sns.boxplot(x= 'status' , y= 'nodes' , data= df)
plt.show();


# #### Pair Plots

# In[ ]:


plt.close();
sns.set_style("whitegrid")
sns.pairplot(df, hue='status', size= 3)
plt.show();


# ## Observation
# 
# As there is considerable overlap between the variables,classifying the survival status is  difficult.

# ### PDF and CDF

# In[ ]:


survived_df = df[df['status'] == 'Yes']


# In[ ]:


not_survived_df = df[df['status'] == 'No']


# In[ ]:


# PDF and CDF of positive nodes of survived people

#survived_nodes, survived_binegdes = np.histogram(survived_df['nodes'], bins=10)

# Calculating the count and bin edges of the nodes variable
nodes_count , binedges = np.histogram(survived_df['nodes'], bins= 10)

# Probablity density function calculation
pdf_survived = nodes_count / sum(nodes_count)  

# Cummulative density function calculation
cdf_survived = np.cumsum(pdf_survived)

plt.plot(binedges[1:], pdf_survived, binedges[1:], cdf_survived)
plt.title("PDF and CDF Survived Patients based on nodes feature")
plt.xlabel("Nodes")
plt.show();


# In[ ]:


survived_df['nodes'].describe().transpose()


# ### Observation
# * Close to 82% percent of the survived people having nodes less than or equal to 5 nodes. Hence patients having less number of nodes having high chances of survival.

# In[ ]:



# Calculating the counts and bin edges of the nodes of not survived
n_count , bin_edges = np.histogram(not_survived_df['nodes'] , bins=10)

# pdf of the not survived based on nodes
pdf_not_survived = n_count / sum(n_count)
cdf_not_survived = np.cumsum(pdf_not_survived)

# Plotting the PDF and CDF of patients not survived based on the nodes

plt.plot(bin_edges[1:], pdf_not_survived , bin_edges[1:] , cdf_not_survived )
plt.title('PDF and CDF of not survived based on nodes')
plt.xlabel('Nodes')
plt.show();


# In[ ]:


not_survived_df['nodes'].describe()


# #### Observation
# 
# * Clearly we can see that people who are not survived having more number of nodes than the people who survived.

# ### Violin plots

# In[ ]:


# Voilin plots
sns.violinplot(x = 'status' , y = 'age' , data= df)
plt.show();


# In[ ]:


sns.violinplot(x = 'status', y='year' , data= df)
plt.show();


# In[ ]:


sns.violinplot(x='status', y = 'nodes' , data= df)
plt.show()

