#!/usr/bin/env python
# coding: utf-8

#  ## **Data Walkthrough**

#  The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer
# *  Number of instances : 306
# *  Number of Attributes : 4 
# 
# 
# ### Attribute Information:-
# 
# 
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 4. Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year

# In[ ]:


#Enivronment Configration 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
import seaborn as sns


# In[ ]:


#loading the dataset into dataframe

haberman_df=pd.read_csv('../input/haberman.csv')
print(haberman_df.head())


# In[ ]:


#Column names are not readable from above view 

haberman_df.columns = ["age", "operation_year", "axillary_lymph_node", "survival_status"]


# In[ ]:


print(haberman_df.head())


# ### Statistics of the data

# In[ ]:


print(haberman_df.info())
      


# In[ ]:


haberman_df.shape


# In[ ]:


haberman_df.describe()


# ### Observation:
# 1. There are no none values in the dataset. 
# 2. The person suffering from the cancer are in the range of 30years to 83 years.
# 3. Number of auxillary lymph node detected is in the range of  0 to 52.
# 4. Survival status is like a categorical value where it has either 1 i.e, the patient survived 5 years or longer and 2 i.e., the patient died within 5 year

# ## Univarte Analysis
# ### 1. Distrubtion Plot (PDF(Probability Density Function))

# Objective is to predict the patient survivied for more than 5 years based on age , year of operation and number of positive auxiliary nodes.
# 
# Here in distrubution plot the data points are grouped into bins and the height of the bar denotes the percentage of data points under the corresponding group i.e, histogram on smoothening the histogram we can arrive at Probability Density Function it is the probability that the variable takes a value x.

# In[ ]:


for idx, feature in enumerate(haberman_df.columns[:-1]):
    g = sns.FacetGrid(haberman_df , hue="survival_status" , size = 5)
    g.map(sns.distplot , feature , label = feature).add_legend()
    plt.show()


# ### 2. CDF

# In[ ]:


plt.figure(figsize=(5,20))
for index, feature in enumerate(haberman_df.columns[:-1]):
    plt.subplot(3,1,index+1)
    counts, bin_edges = np.histogram(haberman_df[feature] , bins = 10 , density = True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(feature)


# ## 3. Box plots

# In[ ]:


fig, ax = plt.subplots(1,3, figsize = (15,5))
for idx, feature in enumerate(haberman_df.columns[:-1]):
    sns.boxplot(x = 'survival_status' , y = feature , data = haberman_df , ax = ax[idx])
plt.show()


# ## 4.Violin Plots

# These are combination of box plots and probability density function.

# In[ ]:


fig, ax = plt.subplots(1, 3, figsize= (15,5))
for idx, feature in enumerate(haberman_df.columns[:-1]):
    sns.violinplot(x = 'survival_status' , y = feature , data = haberman_df , ax = ax[idx])
plt.show()


# ### Observation
# 
# 1. Around 80% of  Cancer suriviors have  0 to 5 postive auxillary lymph node
# 2. Most patients at the age of 50 -60 have survived more than 5 years.

# ## Multivariate Analysis

# Multivariate analysis is the realation between multiple variables and its realtion with the target variable 

# ### 1. Scatter Plot

# Scatter plot is visual reprsetention of relation between two variables and by using the scatter plot we can arrive at the linear non linear realtion between the variables

# In[ ]:


sns.set_style('whitegrid')
g = sns.FacetGrid(haberman_df , hue = 'survival_status' , size = 5 )
g.map(plt.scatter , "age", "operation_year")
g.add_legend()
plt.title("2-D scatter plot for age and operation_year")
plt.show()


# In[ ]:


sns.set_style('whitegrid')
g = sns.FacetGrid(haberman_df , hue = 'survival_status' , size = 5 )
g.map(plt.scatter , "age", "axillary_lymph_node")
g.add_legend()
plt.title("2-D scatter plot for age and operation_year")
plt.show()


# ### Observations:
# 
# 1. In the first plot between opertion year and age its not linearliy separable.
# 2. In the second plot between axillary lymph node and age most of the positive node in the range of 0-5 servived.

# ### Pair Plot

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(haberman_df , hue = 'survival_status' , vars = ['age' , "operation_year", "axillary_lymph_node"] , size = 4)
plt.suptitle("Pair plot of age, opertion year and axillary lymph node with survival status")
plt.show()


# ### Observations
# 1. Most of them overlapping so we cant clearly classify but there is better seperation between operation year and axillary lymph node

# In[ ]:




