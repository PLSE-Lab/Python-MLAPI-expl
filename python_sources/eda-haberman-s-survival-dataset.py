#!/usr/bin/env python
# coding: utf-8

# # Haberman's Survival dataset

# ### ***Description***
# 
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# ### ***About the Dataset***
# 
# **Title**: Haberman's Survival Data
# 
# #### Sources: 
# 
# (a) Donor: Tjen-Sien Lim (limt@stat.wisc.edu) 
# 
# (b) Date: March 4, 1999

# **Relevant Information:** The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# **Number of Instances:** 306
# 
# **Number of Attributes:** 4 (including the class attribute)
# 
# **Attribute Information:**
# 
#     1. Age of patient at time of operation (numerical)
#     2. Patient's year of operation (year - 1900, numerical)
#     3. Number of positive axillary nodes detected (numerical)
#     4. Survival status (class attribute) 1 = the patient survived 5 years or
#        longer 2 = the patient died within 5 year
#     5. Missing Attribute Values: None

# ## ***Objective:***
# 
# **Our objective is to perform EDA on Haberman data. <br>Such that the patients with different survival status could be differentiated easily.**
# 
# This dataset contents records of 306 patients and their survival status after year of operation as:
# 
#     1. Survived more than 5 years
#     2. Died before 5 years

# ## ***Exploratory Data analysis***

# ### Importing Libraries

# In[ ]:


#Importing libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

#Filtering out the warnings
import warnings
warnings.filterwarnings("ignore")
sns.set()


# ### Reading data from  CSV file

# In[ ]:


#Reading haberman.csv to a Dataframe named df
# load the dataset
df = pd.read_csv('../input/haberman.csv', header=None, names=['age', 'year', 'nodes', 'status'])
print(df.head())


# ### Data Preprocessing and High Level Statistics

# In[ ]:


df['status'].replace(2,0, inplace = True)


# ### *Observation:*
# 
# * *Modified target column 'status'*, such that -  
# 
#    * '0' means that the patient did not survive for more than 5 years,
# 
#    * '1' means the patient has survived for more than 5 years.

# In[ ]:


#Dataframe info
df.info()


# ### *Observation:* 
# 
# * From this I can infer than there are 306 rows and all features are non-null.  
# * This is excellent as now we don't have to handle null values.

# In[ ]:


# Dataframe describe
df.describe()


# ### *Observations*
# 
# From the ***describe*** function we now know that -
# 
# ***Age***
# * The average age is 52.4 years
# * The youngest patient is 30 years old
# * The oldest patient is 83 years old
# 
# ***Year***
# * 1958 is the year the survey (data collection) began
# * The data collection went on till 1969
# 
# ***Nodes***
# * There are quite a few patients with 0 positive axillary nodes
# * On average the patients had 4 nodes
# * There was also instances where patients had positive axillary as high as 52

# In[ ]:


#Shape of the dataframe
print("Shape of the dataframe. Number of rows and columns -",df.shape)


# In[ ]:


#Initial 5 rows of the dataframe
df.head()


# In[ ]:


#Last 5 rows of the Dataframe
df.tail()


# In[ ]:


#Column names of Dataframe
df.columns


# In[ ]:


print("No. of rows: "+str(df.shape[0]))
print("No. of columns: "+str(df.shape[1]))
print("Columns: "+", ".join(df.columns))
print("Target variable distribution")
print(df['status'].value_counts())
print("*"*40)
print(df['status'].value_counts(normalize = True))


# * *This is a binary classification problem as there are only 2 classes.*
# * *The dataset is quite imbalanced as one class has 73% records while the other has only 26% records.*

# In[ ]:


#Total no. of patients every year
print(df['year'].value_counts().sort_index());

# As no. of patients every year are not nearly equal there is a high probability   
# that more people might die / survive in year with higher no. of patients.


# In[ ]:


#Patients who survived with 0 axilary nodes
ax_0 = (df['nodes'] == 0).value_counts()[1]
surv_0 = ((df['status'] == 1) & (df['nodes'] == 0)).value_counts()[1]
dead_0 = ((df['status'] == 0) & (df['nodes'] == 0)).value_counts()[1]
print("No. of patients with 0 axiliary nodes : ", ax_0)
print("No. of patients who had 0 axiliary nodes and survived : ",surv_0)
print("No. of patients who had 0 axiliary nodes and still died : ",dead_0)


# In[ ]:


#color co-ordinating 'status' feature as green and red
colors = {1: 'green', 0: 'red'}


# ## (1.) Univariate Analysis   
# 
# ## *(1.1)  Histograms*

# In[ ]:


#plotting histograms for all features
sns.set_style('darkgrid')
for idx,feature in enumerate(df.columns[:-1]):
    f = sns.FacetGrid(df, hue = 'status',height = 6, palette = ['red','green'])           .map(sns.distplot, feature)           .add_legend()
plt.show()


# ### *Observations*
# * From the histogram of age feature it might look like as if patients between age 30 and 34 have high probability of survival,  
#     and patients above 76-77 don't survive. 
# * PDF of patients who died is normally distributed.
# * But that **might not** be the case because considering the **dataset is imbalanced**, these kind of illegible analysis is pretty much futile.
# * If a patient has only 1-2 nodes, his chances of survival increases drastically.
# * Most of the data (especially year and age) is overlapping too much to differentiate between survival_status.

# ## (1.2.) *PDF and CDF (*Cumulative density function*)*

# In[ ]:


#pdf and cdf of all features
plt.figure(figsize=(20,5))
for idx, features in enumerate(df.columns[:-1]):
    plt.subplot(1, 3, idx+1)
    counts, bin_edges = np.histogram(df[features], bins = 10, density = True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,'bo-')
    plt.plot(bin_edges[1:],cdf,'k^-')
    plt.title("pdf and cdf of "+str(features))
    plt.xlabel(features)
    plt.ylabel("Percentage")
    labels = ['pdf','cdf']
    plt.legend(labels)


# ### *Observation:*
# *  *Age*  
#    * Only 15% patients are younger than 40.
#    * More than 60% patients are older than 50.
#    * Around 90% patients are younger than 67.  
# 
# 
# *  *Year*
#    * 20% operations were performed before year 1959.  
# 
# 
# *  *Nodes*
#     * Almost 80% patients had nodes less than 3.
#     * Only 15-17% patients had nodes greater than 10.

# In[ ]:


#Creating seperate dataframes for patients with different status
dead = df.loc[df["status"] == 0]
surv = df.loc[df["status"] == 1]


# In[ ]:


# Status of patients who were dead within 5 years of Operation year
dead.describe()


# ### *Observations :*
# * 81 patients died before 5 years after year of operation
# * Average age of patients who died was 53 (At the time of operation)
# * Youngest patient was 34 years old
# * Oldest patient was 83 years old.

# In[ ]:


# Status of Patients who survived
surv.describe()


# ### *Observations :*
# * 225 patients survived for more than 5 years after year of operation
# * Average age of patients who survived was 52 (At the time of operation)
# * Youngest patient was 30 years old
# * Oldest patient was 77 years old.

# In[ ]:


#PDF's and CDF's of all features for both survived and dead patients
plt.figure(figsize=(20,5))
for idx, features in zip(enumerate(surv.columns[:-1]),enumerate(dead.columns[:-1])):
    plt.subplot(1, 3, idx[0]+1)
    counts, bin_edges = np.histogram(surv[features[1]], bins = 10, density = True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,'g--')
    plt.plot(bin_edges[1:],cdf,'go-')
    
    counts, bin_edges = np.histogram(dead[features[1]], bins = 10, density = True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,'r--')
    plt.plot(bin_edges[1:],cdf,'ro-')
    
    plt.title("PDF and CDF of "+str(features[1]))
    plt.xlabel(features[1])
    plt.ylabel("Pecentage of distribution")
    labels = ['PDF of patients survived','CDF of patients survived',              'PDF of patients dead','CDF of patients dead']
    plt.legend(labels)


# ### *Observations :*
# 
# *   **Age**
#     * More than 40% patients were younger than 50 and died before operation
#     * 30% were younger than 45 who were alive.
#     * 18% were younger than 45 who died before 5 years.
#     
# *   **Year**
#     * Slightly Higher percentage of patients died in operation year 59 - 60 and 65 - 67
#     * Slightly Higher percentage of patients survived between 61 and 65
#     
# *   **Nodes**
#     * Almost 90% patients who survived had less than 5 nodes
#     * 60% who died had less than 5 nodes
#     * Really few % of patients who died had 25+ nodes.

# ## (1.3.) *Box Plots*

# In[ ]:


#Box Plot using patients age.
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.figure(1)
plt.subplot(161)
sns.boxplot(x='status',y='age',data=df,notch = True,palette=['red','green']).set_title("Box plot for age")

#Box Plot using Patients operation year.
plt.subplot(163)
sns.boxplot(x='status',y='year',data=df,notch = True,palette=['red','green']).set_title("Box plot for year")

#Box Plot using no. of positive axillary nodes.
plt.subplot(165)
sns.boxplot(x='status',y='nodes',data=df,notch = True,palette=['red','green']).set_title("Box plot for nodes")
plt.show()


# ### *Observations :*
# 
# *   **Age**
#     * Age box plot overlap almost exactly
#     
# *   **Year**
#     * Year box plot overlap almost exactly, but it seems like the patients who had their operation after 1960   
#         had slightly better chance of survival.
#     
# *   **Nodes**
#     * Quite more than 75% patients who survived has nodes less 4.
#     * Patients who died seems to have more nodes as we can see by comparing the size of the box plots.
#     * Patients who survived also have quite a no. of outliers in case of no. of nodes
#     
#     
# * The notch shows a 95% confidence interval which is a range of values that you can be 95% certain contains the true mean/median of the population. 

# ## *(1.4.) Violin Plots*

# In[ ]:


#violin plots
for idx, features in enumerate(surv.columns[:-1]):
    sns.violinplot(x = 'status', y = features,hue = 'status', data = df,palette=['red','green']).set_title("Violin plot for "+str(features)+" and survival status")
    plt.show()


# #### *Observations :*  
# 
# Remarkably, most of data is overlapping too much to differentiate different survival_status.
# * #### *Nodes*
#     * The plot is quite skewed due to large no. of outliers in the feature
#     * More than 75% patients who survived had nodes less 4.
#     * As 50th percentile for axillary node is zero, from above plots we can see that patients with less number of axillary node 
#       have higher rate of survival.
#     * Patients who died seems to have more nodes as we can see by comparing the size of the box plots.
#     * There seems to be a lot of Nodes as an outlier in case of both who survived and those who didn't.

# ## (2.1.) *Pair Plot*

# In[ ]:


#pairplot
sns.set_style("darkgrid")
sns.pairplot(df, hue = 'status', height = 4,vars = ['age', 'year', 'nodes'],palette=['red','green']);


# ### Observations :
# 
# * Similarly in pair plots, most of data is overlapping too much to differentiate different survival_status.
# * None of the features helps in classifying the data.
# * There are lots of patients with very few axillary nodes (None or 1 node).
# * But with a dataset as imbalanced as this, this might be misleading.
# * We know that No. of patients who had 0 axillary nodes and still died are 19 patients,  
#     but the 0 nodes region doesn't showcase anything like this.  
# * This is a huge drawback of scatterplot (even if it's colored by class label)   
#     and can be very misleading.

# ## (2.2.) *Heatmap*

# In[ ]:


X = df
# print(df.corr())
sns.heatmap(X.corr(),cmap="Blues");


# ### *Observation :*
# * axil_nodes shows some correlation with status.

# ## *Conclusion :*  
# 
# * If a patient has only less than 3 nodes, their chances of survival increases drastically.
# * Most of the data (especially year and age) is overlapping too much to differentiate between survival_status.
# * The dataset is quite imbalanced, hence this kind of illegible analysis is pretty much futile if the objective is classification.

# In[ ]:




