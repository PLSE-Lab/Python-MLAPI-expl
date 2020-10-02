#!/usr/bin/env python
# coding: utf-8

# # Haberman's Cancer Survival Dataset: EDA

# ### Description:
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# ### Attribute Information:
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 4. Survival status (class attribute):
#     1 = the patient survived 5 years or longer,
#     2 = the patient died within 5 year

# ## 0. Environment Configuration and Data Collection

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Load haberman.csv into a pandas dataFrame.
df = pd.read_csv('../input/haberman.csv',                  header=None,                  names=['Age', 'Op_Year', 'axil_nodes', 'Surv_status'])


# In[ ]:


#How many data-points and features
print (df.shape)


# In[ ]:


#Different data-points for Survival status
print(list(df['Surv_status'].unique()))


# In[ ]:


# 1=> positive, 2=> negative
df["Surv_status"]=df["Surv_status"].map({1:'positive',2:'negative'})


# In[ ]:


df.head()


# Changed the Survival_Status column to positive -> patient survived for 5 years or more, negative -> patient survived for less than 5 years.

# In[ ]:


#Balanced dataset or not?
print (df["Surv_status"].value_counts())
print("*"*50)
print (df["Surv_status"].value_counts(normalize= True))


# Target column is imbalanced with more than 73% positive result.

# In[ ]:


#Dividing the dataset into 2 datasets of positive and negative result.
positive=df.loc[df['Surv_status']=='positive']
negative=df.loc[df['Surv_status']=='negative']


# ## 1. Objective
# To identify if a patient will survive cancer treatment for more than 5 years or not
# based on his age, year_of_operation and no_of_positive_axillary_nodes. 

# ## 2. Statistical Analysis

# In[ ]:


print (df.describe())


# In[ ]:


#Mean and Std-Deviation
print ("Age")
print ("  Mean:")
print ("  positive result- "+str(np.mean(positive["Age"])))
print ("  negative result- "+str(np.mean(negative["Age"])))
print ()
print ("  Standard Devation:")
print ("  positive result- "+str(np.std(positive["Age"])))
print ("  negative result- "+str(np.std(negative["Age"])))
print ()
print("*"*50)
print ()
print ("Year of Operation")
print ("  Mean:")
print ("  positive result- "+str(np.mean(positive["Op_Year"])))
print ("  negative result- "+str(np.mean(negative["Op_Year"])))
print ()
print ("  Standard Devation:")
print ("  positive result- "+str(np.std(positive["Op_Year"])))
print ("  negative result- "+str(np.std(negative["Op_Year"])))
print ()
print("*"*50)
print ()
print ("No of Auxillary Nodes")
print ("  Mean:")
print ("  positive result- "+str(np.mean(positive["axil_nodes"])))
print ("  negative result- "+str(np.mean(negative["axil_nodes"])))
print ()
print ("  Standard Devation:")
print ("  positive result- "+str(np.std(positive["axil_nodes"])))
print ("  negative result- "+str(np.std(negative["axil_nodes"])))



# In[ ]:


#90th percentile
print ('90th Percentile')
print ()
print ("Age")
print ("  positive result- "+str(np.percentile(positive["Age"],90)))
print ("  negative result- "+str(np.percentile(negative["Age"],90)))
print("*"*50)
print ("Year of Operation")
print ("  positive result- "+str(np.percentile(positive["Op_Year"],90)))
print ("  negative result- "+str(np.percentile(negative["Op_Year"],90)))
print("*"*50)
print ("No of Auxillary Nodes")
print ("  positive result- "+str(np.percentile(positive["axil_nodes"],90)))
print ("  negative result- "+str(np.percentile(negative["axil_nodes"],90)))
print ("  general result- "+str(np.percentile(df["axil_nodes"],90)))


# ### Observations:
# 
# -  There is no missing data.
# -  The age of patients ranges from 30 years to 83 years
# -  The max no. of axillary nodes detected is 52.
# -  The age and year_of_operation do not have a huge impact on the patient's survival.
# -  The no_of_axillary_nodes_detected are less in patient's with postive result than with negative result having a mean of 2.8 and 7.5 respectively. 
# -  25% of the patients do not have any axillary node while 75% of them have less than 5 and it reaches to 13 for 90% of the patients.

# ## 3. Univariate Analysis
# > ### 3.1 PDF and CDF
# > **Probability Density Function** is a function of a continuous random variable, whose integral across an interval gives the probability that the value of the variable lies within the same interval. <br>
# > **Cumulative Density Function** gives the cumulative sum of the PDF of the variable from minus infinity to that value.

# In[ ]:


#PDF
sns.set_style("whitegrid")
for index, feature in enumerate(list(df.columns)[:-1]):
    sns.FacetGrid(df,hue='Surv_status',height=4).map(sns.distplot,feature).add_legend()
    plt.show()


# In[ ]:


#CDF
plt.figure(figsize=(20,5))
for index, feature in enumerate(list(df.columns)[:-1]):
    plt.subplot(1, 3, index+1)
    print("\n********* "+feature+" *********")
    counts, bin_edges = np.histogram(df[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(feature)


# > ### 3.2 Box Plots and Violin Plots
# > **Box Plot** is a simple way of representing statistical data on a plot in which a rectangle is drawn to represent the second and third quartiles, usually with a vertical line inside to indicate the median value. The lower and upper quartiles are shown as horizontal lines either side of the rectangle.<br>
# > **Violin Plots** are similar to box plots, except that they also show the probability density of the data at different values. It has four layers. The outer shape represents all possible results, with thickness indicating how common. The next layer inside represents the values that occur 95% of the time. The next layer (if it exists) inside represents the values that occur 50% of the time. The central dot represents the median average value.

# In[ ]:


#Box Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(df.columns)[:-1]):
    sns.boxplot( x='Surv_status', y=feature, data=df, ax=axes[idx])
plt.show()


# In[ ]:


#Violin Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(df.columns)[:-1]):
    sns.violinplot( x='Surv_status', y=feature, data=df, ax=axes[idx])
plt.show()


# ### Observations:
# -  The no of axillary nodes in patients with positive result is highly densed from 0 to 5.
# -  80% of patients with positive result have less than 6 axillary nodes.
# -  The patients treated after 1966 have the slighlty higher chance to surive that the rest. The patients treated before 1959 have the slighlty lower chance to surive that the rest.
# -  The patients with age less than 30 years have slightly higher chance of survival while patients with age more than 75 years have slightly lower chance of survival.

# ## 4. Bivariate Analysis
# > ### Pair Plots
# **Pair Plots** represent the relationship between different pair of features and the distribution of every feature over the dataset.

# In[ ]:


sns.set_style("whitegrid")
sns.pairplot(df,hue='Surv_status',height=4)
plt.show()


# ### Observations:
# -  No good separation can be done between the classes based on the pair plots.
# 

# ## Conclusions
# -  Younger people with age less than 30 yrs have a slightly higher chance of survival, whereas older people above the age of 75 years have slightly lower chance of survival.
# -  With better technologies coming the rate of survival increased slightly for patients who where treated after 1966.
# -  The max impact on survival is of the number of positive axillary lymph nodes found in the patient. The lesser the no of lymphs found the higher is the chance of survival.
# -  90% of the patients have less than 8 and 20 axillary nodes for positive and negative result patients respectively.
# -  The mean of no of axillary nodes is 2.8 for patients who survived for atleast 5 years and 8.5 for patients who didn't survive for 5 years.
