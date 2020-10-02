#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
#Haberman Cancer Survival Data Set
"""The dataset contains cases from a study that was conducted between 1958 and 1970 
   at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer."""
import pandas as pd
# Read the data from CSV file
haberman_set = pd.read_csv("../input/haberman.csv",names=['Age', 'Treatment_year', 'Positive_axillary_nodes', 'Survival_status'])
# Any results you write to the current directory are saved as output.


# In[ ]:


# The total number of rows and columns on the table
print("The number of rows and columns are : ",haberman_set.shape)


# Totally there are 306 rows and 4 columns. The dataset is quite small.

# In[ ]:


# Columns present in the dataset
print(haberman_set.columns)


# In[ ]:


# Display few columns of the dataset to understand about the same
haberman_set.head(10)


# All the variables are numbers and can be used for numerical calculations.

# # Understanding of Dataset

# 1. Age - Age of the person when she undergoes the treatment
# 2. Year - Year of the treatment
# 3. Axillary Nodes (Positive) - This implies whether the cancer affected the lymph nodes or not. If there are positive nodes it
#    indicates that cancer has been spread to lymph nodes
# 4. Status - 1 represents patients survived more than 5 years ; 2 represents patients died within 5 years of treatment.

# ### Objective

# Explore the dataset and identify the method for predicting whether the patient survived for more than 5 years or not based on
# given features.

# In[ ]:


# To know the balance of data, check the number of status 1 and status 2 present in the data set
haberman_set["Survival_status"].value_counts()


# ### Observations

# 1. From the above list it is clear that the data set is quite imbalance.
# 2. About 73% of the data are resulting with status 1 (Patients survived more than 5 years) and only 27% data are resulted in status 2(died within 5 years).

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
#Lets plot 2D scatter plot to understand the distribution of status based on age and positive nodes
#Chosen age and nodes since they are resembling quite important than year variable
sns.set_style("whitegrid")
sns.FacetGrid(haberman_set,hue="Survival_status",height=5)    .map(plt.scatter, "Age", "Positive_axillary_nodes")    .add_legend();
plt.show();


# The points are scattered in all the regions and this is very hard to find the separation of status. Lets try to get
# pair plots with the variables to get more insights.

# In[ ]:


# Plot the 2D scatter plots for age, year and nodes based on status
sns.set_style("whitegrid");
sns.pairplot(haberman_set, hue="Survival_status", height=5,vars=['Age','Treatment_year','Positive_axillary_nodes']);
plt.show()


# ### Observations

# 1. From the above pair plots also we didn't find any useful information for bifurcating the statuses based on the provided
#    variables.
# 2. The year data is not providing any useful information since both the points are scattered across almost all mentioned years.
# 3. Now lets try to find the classification based on univariate analysis

# ### Univariate Analysis - Histogram

# In[ ]:


# For doing the univariate analysis, first plot the histogram for all the variables based on status
# to get the distribution
sns.FacetGrid(haberman_set, hue="Survival_status", size = 5)    .map(sns.distplot, "Positive_axillary_nodes")    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(haberman_set, hue="Survival_status", size = 5)    .map(sns.distplot, "Age")    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(haberman_set,hue="Survival_status", size = 5)    .map(sns.distplot, "Treatment_year")    .add_legend()
plt.show()


# ### Observations from Histogram

# 1. From Histogram we can come to a conclusion that year variable is not very useful for classifying the status. Adding to that 
#    the age is also not providing any good insight since both level patients are there in almost all age groups.
# 2. The dataset is ofcourse imbalance, but from the histogram of nodes (axillary positive nodes) we come to know that if the        number of positive axillary nodes are less, then there is a fair chance for the patient to survive more than 5 years. We        cannot conclude the same but we will get some insight if we dig deep on to the nodes variable.
# 3. Number of positive axillary nodes is a good feature for classification in the provided dataset.

# ### Univariate Analysis of Axillary positive nodes

# In[ ]:


#Let's try to plot the PDF and CDF for the positive nodes variable to get some more insights
# For that first we will separate the status 1 and status 2 data in a separate frame for analysis purpose
haberman_1 = haberman_set.loc[haberman_set["Survival_status"] == 1]
haberman_2 = haberman_set.loc[haberman_set["Survival_status"] == 2]


# In[ ]:


import numpy as np
# Get the counts and bin edges by giving the bins required
# Numpy will split the nodes equally into given number of bins. For example bins = 10
# means numpy divides 10 equal parts from 0 to 46. Each reqpresents a bin edge
# PDF contains the percentages of status - 1 on that particular window.

# Divided into 10 bin windows from 0 to 46, since it will give windows between 0 to 46 and it will be easy to find
# the counts in small intervals. From scatter plots it is clear that more data lies between 0 and 10
counts, bin_edges = np.histogram(haberman_1["Positive_axillary_nodes"], bins=10, 
                                 density = True)
print("Divided 10 windows from 0 to 46 : ",bin_edges)
# calculate the percentage or density in a bin edge window
pdf = counts/(sum(counts))
print("PDF : ",pdf);

# CDF tells us the total percentage covered for a number of nodes from the beginning
cdf = np.cumsum(pdf)
print("CDF : ",cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(["PDF","CDF"])
plt.show();


# ### Observation:

# 1. About 93% of survived patients had positive axillary nodes between 0 and 9. This clearly tells the fair amount of survival
#    chance for low positive nodes patients. But conclusion cannot be made before viewing the CDF and PDF for status 2 datasets.

# In[ ]:


counts, bin_edges = np.histogram(haberman_2["Positive_axillary_nodes"], bins=10, 
                                 density = True)
# calculate the percentage or density in a bin edge window
pdf = counts/(sum(counts))
print("PDF : ",pdf);
print("Divided 10 windows from 0 to 52 : ",bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
print("CDF : ",cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(["PDF","CDF"]);
plt.show()


# 1. About 71% of non-survived patients are having nodes from 0 to 10.
# 2. In the bin window 0 to 5 nodes 56% of the status 2 cases are identified. But as mentioned before the dataset is imbalance
#    and we are having only few status 2 sets present in it.

# In[ ]:


fig, (box,violin) = plt.subplots(1,2,figsize=(14,6))
box.set_title("Box Plot for Positive Nodes on Status")
#Plot the box plot which gives the quantiles 25, 50 and 75 percentile
sns.boxplot(x="Survival_status",y="Positive_axillary_nodes", data=haberman_set, ax=box)
violin.set_title("Violin Plot for Positive Nodes on Status")
#Plot violin plots which is an integration of Box and PDF
sns.violinplot(x="Survival_status",y="Positive_axillary_nodes",data=haberman_set, ax=violin)
#sns.violinplot(x="status",y="age",data=haberman_set, ax=age)
plt.show()


# 1. From the box and violin plots its clear that more than 50% of status 1 patients doesn't have positive nodes. Also the area
#    is denser in the bottom which indicates most of the status 1 patients have less nodes identified.
# 2. On the other hand the status 2 violin is thinner and distributed from bottom to top. This indicates that a significant number
#    of nodes are there for status 2 patients.
# 3. To see further deeper we shall analyse the median and quantiles for the nodes in both statuses.

# In[ ]:


from statsmodels import robust
#Median
print("Median of Status 1 nodes is : ",np.median(haberman_1["Positive_axillary_nodes"]))
print("Median of Status 2 nodes is : ",np.median(haberman_2["Positive_axillary_nodes"]))
print("*****************************************************************")
#Percentiles
print("Quantiles (25%, 50% and 75%) of Status 1 nodes are : ", np.percentile(haberman_1["Positive_axillary_nodes"],np.arange(25,100,25)))
print("Quantiles (25%, 50% and 75%) of Status 2 nodes are : ", np.percentile(haberman_2["Positive_axillary_nodes"],np.arange(25,100,25)))
print("*****************************************************************")
#Median Absolute Deviation
print ("Median Absolute Deviation of Status 1 nodes is : ", robust.mad(haberman_1["Positive_axillary_nodes"]))
print ("Median Absolute Deviation of Status 2 nodes is : ", robust.mad(haberman_2["Positive_axillary_nodes"]))


# In[ ]:


#Get the patients with no or zero positive nodes
haber_0 = haberman_set[haberman_set["Positive_axillary_nodes"] == 0]
haber_no_nodes_1 = haber_0[haber_0["Survival_status"] == 1]
haber_no_nodes_2 = haber_0[haber_0["Survival_status"] == 2]
print(haber_no_nodes_1.shape)
print(haber_no_nodes_2.shape)


# In the dataset, about 44% (136 out of 306) of records don't have axillary positive nodes. In that 136 records about 117 records
# are patients who lived more than 5 years.

# #### Further analysis of Data before conclusion

# In[ ]:


fig, (box,violin) = plt.subplots(1,2,figsize=(14,6))
box.set_title("Box Plot for Age on Status")
#Plot the box plot which gives the quantiles 25, 50 and 75 percentile
sns.boxplot(x="Survival_status",y="Age", data=haberman_set, ax=box)
#Plot violin plots which is an integration of Box and PDF
violin.set_title("Violin Plot for Age on Status")
sns.violinplot(x="Survival_status",y="Age",data=haberman_set, ax=violin)
plt.show()

fig1, (box_year,violin_year) = plt.subplots(1,2,figsize=(14,6))
#Plot the box plot which gives the quantiles 25, 50 and 75 percentile for years
box_year.set_title("Box Plot for Year on Status")
sns.boxplot(x="Survival_status",y="Treatment_year", data=haberman_set, ax=box_year)
#Plot violin plots which is an integration of Box and PDF for years
violin_year.set_title("Violin Plot for Year on Status")
sns.violinplot(x="Survival_status",y="Treatment_year",data=haberman_set, ax=violin_year)
plt.show()


# In[ ]:


print("Median of status 1 age is : ",np.median(haberman_1["Age"]))
print("Median of status 2 age is : ",np.median(haberman_2["Age"]))
print("*****************************************************************")
print("Median of status 1 year is : ",np.median(haberman_1["Treatment_year"]))
print("Median of status 1 year is : ",np.median(haberman_2["Treatment_year"]))
print("*****************************************************************")
print("Qunatiles (25%, 50% and 75%) of status 1 age is : ",np.percentile(haberman_1["Age"],np.arange(25,100,25)))
print("Qunatiles (25%, 50% and 75%) of status 2 age is : ",np.percentile(haberman_2["Age"],np.arange(25,100,25)))
print("*****************************************************************")
print("Qunatiles (25%, 50% and 75%) of status 1 year is : ",np.percentile(haberman_1["Treatment_year"],np.arange(25,100,25)))
print("Qunatiles (25%, 50% and 75%) of status 2 year is : ",np.percentile(haberman_2["Treatment_year"],np.arange(25,100,25)))


# In[ ]:


# Information on status 1 records
haberman_1.describe()


# In[ ]:


# Information on status 2 records
haberman_2.describe()


# # Conclusion

# 1. The dataset is imbalance with higher survived patients details
# 2. Based on the data exploration the patients with no or very less number of postive axillary nodes had a higher chance of
#    survival. But it is not true for all the cases, since we are also having no or very less nodes for even un-survived patients.
#    Also the dataset is very small.
# 3. Number of positive axillary nodes field is better than other fields for classification.
# 4. Age field is not very informative since patients are there in all age groups with both statuses. The violin and box plots in    furthur analysis of data section shows it clearly.
# 5. Years also scattered across all intervals for both statuses.

# In[ ]:




