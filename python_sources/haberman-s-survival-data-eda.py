#!/usr/bin/env python
# coding: utf-8

# # Haberman's Survival Data

# ## Data Description

# Relevant Information: The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# ### Attribute Information

# Number of Instances: 306
# 
# Number of Attributes: 4 (including the class attribute)
# 
# Attribute Information:
# 
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 4. Survival status (class attribute):
#         1 = the patient survived 5 years or longer
#         2 = the patient died within 5 year
#    Missing Attribute Values: None

# Downlaod haberman.csv from ''' https://www.kaggle.com/gilsousa/habermans-survival-data-set '''
# Load haberman.csv into a pandas dataFrame.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
hbm = pd.read_csv("../input/haberman.csv")


# In[ ]:


# (Q) how many data-points and features?
#get dimentions of the table
#total number of rows and columns
print (hbm.shape)


# In[ ]:


hbm = hbm.rename(columns = {"30" : "Age", "64" : "Op_Year", "1" : "axil_nodes_det", "1.1" : "Surv_status"})


# In[ ]:


#(Q) What are the column names in our dataset?
print (hbm.columns)


# In[ ]:


#(Q) How many data points for each class are present? 
hbm["Surv_status"].value_counts()


# To know number of data-points for each class.
# As it is not balanced dataset, it is imbalanced dataset because the number of data-points for both of the class are significantly different.
#  we will see how to handle imbalanced data later
# 

# In[ ]:


#if you want to see initial some rows then use head command (default 5 rows)
hbm.head()


# In[ ]:


#if you want to see last few rows then use tail command (default last 5 rows will print)
hbm.tail()


# In[ ]:


#slicing
hbm[30:45]


# In[ ]:


# To know statistical summary of data which is very important
hbm.describe()


# In[ ]:


hbm.Surv_status.value_counts(normalize=True)


# ### Observation

# 1. The age of the patients vary from 30 to 83 with the median of 52.
# 2. The number of detected nodes vary from 0 to 52 (probably an outlier) with the median of 1 and mean of 4.
# 3. 75% of data points have less than 5 detected axilary nodes and nearly 25% have no detected nodes.
# 4. The target column is unbalanced with 73% Yes(1) and 26% No(2).

# # 2-D Scatter Plot

# In[ ]:



hbm.plot(kind='scatter', x='Age', y='Op_Year');
plt.grid()
plt.show()

#cannot make any sence out it.
#what if we color the point by thier class_lebel/Surv_status.


# In[ ]:


#2-D Scatter plot with color-coading for each Surv_status/class.
sns.set_style('whitegrid');
sns.FacetGrid(hbm, hue='Surv_status',size= 4)    .map(plt.scatter, "Age", "Op_Year")    .add_legend();
plt.title('2-D scatter plot for age and opration year')
plt.show();

#blue and orange data point cannot be easily seperated.
#we can draw multiple 2-D scatter plot for each combination of features.


# ## Observation :

# 1. Using Age and Op_Year features, we cannot seperated Surv_status.
# 2. Seperating Survival status (class attribute) 1(the patient survived 5 years) from 2(the patient died within 5 year) is much harder as they have considerable overlap.

# In[ ]:


#2-D scatter plot b/w age and axil_nodes_det
sns.set_style('whitegrid');
sns.FacetGrid(hbm, hue='Surv_status',size= 4)    .map(plt.scatter, "Age", "axil_nodes_det")    .add_legend();
plt.title('2-D scatter plot for age and axillary nodes detected')
plt.show();


# # Observation :

# 1. In the above 2d scatter plot class label(i.e. a person died or survived) is not linearly seprable.
# 2. 0-5 axillary notes detected person survived and died as well but the died ratio is less than survive ratio.

# # Pair-plot

# In[ ]:


#Only possible to view 2D patterns.
plt.close();
sns.set_style('darkgrid');
sns.pairplot(hbm, hue="Surv_status",vars = ["Age", "Op_Year", "axil_nodes_det"], size = 3);
plt.suptitle("pair plot of age, operation_year and axillary_node")
plt.show()

# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.


# ### Observation :

# 1. As we are unable to classify which is the most useful feature because of too much overlapping. But, Somehow we can say, In operation_year, 60-65 more person died who has less than 6 axillary_lymph_node.
# 2. And hence, this plot is not much informative in this case.

# # Histogram, PDF, CDF

# ### 1-D Scatter Plot

# In[ ]:


import numpy as np
hbm_1 = hbm.loc[hbm["Surv_status"] == 1];
hbm_2 = hbm.loc[hbm["Surv_status"] == 2];
plt.plot(hbm_1["Age"], np.zeros_like(hbm_1['Op_Year']), 'o')
plt.plot(hbm_2["Age"], np.zeros_like(hbm_2['Op_Year']), 'o')
plt.show()


# ## 1 - Univariate analysis on Two separated dataframes

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Let's split our dataframe inti two dataframes
yes_hbm = hbm[hbm.Surv_status == 1] 
no_hbm = hbm[hbm.Surv_status == 2]


# In[ ]:


for data in [yes_hbm,no_hbm]:
    for column in hbm.columns[:-1]:
        sns.FacetGrid(data,hue="Surv_status", size = 4).map(sns.distplot,column).add_legend()
        plt.ylabel("Density")
        plt.title("Histogram of " + column)
        plt.show()
        print("*"*75)


# ## 2 - Univariate analysis on the whole data set

# In[ ]:


for column in hbm.columns[:-1]:
    sns.FacetGrid(hbm, hue="Surv_status", size=4).map(sns.distplot,column).add_legend()
    plt.ylabel("Density")
    plt.title("Histogram of "+ column)
    plt.show()
    print("*"*75)


# ### Univariate PDFs/CDFs on the whole Data set 

# In[ ]:


for column in hbm.columns[:-1]:
    counts, bins_edges = np.histogram(hbm[column],bins=10,density=True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    print(counts)
    print("PDF and CDF of ",column," variable.")
    print("PDF: ",pdf)
    print("Bins_Edges: ",bins_edges)
    plt.plot(bins_edges[1:],pdf,'b-', label = 'pdf')
    plt.plot(bins_edges[1:],cdf,'r-', label='cdf')
    plt.legend()
    plt.show()
    print("*"*75)


# ## Univariate PDFs/CDFs on the yes/no datasets(split in two data sets)

# In[ ]:


for data in [yes_hbm,no_hbm]:
    for column in hbm.columns[:-1]:
        counts, bins_edges = np.histogram(data[column],bins=10,density=True)
        pdf = counts/sum(counts)
        cdf = np.cumsum(pdf)
        print("PDF and CDF of ",column," variable In DataFrame:",data.Surv_status.iloc[0],"_DF")
        print("PDF: ",pdf)
        print("Bins_Edges: ",bins_edges)
        plt.plot(bins_edges[1:],pdf,'b-', label='pdf')
        plt.plot(bins_edges[1:],cdf,'r-', label='cdf')
        plt.legend()
        plt.show()
        print("*"*75)


# # Median, Percentile, Quantile, IQR, MAD 

# In[ ]:


from statsmodels import robust
for column in hbm.columns[:-1]:
    print("*"*75)
    print(column," variable : ")
    print("\nMedian :- ")
    print(np.median(hbm[column]))
    print("\nQuantiles :- ")
    print(np.percentile(hbm[column],np.arange(0,100,25)))
    print("\nMedian Absolute Deviation :-")
    print(robust.mad(hbm[column]))


# # Box plot and Whiskers

# In[ ]:


# Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitivey.
# The Concept of median, percentile, quantile.
fig, axes = plt.subplots(1,3,figsize=(10,5))#sharey='all')
for idx, feature in enumerate(list(hbm.columns)[:-1]):
    sns.boxplot(x='Surv_status', y=feature, data=hbm, ax=axes[idx])
plt.show()


# # Violin plots

# In[ ]:


# A violin plot combines the benefits of the previous two plots and simplifies them

# Denser regions of the data are fatter, and sparser ones thinner in a violin plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(hbm.columns)[:-1]):
    sns.violinplot(x="Surv_status", y=feature, data=hbm, ax=axes[idx], kind='kde')
plt.show()


# # Observations

# 1. The number of positive lymph nodes of the survivors is highly densed from 0 to 5.
# 2. Almost 80% of the patients have less than or equal to 5 positive lymph nodea.
# 3. The patients treated after 1966 have the slighlty higher chance to surive that the rest. The patients treated before 1959 have the slighlty lower chance to surive that the rest.

# # Multivariate probability density, contour plot

# In[ ]:


#2D Density plot, contors-plot
fig, axes = plt.subplots(1, 3, figsize=(5, 5))
for idx, feature in enumerate(list(hbm.columns)[:-1]):
    sns.jointplot(x=feature, y="axil_nodes_det", data=hbm_1, kind='kde');
    plt.plot();


# # Conclusions

# 1. Patients who survived have lower no. of auxilary nodes compared patoents who din't survive.
# 
# 2. Out of all 3 features, axil_nodes is most important.
# 
# 3. Most no. of surgeries were performed between 1960 - 64.
# 
# 4. Most of the patients who had undergone surgery were aged between 42-60 years.
# 
# 5. People with auxilary nodes less than 5 had higher chances of survival.
# 
# 6. People with ages less than 40 had 90% chances of survival.

# In[ ]:




