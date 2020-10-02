#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis On Haberman Dataset

# In[ ]:


# Importing some useful packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import robust


# In[ ]:


# Importing/loading the data to our Ipython notebook environment
haberman = pd.read_csv("../input/haberman.csv", header=None, names=["age","op_year","axil_nodes_det","surv_status"])


# In[ ]:


print(haberman.shape)


# Relevant Information about the Data set (https://www.kaggle.com/gilsousa/habermans-survival-data-set): 
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# - Number of Instances: 306
# - Number of Attributes: 4 (including the class attribute)
# - Attribute Information:
# 
#     1-Age of patient at time of operation (numerical) <br/>
#     2-Patient's year of operation (year - 1900, numerical)<br/>
#     3-Number of positive axillary nodes detected (numerical)<br/>
#     4-Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year

# In[ ]:


haberman.head()


# In[ ]:


print(haberman.info())
# There is no missing values in our data set


# In[ ]:


print(haberman.surv_status.unique())


# In[ ]:


# Let's modify the dependant variable(target column) to be categorical/meaningful
haberman.surv_status = haberman.surv_status.map({1:"yes",2:"no"}).astype("category")


# In[ ]:


# Some high level statistics
print(haberman.describe())


# In[ ]:


print("Haberman Data set structure: ")
print("Number of data points: ",haberman.shape[0])
print("Number of features/independant variables: ",haberman.shape[1])
print("-"*70)
print(haberman.surv_status.value_counts(normalize=True))


# ##### Some observations:
# 1. The age of the patients vary from 30 to 83 with the median of 52.
# 2. The number of detected nodes vary from 0 to 52 (probably an outlier) with the median of 1 and mean of 4.
# 3. 75% of data points have less than 5 detected axilary nodes and nearly 25% have no detected nodes
# 4. The target column is unbalanced with 73% yes- 26% no

# In[ ]:


# Let's visualize some 2D Scatter plots with color coding for each class
sns.set_style("darkgrid")
sns.FacetGrid(haberman,hue="surv_status",height=6).map(plt.scatter,"age","axil_nodes_det").add_legend()
plt.show()


# Some observations: as the number of positive axillary nodes detected increases, there are more people who die within 5 years than people who survived. On the opposite we notice that people who survived generally had few axilary detected nodes.
# 

# In[ ]:


sns.FacetGrid(haberman,hue = "surv_status", height = 6).map(plt.scatter,"op_year","axil_nodes_det").add_legend()
plt.show()


# In[ ]:


# For more 2D-Plot insights let's draw our pair plot
plt.close()
sns.set_style("whitegrid")
sns.pairplot(haberman,hue = "surv_status",height = 4)
plt.show()


# 1. The number of detected axilary nodes are the most useful features to identify our target variable value.
# 2. Ther's a lot of overlap between the two classes which can not be simply separated.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

    
xs = haberman["age"]
ys = haberman["op_year"]
zs = haberman["axil_nodes_det"]    

ax.scatter(xs, ys, zs)

ax.set_xlabel('age')
ax.set_ylabel('op_year')
ax.set_zlabel('axil_nodes_det')

plt.show()


# ### Univariate analysis using hsitograms, PDF and CDF
# #### 1 - Univariate analysis on Two separated dataframes 

# In[ ]:


# Let's split our dataframe inti two dataframes
yes_df = haberman[haberman.surv_status == "yes"] 
no_df = haberman[haberman.surv_status == "no"]


# In[ ]:


for dataF in [yes_df,no_df]:
    for column in haberman.columns[:-1]:
        sns.FacetGrid(dataF,hue="surv_status",height = 6).map(sns.distplot,column).add_legend()
        plt.show()


# #### 2 - Univariate analysis on The whole data set 

# In[ ]:


for column in haberman.columns[:-1]:
    sns.FacetGrid(haberman,hue="surv_status",height=5).map(sns.distplot,column).add_legend()
    plt.show()


# ##### PDFs and CDFs visualizations

# 1. Univariate PDFs/CDFs on the whole Data set

# In[ ]:


for column in haberman.columns[:-1]:
    counts, bins_edges = np.histogram(haberman[column],bins=10,density=True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    print("PDF and CDF of ",column," variable.")
    print("PDF: ",pdf)
    print("Bins_Edges: ",bins_edges)
    plt.plot(bins_edges[1:],pdf)
    plt.plot(bins_edges[1:],cdf)
    plt.show()
    print("**********************************************************************")


# 2. Univariate PDFs/CDFs on the yes/no datasets(split in two data sets)

# In[ ]:


for dataF in [yes_df,no_df]:
    for column in haberman.columns[:-1]:
        counts, bins_edges = np.histogram(dataF[column],bins=10,density=True)
        pdf = counts/sum(counts)
        cdf = np.cumsum(pdf)
        print("PDF and CDF of ",column," variable In DataFrame:",dataF.surv_status.iloc[0],"_DF")
        print("PDF: ",pdf)
        print("Bins_Edges: ",bins_edges)
        plt.plot(bins_edges[1:],pdf)
        plt.plot(bins_edges[1:],cdf)
        plt.show()
        print("**************************************************************")
    print("------------------------------------------------------")


# ###### Some statistics on "age" and "axil_nodes_det" variables (mean, std, percentiles, median, IDR, MAD, Quantiles)

# In[ ]:


for column in ["age","axil_nodes_det"]:
    print("*********************************")
    print(column," variable : ")
    print("Mean")
    print(np.mean(haberman[column]))
    print("Std-dev:")
    print(np.std(haberman[column]))
    print("Median:")
    print(np.median(haberman[column]))
    print("Quantiles:")
    print(np.percentile(haberman[column],np.arange(0,100,25)))
    print("90th Percentile:")
    print(np.percentile(haberman[column],90))
    print ("Median Absolute Deviation")
    print(robust.mad(haberman[column]))


# ### Box plots

# In[ ]:


sns.boxplot(x="surv_status",y="axil_nodes_det",data=haberman)
plt.show


# In[ ]:


sns.boxplot(x="surv_status",y="age",data=haberman)
plt.show


# ### Violin plots

# In[ ]:


sns.violinplot(x="surv_status",y="axil_nodes_det",data=haberman,height=8)
plt.show()
sns.violinplot(x="surv_status",y="age",data=haberman,height=8)
plt.show()
sns.violinplot(x="surv_status",y="op_year",data=haberman,height=8)
plt.show()


# #### Multivariate Contour Probability density

# In[ ]:


sns.jointplot(x="axil_nodes_det",y="op_year",data=haberman,kind = "kde")
plt.show()


# In[ ]:


sns.jointplot(x="axil_nodes_det",y="age",data=haberman,kind = "kde")
plt.show()


# #### Observations and conclusions: 
# 1. The variable "number of detected nodes" is the best one to help us separate and differentiate the two classes
# 2. Except for the year 65 (+- 1 year), the chances of surviving are greater than before year 1960
# 3. People who survived had their ages mainly between 45 to 55 years old.

# In[ ]:





# In[ ]:




