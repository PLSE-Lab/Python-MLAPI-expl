#!/usr/bin/env python
# coding: utf-8

# # What is Haberman Dataset?
# -> Consists  of cases of study conducted on Survivals of Cancer surgeries.

# ##### Objective:
#     1) To classify the Data.
#     2) Perform basic statistical operations on dataset.
#     3) Perform Univariate Analysis (PDF,CDF,Box plot, Violin plot,etc.)
#     4) Perform Bivariate Analysis (Scatter plots, Pairplots,etc.)

# In[ ]:


#Importing all the Libraries required for Excercise
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Reading the CSV downloaded from Kaggle
Dataset=pd.read_csv('../input/haberman.csv')

#Giving Column Names to Dataset
Dataset.columns=['age','surgery_year','axil_nodes','status']


# ##### Observations:
#     1) Age ranges from 30 to 83
#     2) Surgery Year from 1958 to 1969
#     3) Status: 1 = Survived (Alive after 5 Years)
#                2 = Not Survived (Died within 5 Years)

# In[ ]:


print("Shape of Dataset is: ",Dataset.shape)


# In[ ]:


print("Head of the Dataset is: \n",Dataset.head())


# In[ ]:


print("Tail of the Dataset is: \n",Dataset.tail())


# In[ ]:


print("Columns of the Dataset is: \n",Dataset.columns)


# In[ ]:


print("Applying Max and Min Functinos")
print("**Minimum Age: ",Dataset['age'].min(),'\t**Maximum Age: ',Dataset['age'].max())
print("**Minimum Axils: ",Dataset['axil_nodes'].min(),'\t**Maximum Axils: ',Dataset['axil_nodes'].max())


# In[ ]:


#Use of Group by:
g=Dataset.groupby('axil_nodes')
for axil_nodes,node_df in g:
    print("Group by: ",axil_nodes)
    print(node_df,'\n')


# In[ ]:


print("Row with maximum Axil Nodes:\n ",Dataset[Dataset.axil_nodes==Dataset.axil_nodes.max()])


# ##### Observation:
#     Returned a Row where Axil Nodes are most

# In[ ]:


#Descibing the Haberman Dataset
Dataset.describe()


# In[ ]:


#Mean of all Input Features
print("Mean of feature Age is: ",Dataset['age'].mean())
print("Mean of feature Year is: ",Dataset['surgery_year'].mean())
print("Mean of feature Axils is: ",Dataset['axil_nodes'].mean())


# In[ ]:


#Median of all Input Features
print("Median of feature Age is: ",Dataset['age'].median())
print("Median of feature Year is: ",Dataset['surgery_year'].median())
print("Median of feature Axils is: ",Dataset['axil_nodes'].median())


# In[ ]:


#Use of Quantile over Dataset
print(np.percentile(Dataset["age"],np.arange(0, 100, 25)))
print(np.percentile(Dataset["axil_nodes"],np.arange(0, 100, 25)))
print(np.percentile(Dataset["axil_nodes"],np.arange(0, 125, 25)))


# # 1)Univariate Analysis

# ### Probability Density Function (PDF) and Histogram

# In[ ]:


for colomn in Dataset.columns[0:3]:
    sns.FacetGrid(Dataset,hue='status',size=5)              .map(sns.distplot,colomn)              .add_legend()
    plt.grid()
plt.show()


# ##### Observations:
#     1) All the PDFs of Survived and Non-Survived patients are overlapping each other.
#     2) None of them are Linearly Separable.

# ### Cumulative Density Function (CDF) with PDF for Survived.

# In[ ]:


survived=Dataset[Dataset['status']==1]
plt.figure(figsize=(17,5))
sns.set_style('whitegrid')      
for i,colomn in enumerate(list(Dataset.columns[:-1])):
    plt.subplot(1,3,i+1)
    counts,bin_edges= np.histogram(survived[colomn],bins=20,density=True)
    print("--------****",colomn,"****--------")
    pdf=counts/sum(counts)
    print("PDF is: ",pdf)
    print("Edges are: ",bin_edges)

    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(colomn)


# ##### Observations:
#     1) Age below 60, Around 75% patients survived. (Lower the Age, More chances of Survival)
#     2) Axil Nodes between range 0 and 10, around 92% patients survived. (Lower the nodes, more chances of survivals)
#     3) Surgery year, can't differentiate the Survivals.

# ### Cumulative Density function (CDF) with PDF for Non-Survived.

# In[ ]:


not_survived=Dataset[Dataset['status']==2]
plt.figure(figsize=(17,5))
sns.set_style('whitegrid')      
for i,colomn in enumerate(list(Dataset.columns[:-1])):
    plt.subplot(1,3,i+1)
    counts,bin_edges= np.histogram(not_survived[colomn],bins=20,density=True)
    print("--------****",colomn,"****--------")
    pdf=counts/sum(counts)
    print("PDF is: ",pdf)
    print("Edges are: ",bin_edges)

    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(colomn)


# ##### Observations:
#     1) Age below 50, Around 40% patients Died. (Lower the Age, More chances of Survival)
#     2) Age above 50, Around 60% patients Died. (Higher the Age, Less chances of Survival)
#     3) Patients with Axil Nodes below 20, around 88% patients died.
#     4) With surgery_year, can't predict the Survivals.

# ### Box-plot

# In[ ]:


fig,axes=plt.subplots(1,3,figsize=(15,5))
for i,colomn in enumerate(list(Dataset.columns[:-1])):
    sns.boxplot(x='status',y=colomn,data=Dataset,ax=axes[i])


# ###### For Age:
#     Survived:                         Not Survived: 
#     0th: 30                           0th: 34 
#     25th: 43                          25th: 46
#     50th: 52                          50th: 53
#     75th: 60                          75th: 61   
#     100th: 77                         100th: 83
# ###### For Year:
#     Survived:                         Not Survived: 
#     0th: 58                           0th: 58 
#     25th: 60                          25th: 59
#     50th: 63                          50th: 63
#     75th: 66                          75th: 65   
#     100th: 69                         100th: 69
# ###### For Axil Nodes:
#     Survived:                         Not Survived: 
#     0th: 0                           0th: 0 
#     25th: 0                          25th: 1
#     50th: 0                          50th: 4
#     75th: 3                          75th: 11   
#     100th: 7                         100th: 24 
# 
# Here, Seems that few Outlier are there in axil_nodes.
# 
# According to PDF on site: https://www.unc.edu/~rls/s151-09/class4.pdf
# 
# Axils_nodes>8.5 are Outliers

# ### Violin Plot

# In[ ]:


fig,axes=plt.subplots(1,3,figsize=(15,5))
for i,colomn in enumerate(list(Dataset.columns[:-1])):
    sns.violinplot(x='status',y=colomn,data=Dataset,ax=axes[i])


# ##### Observations:
# Survived:
#    1) Age ranges from 40 to 60, have high density of survivals.
#    2) Year ranges from 1958 to 1968, have high density of survivals.
#    3) Axil nodes ranges from 0 to 3, have high density of survivals.
# 
# Not Survived:
#    1) Age ranges from 42 to 60, have high density of non-survivals.
#    2) Year ranges from 1958 to 1959 and from 1962 to 1966, have high density of non-survivals.
#    3) Axil nodes from 0 to 3, have high density of non-survivals.
# 
# Here, Seems that few Outlier are there in axil_nodes like values (35,46,52)
# 
# According to PDF on site: https://www.unc.edu/~rls/s151-09/class4.pdf
# 
# Axils_nodes>8.5 are Outliers.

# # 2)Bi-variate Analysis

# ### PairPlot

# In[ ]:


sns.set_style('whitegrid')
sns.pairplot(Dataset,hue='status',vars=[Dataset.columns[0],Dataset.columns[1],Dataset.columns[2]],size=4)
plt.show()


# ##### Observations:
#     1) All graphs shows that both survived and non-survived are not linearly separable.
