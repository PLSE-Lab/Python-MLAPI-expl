#!/usr/bin/env python
# coding: utf-8

# # Plotting for Exploratory data analysis (EDA)

# ## Objective:
# 
# To Predict the patients who had undergone surgery for breast cancer and the dataset taken from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital.

# In[ ]:


# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Loading Datasets into dataframes
df = pd.read_csv('../input/haberman.csv')
df.head()


# In[ ]:


df = df.rename(columns = {"30" : "Age", "64" : "Op_Year", "1" : "Axil_nodes", "1.1" : "Surv_status"})


# In[ ]:


df.shape


# In[ ]:


df['Surv_status'].value_counts()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# # Univariate Analysis

# ## Probablity Density Function (PDF)

# In[ ]:


sns.FacetGrid(df,hue='Surv_status',size=4)    .map(sns.distplot,"Age")    .add_legend();
plt.title("Histogram of Age")
plt.show()


# In[ ]:


sns.FacetGrid(df,hue='Surv_status',size=4)    .map(sns.distplot,"Op_Year")    .add_legend();
plt.title("Histogram of Op_Year")
plt.show()


# In[ ]:


sns.FacetGrid(df,hue='Surv_status',size=4)    .map(sns.distplot,"Axil_nodes")    .add_legend();
plt.title("Histogram of Axil_nodes")
plt.show()


# **Observations**
# 1. The Pdf Curves of Age and Op_year shows that Survey status are overlapped much.
# 2. The Pdf curve for Axil_nodes shows that people had 60% of survival chances at the axillary nodes between 0 - 4.

# ## Cummulative Density Function (CDF)

# In[ ]:


df_surv1 = df.loc[df["Surv_status"] == 1];
df_surv2 = df.loc[df["Surv_status"] == 2];


# In[ ]:


plt.figure(1)

label=["Survived pdf","Survived Cdf","Non-Survived Pdf","Non-Survived Cdf"]
counts, bin_edges = np.histogram(df_surv1['Age'], bins=10,density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges = np.histogram(df_surv2['Age'], bins=10,density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('Age')
plt.ylabel('Survey_Density')
plt.legend(label)

plt.show();


# In[ ]:


plt.figure(1)

label=["Survived pdf","Survived Cdf","Non-Survived Pdf","Non-Survived Cdf"]
counts, bin_edges = np.histogram(df_surv1['Op_Year'], bins=10,density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges = np.histogram(df_surv2['Op_Year'], bins=20,density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('Op_Year')
plt.ylabel('Survey_Density')
plt.legend(label)

plt.show();


# In[ ]:


plt.figure(1)

label=["Survived pdf","Survived Cdf","Non-Survived Pdf","Non-Survived Cdf"]
counts, bin_edges = np.histogram(df_surv1['Axil_nodes'], bins=10,density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

counts, bin_edges = np.histogram(df_surv2['Axil_nodes'], bins=20,density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('Axil_nodes')
plt.ylabel('Survey_Density')
plt.legend(label)

plt.show();


# **Observations**
# 1. Peoples on age of 30-38 has high survival rate from cancer
# 2. The person who have auxillary nodes between 0-9 has high survival rate
# 3. The person who have auxillary nodes above 45 has high death rate

# ## Box Plot

# In[ ]:


sns.boxplot(x='Surv_status',y='Age', data=df)
plt.show()


# In[ ]:


sns.boxplot(x='Surv_status',y='Op_Year', data=df)
plt.show()


# In[ ]:


sns.boxplot(x='Surv_status',y='Axil_nodes', data=df)
plt.show()


# ## Violin Plots

# In[ ]:


sns.violinplot(x='Surv_status',y='Age',hue='Surv_status', data=df,size=8)
plt.show()


# In[ ]:


sns.violinplot(x='Surv_status',y='Op_Year',hue='Surv_status',data=df,size =8)
plt.show()


# In[ ]:


sns.violinplot(x='Surv_status',y='Axil_nodes',hue='Surv_status', data=df,size = 8)
plt.show()


# # Bivariate Analysis

# ## 2D-Scatter Plot

# In[ ]:


df.plot(kind='scatter',x='Age',y='Op_Year');
plt.show()


# In[ ]:


# 2-D Scatter plot with color-coding for Survey Status
sns.set_style("whitegrid");
sns.FacetGrid(df,hue="Surv_status",size=4).map(plt.scatter,"Age","Op_Year").add_legend();
plt.show()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df,hue="Surv_status",size=4).map(plt.scatter,"Age","Axil_nodes").add_legend();
plt.show()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df,hue="Surv_status",size=4).map(plt.scatter,"Op_Year","Axil_nodes",).add_legend();
plt.show()


# **Observations**
# 1. The Patients in the years (58,59,64,65) as well as the axillary nodes between (0-10) predicts the survival rate as high.
# 

# ## Pair Plot

# In[ ]:


sns.set_style("whitegrid");
sns.pairplot(df, hue="Surv_status", vars=["Age","Op_Year","Axil_nodes"],size=3)
plt.show()


# **Observations**
# 1. These plots are much overlapped with all variables. It's difficult to distinguish between classes

# In[ ]:




