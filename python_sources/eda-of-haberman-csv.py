#!/usr/bin/env python
# coding: utf-8

# # Data description :
# 
*Age of the patient
*year of operation has done
*Number of positive nodes
*Survival status  (1 = the patient survived 5 years or more 
                  2 = the patient died within 5 years)
# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
df=pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv",names=["age","year","nodes","status"])


# In[ ]:


#top 5 rows
#last 5 rows
df.head
df.tail


# In[ ]:


#no of columns and rows in the data set
df.shape


# In[ ]:


#column names
df.columns


# # EDA

# In[ ]:


#Null values in the data set
df.isnull().sum()


# In[ ]:


df['status'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'


# # **observation :**
# 1. percentage of categorial data in status variable
# 2. 73.5% = the patient survived 5 years or longer
# 3. 26.5%= the patient died within 5 years

# In[ ]:


# unique values with type in data set 
df["age"].unique()


# In[ ]:


df.describe()


# # observation :
# * total number of datase in Rows = 306 (count)
# * Avg of each column (mean)
# * min number in variable (min)
# * maximum number in variable (max)
# 

# In[ ]:


#types of variable in the data
df.info()


# *Name of the columns and the dtype of columns.

# # pair plot 

# In[ ]:


#scatter plot with pair plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, vars = ["age", "year", "nodes"], hue="status",height=3);
plt.show()


# # observation :

*in this pair plot blus lable is status of survived and the orange is dead.
*age & nodes very usefull feature in the data set.
*pair plot is use to saparation of the data set.

# # Box plot

# In[ ]:


sns.boxplot(x='status',y='age', data=df)
plt.title("Box Plot of status of patients survived or not ", fontsize=20)
plt.show()


# In[ ]:


sns.boxplot(x='status',y='nodes', data=df)
plt.title("Box plot of nodes of patient survived or not",fontsize=20)
plt.show()


# # observation :
# 
*box plot will give the outliers of a variable
*And its define IQR value
*in this positive axillary nodes is consist more outliers
# # PDF(probability density function) & CDF (probability density function)

# In[ ]:


#pdf abd cdf refer by(https://www.kaggle.com/gokulkarthik/haberman-s-survival-exploratory-data-analysis)
plt.figure(figsize=(15,3))
for idx, feature in enumerate(list(df.columns)[:-1]):
    plt.subplot(1, 3, idx+1)
    print("********* "+feature+" *********")
    counts, bin_edges = np.histogram(df[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(feature)
    plt.legend(('PDF','CDF'))


# In[ ]:


#vailin plot is like a box plot
sns.violinplot(x="status",y="age",data=df)
plt.title("voilin plot of age vs status",fontsize=25)


# In[ ]:


sns.violinplot(x="status",y="year",data=df)
plt.title("voilin plot of year vs status",fontsize=25)


# In[ ]:


sns.violinplot(x="status",y="nodes",data=df)
plt.title("voilin plot of Nodes vs status",fontsize=25)


# # observation :
*voilin plot also like a box plot
*it will improve the prasentation and it makes beautifull as compare to box plot.
*difference between box & voilin plot is box plot well understanding plot.
# # PDF

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df , hue = "status" , height=7 ).map(sns.distplot , "age").add_legend();
plt.title("Histogram of age of patients",fontsize=20)
plt.show()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df , hue = "status" , height=7 ).map(sns.distplot , "year").add_legend();
plt.title("Histogram of year of patients ",fontsize=20)
plt.show()


# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df , hue = "status" , height =7 ).map(sns.distplot , "nodes").add_legend();
plt.title("Histogram of status of patients",fontsize=20)
plt.show()


# In[ ]:



counts, bin_edges = np.histogram(df['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# virginica
counts, bin_edges = np.histogram(df['year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#versicolor
counts, bin_edges = np.histogram(df['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(('PDF','CDF'))

plt.show();


# # observation : 
*80% of the patients have <= equal to 5 positive lymph node.
*A Huge no of overlapping can be observed among the classes
*The objective of classifying the survival status of a new patient based on the given features is a little bit difficult task.