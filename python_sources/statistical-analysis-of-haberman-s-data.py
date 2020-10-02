#!/usr/bin/env python
# coding: utf-8

# Data Description The Haberman's survival dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# **Attribute Information:**
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 4. Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 years

# In[ ]:


#importing useful libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[ ]:


#read data from working dir
data = pd.read_csv('../input/habermans-survival-data-set/haberman.csv')
data.head(10)


# #Data Analysis

# In[ ]:


data.columns = ['age', 'year', 'nodes', 'status']


# In[ ]:


#information about data
data.info()


# In[ ]:


# Change lables(status) numerical data y to catagorical data as 0 -->> 'no' and 1 -->> 'yes'
data['status'] = data['status'].map({1:"yes", 2:"no"}).astype('category')
data.head(10)


# In[ ]:


print(data.shape)
print(data.columns)


# ***Observation*** 
# ### Here number of samples are 306 and number of features are three ('age', 'year' 'nodes') and  one is lables ("status")

# ***Observation*** 
# ### All data has non-null values

# In[ ]:


#check names of status
data['status'].unique()


# In[ ]:


#testing null values
data.isnull().sum()


# # Basic Statistical analysis

# In[ ]:


print(data['status'].value_counts(normalize = True))
sns.set_style("whitegrid");
sns.catplot(x="status", kind="count", palette="ch:.25",hue="status", data=data).add_legend();
plt.title("status count",size=20,pad=20)
plt.show();


# ***Observation*** 
# ### Here we can observe in dataset that  Survival status (The target column) is imbalanced with ~74% patient are survived 5 years or longer and ~ 26%  patient died within 5 years

# In[ ]:


# Statistics describe
data.describe()


# ***Observation*** 
# ### 1. Age of patients varing between 30 to 83 with 52.45 mean and 10.80 std div 
# ### 2. Patient's year of operation varing between 1958 to 1969 with ~ 58 mean and ~3.24 std dev 
# ### 3. Number of positive axillary nodes detected between 0 to 52 with ~4.02 mean and ~7.18 std div

# #correlation matrix

# In[ ]:


#correlation matrix
corr = data.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
plt.title("correlation matrix",size=30,pad=30)


# 
# ***Observation***
# ### As can be seen from the spearmann rank-correlation matrix, the most feature that has most correlation with year is positively related with age and negatively related with nodes.nodes is negatively related with age,

# #2-D Scatter Plot

# In[ ]:


#scatter plots
sns.set_style('whitegrid')
sns.FacetGrid(data, hue="status",size = 8 )    .map(plt.scatter, "year", "nodes")    .add_legend();
plt.title("Scatter plot of status corresponding to years and nodes")
plt.show();

sns.set_style('whitegrid')
sns.FacetGrid(data, hue="status", size=8 )    .map(plt.scatter, "age", "nodes")    .add_legend();
plt.title("Scatter plot of status corresponding to age and nodes")
plt.show();

sns.set_style('whitegrid')
sns.FacetGrid(data, hue="status", size=8 )    .map(plt.scatter, "age", "year")    .add_legend();
plt.title("Scatter plot of status corresponding to age and year")
plt.show();


# #Pair-plots

# In[ ]:


#pair plots
plt.close();
sns.set_style("whitegrid");
sns.pairplot(data, hue="status", size=4);
plt.show()


# ***Observation***
# ### By pairplot we can not classify leanerly with any of pair which is the most useful feature because of too much overlapping .we can not build simple model using only if else condition we need to have some more complex technique to handle this dataset. But, Somehow we can say,
# ### In operation year, 65-67 more person died who has less than 6 axillary lymph node.
# ### In the data points between year and nodes, we can see the better seperation between the two clases than other plots.
# ### 0-5 node person survived and died as well but the died ratio is less than survive ratio.

# #Histogram, PDF

# In[ ]:


sns.FacetGrid(data, hue="status", size=8)    .map(sns.distplot, "age")    .add_legend();
plt.title("Histograms and Probability Density of age")
plt.show();

sns.FacetGrid(data, hue="status", size=8)    .map(sns.distplot, "year")    .add_legend();
plt.title("Histograms and Probability Density of year")
plt.show();

sns.FacetGrid(data, hue="status", size=8)    .map(sns.distplot, "nodes")    .add_legend();
plt.title("Histograms and Probability Density of nodes")
plt.show();


# #PDF-CDF

# In[ ]:


# Plots of CDF when patient survived 5 years or longer.

counts, bin_edges = np.histogram(data[data.status == 'yes']['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.gca().legend(('PDF','CDF'))
plt.title('CDF and PDF of nodes when patient survived 5 years or longer')
plt.show();

counts, bin_edges = np.histogram(data[data.status == 'yes']['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf) 
plt.gca().legend(('PDF','CDF'))
plt.title('CDF and PDF of age when patient survived 5 years or longer')
plt.show();

counts, bin_edges = np.histogram(data[data.status == 'yes']['year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf) 
plt.gca().legend(('PDF','CDF'))
plt.title('CDF and PDF of year when patient survived 5 years or longer')

plt.show();


# ***Observation***
# ### By CDF and PDF for nodes we can observe that ~ 93%  patient survived 5 years or longer when node values belongs between 0 to 13.

# In[ ]:


# Plots of CDF when patient not survived 5 years or longer.
counts, bin_edges = np.histogram(data[data.status == 'no']['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.gca().legend(('PDF','CDF'))
plt.title('CDF and PDF of nodes when patient not survived 5 years or longer')
plt.show();

counts, bin_edges = np.histogram(data[data.status == 'no']['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf) 
plt.gca().legend(('PDF','CDF'))
plt.title('CDF and PDF of age when patient not survived 5 years or longer')
plt.show();

counts, bin_edges = np.histogram(data[data.status == 'no']['year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf) 
plt.gca().legend(('PDF','CDF'))
plt.title('CDF and PDF of year when patient not survived 5 years or longer')

plt.show();


# ***Observation***
# ### By CDF and PDF for nodes we can observe that  ~ 98%  patient not survived 5 years or longer when node values belongs between grater then 30

# #Box plot and Whiskers

# In[ ]:


#Boxplots
sns.boxplot(x='status',y='age', data=data,hue="status")
plt.title('Box plot of age')
plt.show()


sns.boxplot(x='status',y='year', data=data,hue="status")
plt.title('Box plot of year')
plt.show()


sns.boxplot(x='status',y='nodes', data=data,hue="status");
plt.title('Box plot of nodes')
plt.show()


# #Violin plots

# In[ ]:


#violinplots
sns.violinplot(x="status", y="nodes", data=data,hue="status", size=8)
plt.title('Violin plot of nodes')
plt.show()

sns.violinplot(x="status", y="age", data=data,hue="status", size=8)
plt.title('Violin plot of age')
plt.show()

sns.violinplot(x="status", y="year", data=data,hue="status", size=8)
plt.title('Violin plot of year')
plt.show()


# #2D Density plot

# In[ ]:


#2D Density plot, contors-plot
sns.jointplot(x="year", y="nodes", data=data[data.status == 'yes'], kind="kde");
plt.title("2D Density plot of year and nodes when patient survived 5 years or longer",loc = 'right')
plt.show();

sns.jointplot(x="year", y="nodes", data=data[data.status == 'no'], kind="kde");
plt.title("2D Density plot of year and nodes when patient not survived 5 years or longer",loc = 'right')
plt.show();

sns.jointplot(x="age", y="nodes", data=data[data.status == 'yes'], kind="kde");
plt.title("2D Density plot of age and nodes when patient survived 5 years or longer",loc = 'right')
plt.show();

sns.jointplot(x="age", y="nodes", data=data[data.status == 'no'], kind="kde");
plt.title("2D Density plot of age and nodes when patient not survived 5 years or longer",loc = 'right')
plt.show();

sns.jointplot(x="year", y="age", data=data[data.status == 'yes'], kind="kde",);
plt.title("2D Density plot of year and age when patient survived 5 years or longer",loc = 'right')
plt.show();

sns.jointplot(x="year", y="age", data=data[data.status == 'no'], kind="kde");
plt.title("2D Density plot of year and age when patient not survived 5 years or longer",loc = 'right')
plt.show();


# ***Observation***
# ### More  patient not survived 5 years or longer when year belongs between ~1963 to ~1967.
# 
# ### More  patient not survived 5 years or longer when age belonge ~42 to ~57 and nodes between less then ~5
# 
# ### More  patient not survived 5 years or longer when age belonge ~42 to ~57 and year between ~1962 to ~1966.
# 

# # Final Conclustion
# ## By plotting all pdf, cdf, box-plot, pair plots, scatter plot etc. we get only one conclusion :
# ## if number of axillary node is less,than survival of patients is more

# In[ ]:





# In[ ]:




