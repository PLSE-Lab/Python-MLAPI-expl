#!/usr/bin/env python
# coding: utf-8

# <h1>**This is an Ipython Notebook for the Haberman Cancer Dataset**<h1>

# In[ ]:


#The Required Libraries for this analysis
import seaborn as sns #For the visualization
import matplotlib.pyplot as plt #For visualization
import numpy as np #For effective Numerical Operation
import pandas as pd #For handling data


# In[ ]:


haberman = pd.read_csv('../input/haberman.csv',names=['Age','Op_Year','axil_nodes','Surv_status'])


# In[ ]:


haberman.describe()


# <h4>OBSERVATIONS:</h4>
#     1. The mean of the axil_nodes is 4.
#     2. 95% of the observations have axil_nodes below 11(mean+std.dev)

# In[ ]:


haberman.info()


# <h4>OBSERVATIONS:</h4>
#     1. All the attributes have a non null value with an integer type.

# In[ ]:


#The Columns present in the dataset
haberman.columns


# In[ ]:


#The number of rows(x)& columns(y)-->(x,y)
haberman.shape


# <h3>We have 306 rows and 4 columns</h3>
# <h3>We have 3 features and one class label(Surv_status)</h3>

# In[ ]:


haberman['Surv_status'].value_counts()


# <h4>OBSERVATIONS:</h4>
#     1. Since we have only 3 features, we would go for individual scatter plot and not for pair plots.
#     2. We See that we have two classes in Surv_status i.e 1 & 2. 1: the patient survived and 2: the patient died.

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Surv_status", height=4)    .map(plt.scatter, "Age", "Op_Year")    .add_legend();
plt.show();


# <h4><b>OBSERVATIONS:</b></h4>
#     1. Most of the operations done for the age group (30-40) were successful.
#     2. Most of the operations in the year 1960,1961,1967 were successful.

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Surv_status", height=4)    .map(plt.scatter, "Op_Year", "axil_nodes")    .add_legend();
plt.show();


# <h4><b>OBSERVATIONS:</b></h4>
#     1. Most of the operations done in the year 1960,1961 were successful.
#     2. Most of the operations done in the year 1958,1965 were unsuccessful.

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Surv_status", height=4)    .map(plt.scatter, "Age", "axil_nodes")    .add_legend();
plt.show();


# <h4><b>OBSERVATIONS:</b></h4>
#     1. Most of the people, who were considered for the dataset had axil_nodes less than 30.
#     2. Most of the patients who survived had an axil_node count of less than 5.
#     3. Most of the patients in the range of age (40-65) were diagnosed with cancer.
#     4. Chances of survival, if the axil_nodes were greater than 20, were slim.

# In[ ]:


sns.FacetGrid(haberman, hue="Surv_status", height=5)    .map(sns.distplot, "Age")    .add_legend();
plt.show();


# <h4>OBSERVATIONS:</h4>
#     1. Chances of survival are high, if age is less than 40.
#     2. Most of the people aged in the range 40-70 were diagnosed with cancer.

# In[ ]:


sns.FacetGrid(haberman, hue="Surv_status", height=5)    .map(sns.distplot, "Op_Year")    .add_legend();
plt.show();


# <h4>OBSERVATION:</h4>
#     1. The probability of the operations being unsuccessful from 1963 to 1967, was the highest.

# In[ ]:


sns.FacetGrid(haberman, hue="Surv_status", height=5)    .map(sns.distplot, "axil_nodes")    .add_legend();
plt.show();


# <h4>OBSERVATIONS:</h4>
#     1. If the patient had axil_nodes less than 3, the probability of survival is very high.
#     2. If the patient had axil_nodes more than 10, the probability of survival is very low.

# In[ ]:


#Survived contains all the observations with the Surv_status as 1.
#Not_Survived contains all the observations with the Surv_status as 2.

survived = haberman.loc[haberman["Surv_status"]==1]
not_survived=haberman.loc[haberman["Surv_status"]==2]


# In[ ]:


survived.describe()


# <h4>OBSERVATION</h4>
# 
#   1. The mean of axil_node count is 2.8, for survival.
#   2. 75% of the survived patients have an axil_node count of less than 3.
#   3. The maximum axil_node count for a survived patient is 46.
#   4. The average age for survival is 52.
#   5. 75% of the survived patients have age less than 60.

# In[ ]:


not_survived.describe()


# <h4>OBSERVATION</h4>
# 
#   1. The mean of axil_node count is 7.5, for not-survival.
#   2. 75% of the not-survived patients have an axil_node count of 11.
#   3. The maximum axil_node count for a not-survived patient is 52.
#   4. The average age for not-survival is 54.
#   5. 75% of the not-survived patients have age less than 61.

# In[ ]:


counts, bin_edges = np.histogram(survived['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='PDF_survives')
plt.plot(bin_edges[1:], cdf,label='CDF_survives')
plt.legend()

counts, bin_edges = np.histogram(not_survived['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='PDF_not_survies')
plt.plot(bin_edges[1:], cdf,label='CDF_not_survies')
plt.legend()

plt.show();


# <h4>OBSERVATIONS:</h4>
#     1. The probability that a patient survives, is lesser than 'not-survival' when the axil_nodes are between 10-30.
#     2. The probability that a patient survives or not, when the axil_node count is greater than 30, is the same.
#     3.The probability drastically changes from 58% to 82% of not-surviving when the axil_count changes from 5 to 15.
#     4.The probability that a person survives is higher than not-surviving, when axil_count is less than 8.

# In[ ]:


counts, bin_edges = np.histogram(survived['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='PDF_survives')
plt.plot(bin_edges[1:], cdf,label='CDF_survives')


counts, bin_edges = np.histogram(not_survived['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='PDF_not_survives')
plt.plot(bin_edges[1:], cdf,label='CDF_not_survives')
plt.legend()

plt.show();


# <h4>OBSERVATIONS:</h4>
#     1. The probability that a patient survives or not, after the age of 48, is the almost same.
#     2. The probability of the patient surviving is higher, when the age is less than 48.

# In[ ]:


counts, bin_edges = np.histogram(survived['Op_Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='PDF_survives')
plt.plot(bin_edges[1:], cdf,label='CDF_survives')


counts, bin_edges = np.histogram(not_survived['Op_Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='PDF_not_survives')
plt.plot(bin_edges[1:], cdf,label='CDF_not_survives')

plt.legend()
plt.show();


# <h4>OBSERVATIONS:</h4>
#     1. The probability that a patient survives, if the operation year is between 1961-1965,is higher than not-surviving.
#     2. The probability of the patient not surviving is higher, if the operation year is before 1960.
#     3. The probability of the patient surviving/not-surviving is the same, if the operation year is after 1968.
#     4. The probability of the patient not surviving is higher, if the operation year is before between 1965-1967.

# <h3>Let us now look at some box plots</h3>

# In[ ]:


#Box plot for Age
sns.boxplot(x='Surv_status',y='Age', data=haberman)
plt.show()


# <h4>OBSERVATIONS</h4>
# 1. 25% of the survived patients have age less than 42.
# 2. 50% of the survived patients have age less than 52.
# 3. 25% of the survived patients have age less than 60.
# 4. 25% of the not-survived patients have age less than 46.
# 5. 50% of the not-survived patients have age less than 53.
# 6. 75% of the not-survived patients have age less than 62.

# In[ ]:


#Box plot for Operation Year
sns.boxplot(x='Surv_status',y='Op_Year', data=haberman)
plt.show()


# <h4>OBSERVATIONS</h4>
# 1. 25% of the survived patients have the operation year 1960 or prior.
# 2. 50% of the survived patients have the operation year 1963 or prior.
# 3. 75% of the survived patients have the operation year 1966 or prior.
# 4. 25% of the not-survived patients have the operation year 1959 or prior.
# 5. 50% of the not-survived patients have the operation year 1963 or prior.
# 6. 75% of the not-survived patients have the operation year 1965 or prior.

# In[ ]:


#Box plot for Axil_Node Count
sns.boxplot(x='Surv_status',y='axil_nodes', data=haberman)
plt.show()


# <h4>OBSERVATIONS</h4>
# 1. 50% of the survived patients have the axil_node count as 0.
# 2. 75% of the survived patients have the axil_node count less than 3.
# 3. 25% of the not-survived patients have the axil_node count less than 1.
# 4. 50% of the not-survived patients have the axil_node count less than 4.
# 5. 75% of the not-survived patients have the axil_node count less than 11.

# <h3>Now we look at Violin Plots.</h3>

# In[ ]:


sns.violinplot(x="Surv_status", y="Age", data=haberman, size=8)
plt.show()


# <h3>OBSERVATION</h3>
# 1. 25% of the survived patients have age less than 42.
# 2. 50% of the survived patients have age less than 52.
# 3. 75% of the survived patients have age less than 60.
# 4. 25% of the not-survived patients have age less than 46.
# 5. 50% of the not-survived patients have age less than 53.
# 6. 75% of the not-survived patients have age less than 62.

# In[ ]:


sns.violinplot(x="Surv_status", y="Op_Year", data=haberman, size=8)
plt.show()


# <h4>OBSERVATIONS</h4>
# 1. 25% of the survived patients have the operation year 1960 or prior.
# 2. 50% of the survived patients have the operation year 1963 or prior.
# 3. 75% of the survived patients have the operation year 1966 or prior.
# 4. 25% of the not-survived patients have the operation year 1959 or prior.
# 5. 50% of the not-survived patients have the operation year 1963 or prior.
# 6. 75% of the not-survived patients have the operation year 1965 or prior.
# 7. Most of the survived patients have the operation year between 1960-1966.
# 8. Most of the not-survived patients have the operation year between 1962-1965 and 1958-1960.

# In[ ]:


sns.violinplot(x="Surv_status", y="axil_nodes", data=haberman, size=8)
plt.show()


# <h4>OBSERVATIONS</h4>
# 1. 50% of the survived patients have the axil_node count as 0.
# 2. 75% of the survived patients have the axil_node count less than 3.
# 3. 25% of the not-survived patients have the axil_node count less than 1.
# 4. 50% of the not-survived patients have the axil_node count less than 4.
# 5. 75% of the not-survived patients have the axil_node count less than 11.
# 6. Most of the patients that survived have an axil_node count of 0.
# 7. Most of the patients that not-survived have an axil_node count of 1-10. 

# <h3>Now Let us look at Contour Plots</h3>

# In[ ]:


plt.close()
sns.jointplot(x="Age", y="Op_Year", data=survived, kind="kde")
plt.show();


# <h3>OBSERVATION:</h3>
# 1. Most of the patients that survived have an age in the range 40-60.
# 2. Most of the patients that survived have the Operation_Year in the range 1961-1967. 

# In[ ]:


sns.jointplot(x="Age", y="axil_nodes", data=survived, kind="kde")
plt.show();


# <h3>Observations</h3>
# 1. Most of the patients that survived have the axil_node count as 0.

# In[ ]:


sns.jointplot(x="Age", y="Op_Year", data=not_survived, kind="kde")
plt.show();


# <h3>Observations</h3>
# 1. Most of the patients which are not-survived have the Operation_Year in the range 1963-1965.
# 2. Most of the patients which are not-survived have the Age in the range 40-60.

# In[ ]:


sns.jointplot(x="Age", y="axil_nodes", data=not_survived, kind="kde")
plt.show();


# <h3>Observations</h3>
# 1. Most of the patients which are not-survived have the Axil_nodes in the range 1-10.

# <h1>FINAL CONCLUSION:</h1>
# <h3>1. 76% of the patients survived.</h3>
# <h3>2. Most of the patients that survived have the axil_node count as 0.</h3>
# <h3>3. Most of the operations done for the age group (30-40) were successful.</h3>
# <h3>4. Most of the patients that survived have the Operation_Year in the range 1961-1967. </h3>
# <h3>5. Most of the 'not-survived' patients have the operation year between 1962-1965 and 1958-1960.</h3>
# <h3>6. Most of the patients which are 'not-survived' have the Axil_nodes in the range 1-10</h3>
# <h3>7. 75% of the 'not-survived' patients have age less than 62.</h3>
# <h3>8. 75% of the survived patients have age less than 60.</h3>

# In[ ]:




