#!/usr/bin/env python
# coding: utf-8

# **Exploratory Data Analysis of Haberman Dataset**
# 
# This is a dataset that is quite easy to understand and it is the first step in machine learning. We basically have 4 columns that are used to define the status of a patient. 
# 
# We plot various graphs that are necessary to select the most important features in the data. We import various standard libraries that are necessary to implement the algorithm
# 
# We use various plotting tools like boxplot, violin plot and so on. We also measure the distribution of data and consider only the important features.
# 
# Below is the code that is used to exploratory data analysis
# 
# 

# In[ ]:


import numpy as np                         #importing the numpy library for scientific calculation
import seaborn as sns                      #importing seaborn for visualization 
import matplotlib.pyplot as plt            #This is also used for plotting 
import pandas as pd                        #This is used to create data frames and also read and write files
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


haberman = pd.read_csv('../input/haberman.csv')       #data is read from the current directory


# In[ ]:


haberman.shape                           #Looking at the current shape of the dataset under consideration


# In[ ]:


haberman.head()                          #Looking at various columns and how the data looks like 


# We are now going to see the number of outputs and see their classification. We have more number of patients who have less chance of being affected than the patients who have more chances of being affected.

# In[ ]:


haberman['status'].value_counts()                  #counting the number of output labels and their distribution


# It looks like there seems to be not a good relationship between age and the year at which they were diagnosed as you can see below. The points are pretty scattered out here.

# In[ ]:


haberman.plot(x = 'age', y = 'operation_year', kind = 'scatter')            #Using a scatter plot to see the distribution


# We can only see some of the outliers in the data but not a good relationship between the nodes and the year. For example, we see a patient who was diagnosed in the year 1960 to be having node of value slightly greater than 50. We could see some of the outliers in the data but not a good relationship between the 2 selected input features.

# In[ ]:


haberman.plot(x = 'operation_year', y = 'axil_nodes', kind = 'scatter')         #Using various other features to check the scatter plot


# We are going to use pairplot to see the relationship between all the parameters and see if there are any linearly separable features. It appears that almost all the features are not linearly separable. However, we could see that there could be some boundary between age on the x-axis and nodes on the y-axis.

# In[ ]:


sns.pairplot(data = haberman, hue = 'status', size = 3)                 #Using pairplot to see the relationship with all the input features
plt.show()


# We are seeing the relationship between year of diagnosis and the nodes. But there seems to be no clear separation of data. Let us explore the data further.

# In[ ]:


sns.set_style(style = "whitegrid")
sns.FacetGrid(haberman, hue = "status", size = 5).map(plt.scatter, 'axil_nodes','operation_year').add_legend()


# We are going to divide the data into two parts namely haberman_1 and haberman_2. Haberman_1 contains those data points where the output status label is 1. Haberman_2 contains those data points where the output status label is 2. We take these two variables and see how the age is spread about the mean

# In[ ]:


#dividing the data into 2 parts namely haberman_1 and haberman_2
haberman_1 = haberman.loc[haberman['status'] == 1]      #storing the values of the input where the status is 1               
haberman_2 = haberman.loc[haberman['status'] == 2]      #storing the values of the input where the status is 2
#Checking how the age is distributed and comparing between status 1 and status 2
plt.plot(haberman_1["age"], np.zeros_like(haberman_1['age']), 'o')   
plt.plot(haberman_2["age"], np.zeros_like(haberman_2['age']), 'o')


# We can see from the above that there seems to be not much seperation between the ages for those with status 1 and 2. There are few age groups below the age say 35 who have a status of 1. However, there are also groups of status 1 in the age of range75 - 80 years. So it is not possible to linearly seperate the data points based on age.

# In[ ]:


haberman.head(1)         #Having a look at the column just to access it later in the next cell


# In[ ]:


sns.FacetGrid(haberman, hue = 'status', size = 5).map(sns.distplot, 'age').add_legend()      #Having a look at the distribution how the age is distributed and trying to see the co-relation between age and status


# **Observations**
# 
# 1. It seems that there is no linearly separable boundary between age and status as their distribution is close.
# 2. The mean of those patients who do not have the disease is centered somewhere around 55 years of age.
# 3. The mean of those patients who have the disease is centered somewhere around 50 years of age.
# 4. The variance of age of those patients who do not have the disease is quite higher than those patients who have the disease.
# 5. The variance of age of those patients who have the disease has more tailedness which means that there are some outliers in the data.

# In[ ]:


sns.FacetGrid(haberman, hue = 'status', size = 5).map(sns.distplot, 'operation_year').add_legend()    #Checking the co-relation between year and status


# **Observations**
# 
# 1. We cannot also seperate the groups based on the year of operation.
# 2. Most of the patients of both the status 1 and status 2 were diagnosed in the year range of 1958 - 1969. 
# 3. There are some outliers in the data for those patients which both status 1 and status 2. For example, there are around 1% of the patients who were diagnosed in the year 1972. It is a very small percentage.
# 4. In addition to this, there are also some other patients (say 1%) who are diagnosed in the year 1956 for both status 1 and status 2.

# In[ ]:


haberman.head(1)


# In[ ]:


sns.FacetGrid(haberman, hue = 'status', size = 5).map(sns.distplot, 'axil_nodes').add_legend()       #Checking the corelation between nodes and status


# **Observations**
# 
# 1. There seems to be a possible separation between the data points in the above plot. 
# 2. Around 70 percent of the patients who did not have the disease (status 1) have axil nodes of range (-2) - 5.
# 3. Around 70 percent of the patients who did have the disease (status 2) have axil nodes in the range (-5) - 20.
# 4. There are some outliers at the far end in the axil nodes of patients who do not have the disease (status 1) as seen from the tailedness of the curve with status 1 (blue).
# 5. There is also very less data points where the patients have axil nodes in the range 50 - 60 with status 2.

# In[ ]:


counts, bin_edges = np.histogram(haberman['axil_nodes'], density = True)       #Creating the values for the histogram
pdf = counts / sum(counts)                                                #Computing the pdf of the above values
cdf = np.cumsum(pdf)                                                      #Computing the cdf based on the pdf calculated above      
print(bin_edges)                                                          #Printing the edges of the histogram
plt.plot(bin_edges[1:], pdf)                                              #Plotting the pdf of the above 
plt.plot(bin_edges[1:], cdf)                                              #Plotting the cdf of the above
counts, bin_edges = np.histogram(haberman['axil_nodes'], density = True, bins = 20) #Creating the values of the histogram for haberman_2
pdf = counts / sum(counts)                                                #Creating the pdf of haberman_2
plt.plot(bin_edges[1:], pdf)                                              #Plotting the pdf of the haberman_2


# **Observations**
# 
# 1. We have plotted the pdf and cumulative distribution function in the graph. We see that as the line is moving towards the right, there is an increase in the cdf (which is intuitive). 
# 2. The green line indicates the pdf of the data when the bins are 20. 
# 3. The blue line indicates the pdf of the data when the bins are 10 (default).
# 4. We plotted the cdf of the line which as the blue line as the pdf.
# 

# In[ ]:


#Repeating the steps in the above cells with slight modification
counts1, bin_edges = np.histogram(haberman_1['axil_nodes'], density = True)
pdf1 = counts1 / sum(counts1)
cdf1 = np.cumsum(pdf1)
counts2, bin_edges = np.histogram(haberman_2['axil_nodes'], density = True)
pdf2 = counts2 / sum(counts2)
cdf2 = np.cumsum(pdf2)
plt.plot(bin_edges[1:], pdf1)
plt.plot(bin_edges[1:], cdf1)
plt.plot(bin_edges[1:], pdf2)
plt.plot(bin_edges[1:], cdf2)


# We repeated the same code as the above cell except that we took haberman_1 and haberman_2 as separate values.

# In[ ]:


print('The mean of the nodes of haberman dataset which has status 1 and status 2 are given below')
print(np.mean(haberman_1['axil_nodes']))         #calculating the mean of the nodes with status 1
print(np.mean(haberman_2['axil_nodes']))         #calculating the mean of the nodes with status 2
print('The standard deviation of the nodes of haberman dataset which has status 1 and status 2 are given below')
print(np.std(haberman_1['axil_nodes']))          #calculating the standard deviation with status 1
print(np.std(haberman_2['axil_nodes']))          #calculating the standard deviation with status 2


# **Observations**
# 
# 1. The mean of the patients who have the disease is quite higher (7.5) those who do not the disease (2.8). 
# 2. The standard deviation and variance is higher for the patients who have the disease (status 2).
# 3. The standard deviation and variance is relatively low for the patients who do not have the disease (status 1).
# 

# In[ ]:


sns.distplot(haberman_1['axil_nodes'])           #Plotting the distribution of nodes with status = 1


# Having a look at the distribution of axil nodes of patients who do not have the disease (status 1)

# In[ ]:


sns.distplot(haberman_2['axil_nodes'], color = 'green')    #Plotting the distribution of nodes with status = 2


# Having a look at the axil nodes of the patients who have the heart disease (status 2).

# In[ ]:


print('The 0, 25 and 75th percentiles of the given nodes of haberman dataset with status 1 and status 2 is')
print(np.percentile(haberman_1['axil_nodes'], np.arange(0, 100, 25)))       #printing the 0th, 25th, 50th and 75th percentile of nodes with status = 1
print(np.percentile(haberman_2['axil_nodes'], np.arange(0, 100, 25)))       #printing the 0th, 25th, 50th and 75th percentile of nodes with status = 2


# **Observations**
# 
# 1. We see that 1st value and 25tht percentile of the data have almost equal values.
# 2. But when we move to the 50th and 75th percentile, there is a significant variation in the values (axil nodes) of patients.
# 3. This implies that there is more tailedness in the data of patients with axil nodes and having a disease (status 2).
# 
# 
# 

# In[ ]:


print('the 95th percentile of the nodes that are present in haberman data set with status 1 and 2 are given below')
print(np.percentile(haberman_1['axil_nodes'], 95))     #printing the 95th percentile of nodes with status = 1
print(np.percentile(haberman_2['axil_nodes'], 95))     #printing the 95th percentile of nodes with status = 2


# **Observations**
# 
# 1. We see that 95th percentile of patients who do not have the disease (status 1) have axil node value of 14.
# 2. We see that 95th percentile of patients who have the disease (status 2) have axil node value of 23. 
# 3. We can see from the above that axil node values of patients who have the disease (status 2) seems to be more spread out as compared to axil nodes of those who do not have the disease (status 1).

# In[ ]:


print('This is a box plot taking nodes as the feature and seperating them based on hue = status')
sns.boxplot(x = 'status', y = 'axil_nodes', data = haberman)    #Plotting the box plot for x as status and y as nodes


# This is another way of representing the spread in the data. 
# 
# **Observations**
# 
# 1. We see that there are very few outliers in the axil nodes of patients who do not have the disease (status 1). Mean would be affected by such outliers but not the median.
# 2. We also see quite a few outliers in the axil nodes of patients who have the disease (status 2). 

# In[ ]:


print('Let us also use the violen plot that would also take into account how the points are spread')
sns.violinplot(x = 'status', y = 'axil_nodes', data = haberman)    #also plotting the violin plot for x as status and y as nodes


# **Observations**
# 
# 1. We also see the distribution plot along with box plot (sort of). 
# 2. This is the violin plot that gives rise to box plot and distribution plot. 

# **Conclusion**
# 
# We tried to find the features that could be separated easily. We first tried to see the features like age, year of diagnosis, axial nodes and the status and tried to find the relationship between them. 
# We found that the most significant feature that could give us an idea about the status of a patient is the number of axial nodes.
# We have plotted the distribution of the axial nodes and saw how the data is spread

# In[ ]:




