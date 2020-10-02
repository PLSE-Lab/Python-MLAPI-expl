#!/usr/bin/env python
# coding: utf-8

# # Data Visualization with Haberman Dataset 

# # Habermans Survival Data Set

# About the Data Set:- 
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.

# Attribute Information:
# 
# 1.Age of patient at time of operation (numerical)            
# 2.Patient's year of operation (year - 1900, numerical)                             
# 3.Number of positive axillary nodes detected (numerical)                    
# 4.Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''downlaod Habernets csv  from https://www.kaggle.com/gilsousa/habermans-survival-data-set/version/1#haberman.csv'''
#Load habernets into a pandas dataFrame.
data = pd.read_csv("../input/haberman.csv", names=['age', 'year', 'nodes', 'status'])#loads the data in haberman.csv file into data


# In[ ]:


#Number of data points(rows) and Features(columns)
print(data.shape)


# ***Observations***                                          
#  There are 306 datapoints(rows) and 4 features(columns)

# In[ ]:


#Column names in our dataset
print (data.columns)


# In[ ]:


'''to change the names of the columns
data.columns=['age','year','nodes','Status']
print(data.columns)
'''


# In[ ]:


#to get information about the data set like the type of the data in the file(int)
#we can also check weather their is a null value or not by looking at the type if there is null value if it is 'object' it has NULL value
data.info()


# In[ ]:


#to check the number of Null values
np.sum(data.isna())


# ***Observations***
#  There are no null values the Habermans dataset 

# In[ ]:


#to get the details like number of observations, min,max,25%,50%,75% ,mean,std
data.describe()


# ***Observations:-***            
# 1.The average of the patients is 52                                
# 2.Most of the patients have 4 nodes on an average                          
# 3.75% of patients have below 60 years of age                                                   
# 4.75% of the patients have 4 nodes                                     
# 5.75% of the patients Died within 5 years           

# In[ ]:


#to check number of patients survived 5 years or longer and number of patients died within 5 year
data['status'].value_counts()
#slightly Balanced data Set


# In[ ]:


#pie chart representation
status=data['status'].value_counts()

labels='Patients survived 5 years or longer','patients died within 5 years'
sizes=[status[1],status[2]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# ***Observations:***                           
# 1.255 Patients survived 5 years or longer                      
# 2.81 patients died within 5 years               
# 

# In[ ]:


#to find the corelation between the columns
data.corr()


# **Obeservation:**                                        
# From the above table it is clear that nodes and status are corelated                       
# age and status are also corelated

# ***Observations***                  
# 1.75% of the patients who had undergone surgery for breast cancer died within 5years            
# 2.75% of the patients are of age 60                            
# 3.Most patients(75%) have 4 nodes

# #  2-D Scatter Plot

# ***Observation:***                                        
# from the above scatter plot of age and status we can't clearly state the relation between the age and status 

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(data,hue='status',size=6).map(plt.scatter,'age','nodes').add_legend()
plt.show()
#1->the patient survived 5 years or longer
#2->patient died within 5 year


# ***Observation:***                                        
# Scatter plots of the features of Haberman dataset does'nt clearly depict the relation between the features and Status of the patient 

# ## 3D Scatter plot
# 
# https://plot.ly/pandas/3d-scatter-plots/

# # Pair-plot

# In[ ]:


# pairwise scatter plot: Pair-Plot.
plt.close();
sns.set_style("whitegrid");
sns.pairplot(data,hue='status',vars=['age','year','nodes'],size=5,diag_kind='kde');
plt.legend()
plt.show() 
# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.


# ***Observation:***                         
# 1.Almost all the Patients are of age 30 - 80       
# 2.Number of nodes many of the patients have are below 20

# # (3.4) Histogram, PDF, CDF

# In[ ]:


counts,bin_edges=np.histogram(data['age'],bins=10,density=True)
pdf=counts/(sum(counts))
#print(pdf)
#print(bin_edges) 
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()


# In[ ]:


# Or we can use the percentile concept to get the same information
print(np.percentile(data['age'],np.arange(0,100,25)))#for calculating quantiles(0,25,50,75)
print(np.percentile(data['age'],95))#for caluculating 95th percentile


# ***Observations:***                                                                        
# 1.50% of the patients have the age of less than 52                                  
# 2.95% of the patients have age less than 70          

# In[ ]:


counts,bin_edges=np.histogram(data['nodes'],bins=10,density=True)
pdf=counts/(sum(counts))
#print(pdf)
#print(bin_edges) 
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()

print(np.percentile(data['nodes'],95))#for caluculating 95th percentile


# ***Observations:***                                        
#   95% of the patients have less than 20 nodes

# In[ ]:


sns.FacetGrid(data,hue='status',size=5).map(sns.distplot,'age').add_legend()
plt.show()


# ***Obesrvations***                 
#  Patients of age 30-40, 60-68 and 70-76 survived for more than 5 years  

# In[ ]:


sns.FacetGrid(data, hue='status', size=5).map(sns.distplot, 'nodes').add_legend();
plt.show();


# ***Obesrvations***                                             
#  patients who had less number of nodes lived for more than 5 years  

# In[ ]:


sns.FacetGrid(data,hue='status',size=5).map(sns.distplot,'year').add_legend();
plt.show()


# ***Obesrvations***                          
#     patients with year of treatement 61-64 and 66-67 lived for more than 5year compared to others

# ***Obesrvations***                 
# 1.Patients of age 30-40, 60-68 and 70-76 survived for more than 5 years                    
# 2.patients who had less number of nodes lived for more than 5 years              
# 3.patients with year of treatement 61-64 and 66-67 lived for more than 5year compared to others

# # Box plot and Whiskers

# In[ ]:


#NOTE: IN the plot below, a technique call inter-quartile range is used in plotting the whiskers. 
#Whiskers in the plot below donot correposnd to the min and max values.

#Box-plot can be visualized as a PDF on the side-ways.

sns.boxplot(x='status',y='age', data=data)
plt.show()


# ***Observation***                           
# Patients who lived  for 5 years or more  have age ranging from 30 to 78                                       
# Patients who died within 5 years have age ranging ranging from 33 to 83 

# In[ ]:


sns.boxplot(x='status',y='nodes', data=data)
plt.show()


# ***Observation***                           
# Most of the Patients who lived  for 5 years or more  had nodes less than 7                 
# 75% of the Patients who lived  for 5 years or more  had 4 nodes                                   
# 50%  of the Patients who died within 5 years had more than 4 nodes 

# # Violin plots

# In[ ]:


sns.violinplot(x="status", y="age", data=data, size=10)
plt.show()


# ***Observations:***                                       
# 1.75% of the patients who lived for 5 years or more than  have age below 60                    
# 2.75% of the patients who died within  5 years have age below 63

# In[ ]:


sns.violinplot(x="status", y="nodes", data=data, size=10)
plt.show()


# ***Observations:***                                
# 1.75% of the patients who died within 5 year of operation have less than 11 nodes                                       
# 2.75% of the patients who lived for 5 years or more have nodes less than 4

# # Overall Summary

# 1.Patients who suffer from breast cancer are of age 30 - 80                
# 2.75% patients had 4 nodes                                      
# 3.Status of a patients who had undergone surgery for breast cancer depends on age and number of nodes                          
# 4.75% of patients who had less number of nodes(4 nodes) lived for more than 5 years                       
# 5.75% of the patients who had undergone surgery for breast cancer died within 5years                                            
# 6.Patients of age 30-40, 60-68 and 70-76 survived for more than 5 year 

# 1.There are 75% chances that the patients who will undergo surgery for breast cancer will die                
# 2.patient who has less number of nodes has more chances of survival

# In[ ]:




