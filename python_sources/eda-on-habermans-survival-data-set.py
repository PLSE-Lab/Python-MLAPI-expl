#!/usr/bin/env python
# coding: utf-8

#                                **Title: Haberman's Survival Data**
# Source: Kaggle
# 
# Relevant Information: The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# Number of Instances: 306
# 
# Number of Attributes: 4 (including the class attribute)
# 
# Attribute Information:
#     Age of patient at time of operation (numerical)
#     Patient's year of operation (year - 1900, numerical)
#     Number of positive axillary nodes detected (numerical)
#     Survival status (class attribute) 1 = the patient survived 5 years or longer & Class attribute 2 = the patient died within 5 year
# 
# Missing Attribute Values: None 

# In[ ]:


# first thing first "Import all the related libraries"

#import pandas for high performance data analysis tools
import pandas as pd

# import seaborn for statistical data visualization
import seaborn as sns

#import numpy for array-processing package
import numpy as np

#import matplotlib.pyplot for MATLAB like interface 
import matplotlib.pyplot as plt

import warnings   #Learnt from CancernDiagonostic Case Study
warnings.filterwarnings("ignore")
import os
print(os.listdir('../input'))


# In[ ]:


#import the data and store it as data frame

haberman_df= pd.read_csv('../input/haberman.csv')

#print(haberman_df) 

#check the total number of data points, features 

print(haberman_df.shape)

print(haberman_df.columns)


# In[ ]:


# Clearly above code displays just 305 rows and 4 columns. As per data info , we have 306 rows.  Also, there are no column names here, just data displayed

haberman_df= pd.read_csv('../input/haberman.csv', header=None)

print(haberman_df.shape)


# In[ ]:


#We can start mentioning the feature names as there is none.
# Age of patient at time of operation (numerical)  : Age
#Patient's year of operation (year - 1900, numerical)  : Year
#Number of positive axillary nodes detected (numerical) : Nodes
#Survival status : Status

column_names= ['Age', 'Year' , 'Nodes' , 'Status']

#infusing column names in the dataset.

haberman_df= pd.read_csv('../input/haberman.csv', header=None, names=column_names )
# sample checkout for  first five rows with infused column names

print(haberman_df.head(10))


# In[ ]:


print(haberman_df.info())


# 
# ## Observations on the above data
# 
# ###   1. No missing values in dataset.
# ###  2.  Data of every column is integer type, this contradi****cts the statement in the top("Survival status (class attribute) 1 = the patient survived 5 years or longer & Class attribute 2 = the patient died within 5 year")
# ###   3. We have to see the number of points for column Status. 

# In[ ]:


haberman_df['Status'].value_counts()


# ## Observations on the above data
# 
# ###   1. class attribute 1 has 225 points i.e. " 225 patients survived 5 years or longer"  This is around 73.5 percent
# ###   2. class attribute 2 has 81 points i.e. " 81  patients died within 5 year" This is around 26.5 percent

# In[ ]:


#Series has a map method for applying an element-wise function
#I will map the Status to categories : Survived, Died   using that method

#haberman_df['Status']= haberman_df['Status'].map({1:'Survived ', 2:'Died'})
#haberman_df['Status']= haberman_df['Status'].astype('category')

haberman_df.head(10)


# In[ ]:


#Univariate Analysis

#      1. PDF  

sns.FacetGrid(haberman_df, hue="Status", size=5).map(sns.distplot, "Age").add_legend();
sns.FacetGrid(haberman_df, hue="Status", size=5).map(sns.distplot, "Year").add_legend();
sns.FacetGrid(haberman_df, hue="Status", size=5).map(sns.distplot, "Nodes").add_legend();
warnings.filterwarnings("ignore")
plt.show();


# In[ ]:


# Observations

# 1. Nodes(axillary nodes) found useful in identifying the status.
# 2. Rest of the classes cannot be used for analysis as they are overlapping here.

print(haberman_df['Nodes'].mean())

# 3. "Survived" status falls mostly from 0 to 4 Nodes range. Those falling under mean have great chances of survival 


# In[ ]:


chance = haberman_df.loc[haberman_df["Status"] == 1];

#Summary for People belonging to class 1 (the patient survived 5 years after opertn)

chance.describe()


# In[ ]:


no_chance = haberman_df.loc[haberman_df["Status"] == 2];

#Summary for People belonging to class 1 (the patient who died within 5 years of opertn)

no_chance.describe()


# In[ ]:


#CDF of axil_nodes for two classes



#patient who Survived 5 years or longer

counts, bin_edges = np.histogram(chance['Nodes'], bins=20, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label= "PDF-axil nodes for the patient who Survived 5 years or longer")
plt.plot(bin_edges[1:], cdf, label= "CDF-axil nodes for the patient who Survived 5 years or longer")

plt.legend(loc='best')
plt.show();

# patient who died within 5 years of opertn

counts, bin_edges = np.histogram(no_chance['Nodes'], bins=20, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label= "PDF-axil nodes for the patient who died within 5 years of opertn")
plt.plot(bin_edges[1:], cdf, label= "CDF-axil nodes for the patient who died within 5 years of oprtn")

plt.legend(loc='best')
plt.show()


# In[ ]:


#Observarions using Describe and CDFs

#  None of the patient survived more than 5 year with Nodes greater than 46.


# In[ ]:


#BOX PLOT And WHISKERS

sns.boxplot(x='Status', y='Nodes', data = haberman_df)
    
plt.show()

sns.boxplot(x='Status', y='Age', data = haberman_df)
plt.show()

sns.boxplot(x='Status', y='Year', data = haberman_df)
plt.show()


# In[ ]:


#Observations

#1. Survival rate is great for people having less than 4 nodes
#2. People with age above 78 year have no chance of survival(belongs to class 2)


# In[ ]:


#VIOLIN PLOTS


sns.violinplot(x='Status',y='Nodes', data=haberman_df)

plt.show()

sns.violinplot(x='Status',y='Age', data=haberman_df)
plt.show()

sns.violinplot(x='Status',y='Year', data=haberman_df)
plt.show()


# In[ ]:


#Observations

#1. Status Vs Nodes Violin plot indicates their is fatter region around 0-4 region,which indicates Nodes as the useful feature
#2. More than 50 percent of people who didn't survived is having aux nodes greater than 4 


# In[ ]:


#Univariate Analysis

#2-D Scatter Plot
sns.set_style("whitegrid");
sns.FacetGrid(haberman_df, hue="Status", size=4).map(plt.scatter, "Age", "Nodes").add_legend();
plt.show();


# In[ ]:


#Observations

#1. Age group from 51 to 59 having zero Nodes has high survival rate among any other age group 


# In[ ]:


# PAIR PLOTS

sns.pairplot(haberman_df, hue="Status",vars=['Age','Year','Nodes'] );
plt.show()


# In[ ]:


##Observation

1.No clear observations can be made using Pair Plot  as we are not able to  distinguish data easily here.
2.Still Nodes Vs Year is better in classifying the data here.

#End Of Observation.


# In[ ]:




