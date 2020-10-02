#!/usr/bin/env python
# coding: utf-8

# ![](http://)Exploratory Data Analysis
# 
# Dataset : Habermans Cancer Survival Dataset

# #### Importing Packages :

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels import robust
import matplotlib.pyplot as plt 


# #### Reading Data From CSV File :

# In[2]:


haberman=pd.read_csv('../input/haberman.csv')


# In[3]:


haberman.columns=['age','year','node','status']


# In[4]:


haberman.columns


# In[5]:


haberman.status=haberman.status.replace({1:"survived",2:"died"})


# #### Knowing About the Dataset :
# 
# Features :
# 
#            Age :- Age Of The Patient At The Time Of Operation
# 
#            year :- Patients Year Of Operation
#            
#            Nodes :- Number Of Positive Auxilary Nodes Detected
#            
#            Status :- 1) The Patients Survived 5 years or Longer
#            
#                      2) The Patients Died Within 5 Years
#            
# Its an Unbalanced Dataset, Datapoints for Each class are not Equal in Number
# 
# Status is the Output/Dependent/Class Variable
# 
# Age, Year, Node are the Input/Independent Variable
# 
# Its The Three Dimentional Dataset

# In[6]:


print("Number Of DataPoints : {} ".format(haberman.shape[0]))


# In[7]:


print("Number Of Features : {}".format(haberman.shape[1]))


# In[8]:


print("Number Of Classes : {}".format(len(haberman.status.value_counts())))


# In[9]:


print(haberman.status.value_counts())


# In[10]:


print("Independent Features Are : ")
i=0
for feature in haberman.columns:
    if i<3:
        print(str(i+1)+") "+feature)
    else:
        print("Dependent Feature Is : {}".format(feature))
    i+=1


# ### Objective :
# 
# To Perform Exploratory Data Analysis To Find Useful Features Inorder To Classify Will The Patient Survive The Cancer Treatment Or Not. 

# #### 1. Univariate Ananlysis :

# ##### 1.1 1-D Scatter Plot :

# In[11]:


patient_survived=haberman.loc[haberman.status=='survived']
patient_died=haberman.loc[haberman.status=='died']
plt.plot(patient_survived['age'],np.zeros_like(patient_survived['age']),'o',
         label='Survived')
plt.plot(patient_died['age'],np.zeros_like(patient_died['age']),'o',
         label='Died')
plt.xlabel("Age")
plt.title("1D Age Scatter Plot")
plt.legend()
plt.grid()
plt.show()


# In[12]:


plt.plot(patient_survived['node'],np.zeros_like(patient_survived['node']),'o',label='Survived')
plt.plot(patient_died['node'],np.zeros_like(patient_died['node']),'o',label='Died')
plt.xlabel("Auxillary Nodes")
plt.title("1D Number of Nodes Scatter Plot")
plt.legend()
plt.grid()
plt.show()


# ##### Observation : 
# 1. The Points on 1-D Scatter Plot are hard to diffrentiate because two many points are overlapping and number pf points overlapped aslo cant be known.
# 2. Year of Patient's operation is not used here because there are people survived and died in the same year.

# ##### 1.2 Univariate Analysis Using Probability Density Function(PDA) :

# In[13]:


sns.FacetGrid(haberman,hue='status',height=5).map(sns.distplot,"age").add_legend()
plt.grid()
plt.title("Age PDF")
plt.show()


# In[14]:


sns.FacetGrid(haberman,hue='status',height=5).map(sns.distplot,"year").add_legend()
plt.grid()
plt.title("Year Of Operation PDF")
plt.show()


# In[15]:


sns.FacetGrid(haberman,hue='status',height=6).map(sns.distplot,"node").add_legend()
plt.grid()
plt.title("Number Of Nodes PDF")
plt.show()


# ##### Observation : 
# 1. The First Two plots year-pdf and age-pdf are overlapping each other, its difficult to diffrentiate.
# 2. the Third Plot tells if number of nodes is less than 5 then there is more chance of survival than greater than 5.
# 3. Node is the Better Feature Compared To Age And Year.

# ##### 1.3 Cummulative Density Function (CDF) :

# In[16]:


counts,bin_edges=np.histogram(patient_survived['age'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="pdf-survived")
plt.plot(bin_edges[1:],cdf,label="cdf-survived")
counts,bin_edges=np.histogram(patient_died['age'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="pdf-died")
plt.plot(bin_edges[1:],cdf,label="cdf-died")
plt.xlabel("Age")
plt.ylabel("Probabilities")
plt.legend()
plt.grid()
plt.title("Age PDF CDF")
plt.show()


# In[17]:


counts,bin_edges=np.histogram(patient_survived['year'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="pdf-survived")
plt.plot(bin_edges[1:],cdf,label="cdf-survived")
counts,bin_edges=np.histogram(patient_died['year'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="pdf-died")
plt.plot(bin_edges[1:],cdf,label="cdf-died")
plt.xlabel("Year")
plt.ylabel("Probabilities")
plt.grid()
plt.title("Year Of Operation PDF CDF")
plt.legend()
plt.show()


# In[18]:


counts,bin_edges=np.histogram(patient_survived['node'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="pdf-survived")
plt.plot(bin_edges[1:],cdf,label="cdf-survived")
counts,bin_edges=np.histogram(patient_died['node'],bins=10,density=True)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label="pdf-died")
plt.plot(bin_edges[1:],cdf,label="cdf-died")
plt.legend()
plt.xlabel("Nodes")
plt.ylabel("Probabilities")
plt.grid()
plt.title("Number Of Nodes PDF CDF")
plt.show()


# ##### Observation : 
# 1. The First Two plots year-pdf-cdf and age-pdf-cdf are overlapping each other, its difficult to diffrentiate.
# 2. The Thrid Plot Nodes-pdf-cdf tells that approximately 90% of the survived points lie bellow 10 nodes and 70% of the died     points lie bellow 10 nodes.
# 3. the number of points bellow 10 nodes are far greater than above 10 , so 10 Nodes can be taken diffrentiation point.
# 4. Number of nodes is the better feature compared to age and year.
# 5. No Person Survived after 76 year of age.
# 6. No Person Survived with 45 or more nodes. 

# ##### 1.4 Box Plot :

# In[19]:


sns.boxplot(x='status',y='age',data=haberman)
plt.grid()
plt.title("Age Box Plot")
plt.show()


# In[20]:


sns.boxplot(x='status',y='year',data=haberman)
plt.grid()
plt.title("Year Of Operation Box Plot")
plt.show()


# In[21]:


sns.boxplot(x='status',y='node',data=haberman)
plt.grid()
plt.title("Number Of Nodes Box Plot")
plt.show()


# ##### Observation : 
# 1. First Plot tells that 25 percentile of survived is at 42 age and died is at 46 age, so higher age is less probable of surviving.
# 2. Second Plot tells that 25 percentile of survived is at 1960 and died is at 1955, so if year of operation is greater than 1960 you are more likley to survive the operation than bellow 1960.
# 3. Third Plot tells that 75 percentile people survived who have less than 3 nodes and 75 percentile people have died who have less than 11 nodes, So if number of nodes is less than there is more chance to survive.

# ##### 1.5 Violin Plot :

# In[22]:


sns.violinplot(x='status',y='age',data=haberman,size=10)
plt.grid()
plt.title("Age Violin Plot")
plt.show()


# In[23]:


sns.violinplot(x='status',y='year',data=haberman,size=10)
plt.grid()
plt.title("Year Of Operation Violin Plot")
plt.show()


# In[24]:


sns.violinplot(x='status',y='node',data=haberman,size=10)
plt.grid()
plt.title("Number Of Nodes Violin Plot")
plt.show()


# ##### Observation : 
# 1. First Plot Tells There are more number of people of age 53 and died of age 50.At the age 27 people started dying.
# 2. Second Plot Tells There are more Survivour in 1959 and There are more Deaths in 1964.
# 3. Third Plot Tells There are more number of people Survived with 0 Nodes and There are more Number of people deaths with 3 Nodes 

# #### 2. Bi-variate analysis :

# ##### 2.1 Pair Plots : 

# In[25]:


plt.close()
sns.set_style("whitegrid")
sns.pairplot(haberman,hue="status",height=5)
plt.show()


# ##### Observation : 
# 1. The Two-D Scatter Plots Are Overlapping so much.
# 2. The Age between 30-40 survived even though they have more number of nodes.
# 3. If The Age is Less and Number of nodes is less there is hight chance of survival.
# 4. Age is directly proportional to number of nodes probability of Survival and Death wise.

# ### Conclusion:
# 1. If The Age is Less,Number Of Nodes is Less and Year Of Operation is Higher than you have more chance of survival.
# 2. The Data is not Linearly Seperable, Need Something more complicated algorithm to seperate this data.
# 3. One of the Best features from univariate analysis is Number Of Nodes and From Bivariate Analyis is Age and Number Of Nodes.
# 4. The Data is Overlapping so much, so classification is difficult with these features.
