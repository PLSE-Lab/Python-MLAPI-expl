#!/usr/bin/env python
# coding: utf-8

# # Haberman's Survival Data Assignment
# 

# 
# The Haberman's dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 
# Attribute Information:
# 
#     Age of patient at time of operation (numerical)
#     Patient's year of operation (year - 1900, numerical)
#     Number of positive axillary nodes detected (numerical)
#     Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year
#  
# 
# Objective :- Find the model to classify the Patient Survival from the given attributes 

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
'''Reading the Haberman's Data'''
#Load haberman.csv into a pandas dataFrame.
#To read the data from the specified convert the data file path from normal string to raw string
import os
print(os.listdir("../input"))
Haber = pd.read_csv("../input/haberman.csv",header = None)


# #View the dataframe shape and colums details

# In[ ]:


print (Haber.shape)
print (Haber.columns)


# In[ ]:


#Adding headder to the Haber dataframe
Haber.columns=["Age","Operation_year","axil_nodes","Surv_Status"]


# In[ ]:


print (Haber.columns)


# In[ ]:


Haber.head(5)


# In[ ]:


Haber.describe()


# Observation :-
# 
#     1.The Age tells us the Dataset is between the Age the age group (30-83) with an average age of 52
#     2.The Data set has been collected during the time priod (58-69) 12 years of data 
#     3.The positive axillary nodes are ranging from (0-52) looking at the 75% value and the Max there should be some
#     data currepted
#     4.The survival status has only two values 1 and 2
# 

# In[ ]:


Haber.plot(kind='Scatter',x='Age',y='Operation_year') 
plt.show()


# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(Haber,hue='Age',size=4)  .map(plt.scatter,"Age","axil_nodes")
plt.show()


# In[ ]:


Haber["Surv_Status"] = Haber["Surv_Status"].apply(lambda y: "Survived" if y == 1 else "Died")
Survive_long=Haber.loc[Haber["Surv_Status"] == "Survived"]
Survive_short=Haber.loc[Haber["Surv_Status"] == "Died"]
plt.plot(Survive_long["Age"],np.zeros_like(Survive_long['axil_nodes']),'o')
plt.plot(Survive_short["Age"],np.zeros_like(Survive_short['axil_nodes']),'x')
plt.show()


# In[ ]:


plt.close()
sns.set_style("whitegrid")
sns.pairplot(Haber,hue='Surv_Status',size=3)

plt.show()


# Observation:
# 
# We are not able to clearly seperate the Survival catagory but looking at the axillary nodes plot graph it is showing some better classification

# In[ ]:


import numpy as np
Survive_long=Haber.loc[Haber["Surv_Status"] == "Survived"]
Survive_short=Haber.loc[Haber["Surv_Status"] == "Died"]
plt.plot(Survive_long["Age"],np.zeros_like(Survive_long['axil_nodes']),'o')
plt.plot(Survive_short["Age"],np.zeros_like(Survive_short['axil_nodes']),'x')
plt.show()


# In[ ]:


Survive_long.describe()


# In[ ]:


Survive_short.describe()


# Observation :-
# 
# Looking at the data above we can clearly say that axillary nodes is much differed between the two catagories
# 
#     1.Though the Pactients survived long has the axillary nodes max as 46 25%-75% of the pactients survived has the
#        rage of [0-3]   axillary nodes 
#     2.The Pactients Survived short has the axillary nodes rage  [0-11] observing the the 25%-75%
# 
# We can Build a simple model on the axillary nodes to by if else codition to say all the patients with less than the 3 can Survive more than 5 years further we can look at the other attribute combinations to improvise our model 

# In[ ]:


sns.FacetGrid(Haber,hue="Surv_Status",size=5)    .map(sns.distplot,"Operation_year")    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(Haber,hue="Surv_Status",size=5)    .map(sns.distplot,"Age")    .add_legend()
plt.show()


# In[ ]:


sns.FacetGrid(Haber,hue="Surv_Status",size=5)    .map(sns.distplot,"axil_nodes")    .add_legend()
plt.show()


# In[ ]:


#Plot CDF of Survive_Status

# Survive_short
counts, bin_edges = np.histogram(Survive_short['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
 

#Survive_long
counts, bin_edges = np.histogram(Survive_long['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();


# In[ ]:


#Plot CDF of Survive_Status

# Survive_short
counts, bin_edges = np.histogram(Survive_short['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
 

#Survive_long
counts, bin_edges = np.histogram(Survive_long['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();


# In[ ]:


# virginica
counts, bin_edges = np.histogram(Survive_short['Operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
 

#versicolor
counts, bin_edges = np.histogram(Survive_long['Operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();


# In[ ]:


sns.boxplot(x='Surv_Status',y='Age', data=Haber)
plt.show()
sns.boxplot(x='Surv_Status',y='Operation_year', data=Haber)
plt.show()
sns.boxplot(x='Surv_Status',y='axil_nodes', data=Haber)
plt.show()


# In[ ]:


sns.violinplot(x="Surv_Status", y="Age", data=Haber, size=8)
plt.show()
sns.violinplot(x="Surv_Status", y="Operation_year", data=Haber, size=8)
plt.show()
sns.violinplot(x="Surv_Status", y="axil_nodes", data=Haber, size=8)
plt.show()


# Observation:-
# 
#     1.The patients treated after 1966 have higher chance to surive than the rest
#     2. Age group of 30 -34 are in the survived region 
#     3.Age group of the 78-83 are in dead reagion 

# Final Thoughts:-
# 
# The Dataset is an imbalanced dataset and based on the observations we can build a model with the below conditions for chances of Survival and Non Survival 
# 
# Survival :-
#     
#     1.Axillary nodes value less than 3 Can  survive
#     2.Axillary nodes value less than 3 and treated after 1966 has higher chance to Survive 
#     3.Axillary nodes value less than 3 and treated after 1966 and patient's with in the age less than 34 
#     Can definitely Survive
#     
# Non Survival:-
# 
#     1.Axillary nodes value grater than 3 chances of surviving  is less
#     2.Axillary nodes value grater than 3 and treated before 1966 has veryless chances of surviving 
#     3.Axillary nodes value grater than 3 and treated before 1966 and patient's with age gretar than 83
#     definitely Can't Survive
# 
