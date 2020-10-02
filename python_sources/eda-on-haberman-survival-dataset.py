#!/usr/bin/env python
# coding: utf-8

# # Data Description
# 
# Haberman dataset: Dataset is to classify survival of patients who had undergone breast cancer surgery.
# 
# Dataset contains 3 featuresa and 1 class:
# Age: Age of patient at the time of surgery
# Op_Year: Patient's year of surgery(operation)
# axil_nodes_det:Number of positive axillary nodes detected
# Surv_status: It is a class feature
#                - 1(Patient survived 5 or more years)
#                - 2 (the patient died with in 5 years of surgery)
#                
# 
# Objective: To classify new patient will survive for 5 or more years or not after breast cancer surgery.

# # Data Preparetion

# In[ ]:


#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Load data

data=pd.read_csv("../input/haberman.csv",header=None, names=['Age', 'Op_Year', 'axil_nodes_det', 'Surv_status'])

data.shape
data.columns


# There are 306 data points and 4 features.
# Features name/columns names are Age, Op_Year, axil_nodes_det, Surv_status

# In[ ]:


# Printing few data points
data.head()


# In[ ]:


# Calculate no of classes and data points per class
data['Surv_status'].value_counts()


# Class value 1 means Patient survived 5 or more years  and 2 means the patient died with in 5 years of surgery.
# 
# Here for class value 1 counts are 225 means 225 patients survived and 81 patients not.
# 

# # Exploratory Data Analysis
# 
# # Objective: To classify new patient will survive for 5 or more years or not after breast cancer surgery.

# In[ ]:


#Data Analysis
data.describe()


# Axil_nodes_det seems to be very weird.. may be some outlier as min value is 0.0 and max is 52 but mean value is 4.02 and 75% is 4 meand 75% data lies below value 4 of axilary nodes.

# In[ ]:


data['Op_Year'].value_counts()


# Maximum operations done in year 58 and minimum operations done in year 69

# In[ ]:


data['axil_nodes_det'].value_counts()


# Analysis: Most patients have axilary nodes 4 or less than 4

# # Bi-varate Analysis

# In[ ]:


data.plot(kind='scatter',x='axil_nodes_det',y='Surv_status')
plt.show()


# Not able to conclude from scatter plots.

# In[ ]:


#Pair plot

plt.close();
sns.set_style("whitegrid")
sns.pairplot(data,hue='Surv_status',vars=["Age", "Op_Year","axil_nodes_det"],height=3);
plt.show()


# Analysis:From pdf of axil_nodes_det we can conclude that survivalcounts is maximum at axil_nodes_det=0 and 
# Age and Op_Year pdf are overlaping.

# # Univarate Analysis

# In[ ]:


#Histogram(1-D sctter plot kind of)

sns.FacetGrid(data,hue="Surv_status",size=5)    .map(sns.distplot, "axil_nodes_det")    .add_legend();
plt.ylabel('Frequency')
plt.show();


# For axil_nodes_det>=30, surv_status is 2 only 1 patient is survived (but there are only 4 patients for 30 or above 30)

# In[ ]:


#PDF AND CDF For feature Age

data_1=data[data['Surv_status']==1]
counts, bin_edges= np.histogram(data_1['Age'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.plot(bin_edges[1:],pdf,label='pdf_survived')
plt.plot(bin_edges[1:],cdf,label='cdf_survived')


data_2=data[data['Surv_status']==2]
counts_2, bin_edges_2= np.histogram(data_2['Age'],bins=10, density= True)
pdf_2=counts_2/(sum(counts_2))
print(pdf_2)

print(bin_edges_2)

cdf_2=np.cumsum(pdf_2)
print(cdf_2)

plt.plot(bin_edges_2[1:],pdf_2,label='pdf_not-survived')
plt.plot(bin_edges_2[1:],cdf_2,label='cdf_not-survived')


plt.xlabel("Age")
plt.ylabel("Probabilty")
plt.legend()
plt.title("pdf and cdf of Age")
plt.show();


# From above cdf and pdf graph.. we can in ititial years all patients survived.

# In[ ]:


#PDF AND CDF For feature Op_Year
plt.close();
data_1=data[data['Surv_status']==1]
counts, bin_edges= np.histogram(data_1['Op_Year'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.figure(1)


plt.subplot(211)
plt.plot(bin_edges[1:],pdf,label='pdf_survived')
plt.plot(bin_edges[1:],cdf,label='cdf_survived')
plt.xlabel("Op_Year")
plt.ylabel("Probabilty")
plt.legend()
plt.title("Pdf and cdf plot of Op_Year")



data_2=data[data['Surv_status']==2]
counts_2, bin_edges_2= np.histogram(data_2['Op_Year'],bins=10, density= True)
pdf_2=counts_2/(sum(counts_2))
print(pdf_2)

print(bin_edges_2)

cdf_2=np.cumsum(pdf_2)
print(cdf_2)

plt.subplot(212)
plt.plot(bin_edges_2[1:],pdf_2,label='pdf_not-survived')
plt.plot(bin_edges_2[1:],cdf_2,label='cdf_not-survived')



plt.xlabel("Op_Year")
plt.ylabel("Probabilty")
plt.legend()

plt.show();


# Graphs are almost similar for Op_year

# In[ ]:


#PDF AND CDF For feature axil_nodes_det

data_1=data[data['Surv_status']==1]
counts, bin_edges= np.histogram(data_1['axil_nodes_det'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.plot(bin_edges[1:],pdf,label='pdf_survived')
plt.plot(bin_edges[1:],cdf,label='cdf_survived')


data_2=data[data['Surv_status']==2]
counts_2, bin_edges_2= np.histogram(data_2['axil_nodes_det'],bins=10, density= True)
pdf_2=counts_2/(sum(counts_2))
print(pdf_2)

print(bin_edges_2)

cdf_2=np.cumsum(pdf_2)
print(cdf_2)
plt.plot(bin_edges_2[1:],pdf_2,label='pdf_survived')
plt.plot(bin_edges_2[1:],cdf_2,label='cdf_survived')



plt.xlabel("axil_nodes_det")
plt.ylabel("Probabilty")
plt.legend()
plt.title("Pdf and cdf plot of axil_nodes_det")
plt.show();


# using Pdf,Cdf ,scatter plot, pair plot we analyse that maximum patients went for surgent have Axilary nodes less than 10 i.e. 
# axil_nodel_det < 10

# In[ ]:


#BoxPlot for Age
sns.boxplot(x="Surv_status",y="Age",data=data)
plt.show()


# In[ ]:


#BoxPlot for Op_Year
sns.boxplot(x="Surv_status",y="Op_Year",data=data)
plt.show()


# In[ ]:


#BoxPlot for axil_nodes_det
sns.boxplot(x="Surv_status",y="axil_nodes_det",data=data)
plt.show()


# In[ ]:


#violinPlot for axil_nodes_det
sns.violinplot(x="Surv_status",y="axil_nodes_det",data=data)
plt.show()


# In[ ]:


#violinPlot for Op_Year
sns.violinplot(x="Surv_status",y="Op_Year",data=data)
plt.show()


# In[ ]:


#violinPlot for AgeS
sns.violinplot(x="Surv_status",y="Age",data=data)
plt.show()


# # Observations:
# 1. Number of survivors are highly dense when axiliary nodes are btween 0 to 5.
# 2. 75% of patients have 4 or less axillary nodes.
# 3. Patients operated before 1960 didnt survived, pateints survived when operated in 1960 or after
# 4. Patients suvival percentage is higher after 1966 than before.
# 5. Age is not play an important feature role in this dataset.
