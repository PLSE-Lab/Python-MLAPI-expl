#!/usr/bin/env python
# coding: utf-8

# # **Data Description**
# The Haberman's survival dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 

# In[ ]:


#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


# In[ ]:


#load the data set
haberman=pd.read_csv('../input/haberman.csv')
haberman
haberman.head()


# In[ ]:


#assign coloumn names manually
haberman.columns = ["Age", "Year of operation", "axillary nodes", "Survival status after 5 years"]
haberman.columns


# In[ ]:


#Top 5 data elements with header
haberman.head()


# In[ ]:


# Get how many data elements and columns are there
haberman.shape


# In[ ]:


# Get  data type of variables
haberman.dtypes


# # observation:
# - We have total 305 data elements and it has 4 columns 
# - The survival status after 5 years is an integer type so it needs to be converted into category type,and it's values are ,mapped as survived and not survived
# 

# In[ ]:


# modify the target column values to be meaningful as well as categorical
haberman['Survival status after 5 years'] = haberman['Survival status after 5 years'].map({1:"survived", 2:"not survived"})
haberman['Survival status after 5 years'] = haberman['Survival status after 5 years'].astype('category')
haberman.head()


# In[ ]:


print(haberman.iloc[:,-1].value_counts())
haberman.dtypes


# In[ ]:


haberman["Age"].value_counts()[:10]


# # observation
# - There are 224 people who have survived and 81 people who have not survived
# - From the above data we can say that the maximum no.of people who are affected with cancer are of age 47yrs-57yrs

# # OBJECTIVE
# - To predict whether the patient will survive after 5 years or not based upon the patient's age, year of operation and the number of axillary nodes

# # 1) UNIVARIATE ANALYSIS

# In[ ]:


#PDF AND HISTOGRAM PLOTS
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.set_style(style="whitegrid")
    sns.FacetGrid(haberman,hue=('Survival status after 5 years'),size=5)             .map(sns.distplot,feature)             .add_legend()
    plt.xlabel(feature)         
    plt.ylabel('PDF')
    plt.legend()
    plt.title("survival status with respective {}".format(feature))
plt.show()


# # Observations
# - most of the people between the age 30-40 years who are affected with cancer survived

# In[ ]:


#cdf 
#plot cdf for age
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    count,bin_edges=np.histogram(haberman[feature],bins=10,density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = count/sum(count)
    print("PDF: {}".format(pdf))
    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(feature)
    plt.ylabel('PDF')
    plt.legend(['Pdf for the patients who dead Within 5 years',
            'Cdf for the patients who dead within 5 years'])
    plt.show()


# In[ ]:


#Statistical Description data of people who survived more than 5 years
haberman_Survived = haberman[haberman["Survival status after 5 years"] == 'survived']
print ("Summary of patients who are survived more than 5 years")
haberman_Survived.describe()


# In[ ]:


#Statistical Description data of people who survived less than 5 years
haberman_notSurvived = haberman[haberman["Survival status after 5 years"] == 'not survived']
print ("Summary of patients who are survived not more than 5 years")
haberman_notSurvived.describe()


# In[ ]:


haberman['Age'].mean()


# In[ ]:


haberman.Age[haberman.Age == haberman.Age.max()]


# In[ ]:


haberman.Age[haberman.Age == haberman.Age.min()]


# # Observation
#  from the above data
# - The average age to get affected with cancer is 52-53yrs
# - Minimum age to get affected with cancer is 30-40 yrs 
# - Mximum age to get affected with cancer is 75-80 yrs 

# In[ ]:


#box plots
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.boxplot(x="Survival status after 5 years",y=feature,data=haberman)
    plt.show()


# In[ ]:


#violin plot
for idx, feature in enumerate(list(haberman.columns)[:-1]):
    sns.violinplot(x="Survival status after 5 years",y=feature,data=haberman,size=8)
    plt.show()
    


# # Observations :
# 
# <font size=4> 
# - survived and non survived are almost equal when we comare with Age and Year of operation.
# <br>
# - we can see most Survival patients fall has Zero Auxillary nodes
#     <br>
# - The number of auxillary nodes is inversely proportional to the chance of survival</font>
# 

# # MULTIVARIATE ANALYSIS

# In[ ]:


#PAIR PLOTS
sns.pairplot(haberman, hue='Survival status after 5 years', size=4)

plt.legend(['not survived','survived'])
plt.show()


# # Obsevations
# - we can get better survival status by the graph between auxillary nodes and year of operation

# # Contour plot

# In[ ]:


#contour plot
sns.jointplot(x="Age",y="Year of operation",data=haberman,kind='kde')
plt.show()


# # Observation
# 
# - From the above contour plot, There are many number of patients who has undergone operation is between Year 59 to 64
# - And the age of them is between 45 to 55 approximately
