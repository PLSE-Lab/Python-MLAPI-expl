#!/usr/bin/env python
# coding: utf-8

# ## Exercise
# ### 1. Download Haberman Cancer Survival dataset from Kaggle. You may have to create a Kaggle account to donwload data (https://www.kaggle.com/gilsousa/habermans-survival-data-set)
# 
# ### 2. Perform a similar alanlaysis as above on this dataset with the following sections:
# 
# --> High level statistics of the dataset: number of points, numer of features, number of classes, data-points per class.
# 
# -->Explain our objective.
# 
# --> Perform Univaraite analysis(PDF, CDF, Boxplot, Voilin plots) to understand which features are useful towards classification.
# 
# --> Perform Bi-variate analysis (scatter plots, pair-plots) to see if combinations of features are useful in classfication.
# 
# --> Write your observations in english as crisply and unambigously as possible. Always quantify your results.

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pylab as pylab


# In[ ]:


## Loading Python Data Set
df_hb = pd.read_csv("../input/habermans-survival-data-set/haberman.csv",header=None,names=['Age','Op_Year','axil_nodes','Surv_status'])


# In[ ]:


df_hb.head()


# In[ ]:


df_hb['Surv_status'].value_counts()


# In[ ]:


df_hb['Op_Year'].value_counts()


# In[ ]:


df_hb.describe()


# In[ ]:


print("Median of Positive Axillary nodes is {0}".format(np.median(df_hb["axil_nodes"])))
print("Median of Age is {0}".format(np.median(df_hb["Age"])))


# In[ ]:


df_hb.info()


# In[ ]:


hb_SurvStatus_1 = df_hb.loc[df_hb["Surv_status"] == 1];
hb_SurvStatus_2 = df_hb.loc[df_hb["Surv_status"] == 2];
print(hb_SurvStatus_1.describe())
print(hb_SurvStatus_2.describe())


# In[ ]:


print("*************Median**************")
print("Median of Age for Survival Status 1 is {0} & 2 is {1}".format(np.median(hb_SurvStatus_1["Age"]),np.median(hb_SurvStatus_2["Age"])))
print("Median of Operation year for Survival Status 1 is {0} & 2 is {1}".format(np.median(hb_SurvStatus_1["Op_Year"]),np.median(hb_SurvStatus_2["Op_Year"])))
print("Median of Positive Axillary nodes for Survival Status 1 is {0} & 2 is {1}".format(np.median(hb_SurvStatus_1["axil_nodes"]),np.median(hb_SurvStatus_2["axil_nodes"])))


# In[ ]:



print("Positive Axillary Nodes")
print(np.percentile(hb_SurvStatus_1["axil_nodes"],np.arange(0, 100, 10)))
print(np.percentile(hb_SurvStatus_2["axil_nodes"],np.arange(0, 100, 10)))

print("\nQuantiles:")
print(np.percentile(hb_SurvStatus_1["axil_nodes"],np.arange(0, 100, 25)))
print(np.percentile(hb_SurvStatus_2["axil_nodes"],np.arange(0, 100, 25)))

print("\n95% Percentile:")
print(np.percentile(hb_SurvStatus_1["axil_nodes"],95))
print(np.percentile(hb_SurvStatus_2["axil_nodes"],95))

print("\nAge")
print(np.percentile(hb_SurvStatus_1["Age"],np.arange(0, 100, 10)))
print(np.percentile(hb_SurvStatus_2["Age"],np.arange(0, 100, 10)))


print("\nOp_Year")
print(np.percentile(hb_SurvStatus_1["Op_Year"],np.arange(0, 100, 10)))
print(np.percentile(hb_SurvStatus_2["Op_Year"],np.arange(0, 100, 10)))


# ### Basic Observations from above Analysis is
# --> the Studay has been done from Year 1958 to 1969
# 
# --> out of 306 Patient who undergone Surgery during the study Period 225(73%) has Survived for 5 years or Longer
# 
# -->if we look at the Mean and Median of all the Patient and Mean and Median of the Patient in both the Survival Categories. the value is ranges between 52-53. which means there is not a large outlier for the Age Feature wven though when the Patient underwent surgey has has age from 30 to 83 years
# 
# --> One Important Point to note in the Above analsis is if we see the Dataset for Survival Status 1 the Maximum number of Postivie Axilary node is 46 but the Axil Node value for 75 Percentile is just 3 which says about 75% patient who Survivied for 5 year or More has Positive axilay nodes less than or equal to 3 and 90% Patient who survivied for 5 or More years have Positivie axilary nodes of less than equal to 8

# ## 3. High level statistics of the dataset: number of points, numer of features, number of classes, data-points per class.

# In[ ]:


print ("Number of Data Points {0}". format(df_hb.shape[0]))
print ("Number of Features {0}". format(df_hb.shape[1]))
print ("List of Columns {0}". format(df_hb.columns[:]))
print("\nClasses and Data Points Per class",df_hb.groupby(df_hb['Surv_status'])['Surv_status'].count())


# ### basic Observation from above analysis
# --> there are total 306 Data Points in the Dataset 
# 
# --> there are total of 4 columns out of which three are feature column and one column is Class Column
# 
# --> Features are : Age, Operation year, Number of Positive Axillary Nodes
# 
# --> class Column : Surv_Status which says the Survival Status of the indvidual rows
# 
# --> there are two categories in "Survival Status" column i.e, 1 and 2 with total records of 225 and 81 Respectively for each classess

# ## 4. Explain our Objective

# our Primary objective is to categories the Survival Status of a Patient among value 1 & 2 for a given value of Age, Operation Year and number Positive Axillary Nodes feature

# ## 5. Perform Univaraite analysis(PDF, CDF, Boxplot, Voilin plots) to understand which features are useful towards classification.

# In[ ]:


hb_SurvStatus_1 = df_hb.loc[df_hb["Surv_status"] == 1];
hb_SurvStatus_2 = df_hb.loc[df_hb["Surv_status"] == 2];


# we can Start with Histogram plot to look into data Disctribution for all the three Features

# In[ ]:


sns.FacetGrid(df_hb,hue='Surv_status',size=8).map(sns.distplot,'Age').add_legend();
plt.title("Histogram of Age")

sns.FacetGrid(df_hb,hue='Surv_status',size=5).map(sns.distplot,'Op_Year').add_legend();
plt.title("Histogram of Year of Operation")

sns.FacetGrid(df_hb,hue='Surv_status',size=5).map(sns.distplot,'axil_nodes').add_legend();
plt.title("Histogram of Axil Nodes")
plt.show()


# ## Observation from Histogram
# ### Age
# --> if the Age is less than 34 then there is full possbility of Patient to Survive for 5 years and Longer
# 
# --> For Patient with Age >=78, there are no chances of Survival for more than 5 years
# 
# --> From Age 34 to 40 The Possbility of Survival for more than 5 Years are higher as compared to Survival Status 2
# 
# --> From Age 40 The chances of longer survival Decreases except from 59 to 61 we can see there is again a larger possbility of longer survival for the patient
# 
# ### Year Of Operation
# --> from the Histogram Plot. we cannot make analysis on the Survival Status since there seems similar kind of plot for both the categories
# 
# ### Axillary Nodes
# -->it is not possbile to make an absolutely clear analysis for the Survival status of a Patient based on Positive Axilary Nodes
# 
# --> but for Patient with Axillary Nodes <=4, the Chances for longer survival are much higher and for Patient with Xillary nodes>= 8 the lnger survival chances are very less

# ### Univariate Analysis using CDF

# In[ ]:


plt.figure(figsize=(20,5))
index=1
for Feature in (list(df_hb.columns)[:-1]):
    plt.subplot(1,3,index)
    Counts , bin_edges = np.histogram(hb_SurvStatus_1[Feature],bins=20,density=True)
    pdf=Counts/sum(Counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],cdf,label="cdf of Surv. status 1",color="red")
    plt.plot(bin_edges[1:],pdf,label="pdf of Surv. status 1",color="green")

#Shorter Survival Status
    #plt.subplot(1,3,i)
    Counts1 , bin_edges1 = np.histogram(hb_SurvStatus_2[Feature],bins=20,density=True)
    pdf1=Counts1/sum(Counts1)
    cdf1 = np.cumsum(pdf1)
    plt.plot(bin_edges1[1:],cdf1,label="cdf of Surv. status 2",color="blue")
    plt.plot(bin_edges1[1:],pdf1,label="pdf of Surv. status 2",color="orange")
    plt.xlabel(Feature)
    plt.ylabel("Density Function")
    pylab.legend(loc='center')
    plt.grid(which='both')
    index+=1
plt.show()


# ## Observation
# 
# --> 75% of the Patient who survived for 5 years or longer has axil nodes of 4 or less and 90% of Patient whos survived has axil nodes of 9 or less
# 
# --> between Axilary Nodes 4-24, as the number of Positive Axillary Nodes Increases, the Chances of Less Survival grow from 40 to 90% 

# ## Box Plot

# In[ ]:


plt.figure(figsize=(20,5))
i=1
for Feature in (list(df_hb.columns)[:-1]):
    plt.subplot(1,3,i)
    sns.boxplot(x='Surv_status',y=Feature,data=df_hb)
    i+=1
    plt.grid()
plt.show()


# ## Violin PLot

# In[ ]:


plt.figure(figsize=(20,5))
i=1
for Feature in (list(df_hb.columns)[:-1]):
    plt.subplot(1,3,i)
    sns.violinplot(x='Surv_status',y=Feature,data=df_hb,size=8)
    i+=1
    plt.grid()
plt.show()


# ## Box Plot & Violin Plot
# --> if we see the box plot between Axil_nodes and Survival Status , there Seems to be to many Outliers, especially for long Survival Status. it is always good to remove these Outliers for better analysis
# 
# --> about 50% of the Patient who survived has no postive axillay nodes 
# 
# --> if you look at the Box-plot of Non_Surivval Patient (less than 5 years). the 50% percentile and 75% gap seems almose thrice as gap between 25% to 50%, in other way about 50% of Patient who Survived less has Axillary nodes of 4 or less and other 25% Patient has axillary nodes from 4-11. and once it crossess 11, the chances of Survival are very very less

# ## Scatter Plot

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df_hb, hue="Surv_status", size=4)    .map(plt.scatter, "Age", "Op_Year")    .add_legend();
plt.show();

sns.set_style("whitegrid");
sns.FacetGrid(df_hb, hue="Surv_status", size=4)    .map(plt.scatter, "Op_Year", "axil_nodes")    .add_legend();
plt.show();

sns.set_style("whitegrid");
sns.FacetGrid(df_hb, hue="Surv_status", size=4)    .map(plt.scatter, "Age", "axil_nodes")    .add_legend();
plt.show();


# ## Pair Plot

# In[ ]:


#df_hb["Surv_status"][df_hb["Surv_status"] == 1]='Yes';
#df_hb["Surv_status"][df_hb["Surv_status"] == 2]='No';
#print(df_hb)
plt.close() # it close a figure window
sns.set_style("whitegrid")
sns.pairplot(df_hb,vars=["Age","Op_Year","axil_nodes"],hue="Surv_status",size=3)
plt.show()


# ## Observations
# --> looking at the Pair plots of Heberman data, its Tough to Perform Multivariate analysis ,the only pair where we can find a data Seperation is between Acillary Nodes and Year of Operation
# --> even though it tough to Quantify the analysis , exp using a if else condition but we can say that between Year 60 to 65 Patient with axil nodes of 8 or less has higher chances of long Survival

# ## Contour plot

# In[ ]:


sns.jointplot(x="Op_Year", y="axil_nodes", data=df_hb, kind="kde");
plt.show();

sns.jointplot(x="Age", y="axil_nodes", data=df_hb, kind="kde");
plt.show();


# ## Conclusion
# -->with this Data it is not possbile to analyse the effect of parameters i,e Age, Operation Year and Axillary nodes on Survival Statu
# 
# -->if we look at the cdf and pdf Analysis we find that 75% of the Patient who survived for 5 years or longer has axil nodes of 4 or less and 90% of Patient whos survived has axil nodes of 9 or less
# 
# --> none of the variable either single or pair can help in categorizing the survivial status using if else condition
# 
# --> if there would have been large quantity of data, the analysis might have been different.

# In[ ]:




