#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis(EDA) For Haberman's Servival Data Set
# 

# ### Objective / About the DataSet :-

# * The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 

# source - https://www.kaggle.com/gilsousa/habermans-survival-data-set/data 

# ### Attributes Information: 

# There are 4 attribute as per the dataset above
# 
# 1. Age of patient at time of operation (numerical) - 30
# 2. Patient's year of operation (year - 1900, numerical) - 64
# 3. Number of positive axillary nodes detected (numerical) - 1 
# 4. Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year - 1.1

# In[ ]:


#importing the require packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#haberman's dataframe information from pandas
haberman_df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv')


# In[ ]:


haberman_df.head()


# ##### Observation

# The Attributes are numerical representation for this dataset. Information regarding these numeric is corresponding to the each attribute define in above attributes Information section.

# * Let's rename the column for better Understanding.
# 

# In[ ]:


haberman_df = haberman_df.rename(columns={"30": "Age_of_patient","64":"Patient_year_of_operation","1":"positive_axillary_nodes","1.1":"Survival_status"})


# In[ ]:


haberman_df.shape


# In[ ]:


haberman_df.columns


# In[ ]:


haberman_df.describe()


# In[ ]:


haberman_df["Survival_status"].value_counts()


# ##### Observation :-
# 1. Number of points - It Contains 304 data points/Rows
# 2. Numer of features - There are 3 Features/Independent variables (Age of patient, Patient's yearofOperation, axillary nodes detected )
# 3. Number of classes - There are 2 classes Patient Survival Status.
#     1 -  the patient survived 5 years or longer 
#     2 -  the patient died within 5 year 
# 4.  Data-points per class - 
#     There are 224 datapoints for 1 and 81 for 2 . It's Look like this is imbalanced dataset

# ## Univaraite Analysis 

# ### (1.1) PDF (Probability Density Function) 

# #### (1.1.1) PDF For  Age of Patient Feature Analysis:-

# In[ ]:


#using seaborn
sns.set_style("whitegrid")
sns.FacetGrid(haberman_df,hue="Survival_status",size=5)    .map(sns.distplot,"Age_of_patient")     .add_legend()

plt.show()


#  #### (1.1.2) PDF For Patient year of operation Feature Analysis :-

# In[ ]:


sns.FacetGrid(haberman_df,hue="Survival_status",size=5)    .map(sns.distplot,"Patient_year_of_operation")     .add_legend()

plt.show()


# #### (1.1.3) PDF For Positive Axillary Nodes  Feature Analysis :-

# In[ ]:


#using seaborn
sns.FacetGrid(haberman_df,hue="Survival_status",size=5)    .map(sns.distplot,"positive_axillary_nodes")     .add_legend()

plt.show()


# ##### Observation :-

# 1. For all three plots the features are overlap each other massively.
# 2. Still From the positive axillary nodes we can conclude that the patient had positive axillary nodes between the 0 to 4 having more probabilty of  the patient survived 5 years or longer and less  probabilty of the patient died within 5 year
# 3. Most of the patient died within 5 year whose age is in between 40 to 50

# ### (1.2) Cumulative Distribution Fuction (CDF)

# In[ ]:


haberman_Servive = haberman_df.loc[haberman_df["Survival_status"] == 1];
haberman_NotServive = haberman_df.loc[haberman_df["Survival_status"] == 2];


# #### (1.2.1) CDF For Age of Patient Feature with haberman Servive Analysis :-

# In[ ]:



counts, bin_edges = np.histogram(haberman_Servive["Age_of_patient"],bins=10,density=True)
#(max-min)/bins = bin_edges 
#number of point fall in each bean = counts (eg -1----11 - 5 counts ,11---21 - 3 counts..etc)
pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)# matchmatical calculation of cdf
cdf
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.xlabel("Age_of_patient")
plt.show()


#   #### (1.2.2) CDF For Positive Axillary Node Feature with haberman Servive Analysis :-

# In[ ]:


counts, bin_edges = np.histogram(haberman_Servive["positive_axillary_nodes"],bins=10
                                 ,density=True)
pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)# matchmatical calculation of cdf
cdf
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.xlabel("positive_axillary_nodes")
plt.show()


# #### (1.2.3) CDF For Patient Year Of operation Feature with haberman Servive Analysis:-

# In[ ]:


counts, bin_edges = np.histogram(haberman_Servive["Patient_year_of_operation"],bins=10,density=True)
#(max-min)/bins = bin_edges 
#number of point fall in each bean = counts (eg -1----11 - 5 counts ,11---21 - 3 counts..etc)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)# matchmatical calculation of cdf
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.xlabel("Patient_year_of_operation")
plt.show()


# ##### Observation :-

# 1. 5% of  the patient survived 5 years or longer whose Age <=  32 years .
# 2. There are 80% of the patient survived 5 years or longer that have Age <= 63
# 3. There are 100% of the patient survived 5 years or longer that have Age <= 78
# 3. There are 20% of patient survived 5 years or longer whose year of operation is <=1957
# 4. 100% of probability patient survived 5 years or longer whose year of operation is <=1969 may be due to advance medical technology
# 5. There are 82% probability  patient survived 5 years or longer whose positive axillary nodes <= 4
# 6. There are 100% probability  patient survived 5 years or longer whose positive axillary nodes <= 30

# #### (1.2.4) CDF For All Feature with haberman Not Servive Analysis :-

# In[ ]:


counts, bin_edges = np.histogram(haberman_NotServive["Age_of_patient"],bins=10,density=True)
pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)# matchmatical calculation of cdf
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.xlabel("Age_of_patient")
plt.show()


counts, bin_edges = np.histogram(haberman_NotServive["Patient_year_of_operation"],bins=10,density=True)
pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)# matchmatical calculation of cdf
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.xlabel("Patient_year_of_operation")
plt.show()


counts, bin_edges = np.histogram(haberman_NotServive["positive_axillary_nodes"],bins=10,density=True)
pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)# matchmatical calculation of cdf
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.xlabel("positive_axillary_nodes")

plt.show()


#  ##### Observation :-
# 1. 5% probability of the patient died within 5 year whose age<=35
# 2. 100% probability of the patient died within 5 year whose age<=85
# 3. 23% probability of the patient died within 5 year whose year of operation <= 1959
# 4. 100%  probability of the patient died within 5 year whose year of operation <= 1969
# 5. 58% probability of the patient died within 5 year whose positive axilary node <= 5
# 6. 100% probability of the patient died within 5 year whose positive axilary node <= 100

# ### (1.3) Box Plot :-

# #### (1.3.1) Box Plot For Age of patient Feature Analysis :-

# In[ ]:


sns.boxplot(x="Survival_status",y="Age_of_patient",data=haberman_df)
plt.show()


# #### (1.3.2) Box Plot For Patient Year of Operation Feature Analysis :-

# In[ ]:


sns.boxplot(x="Survival_status",y="Patient_year_of_operation",data=haberman_df)
plt.show()


# #### (1.3.3) Box Plot For Positive Axilary Node Feature Analysis :-
# 

# In[ ]:


sns.boxplot(x="Survival_status",y="positive_axillary_nodes",data=haberman_df)
plt.show()


# ### (1.4) Violin Plot

# #### (1.4.1) Violin Plot For Age of Patient Feature Analysis :-

# In[ ]:


sns.violinplot(x="Survival_status",y="Age_of_patient",data=haberman_df,hue="Survival_status")
plt.show()


# #### (1.4.2) Violin Plot For Patient Year Of Opeartion Feature Analysis :-

# In[ ]:


sns.violinplot(x="Survival_status",y="Patient_year_of_operation",data=haberman_df,hue="Survival_status")
plt.show()


# #### (1.4.3) Violin Plot For Positive Axillary Nodes Feature Analysis :-

# In[ ]:


sns.violinplot(x="Survival_status",y="positive_axillary_nodes",data=haberman_df,hue="Survival_status")
plt.show()


# ##### Observation :-

# 1. There are massive overlap for all features
# 2. Still We can classify using positive axillary nodes value with some error in it. 
# 3. Most of the Points on the patient survived 5 years or longer lies auxilary value <=7
# 4. 75% of the patient survived 5 years or longer lies auxilary value <=4 still less then 50% of the patient died within 5 year so there is possibility of having error of 50% if we write simple model using if else condition
# 5. there are 75% probability the patient died within 5 year whose auxilary value <= 12

# ## Bivariate Analysis :-

# ### (1.1) Scatter Plot :-

# #### (2.1.1) Scatter Plot For Patient Year of Operation Feature Analysis:-[](http://)

# In[ ]:


sns.FacetGrid(haberman_df,hue="Survival_status",size=5)     .map(plt.scatter,"Patient_year_of_operation","Age_of_patient")     .add_legend()
plt.show()


# #### (2.1.2) Scatter Plot For Postitive Axillary Nodes Feature Analysis :-

# In[ ]:


sns.FacetGrid(haberman_df,hue="Survival_status",size=5)     .map(plt.scatter,"Patient_year_of_operation","positive_axillary_nodes")     .add_legend()
plt.show()


# #### (2.1.3) Scatter Plot For Age of Patient Feature Analysis :-

# In[ ]:


sns.FacetGrid(haberman_df,hue="Survival_status",size=5)     .map(plt.scatter,"Age_of_patient","positive_axillary_nodes")     .add_legend()
plt.show()


# ##### Observation :-

# 1. We can not predict anything using these scatter plot as there is always an error 
# 2. Still with positive auxilary node , We can Predict  the patient survived 5 years or longer whose auxilary value is <=4 with any age between 30 to 40 , 50 to 60 and 70 t0 80 . There are more chances of the patient died within 5 year whose age is 40 to 50 and 60 to 70

# ### (2.2) Pair Plot :-

# In[ ]:


haberman_df.loc[:]
sns.pairplot(haberman_df,hue="Survival_status",size=4, vars= ['Age_of_patient','Patient_year_of_operation','positive_axillary_nodes'])
plt.show()


# ##### Observation :-

# 1. Here we cannot classify between any feature without having error. There are some error during classification
# 2. Still , We can somehow classify between the positive axilary value with the age of patient with having some error
# 3. there are more probability of the patient survived 5 years or longer whose auxilary value is <=4

# ## Conclusion :-

# 1. This data set is imbalanced data set There are 224 datapoints for 1 and 81 for 2
# 2. Patient admited for the breast cancer are in between the age of 30 to 85
# 3. Most of the patient died within 5 year whose age is in between 40 to 50
# 4. The Given data set is not linearly separable as there are lots of overlaping in the datapoints and hence It is very difficult to classify
# 5. There are still positive axillary nodes feature which give some intution for this data set comapare to other features.
# 6. We cannot build simple if else model for this dataset because there are more probabilty of getting error on this . For this we need complex model to solve this data set  
