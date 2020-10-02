#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on Haberman's Survival Data

# # Dataset Description :
# 
# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# 

# **# Objective :
# 
# 1. To classify a patient  who had undergone surgery for breast cancer had survived or not by analysing all given 3 features.
# 
# 2. 'Survival' is a feature which is class attribute.

# In[3]:


#Reading Dataset into pandas dataframe

import pandas as p

hab = p.read_csv("haberman.csv")

hab


# # Attribute Description :
# 
# **1. Patient_Age **           :=> Age of patient at time of operation (numerical)
# 
# **2. Year_of_operation **     :=> Patient's year of operation (year - 1900, numerical)
# 
# **3. #PositiveAxiliaryNode**  :=> Number of positive axillary nodes detected (numerical)
# 
# **A positive axillary lymph node** is a lymph node in the area of the armpit (axilla) to which cancer has spread. This spread is determined by surgically removing some of the lymph nodes and examining them under a microscope to see whether cancer cells are present.
# 
# **4. Survival**               :=> Survival status (class attribute)
#             
#                              1 = the patient survived 5 years or longer 
#                              2 = the patient died within 5 year

# In[4]:


# Number of data Points and Features

print(hab.shape)


# In[5]:


# Features/Attributes/Column name in our dataset

print(hab.columns)


# In[6]:


# Data points for each class or How many patient survived and how many not

hab['Survival'].value_counts()  # 1 :=> the patient survived 5 years or longer 
                                # 2 :=> the patient died within 5 year


#    => Here we got , 224 patients survived 5 years or longer and 81 patients died within 5 year when each one for them undergone surgery of breast cancer.

# # Bivariate Analysis 
# 
# **(Two variable Analysis for the purpose of determining the empirical relationship between them)**
# 
# **Using :=> **
# 
# ** 1.   2-D Scatter plot **
# 
# ** 2.   Pair-Plot **

# # (1) 2-D Scatter Plot

# In[21]:


#2-D Scatter plot when we did not color the points by their class-label/Survival-type
#Analysing by taking two feature 'Patient_Age' on x-axis and '#PositiveAxiliaryNode' on y-axis

import seaborn as sb
import matplotlib.pyplot as mp

sb.set_style('whitegrid')

hab.plot(kind='scatter', x='Patient_Age',y='#PositiveAxiliaryNode')

mp.show()


# **Observation(s):**
# 
# 1. Here we are not able to classify or make any decision regarding patient survival as here class-label is not  colored by which we are not able tell who survived or who not.

# In[22]:


#2-D scatter plot when we did color the points by their class-label/Survival-type
#Analysing by taking two feature 'Patient_Age' on x-axis and '#PositiveAxiliaryNode' on y-axis

sb.set_style('whitegrid')

sb.FacetGrid(hab,hue='Survival',size=4)  .map(mp.scatter,'Patient_Age','#PositiveAxiliaryNode')  .add_legend()

mp.show()


# **Observation(s):**
# 
# 1. Here we clearly see that blue points are not seperated from orange points.
# 
# 2. So, by looking this 2-D scatter plot between 'Patient_Age' and '#PositiveAxiliaryNode' we cannot make any decision regarding patient will survive or not.
# 
# 3. Therefore, we have to check all combination/pair of features to make good classification/decision.
# 
# 4. Number of Combinations of features : 3C2 = 3 (excluding class-attribute 'Survival')
# 
# 5. Now, for these combination to analyse we get help from Pair-Plot concept.

# # (2) Pair-Plot 

# In[9]:


#Pairwise combination of each of features

mp.close() #by itself closes the current figure
sb.set_style('whitegrid')
sb.pairplot(hab,hue='Survival',size=3)
mp.show()


# **Observation(s) :**
# 
# 1. Here we can clearly see that in any of the combinations of all the 3 features blue points and orange points are not seperated.
# 2. So, we conclude that using bivariate analysis we cannot classify that patient will survive or not who had undergone surgery for breast cancer.
# 3. If bivariate analysis did not help us in making classification than we do Univariate Analysis(Where we analyse only one variable).

# # Univariate Analysis (one variable Analysis)
# 
# **Using :=> **
# 
# 1. Histogram/PDF
# 2. CDF
# 3. Mean,Variance and Standard-Deviation,Median,Quantiles,Percentiles,IQR,MAD
# 
# 4. Box-Plot
# 5. Violin Plot

# # (1) Histogram ,PDF

# In[10]:


#1-D Scatter Plot for 'Patient_Age'

import numpy as np

hab_survive=hab.loc[hab["Survival"]==1]
hab_notsurv=hab.loc[hab["Survival"]==2]

mp.plot(hab_survive["Patient_Age"],np.zeros_like(hab_survive["Patient_Age"]),'o') # 'o' is there to show plot as dot/circle shape/o shape instead of just line

mp.plot(hab_notsurv["Patient_Age"],np.zeros_like(hab_notsurv["Patient_Age"]),'o')

mp.show()


# **Observation(s) : **
# 1. Here we can see that blue points and orange points are overlapping so, we cannot say anything from it.
# 2. Now, if we can draw  the distibution of both class-label/survival, we can than predict that patient will survive or not.
# 3. So, try to do this by histogram and PDF which help us to predict such things.

# In[11]:


#Histogram/PDF for 'Patient_Age'

mp.close()

sb.FacetGrid(hab,hue="Survival",size=5)  .map(sb.distplot,"Patient_Age")  .add_legend()

mp.show()


# **Observation(s) :**
#  
# 1. As, we clearly see by using 'patient_Age' most of the data points overlapped .
# 2. So, by using 'patient_Age' we cannot say anything about survival of patient.

# In[12]:


#Histogram/PDF for 'Year_of_operation'

mp.close()

sb.FacetGrid(hab,hue="Survival",size=5)  .map(sb.distplot,"Year_of_operation")  .add_legend()

mp.show()


# **Observation(s) :**
#  
# 1. As, we clearly see by using 'Year_of_operation' most of the data points overlapped .
# 2. So, by using 'Year_of_operation' we cannot say anything about survival of patient.

# In[13]:


#Histogram/PDF for '#PositiveAxiliaryNode'

mp.close()

sb.FacetGrid(hab,hue="Survival",size=5)  .map(sb.distplot,"#PositiveAxiliaryNode")  .add_legend()

mp.show()


# **Observation(s) :**
# 
# 1. Here, we not see that much overlapping as there was in 'Patient_Age' and 'Year_of_operation' feature , but than to there is overlapping by which here also we will get diffculty in classifing the survival of patient.

# ** >As Histogram/PDF will tell us that what is the probability/Height (how many of our data points)  that lie at certain value of feature (That's why it is called density plot or prbability plot**
# 
# ** >But disadvantage of PDF is that we cannot get how much percentage of our date points are less than the certain value of an feature. This disadvantage can be overcome by CDF.**

# # (2) CDF

# In[14]:


#PDF calculated as counts/frequencies of data points in each window.

#Plot CDF of 'Patient_Age' for survived Patient

counts, bin_edges = np.histogram(hab_survive['Patient_Age'], bins=10, 
                                 density = True)

print(counts)
print(sum(counts))

pdf = counts/(sum(counts))

print('********************')
print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

mp.plot(bin_edges[1:],pdf);
mp.plot(bin_edges[1:], cdf)

mp.show()


# **Observation(s) :**
# 
# 1. Orange curve is for CDF and Blue curve for PDF .
# 2. Here we can get to know suppose how many patient survived have age less than 60 ?
#    we got from CDF curve that about 78 percentage of patients survived having age less than 60!

# In[15]:


#Plot CDF of 'patient_Age' for not-survived Patient

counts, bin_edges = np.histogram(hab_notsurv['Patient_Age'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))

print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

mp.plot(bin_edges[1:],pdf);
mp.plot(bin_edges[1:], cdf)


mp.show()


# **Observation(s) :**
# 
# 1. Orange curve is for CDF and Blue curve for PDF .
# 2. Here we can get to know suppose, how many patient not-survived having age less than 50 ?
#    we got from CDF curve that about 40 percentage of patients not-survived having age less than 50!

# In[16]:


#plot of data points of both class as together


#Plot CDF of 'patient_Age' for survived Patient

counts, bin_edges = np.histogram(hab_survive['Patient_Age'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))

print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

mp.plot(bin_edges[1:],pdf);
mp.plot(bin_edges[1:], cdf)


#Plot CDF of 'patient_Age' for not-survived Patient

counts, bin_edges = np.histogram(hab_notsurv['Patient_Age'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))

print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

mp.plot(bin_edges[1:],pdf);
mp.plot(bin_edges[1:], cdf)


mp.show()


# **Observation(s) :**
# 
# 1. For Survived-patient > Orange curve is for CDF and Blue curve for PDF .
# 2. For Not-Survived-patient > Red curve if for CDF and Green curve for PDF. 
# 2. Here we can get to know suppose how many patient survived and not survived having  age less than 70 ?
#    we got from CDF curve that about 95 percentage of patients will survived having age less than 70 and 94 percentage pf patients will not survived having age less than 70!

# # (3) Mean,Variance and Standard-Deviation

# **Till now we have studied about our observations by plotting graphs like 2-D,3-D,Pair-plot Scatter plots,histogram/PDF,CDF But If we have no facilities of graphs than how we do our observations, is as follows --------** 

# In[17]:


#Means ,calculate the Central Tendency of the observation of a feature

print('Means:')
print(np.mean(hab_survive['Patient_Age']))
print(np.mean(hab_notsurv['Patient_Age']))

#Standard-Deviation will tell us about Spread i.e what are the average square distance of each point from the Mean

print("\nStd-dev:");
print(np.std(hab_survive['Patient_Age']))
print(np.std(hab_notsurv['Patient_Age']))

#But one or small number of Outlier(an error) can corrupt both mean and standard-deviation

print('\nFor mean')
print('For survived Patients :'+str(np.mean(np.append(hab_survive['Patient_Age'],300))))

print('\nStd-dev')
print('For survived Patients :' + str(np.std(np.append(hab_survive['Patient_Age'],200))))


# # Median, Percentile, Quantile, IQR, MAD

# **As Mean and Standard-deviation gets corrupted by one or small number of outliers, so we go for other methods by which these corruption should not occur.**

# In[18]:


#Median is equivalent to Mean, which calculate the middle value of an obervastion

print('\nMedian:')
print('Median of patient age for survived patient:'+str(np.median(hab_survive['Patient_Age'])))
print('Median of patient age for not survived patient:'+str(np.median(hab_notsurv['Patient_Age'])))

#Median with an outlier
print('\nMedian with an outlier for survived patients')
print(np.median(np.append(hab_survive['Patient_Age'],300))) #we can see that there is no corruption by an outlier

print('\nQuantiles')
print('0th,25th,50th,75th percentile value of patient age of survived patient:'+str(np.percentile(hab_survive['Patient_Age'],np.arange(0,100,25))))
print('0th,25th,50th,75th percentile value of patient age of not survived patient:'+str(np.percentile(hab_notsurv['Patient_Age'],np.arange(0,100,25))))

print('\n90th Percentile')
print('90th percentile value of patient age for survived patient:'+str(np.percentile(hab_survive['Patient_Age'],90)))
print('90th percentile value of patient age for not survived patient:'+str(np.percentile(hab_notsurv['Patient_Age'],90)))

#MAD-Median Absolute Deviation is equivalent to standard-deviation, which measure that how faraway our data points from central tendency(here is median)

print('\nMedian Abolute Deviation')
from statsmodels import robust
print('MAD value of patient age for survived patient:'+str(robust.mad(hab_survive['Patient_Age'])))
print('MAD value of patient age for not survived patient:'+str(robust.mad(hab_notsurv['Patient_Age'])))

print('\nMAD with an outlier for survived patient')
print(robust.mad(np.append(hab_survive['Patient_Age'],200))) #we can see that there is no corruption by an outlier


# # (4) Box-Plot

# In[19]:


#Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitivey.

sb.boxplot(x='Survival',y='Patient_Age',data=hab)
mp.show()


# **Observation(s) :**
# 1. As, we know Histogram/PDF fail to give us the value of 25th,50th,75th percentile value.
# 2. So, from Box-Plot we can tell those values as-----
# 
#    For Patient that survived (1)      :=>  [25th,50th,75th] = [44,53,60]
#    
#    For Patient that not Survived (2)  :=>  [25th,50th,75th] = [47,54,61]  
#    
# 3. And, also we cannot observe/say wheater patient will survive or not because of overlapping of data points of both survival-type.   

# # (5) Violin Plot

# In[20]:


#A violin plot combines both Histogram/PDF and box-plot 

sb.violinplot(x='Survival', y='Patient_Age', data=hab,size=8)

mp.show()


# **Observation(s) :**
# 
# 1. In middle black strip is our Box-plot which gives all percentile value that we want , and here also in sideways we got our PDF/distibution , and by seeing it we can say that most of the data points of survival-type are overlapping.
# 2. So, here also we cannot say Patient will survive the surgery or not.

# # Conclusion

# 
# **1. On this Haberman's Survival Dataset we had performed both bivariate and univariate analysis , and  by analysing the dataset we conclude that , for this dataset classification is not possible that means we cannot classify that Patient will either survive who had undergone surgery for breast cancer  or will not survive.**
# 
# **2.The reason we not able to classify this problem set is because for both class-label/Survival-type most of the data points/obervations are overlapped , but some of questions on these dataset can be solved ,like an example, from CDF we got to know that what percentage of patients survived having there age less than 60 or by using Box-plot we have calculated what are 25th,50th,745th percentile value for the obervation,etc.**
# 
# 

# In[ ]:




