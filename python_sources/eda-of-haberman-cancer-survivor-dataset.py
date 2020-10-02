#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Primary Objective:**
# Given the data set, analyze the dependant variables to understand how the features influence the probability of a patient to survive more than 5 years after the cancer operation.

# In[ ]:


#Import the required libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

print('Libraries Imported')
import os
print(os.listdir('../input'))


# In[ ]:


#Import the data set:
haberman = pd.read_csv('../input/haberman.csv')


# In[ ]:


#Describe the overall data set:

haberman.info()


# In[ ]:


haberman.columns


# In[ ]:


haberman.columns = ['PatientAge','OperationYear','PositiveAxillaryNodes','SurvivalStatus']


# In[ ]:


haberman.describe().transpose()


# **Observation:**
# 
# 1) The data set has 305 data points of the cancer patients. 
# 
# 2) For each patients there are 3 features that possibly influences the patients probability to survive more than 5 years.
# 
# 3) These features has been renamed as 'PatientAge','OperationYear','PositiveAxillaryNodes','SurvivalStatus'
# 
# 4) From the above summary table we can conclude the following:
#     a) Of all the patient the mean age is 52 yrs with minimum age being 30 and maximum age being 83
#     b) The patient data was captured between the year 1958 to 1996
#     c) Of all the patient the mean Positive Axillary Nodes is 4 but the median is 1 with standard deviation of 7. This                could indicate that there are many outliers for this data point.

# 

# In[ ]:


#Lets segragate the data between patients who survived more than 5 years and who could not.

surviveMoreThan5Years = haberman[haberman["SurvivalStatus"] == 1]
surviveLessThan5Years = haberman[haberman["SurvivalStatus"] == 2]


# In[ ]:


#Describe "surviveMoreThan5Years":

surviveMoreThan5Years.describe().transpose()


# In[ ]:


np.percentile(surviveMoreThan5Years["PositiveAxillaryNodes"],90)


# In[ ]:


#Describe "surviveLessThan5Years":

surviveLessThan5Years.describe().transpose()


# In[ ]:


np.percentile(surviveLessThan5Years["PositiveAxillaryNodes"],90)


# **Observation:**
# 
# 1) There are 224 patients who, after the cancer operation, survived more than 5 years and there are 81 patients who survived less than 5 years.
# 
# 2) Overall, in 73% of operations, the patient survived more than 5 years.
# 
# 3) Between the 2 data sets, the PatientAge and the OperationYear does not seem to be an influencer of the survival probability.
# 
# 4) However, Positive Axillary Nodes does seems as an influencer of the survival probability. Lower the value indicates better chance at survival.
# 
# 5) Of all the patients who survived more than 5 years, the feature Positive Axillary Nodes has a mean of 2.7 with median as 0. This means that 50% of survivors did not have any Positive Axillary Nodes. 75% of survivors had Positive Axillary Nodes as 3 or less than 3. 90% of the survivors had Positive Axillary Nodes as less than equal to 8.
# 
# 6) Of all the patients who survived less than 5 years, the feature Positive Axillary Nodes has a mean of 7.4 with median as 4. This means that 50% of patients had 4 Positive Axillary Nodes. Between 75% of patients had Positive Axillary Nodes had Positive Axillary Nodes as 11 or less than 11. 90% of the patients had Positive Axillary Nodes as less than equal to 20.

# In[ ]:


#Lets run a pair plot to see if there are any combination of features that can standout as an influencer of the survival probabilty.

sns.pairplot(haberman,hue="SurvivalStatus")


# **Observation:**
# From all the above charts, it is not clear if there is a combination of features that affects the survival probability the most. 

# In[ ]:


#Since the feature Positive Axillary Nodes seems to be an influencer of the survival probability, lets segragate whole data into 3 parts. 
#Patients in the 1st data set will have Positive Axillary Nodes upto 3.
#Patients in the 2nd data set will have Positive Axillary Nodes more than 3 but less than 8.
#Patients in the 3rd data set will have Positive Axillary Nodes more than 8.

FirstDS = haberman[haberman["PositiveAxillaryNodes"] <= 3]
SecondDS = haberman[(haberman["PositiveAxillaryNodes"] > 3) & (haberman["PositiveAxillaryNodes"] <= 8)]
ThirdDS = haberman[haberman["PositiveAxillaryNodes"] > 8]


# In[ ]:


#From the newly created 3 data sets, lets find out the probability of the patient surviving more than 5 years. 
#The probabilty will be calculated using the CDF and PDF charts

plt.figure(figsize=(20,10))

count,bin_edges = np.histogram(FirstDS["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

count,bin_edges = np.histogram(SecondDS["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

count,bin_edges = np.histogram(ThirdDS["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


# **Observation:**
# 
# The above chart gives a clear indication on the probability of the patient surviving more than 5 years:
# 
#     a) If the patient has PositiveAxillaryNodes upto 3, then the survival probabilty is about 82%
#     
#     b) If the patient has PositiveAxillaryNodes between 3 and 8, then the survival probabilty is about 63%
#     
#     c) If the patient has PositiveAxillaryNodes more than 8, then the survival probabilty is about 42%

# In[ ]:


#We have seen, from the overall data, that Patient Age does not affect the survival probability. 
#However, lets check if Patient Age does have an impact when all the patients are segragated into the 3 data set created above.
#For each newly created data set, we classify the patient to be either more than 50 years of age or not.

FirstDS_AgeLessThan50 = FirstDS[(FirstDS["PatientAge"] <= 50)]
FirstDS_AgeMoreThan50 = FirstDS[(FirstDS["PatientAge"] > 50)]

SecondDS_AgeLessThan50 = SecondDS[(SecondDS["PatientAge"] <= 50)]
SecondDS_AgeMoreThan50 = SecondDS[(SecondDS["PatientAge"] > 50)]

ThirdDS_AgeLessThan50 = ThirdDS[(ThirdDS["PatientAge"] <= 50)]
ThirdDS_AgeMoreThan50 = ThirdDS[(ThirdDS["PatientAge"] > 50)]


# In[ ]:


#For the 1st data set, when patients has less than 3 Positive Axillary Nodes, does age plays a factor?
#The probabilty will be calculated using the CDF and PDF charts

plt.figure(figsize=(20,10))
count,bin_edges = np.histogram(FirstDS_AgeLessThan50["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

count,bin_edges = np.histogram(FirstDS_AgeMoreThan50["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


# **Observation:**
# The above chart shows that Patient Age does not influence the survival probability if the patient has upto 3 Positive Axillary Nodes

# In[ ]:


#For the 2nd data set, when patients has Positive Axillary Nodes between 3 and 8, does age plays a factor?
#The probabilty will be calculated using the CDF and PDF charts

plt.figure(figsize=(20,10))
count,bin_edges = np.histogram(SecondDS_AgeLessThan50["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

count,bin_edges = np.histogram(SecondDS_AgeMoreThan50["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


# **Observation:**
# The above chart shows that the Patient Age is a crucial factor when patients has PositiveAxillaryNodes between 3 and 8.
# 
#     a) When the patient age is less than 50 then the survival probability is about 82%
#     
#     b) when the patient age is more than 50 then the survival probability is about 48%

# In[ ]:


#For the 3rd data set, when patients has Positive Axillary Nodes more than 8, does age plays a factor?
#The probabilty will be calculated using the CDF and PDF charts

plt.figure(figsize=(20,10))
count,bin_edges = np.histogram(ThirdDS_AgeLessThan50["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

count,bin_edges = np.histogram(ThirdDS_AgeMoreThan50["SurvivalStatus"] == 1,bins=10,density=True)
pdf = count/sum(count)
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


# **Observation:**
# 
# The above chart shows that the Patient Age has smaller impact when patients has PositiveAxillaryNodes more than 8. a) When the patient age is less than 50 then the survival probability is about 45% b) when the patient age is more than 50 then the survival probability is about 38%

# **Final Conclusion:**
# 
# 1. With all the above analysis, we can conclude that the feature "Positive Axillary Nodes" plays a crucial role in determining the survival probability of the patient. I,e will the patient survive for more than 5 years or not. We observe that the survival probability drops from 82% to 42% when the number of "Positive Axillary Nodes" increases from 3 to more than 8.
# 
# 2. Patient Age on the other hand, is a big factor that impacts the survival probability but only when the patient has "Positive Axillary Nodes" between 3 and 8. In this group of patients, the survival probability drops from 82% to 48% when the patient age increases to more than 50.

# In[ ]:




