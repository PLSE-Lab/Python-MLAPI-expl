#!/usr/bin/env python
# coding: utf-8

# <h2>This notebook will answer some questions about COVID-19 through explanatory data analysis. Feel free to provide feedbacks. </h2>

# Importing the libraries we need

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
plt.style.use('fivethirtyeight')


# Question: Which populations have contracted COVID-19 who require the ICU?
# 
# Let's look at the Einstein dataset. 

# In[ ]:


patient_admission = pd.read_csv('../input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv')
patient_admission.head()


# In[ ]:


patient_admission.describe()


# In[ ]:


patient_admission.sars_cov_2_exam_result.value_counts()


# Getting information about the age of covid patients and their ICU status

# In[ ]:


patient_age = patient_admission[patient_admission.sars_cov_2_exam_result=='positive'].patient_age_quantile
patient_icu = patient_admission[patient_admission.sars_cov_2_exam_result=='positive'].patient_addmited_to_intensive_care_unit_1_yes_0_no


# In[ ]:


patient_icu.value_counts()


# Visualizations 

# In[ ]:


plt.figure(figsize=(10, 5))
patient_icu.hist()
plt.title('ICU Admissions Due to COVID-19')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 5))
plt.xlabel('Age Quantile')
plt.ylabel('Frequency')
patient_age.hist()
plt.title('Age Quantile Distribution')
plt.show()


# Age quantile of 0-2.5 and 17.5+ have a disproportionate high ICU admissions.

# In[ ]:


plt.figure(figsize=(10, 5))
plt.title('ICU Admission by Age Quantile Due to COVID-19')
plt.xlabel('Age Quantile')
plt.ylabel('Number of Admissions')
patient_icu_age = patient_admission[patient_admission.sars_cov_2_exam_result=='positive'][patient_admission.patient_addmited_to_intensive_care_unit_1_yes_0_no=='t'].patient_age_quantile
patient_icu_age.hist()


# Question: Which populations of clinicians and patients require protective equipment?

# In[ ]:


hospital_capacity = pd.read_csv('../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-40-population-contracted.csv')
hospital_capacity


# In[ ]:


max_icu_bed_occupancy_rate = np.max(hospital_capacity.icu_bed_occupancy_rate)
state_highest_icu_rate = hospital_capacity[hospital_capacity.icu_bed_occupancy_rate==max_icu_bed_occupancy_rate]


# State with the highest ICU rate 

# In[ ]:


state_highest_icu_rate


# State with the highest ICU occupancy

# In[ ]:


hospital_capacity['ICU Beds Taken'] = np.round(hospital_capacity.icu_bed_occupancy_rate * hospital_capacity.total_icu_beds)
max_icu_bed_occupancy = np.max(hospital_capacity['ICU Beds Taken'] )
state_highest_icu = hospital_capacity[hospital_capacity['ICU Beds Taken']==max_icu_bed_occupancy]


# In[ ]:


state_highest_icu


# Both states require more protective equipments due to their proportion/total icu bed occupancies. 
