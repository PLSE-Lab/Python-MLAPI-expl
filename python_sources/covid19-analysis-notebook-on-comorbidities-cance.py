#!/usr/bin/env python
# coding: utf-8

# <h1> COVID-19 Analysis Notebook on Comorbidities (CANCe) </h1>
# 
# <h2> What is the incident of COVID-19 Infections in Cancer Patients? </h2>

# <H3> Task Details </H3>
# 
# The Roche Data Science Coalition is a group of like-minded public and private organizations with a common mission and vision to bring actionable intelligence to patients, frontline healthcare providers, institutions, supply chains, and government. The tasks associated with this dataset were developed and evaluated by global frontline healthcare providers, hospitals, suppliers, and policy makers. They represent key research questions where insights developed by the Kaggle community can be most impactful in the areas of at-risk population evaluation and capacity management. - COVID19 Uncover Challenge
# 
# <h3> Notebook Detail Study </h3>
# 
# Not much detail on COVID-19 Specific association with certain diseases like cancer is available publically. Datasets regarding this particular condition isn't available much as of now, and majority of the literary references found point to study of only a certain restticted number of population groups, where the sample size is very less. This notebook wrangles all available datasets and visualizations that are built over the dataset under a single house, to study the effect of COVID-19 on cancer patients.
# 
# This notebook would be updated by me as I find newer analyses into this subject. Special thanks to Samar Mahmoud for providing relevant details.
# 
# 

# # <a id='main'><h3>Table of Contents</h3></a>
# - [Importing the Essential Libraries](#lib)
# - [Datasets used in notebook](#data)
# - [Getting information with the Clinical Data Available](#conc)
# 

# # <a id='lib'><h3>Importing the essential libraries</h3></a>

# In[ ]:


#Data Analyses Libraries
import pandas as pd                
import numpy as np    
from urllib.request import urlopen
import json
import glob
import os

#Importing Data plotting libraries
import matplotlib.pyplot as plt     
import plotly.express as px       
import plotly.offline as py       
import seaborn as sns             
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
import matplotlib.animation as animation

#Other Miscallaneous Libraries
import warnings
warnings.filterwarnings('ignore')
from IPython.display import HTML
import matplotlib.colors as mc
import colorsys
from random import randint
import re


# # <a id='data'><h3>Datasets used for analyses in this notebook</h3></a>
# 
# The various datasets that we take under consideration for this particular notebook are mentioned underneath:
# 
# 1. UNCOVER Dataset uploaded under the UNCOVER Covid-19 Challenge. Specifically, the clinical trials data available under einstein organization.
# 
# # <a id='conc'><h3>Understanding the Clinical Trails data of BRAZIL</h3></a>

# In[ ]:


#Importing the clinical spectrum data
clinical_spectrum = pd.read_csv('../input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv')

#Filtering the data to contain the values only for the confirmed COVID-19 Tests
confirmed = clinical_spectrum['sars_cov_2_exam_result'] == 'positive'
clinical_spectrum = clinical_spectrum[confirmed]

#Viewing the dataset statistics
clinical_spectrum.head()


# The dataset contains information on the people admission to hospitals and the severity. The most severe ones were admitted to intensive care units. The later columns holds information about other parameters that were tested for in the patients.
# 
# For the simplicity we filter the data for the patients that were suffering from COVID-19 and patients which weren't. We then analyze both the datasets against the available values to figure out difference if any.
# 
# <h3> Filtering the dataset for COVID-19 Positive and Negative Patients </h3>

# In[ ]:


#Filetering the datasets
positive_condition = clinical_spectrum['sars_cov_2_exam_result'] == 'positive'
positive_condition_spectra = clinical_spectrum[positive_condition]


negative_condition = clinical_spectrum['sars_cov_2_exam_result'] == 'negative'
negative_condition_spectra = clinical_spectrum[negative_condition]

#Taking mean value of the spectra conditions
positive_mean = clinical_spectrum.mean(axis = 0, skipna = True) 
negative_mean = clinical_spectrum.mean(axis = 0, skipna = True) 


# <h3> Blending the datasets together to analyze positive and negative figures </h3>

# In[ ]:


#Making columns for the dataset
positive_mean = positive_mean.to_frame()
positive_mean = positive_mean.reset_index()
positive_mean.columns = ['Parameter','Positive_figures']

negative_mean = negative_mean.to_frame()
negative_mean = negative_mean.reset_index()
negative_mean.columns = ['Parameter','Negative_figures']

#Merging both the dataframes together
positive_mean['Negative_figures'] = negative_mean['Negative_figures']

#Viewing the dataset
positive_mean.dropna()
positive_mean.head()


# <h3> Which are the most definitive clinical parameters that seperates poisitive and negative cases? </h3>
# 
# Giving the extreme changes in clinical factors between confirmed and negative COVID-19 Cases
# 
# 

# In[ ]:


#The most important clinical factors
positive_mean['Change'] =  positive_mean['Positive_figures'] - positive_mean['Negative_figures']
positive_mean.sort_values(['Change'], axis=0, ascending=True, inplace=True) 

#Getting to know the health factors that define HCP Requirement for a patient
lower = positive_mean.head(15)
higher = positive_mean.tail(15)

#Printing the values
for i in lower['Parameter']:
    print('For lower value of {}, the patient is Prone to COVID-19'.format(i))
    
for i in higher['Parameter']:
    print('For higher value of {}, the patient is Prone to COVID-19'.format(i))


# <iframe src='https://flo.uri.sh/visualisation/2260261/embed' frameborder='0' scrolling='no' style='width:100%;height:600px;'></iframe><div style='width:100%!;margin-top:4px!important;text-align:right!important;'><a class='flourish-credit' href='https://public.flourish.studio/visualisation/2260261/?utm_source=embed&utm_campaign=visualisation/2260261' target='_top' style='text-decoration:none!important'><img alt='Made with Flourish' src='https://public.flourish.studio/resources/made_with_flourish.svg' style='width:105px!important;height:16px!important;border:none!important;margin:0!important;'> </a></div>

# Further research needs to be incorporated on does anyone of this figures leads to Cancer or these clincal information provides predictable results for cancer. As per multiple reports the figures of deviation of the above data is prominent in lung cancer. i.e. Similar symptoms are observed in Lung Cancer as seen in Covid-19 cases.
# 
# # Getting to know the details
# 
# <H3> 1. The count of blood components </H3>
# 
# Patients tested positive for COVID-19 show significantly lower levels of platelets, leukocytes and eosinophils. We can also observe lower levels of calcium and magnesium, but significantly higher concentrations of ferritin, in comparison to patients tested negative for COVID-19.
# 
# The decrease in white blood cells (leukocytes) can either be a direct result of the COVID-19 infection or other pre-existing conditions. This decrease lowers patient immunity against COVID-19 and other infections in general. Additionally, electrolytes and minerals such as potassium, magnesium and zinc play a critical role in maintaining vitality, repairing cell damage, as well as immunity.
# 
# <H3> 2. The Associations </H3>
# 
# By extracting clinical data from the literature, we can examine the abnormalities in blood tests in patients diagnosed with cancer and patients testing positive for COVID-19. Blood test results for multiple minerals were extracted for patients with lung cancer and compared with levels seen in COVID-19 patients. [1,2]
# 
# Because the clinical data available for patients tested for COVID-19 is normalised, measurements retrieved for cancer patients were also normalised using blood concentration in control groups as the mean. The published data has reported the mean and standard deviation of the measurements instead of idividual readings, we use these metrics to simulate mineral concentrations in cancer patients assuming a normal distribution. The measurements retrieved from studies were for Magnesium, Ferritin and Zinc in both cancer patients as well as control groups.
# 
# <img src="https://i.ibb.co/3WgqNGV/2.png" alt="2" border="0">

# <H3> Analysis of Electrolyte Abnormalities </h3>
# 
# 
# <img src="https://i.ibb.co/gTjCb0X/4.png" alt="4" border="0">
# 
# Patients diagnosed with lung cancer show a wide distribution of ferritin levels, however, it peaks around +0.5 standard deviation unites which is similar to the peak of ferritin levels of patients tested positive for COVID-19. These levels are higher than what is observed for patients tested negative in the COVID-19 test (mean equivalent to -0.5 standard deviation units).
# 
# We can conclude from these observations that patients who retracted COVID-19 have similar electrolyte abnormalities as patients with lung cancer.

# # Verifying the Results : Cancer and COVID-19

# <h3> Breakdown of COVID-19 Death Cases in South Korea </h3>
# 
# The following is the dataset that shows the deaths in South Korea by diseases that the patients had suffered. Almost for the **13.3% cases** of fatalities in South Korea, cancer played a role. We hunt down more for similar graphs available online to study this in much greater detail.

# <img src="https://www.statista.com/graphic/1/1102796/south-korea-covid-19-deaths-by-chronic-disease.jpg" alt="Statistic: Breakdown of coronavirus (COVID-19) deaths in South Korea as of March 16, 2020, by chronic disease | Statista" style="width: 100%; height: auto !important; max-width:1000px;-ms-interpolation-mode: bicubic;"/></a>
# 
# 

# <h3> Comorbidites of deaths in China because of COVID-19 </H3>
# 
# For the case of China amongst the total deaths in COVID-19, Cancer was found as a comorbidity in **7.6% of the cases. **

# <img src="https://www.statista.com/graphic/1/1108836/china-coronavirus-covid-19-fatality-rate-by-health-condition.jpg" alt="Statistic: Crude fatality rate of novel coronavirus COVID-19 in China as of February 20, 2020, by health condition | Statista" style="width: 100%; height: auto !important; max-width:1000px;-ms-interpolation-mode: bicubic;"/></a>

# <h3> Breakdown of COVID-19 Death Cases in Italy </h3>
# 
# For the case of Italy **16.2%** of the total deaths due to COVID-19 were related somewhere to Cancer cases. A noticable factor is that it is slightly less than COPD (Chronic Obstructive Pulmonary Disorder) despite pneumonia which is the most common and deadly symptom in COVID-19

# <img src="https://www.statista.com/graphic/1/1110949/common-comorbidities-in-covid-19-deceased-patients-in-italy.jpg" alt="Statistic: Most common comorbidities observed in coronavirus (COVID-19) deceased patients in Italy as of April 16, 2020 | Statista" style="width: 100%; height: auto !important; max-width:1000px;-ms-interpolation-mode: bicubic;"/></a>

# <h3> The next big steps </h3>
# 
# We analyzed and found out for roughly 17% of the deaths in COVID-19 Cancer plays a role. We found out certain medical parameters that can contribute to confirmed cases of COVID-19. Research needs to be implemented to find out if any of the factors mentioned above leads to Cancer or is a clinical figure if a person is suffering from cancer. I'll love to further investigate into this and use this notebook as a base to deploy other notebooks on COVID-19 Concern.
# 
# This notebook would be regular updated by me to check for much newer and diverse data to analyze more trends in spread of COVID-19 and understand it through the terms of spread in Cancer Patients. I would love to further test on more datasets across countries. Would update the notebooks with the new findings.
# Contact LinkedIn - https://www.linkedin.com/in/amankumar01/
# 
# Do upvote and comment if you like or wish to suggest something.

# In[ ]:




