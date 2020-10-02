#!/usr/bin/env python
# coding: utf-8

# # To do
# 1. Analyse data by age, especially recovery and deceased.
# 2. Analy data by state/cities.
# 3. Average days for recovery.
# 4. Average days for death.

# # First look at data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
            print(os.path.join(dirname, filename))


# In[ ]:


file_location = '/kaggle/input/covid19india/covidDatabaseIndia.xls'
file = pd.ExcelFile(file_location)
sheet_names = file.sheet_names


# In[ ]:


raw_data = pd.read_excel(io=file_location, sheet_name = 'Raw_Data', header = 0)
raw_data.head(5)


# # Patient status analysis

# In[ ]:


sns.countplot(raw_data['Current Status'])
total_cases = raw_data['Current Status'].count()
recovered = raw_data['Current Status'][raw_data['Current Status']=='Recovered'].count()
deceased = raw_data['Current Status'][raw_data['Current Status']=='Deceased'].count()
percent_recovered = recovered/total_cases
percent_deceased = deceased/total_cases
plt.title('Breakdown by patient status')
plt.show()
print('Total cases: '+str(total_cases))
print('Percent recovered: ' +str(round(percent_recovered*100,2))+'%')
print('Percent death: ' +str(round(percent_deceased*100,2))+'%')


# The majority of the COVID-19 cases as still active with only 2.8% of the total cases recovered and 0.87% deaths.

# # Age Brackets

# In[ ]:


raw_data[raw_data['Age Bracket'] == '28-35']


# Is this possible duplicate data or the source combined 4 patients in one line, as evidenced by age bracket?

# # Time series analysis

# In[ ]:


raw_data['Date Announced'].value_counts().plot(figsize=(18,6))
plt.title('New patients by date')
plt.show()


# In[ ]:


raw_data['Date Announced'].value_counts().sort_index().cumsum().plot(figsize=(18,6))
plt.title('Cumulative cases by date')
plt.show()


# In[ ]:




