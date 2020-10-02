#!/usr/bin/env python
# coding: utf-8

# # Which populations have contracted COVID-19 who require the ICU?
# 

# Task Details
# The Roche Data Science Coalition is a group of like-minded public and private organizations with a common mission and vision to bring actionable intelligence to patients, frontline healthcare providers, institutions, supply chains, and government. The tasks associated with this dataset were developed and evaluated by global frontline healthcare providers, hospitals, suppliers, and policy makers. They represent key research questions where insights developed by the Kaggle community can be most impactful in the areas of at-risk population evaluation and capacity management.

# In[ ]:


from IPython.display import Image
import os
get_ipython().system('ls ../input/')
Image("../input/images/corona.png")


# # Neural Network

# > ****Artificial neural networks are relatively crude electronic networks of neurons based on the neural structure of the brain. They process records one at a time, and learn by comparing their prediction of the record (largely arbitrary) with the known actual record. The errors from the initial prediction of the first record is fed back to the network and used to modify the network's algorithm for the second iteration. These steps are repeated multiple times.
# > 
# A neuron in an artificial neural network is:
# **1.  A set of input values (xi) with associated weights (wi)
# ****2. A input function (g) that sums the weights and maps the results to an output function(y).**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import seaborn as sb


# In[ ]:


hostipal_beds = pd.read_csv('../input/uploads/hospital-capacity-by-state-40-population-contracted.csv')
hostipal_beds['bed_to_people_ratio'] = (hostipal_beds['total_hospital_beds'] + hostipal_beds['total_icu_beds'])/(hostipal_beds['adult_population'] + hostipal_beds['population_65'])
cases_by_country = pd.read_csv('../input/uploads/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv')
cases_by_state = pd.read_csv('../input/uploads/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')
cases_over_time = pd.read_csv('../input/uploads/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv')


# In[ ]:


cases_over_time = cases_over_time.loc[cases_over_time['country_region'] == 'US']


# In[ ]:


states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}


# In[ ]:



risk = hostipal_beds.sort_values(by=['bed_to_people_ratio'], ascending=False).head(52)
risk['state'] = pd.Series([states[x] for x in risk['state']], index=risk.index)
risk = risk[['state','total_hospital_beds', 'total_icu_beds','adult_population','population_65','bed_to_people_ratio']]
risk.index = risk.state
risk = risk.drop(['state'], axis=1)


# In[ ]:


s = cases_by_state.loc[cases_by_state['country_region'] == 'US']
s = s.rename(columns={"province_state": "state"})
s.index = s.state


# In[ ]:


risk['confirmed'] = pd.Series([x for x in s['confirmed']], index=s.index)
risk['deaths'] = pd.Series([x for x in s['deaths']], index=s.index)


# In[ ]:


risk['infected_ratio'] = risk['confirmed']/(risk['population_65'] + risk['adult_population'])
risk['hypothetical_beds_in_use'] = (risk['total_hospital_beds'] + risk['total_icu_beds']) - risk['confirmed']
risk = risk[['total_hospital_beds', 'total_icu_beds', 'adult_population','population_65', 'confirmed', 'deaths','hypothetical_beds_in_use', 'bed_to_people_ratio', 'infected_ratio']]
risk.head(5)


# In[ ]:


risk['M_1'] = risk['deaths'] / (risk['adult_population'] + risk['population_65'])
risk['M_2'] = risk['deaths'] / risk['confirmed']

fig, ax =plt.subplots(1,2, figsize=(20, 5))
_ = risk.sort_values(by=['M_1'], ascending=False).head(10)
ax[0].set_title('Ratio of deaths to total population')
sb.barplot(x=_['M_1'], y=_.index, palette='Blues_r',  orient='h', ax=ax[0])
_ = risk.sort_values(by=['M_2'], ascending=False).head(10)
ax[1].set_title('Ratio of deaths to confirmed cases')
sb.barplot(x=_['M_2'], y=_.index, palette='Greens_r',  orient='h', ax=ax[1]);


# # By using IBM SPSS Modeler

# IBM SPSS Modeler is a data mining and text analytics software application from IBM. It is used to build predictive models and conduct other analytic tasks. It has a visual interface which allows users to leverage statistical and data mining algorithms without programming.

# In[ ]:


#Summary of Model
Image("../input/images/Task_6_1.png")


# In[ ]:


Image("../input/images/Task_6_2.png")


# In[ ]:


Image("../input/images/Task_6_3.png")


# In[ ]:


Image("../input/images/Task_6_4.png")


# In[ ]:


Image("../input/images/Task_6_5.png")


# In[ ]:


Image("../input/images/Task_6_6.png")


# In[ ]:




