#!/usr/bin/env python
# coding: utf-8

# # Losing Your Lunch in La La Land?
# ### Exploring LA County Restaurant Inspections and Violations
# 
# <table class="image">
# <tr>
# <td>
# <img src="https://i.imgur.com/GPpwGRc.jpg" />
# <br />
# Image Credit: <a href="https://www.flickr.com/photos/96227967@N05/15165534560/in/photolist-p78k4N-F4B1VL-YTYcrh-XeA5tU-XXabwi-SfcFqc-XT44os-BQTSeP-e2YaV7-26aUXfP-7HZY8s-ZBaaYE-XGavvs-8JvRsG-8e5Qh6-8ioaxT-ETNFfE-22sGC7c-F6QDVQ-Ebo87u-iwTv5G-212v9E9-DfzWbi-22gNqyA-22B8irh-XVr6CM-DmXTqE-YTWrYg-NPaf79-224851M-Y6ih7d-2249CCP-GgcvSc-23tWxMc-BuKQuV-9gjDAQ-s7cvWU-G9r1U4-9eVxUz-VQF7yh-7fwHVt-VGpkY8-eUtkSg-aPzNxx-nTTCDt-Fd6sqP-qjdyFC-4HsYSk-bma22w-Hpt7Km">Stephan Harris</a>    
# </td>
# </tr>
# </table>

# I really need to spend more time on my EDA skills. So I'm going to do that using this data set.
# 
# I'll be adding quite a bit to this over time. Check back, share ideas, offer suggestions, etc. Thank you for coming!
# 
# 1. [__The Data__](#the-data)<br>
# > [__In the Raw__](#in-the-raw) &mdash; No frills<br>
# > [__inspections.csv__](#inspections)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[activity_date](#activity-date)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[employee_id](#employee-id)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[facility_name](#facility-name)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[facility_address](#facility-address)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[facility_city](#facility-city)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[facility_state](#facility-state)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[facility_zip](#facility-zip)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[owner_id](#owner-id)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[owner_name](#owner-name)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[pe_description](#pe-description)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[program_element_pe](#program-element-pe)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[program_name](#program-name)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[program_status](#program-status)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[record_id](#record-id)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[grade](#grade)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[score](#score)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[serial_number](#serial-number)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[service_code](#service-code)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[service_description](#service-description)<br>
# > [__violations.csv__](#violations)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[serial_number](#vserial-number)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[points](#points)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[violation_code](#violation-code)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[violation_description](#violation-description)<br>
# > &nbsp;&nbsp;&nbsp;&nbsp;[violation_status](#violation-status)<br>
# > [__Make it Your Own__](#make-it-your-own) &mdash; Prepare the data for analysis ...SOON<br>
# > [__Kick it Up a Notch__](#kick-it-up-a-notch) &mdash; Add more data ...SOON<br>

# In[100]:


#import warnings
#warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import re
import gc

path = '../input/'

inspections = pd.read_csv(path+'inspections.csv')
violations = pd.read_csv(path+'violations.csv')

x_size = 10
y_size = 6


# ## Section 1: The Data<a id="the-data"></a>

# ## In the Raw<a id="in-the-raw"></a>

# ## inspections.csv<a id="inspections.csv"></a>

# In[101]:


inspections.head(3)


# In[102]:


inspections.dtypes


# __inspections.csv: activity_date__<a id="activity-date"></a>

# In[103]:


cntIns = inspections.groupby(['activity_date']).size().reset_index(name='count')

x = pd.DataFrame(pd.to_datetime(cntIns['activity_date']).dt.date)
y = pd.DataFrame(cntIns['count'])

timePlot = pd.concat([x,y], axis=1)

cntObs = timePlot['count'].sum() # count of observations
cntDays = y.shape[0] # count of days

minDate = timePlot['activity_date'].min() # date of first observation
maxDate = timePlot['activity_date'].max() # datet of last observation

dateRange = re.split('\,', str(maxDate - minDate))
dateRange = dateRange[0]

print("\n\nThe data includes", "{:,}".format(cntObs), "inspections conducted on", cntDays, "days. The date range \nspans", dateRange, "from", minDate, "to", maxDate, ".\n")

fig = plt.figure(figsize=(x_size,y_size))
ax = fig.add_subplot(111)
ax.set(xlabel='Date', ylabel='# Inspections')
ax.plot_date(x=timePlot['activity_date'], y=timePlot['count'], marker='o')

del x, y, timePlot, fig, ax
gc.collect()


# __inspections.csv: employee_id__<a id="employee-id"></a>

# In[104]:


print(inspections['employee_id'].describe())

print('\nThere are', inspections['employee_id'].describe()[1], 'inspectors.\n')

# Any entries that don't conform to the EE01234567 format?
badids = 0

for e in inspections['employee_id']:
    if not re.match('^EE\d{7}$', e):
        badids += 1
        
print('Number of Employee IDs that don\'t match the EE0123456 ID format: ', badids)


# In[105]:


# Everything matches. Just drop the "EE" part to make feature numeric

inspections['employee_id'] = inspections['employee_id'].apply(lambda x: x.split('EE', 1)[1])
print(inspections['employee_id'].head(5),'\n')


# __inspections.csv: facility_id__<a id="facility-id"></a>

# In[106]:


print(inspections['facility_id'].describe())

print('\nThere are', "{:,}".format(inspections['facility_id'].describe()[1]), 'facilities.\n')

# Any entries that don't conform to the EE01234567 format?
badfa = 0

for f in inspections['facility_id']:
    if not re.match('^FA\d{7}$', f):
        badfa += 1
        
print('Number of Facility IDs that don\'t match the FA0123456 ID format: ', badfa)


# In[107]:


# Everything matches. Just drop the "FA" part to make feature numeric

inspections['facility_id'] = inspections['facility_id'].apply(lambda x: x.split('FA', 1)[1])
print(inspections['facility_id'].head(5),'\n')


# __inspections.csv: facility_address__<a id="facility-address"></a>

# In[108]:


print(inspections['facility_address'].describe())

print('\nThere are', "{:,}".format(inspections['facility_address'].describe()[1]), 'locations.\n')


# __inspections.csv: facility_city__<a id="facility-city"></a>

# In[109]:


print(inspections['facility_city'].describe())

print('\nThere are', "{:,}".format(inspections['facility_city'].describe()[1]), 'cities in the data. '
      'According to Wikipedia, there are 88 cities in LA county. There is a similar '
      'number of CDPs (Census-Designated Places), so we\'re in the right range.'
     )

print('\nThe cities are:\n\n',sorted(inspections['facility_city'].unique()))


# Immediate suspects for misspelling or bad data entry:
# 
#     Kern (CAPS)
#     NORTHRISGE (should be NORTHRIDGE)
#     Rowland Heights (CAPS)
#     WINNEKA (should be WINNETKA)

# In[110]:


# Fix bad data
inspections['facility_city'] = inspections['facility_city'].str.replace('Kern', 'KERN')
inspections['facility_city'] = inspections['facility_city'].str.replace('NORTHRISGE', 'NORTHRIDGE')
inspections['facility_city'] = inspections['facility_city'].str.replace('Rowland Heights', 'ROWLAND HEIGHTS')
inspections['facility_city'] = inspections['facility_city'].str.replace('WINNEKA', 'WINNETKA')


# Take a quick peek at the frequencies for this feature.

# In[111]:


pd.crosstab(index=inspections['facility_city'], columns='count').head(10)


# Unfortunately, many cities have few observations. We'll try to compensate with other location features.

# __inspections.csv: facility_state__<a id="facility-state"></a>

# In[112]:


print(inspections['facility_state'].describe())

print('\nThere are', "{:,}".format(inspections['facility_state'].describe()[1]), 'states. That doesn\'t seem right.\n')

print('Uniqe States:\n',inspections['facility_state'].unique())


# In[113]:


pd.crosstab(index=inspections['facility_state'], columns='count')


# Basically, everything is in CA (as you would expect). Since this is a zero-variance column, we will just drop it.

# In[114]:


inspections = inspections.drop(['facility_state'], axis=1)


# __inspections.csv: facility_zip__<a id="facility-zip"></a>

# In[115]:


print(inspections['facility_zip'].describe())

print('\nThere are', "{:,}".format(inspections['facility_zip'].describe()[1]), 'zip codes. Are they formatted correctly?')


# In[116]:


inspections['facility_zip'].head(10)


# It looks like some entries use the "Zip+4" format. Let's make sure this feature is standardized.

# In[117]:


# Get rid of +4 in zip
inspections['facility_zip'] = inspections['facility_zip'].apply(lambda x: x.split('-', 1)[0] if x.find('-') > -1 else x)

inspections['facility_zip'].head(10)

# Any entries that don't conform to zip format?
badzips = 0

for z in inspections['facility_zip']:
    if not re.match('\d{5}$', z):
        badzips += 1
        
print('\nNumber of zips that don\'t match 5 digit format: ', badzips)


# __inspections.csv: owner_id__<a id="owner-id"></a>

# In[118]:


print(inspections['owner_id'].describe())

print('\nThere are', "{:,}".format(inspections['owner_id'].describe()[1]), 'owners.\n')

# Any entries that don't conform to the EE01234567 format?
badow = 0

for o in inspections['owner_id']:
    if not re.match('^OW\d{7}$', o):
        badow += 1
        
print('Number of Owner IDs that don\'t match the OW0123456 ID format: ', badow)


# In[119]:


# Everything matches. Just drop the "OW" part to make feature numeric

inspections['owner_id'] = inspections['owner_id'].apply(lambda x: x.split('OW', 1)[1])
print(inspections['owner_id'].head(5),'\n')


# __inspections.csv: owner_name__<a id="owner-name"></a>

# In[120]:


print(inspections['owner_name'].describe(), '\n')

print(inspections['owner_name'].head(10))


# We have LLCs and corporation names mixed in with individuals' names (presumably sole proprieters). __[More on Californina business entity types.](http://www.sos.ca.gov/business-programs/business-entities/starting-business/types/)__
# 
# This is tough. We could label or one hot encode this feature but there are a lot of unique values. There is probably valuable information here. We'll get creative in the next section.

# __inspections.csv: pe_description__<a id="pe-description"></a>
# 
# Businesses seem to be categorized into food markets and restaurants, then by the amount of square footage or number of seats respectively. What does the __risk__ designation mean?

# In[121]:


print(inspections['pe_description'].describe())


# In[122]:


sorted(inspections['pe_description'].unique())


# It seems like the majority of inspections occur at "high risk" businesses. Or, maybe businesses with high risk observations get inspected more frequently than those that don't. We'll find out later.

# In[123]:


pd.crosstab(index=inspections['pe_description'], columns='count')


# __inspections.csv: program_element_pe__<a id="program-element-pe"></a>
# 
# This is an ID corresponding to the pe_description above.

# In[124]:


inspections['program_element_pe'].unique().shape


# There are 18 unique IDs, just like the 18 unique entries in pe_description.

# __inspections.csv: program_name__<a id="program-name"></a>
# 
# This is the name of the business.

# In[125]:


print(inspections['program_name'].describe())

print('\nThere are', "{:,}".format(inspections['program_name'].describe()[1]), 'business names. '
      'A single name, like SUBWAY, may '
      '\nrepresent numerous locations and owners.')


# __inspections.csv: program_status__<a id="program-status"></a>
# 
# Program status seems to be synonymous with "Open" and "Closed."

# In[126]:


print('These are the unique values for program_status:\n',inspections['program_status'].unique())

print('\nThe top value, ACTIVE, accounts \nfor',"{:.2f}".format(inspections['program_status'].describe()[3]/inspections['program_status'].describe()[0]),
      '% of the observations.'
     )


# __inspections.csv: record_id__<a id="record-id"></a>
# 
# Unique ID for each program at a facility.

# In[127]:


print(inspections['record_id'].describe())


# __inspections.csv: serial_number__<a id="serial-number"></a>
# 
# This is the unique ID for each observation (inspection). Use it to join violations data.

# In[128]:


print(inspections['serial_number'].describe())


# __inspections.csv: service_code & service_description__<a id="service-code"></a><a id="service-description"></a>
# 
# service_code is an ID for service_description. There are only two types of service.
# 
# Guess what? Almost none of the inspections were requested by the owner!

# In[129]:


print(inspections['service_code'].unique())

print(inspections['service_description'].unique())

pd.crosstab(index=inspections['service_description'], columns='count')


# __inspections.csv: grade__<a id="grade"></a>
# 
# Grade would make a good categorical target variable for classification problems but it's less granular than __score__.

# In[130]:


print(inspections['grade'].describe())

print('\nThere are',inspections['grade'].describe()[1],'values for grade. '
      'The top value, A, accounts \nfor',"{:.2f}".format(inspections['grade'].describe()[3]/inspections['grade'].describe()[0]),
      '% of the values. That\'s a big class imbalance.'
     )


# __inspections.csv: score__<a id="score"></a>
# 
# We have a bit of a left skew to this, our likely target variable. Depending on how we analyze things later on, we may want to take the natural log of __score__ in order to obtain a normal distribution.

# In[131]:


print(inspections['score'].describe())


# In[132]:


inspections.hist('score')


# ## violations.csv<a id="violations"></a>

# In[133]:


violations.dtypes


# In[134]:


violations.head(3)


# __violations.csv: serial_number__<a id="vserial-number"></a>
#     
# Join the violations to the inspections using this value.

# __violations.csv: points__<a id="points"></a>
# 
# The points from each inspection are tallied and subtracted from 100 to give the inspection score. A score from 90 - 100 (10 points deducted) gives a letter grade of __A__, and so on.
# 
# The vast majority of infractions are for a single point deduction. So, it isn't a major violation that will get you in trouble; it's an accumulation of little problems.

# In[135]:


print(violations['points'].describe())


# In[136]:


violations.hist('points')


# Looks like there's an 11 in there. Either someone is serving __Soylent Green__ or we have an error in data entry... Right? Let's look at a frequency table.

# In[137]:


pd.crosstab(index=violations['points'], columns='count')


# Hmmm. That's quite a few 11s. Let's take a closer look by pulling the first record with a value of 11 and comparing it to other records with the same __violation_code__.

# In[138]:


violations[(violations['points'] == 11).idxmax():].head(1)


# In[139]:


violations[(violations['violation_code'] == 'F023').idxmax():].head(1)


# Sadly, rodents and the like are only a 2 point deduction. Feeling hungry?
# 
# Still, we need to figure out what's going on with the 11s. Looking at a few of them,  we see that several 11 point deductions coincide with __violation_code__ F023.

# In[140]:


violations[(violations['points'] == 11)].head(5)


# Ah ha! It turns out that there are situations in which a __[4 + 7 point deduction](http://publichealth.lacounty.gov/eh/docs/ffipFinal.pdf)__ is levied on a restaurant that has been previously closed due to severe violations.
# 
# Good thing we did our homework on this one!

# __violations.csv: violation_code__<a id="violation-code"></a>

# In[141]:


print(violations['violation_code'].describe())

print('\nThere are',violations['violation_code'].describe()[1],'values for '
      'violation code.')


# In[142]:


pd.crosstab(index=violations['violation_code'], columns='count').head(3)


# In[143]:


pd.crosstab(index=violations['violation_code'], columns='count').tail(3)


# This is really a categorical feature. Frequency may be an interesting value for feature engineering.

# __violations.csv: violation_description__<a id="violation-description"></a>
# 
# This is the only true text feature in the dataset.

# In[144]:


violations['violation_description'].head(5)


# __violations.csv: violation_status__<a id="violation-status"></a>

# In[145]:


print(violations['violation_status'].describe())


# There is only one entry with a value other than OUT OF COMPLIANCE. We'll drop this feature.

# In[146]:


violations = violations.drop(['violation_status'], axis=1)


# ### Make It Your Own<a id="make-it-your-own"></a>
# This section will focus on pre-processing and feature engineering.

# In[ ]:




