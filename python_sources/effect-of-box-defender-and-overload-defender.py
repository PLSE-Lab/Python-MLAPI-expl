#!/usr/bin/env python
# coding: utf-8

# Here is the  kernel to show how punt formation affect the rate of concussion

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# Load player role and concussion data

# In[ ]:


player_role_data = pd.read_csv('../input/play_player_role_data.csv')


# In[ ]:


player_role_data.head()


# In[ ]:


play_information_data = pd.read_csv('../input/play_information.csv')


# In[ ]:


play_information_data.head()


# In[ ]:


concussion_data = pd.read_csv('../input/video_review.csv')


# In[ ]:


concussion_data.head()


# In[ ]:


len(concussion_data)


# In[ ]:


concussion_data['concussed'] = 1


# Create a pivot table for all punt play data and merge it with concussion data

# In[ ]:


table = pd.pivot_table(player_role_data,index=['GameKey', 'PlayID'],columns=['Role'], aggfunc=lambda x: len(x.unique()))['GSISID'].fillna(0)

table.reset_index(inplace=True)


# In[ ]:


table.head()


# In[ ]:


merged_data = pd.merge(table,play_information_data)
merged_data = pd.merge(merged_data,concussion_data,how='outer')
merged_data.concussed.fillna(0, inplace=True)


# Check the number of concussed player  in the new dataframe

# In[ ]:


len(merged_data[merged_data['Primary_Impact_Type'].notnull()])


# Here we would like to find number of defender in box and if the receiver team is overloading one side. 

# In[ ]:


merged_data['overload'] =  ((merged_data['PDL1'] + merged_data['PDL2'] + merged_data['PDL3'] + merged_data['PDL4'] + merged_data['PDL5'] + merged_data['PDL6']) - (merged_data['PDR1'] + merged_data['PDR2'] + merged_data['PDR3'] + merged_data['PDR4'] + merged_data['PDR5'] + merged_data['PDR6']) + (merged_data['PLL1'] + merged_data['PLL2'] + merged_data['PLL3']) - (merged_data['PLR1'] + merged_data['PLR2'] + merged_data['PLR3'])).abs()


# In[ ]:


merged_data['box_defender'] =  ((merged_data['PDL1'] + merged_data['PDL2'] + merged_data['PDL3'] + merged_data['PDL4'] + merged_data['PDL5'] + merged_data['PDL6']) + (merged_data['PDR1'] + merged_data['PDR2'] + merged_data['PDR3'] + merged_data['PDR4'] + merged_data['PDR5'] + merged_data['PDR6']) + (merged_data['PLL1'] + merged_data['PLL2'] + merged_data['PLL3']) + (merged_data['PLR1'] + merged_data['PLR2'] + merged_data['PLR3']) + 
(merged_data['PLM1'] + merged_data['PLM'] + merged_data['PDM']))


# Also remove punt plays that are blocked or killed by penalties

# In[ ]:


yards_list = []

for i,yards in enumerate(merged_data.PlayDescription.str.split(' yard').str[0].str[-2:]):
    try:
        yards_list.append(float(yards))
    except ValueError:
        yards_list.append('NaN')
merged_data['punt_yards'] = yards_list
merged_data['no_play'] = merged_data.PlayDescription.str.contains('No Play', regex=True)
merged_data['blocked'] = merged_data.PlayDescription.str.contains('BLOCKED', regex=True)


# In[ ]:


merged_data = merged_data[(merged_data.box_defender > 3) & (merged_data.box_defender  <9) & (merged_data.punt_yards != 'NaN') & (merged_data.no_play == False) & (merged_data.blocked == False)]


# In[ ]:


merged_data.head()


# We now load the statsmodels module for logistic regression to determine whether no. of box defender and overload players would affect

# In[ ]:


import statsmodels
import statsmodels.api as sm

import statsmodels.formula.api as smf


results = smf.logit(formula='concussed ~ box_defender + overload', data=merged_data).fit()


# In[ ]:


results.summary()


# From the result we can see that no. of box defender may has some effect on concussion chance, but overloading one side by receiving team seems to not making any difference.
# Finally we plot the 95% Wilson convidence interval for each case

# In[ ]:


zero_overload = merged_data[merged_data['overload'] == 0]
lower_zero,upper_zero = statsmodels.stats.proportion.proportion_confint(len(zero_overload[zero_overload['concussed'] == 1]), len(zero_overload['concussed']), alpha=0.05, method='wilson')
one_overload = merged_data[merged_data['overload'] == 1]
lower_one,upper_one = statsmodels.stats.proportion.proportion_confint(len(one_overload[one_overload['concussed'] == 1]), len(one_overload['concussed']), alpha=0.05, method='wilson')
two_overload = merged_data[merged_data['overload'] == 2]
lower_two,upper_two = statsmodels.stats.proportion.proportion_confint(len(two_overload[two_overload['concussed'] == 1]), len(two_overload['concussed']), alpha=0.05, method='wilson')
three_overload = merged_data[merged_data['overload'] == 3]
lower_three,upper_three = statsmodels.stats.proportion.proportion_confint(len(three_overload[three_overload['concussed'] == 1]), len(three_overload['concussed']), alpha=0.05, method='wilson')


# In[ ]:


x = [0,1,2,3]
y = [np.mean(zero_overload['concussed']),np.mean(one_overload['concussed']),np.mean(two_overload['concussed']),np.mean(three_overload['concussed'])]

yerr = [[y[0] - lower_zero, y[1] - lower_one, y[2] - lower_two, y[3] - lower_three ], [upper_zero - y[0], upper_one - y[1], upper_two - y[2], upper_three - y[3]]]


# In[ ]:


plt.errorbar(x,y,yerr, capsize=3, elinewidth=1)
plt.xlabel('No. of overload defender')
plt.ylabel('Concussion chance')
plt.title('Error of concussion chance vs overload defender')
plt.xticks(np.arange(0, 4, step=1))


# In[ ]:


six_box = merged_data[merged_data['box_defender'] == 6]
lower_six,upper_six = statsmodels.stats.proportion.proportion_confint(len(six_box[six_box['concussed'] == 1]), len(six_box['concussed']), alpha=0.05, method='wilson')
seven_box = merged_data[merged_data['box_defender'] == 7]
lower_seven,upper_seven = statsmodels.stats.proportion.proportion_confint(len(seven_box[seven_box['concussed'] == 1]), len(seven_box['concussed']), alpha=0.05, method='wilson')
eight_box = merged_data[merged_data['box_defender'] == 8]
lower_eight,upper_eight = statsmodels.stats.proportion.proportion_confint(len(eight_box[eight_box['concussed'] == 1]), len(eight_box['concussed']), alpha=0.05, method='wilson')


# In[ ]:


x = [6,7,8]
y = [np.mean(six_box['concussed']),np.mean(seven_box['concussed']),np.mean(eight_box['concussed'])]

yerr = [[y[0] - lower_six, y[1] - lower_seven, y[2] - lower_eight], [upper_six - y[0], upper_seven - y[1], upper_eight - y[2]]]


# In[ ]:


plt.errorbar(x,y,yerr, capsize=3, elinewidth=1)
plt.xlabel('No. of box defender')
plt.ylabel('Concussion chance')
plt.title('Error of concussion chance vs box defender')
plt.xticks(np.arange(6,9, step=1))


# In[ ]:




