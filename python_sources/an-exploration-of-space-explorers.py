#!/usr/bin/env python
# coding: utf-8

# #Overview
# An EDA of NASA's astronaut profiles is provided in this notebook with a particular exploration regarding:
# 
# * Summaries of experiences and backgrounds. 
#        * Evidence of targeted schools? 
#        * Distribution of majors? 
#        * Popular branches of the Armed Services?

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from __future__ import division
sns.set_context('notebook', font_scale=1.25)


# In[ ]:


astronauts_df = pd.read_csv('../input/astronauts.csv',
                            index_col=0,
                            skipinitialspace=True,
                            dtype={'Year': object,
                                   'Group': object},
                            parse_dates=['Birth Date', 'Death Date'],
                            na_values='')
astronauts_df.info()


# In[ ]:


undergrad_majors = astronauts_df['Undergraduate Major'].str.split(r';\s')
first_major =pd.Series(index=undergrad_majors.index.values,
                       data=[x[0] if type(x) != float else np.nan for x in undergrad_majors])
second_major =pd.Series(index=undergrad_majors.index.values,
                        data=[x[1] if type(x) != float and len(x) == 2 else np.nan for x in undergrad_majors])
majors = first_major.value_counts().add(second_major.value_counts(), fill_value=0).divide(335).sort_values(ascending=False)*100
majors.name = 'Undergraduate Major' 


# ##Educating an Astronaut

# ###Undergraduate Studies
# A fundamental purpose of astronauts (besides their initial function as tools in great power politics) is to conduct experiments in space.  As a result, there has always been a clear preference for those applicants with backgrounds in science and engineering disciplines.  Looking at the top ten most common degrees for selected astronauts, that preference is clearly evident.  The top five most common undergraduate degrees account for nearly half (42.857%) of all such degrees in the data set (335 total).  Meanwhile, the other half quickly drops to only applying to 16.246% of undergraduate studies for all selected astronauts.

# In[ ]:


print('Top 10 Most Common Undergrad Majors (Normalized %)')
print('='*10)
print(majors[:10])
print('='*10)
print('Proportion of Astronauts w/ Top 5 Most Common Undergrad Majors {0:.3f}%'.format(np.sum(majors[:5])))
print('Proportion of Astronauts w/ Top 6-10 Most Common Undergrad Majors {0:.3f}%'.format(np.sum(majors[5:10])))


# In[ ]:


grad_majors = astronauts_df['Graduate Major'].str.split(r';\s')
first_major = pd.Series(index=grad_majors.index.values,
                        data=[x[0] if type(x) != float else np.nan for x in grad_majors])
second_major = pd.Series(index=grad_majors.index.values,
                         data=[x[1] if type(x) != float and len(x) > 1 else np.nan for x in grad_majors])
third_major = pd.Series(index=grad_majors.index.values,
                        data=[x[2] if type(x) != float and len(x) > 2 else np.nan for x in grad_majors])
fourth_major = pd.Series(index=grad_majors.index.values,
                         data=[x[3] if type(x) != float and len(x) > 3 else np.nan for x in grad_majors])
majors = first_major.value_counts().add(second_major.value_counts(), fill_value=0).add(third_major.value_counts(), fill_value=0).add(fourth_major.value_counts(), fill_value=0).divide(298).sort_values(ascending=False)*100
majors.name = 'Graduate Major'


# ### Graduate Studies
# At least 298 astronauts have at least one graduate degree at some level (from a master's degree upwards).  Once again, we see that the top five most common majors encompass nearly half (43.289%) of all graduate degrees earned.  The remaining bottom half of the top ten account for 18.456% of these degrees.
# 
# There is a bit more diversity in the ten most popular graduate degrees compared to those for undergraduate studies: approx. 3% of astronauts' advanced degrees are MBAs and 10% are medical degrees.  

# In[ ]:


print('Top 10 Most Common Grad Majors (Normalized %)')
print('='*10)
print(majors[:10])
print('='*10)
print('Proportion of Astronauts w/ Top 5 Most Common Grad Majors {0:.3f}%'.format(np.sum(majors[:5])))
print('Proportion of Astronauts w/ Top 6-10 Most Common Grad Majors {0:.3f}%'.format(np.sum(majors[5:10])))


# In[ ]:




