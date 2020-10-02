#!/usr/bin/env python
# coding: utf-8

# 
# Here is data from data.world which represents drug-induced deaths within the USA from 1999 - 2015. My goal is to perform a level of analytics on this data set to hopefully abstract some interesting observations.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


mydata = pd.read_csv('../input/drug_induced_deaths_1999-2015.csv')
mydata.head()


# Here it's difficult to understand the data, with multiple variables it seems difficult and not highly intuitive to utilize Seaborn or Matplotlib. This is a personal reason and due to a lack of experience with both packages. I will switch over to Tableau and explain my further observations regarding this data.

# In[ ]:


vis1 = sns.lmplot( data = mydata, x = 'Year', y = 'Deaths', fit_reg = False, hue = 'State', size = 10 )


# Here is a geographical representation of the data which highlights the number of deaths in accordance with the geographical location of them. In addition, it is possible to notice the change in number of deaths as the year changes from 1999 - 2015. A clear observation is that for the large part, the number of deaths has been steadily increasing with no explanation.

# In[ ]:


from IPython.display import IFrame
IFrame('https://public.tableau.com/profile/jared.yu#!/vizhome/DrugDeathRateAnalytics0/Growth?publish=yes', width=900, height=550)


# From this 'viz,' it becomes quite apparent that the number of deaths have been increasing greatly directly alongside the growth in population for various states. California which has the highest number of drug-induced deaths coincidentally has the largest population amongst the other states. Does this mean that there is nothing to worry about? Obviously not. The correlation between the two variables indicate that further investigation is necessary in order to uncover any significant information or findings.

# In[ ]:


from IPython.display import IFrame
IFrame('https://public.tableau.com/profile/jared.yu#!/vizhome/DrugDeathRateAnalytics1/Correlation?publish=yes', width=900, height=550)


# Here I've excluded the states where the convergence or divergence for deaths and population is either non-existent or too recent for significant observations to be made. By too recent, I mean that although the deaths may be increasing, the increase is so recent that it would not be fair to make an assumption based on a recent divergence which is only minor in comparison to the change seen in other states.
# 
# If you scroll the 'Year' bar to the most recent date of 2016, it is possible to see that these states seem to have a noticable divergence in the number of deaths in relation to their population. As population seems to have grown dramatically in the past decade (globally), here is evidence that drug-induced death rates are beginning to climb increasingly higher, perhaps reaching a statistical limit in some areas.
# 
# For instance will the deaths peak in Ohio or Pennsylvania? Will something else happen to change the direction of growth towards a more sensible increase? Perhaps the legalization of Marijuana in California as a recreational substance will either send the drug-induced death rate over the population growth line, or help to alleviate it (the debate has arguments from both sides)?

# In[ ]:


from IPython.display import IFrame
IFrame('https://public.tableau.com/profile/jared.yu#!/vizhome/DrugDeathRateAnalytics/Divergence?publish=yes', width=900, height=550)


# My conclusion is that in recent years, population growth has increased dramatically across the country. With this welcoming increase in number of citizens are the problems that come with such a growth. It would be wise to analyze where certain significant observations are found, no matter how few or far between to better understand the cracks that seem to be widening as the United States seems to progress as a growing nation state.
# 
# I would like to stop short of saying or giving the notion that 'something' is happening around the corner, and we must act in some way to stop it. Rather I am indicating that there is a significant change happening to our population, and if the country is not mindful of the difference between now and before then trouble is inevitable.
# 
# The goal for the analytics performed here has not been to develop a highly elaborate or descriptive explanation for any incredibly obvious piece of data. Rather it was to produce simple, clear, and easy to understand descriptions for any interesting insights uncoverable from the data that was readily available.
# 
# I would like to thank user 'Himanshu Chaudhary' for having examples of Tableau, and allowing for me to be able to share my own insights into data utilizing this valuable new software from within the data analytics industry.
