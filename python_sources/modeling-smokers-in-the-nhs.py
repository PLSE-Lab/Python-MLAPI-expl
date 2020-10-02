#!/usr/bin/env python
# coding: utf-8

# # Modeling Smokers in the NHS
# 
# Hi! This notebook is a partial reproduction of an analysis of this data that I did in [a Google Colab notebook](https://colab.research.google.com/drive/14cB6EIaGBJUvDWHS2UH-lupa0BPeDHoB). I really recommend you check it out there, as you can actually interact with my results much more fully.  The only reason for this is that I am leveraging a development version of `pgmpy`, a python package focused on probabilistic graphical models and Kaggle really doesn't like you install extra software on their VMs.
# 
# All that said, I think this approach gives a nice way to handle such a small record set and I wanted to contribute it in some form.  Since data preparation is not the focus of the notebook, I have [prepared it in advance](https://colab.research.google.com/drive/1viCPLAw-CBGOPXtcrH1CEWvJ2Cw8yiRY) and uploaded a flat dataset that has all of the records and fields that I will analyze. Okay, let's get into it.
# 
# Finally, I should preface this by saying that **I'm not really going to talk about why I'm analyzing the data this way.**  If this topic interests you, I really recommend that you start with [this book from Judea Pearl, called The Book of Why.](https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X)
# 
# ## Getting Into It
# 
# The data is only available at an aggregated, typically yearly granularity. It's provided in 5 different files, which when flattened yield 110 complete records from which we hope to derive information.  This, I likely don't need to tell you, is extremely small to try to learn from. 
# 
# The most important thing to know about how it was prepared is that all data has been centered and standardized by deviding by two times the standard deviation, [per Gelman's recommendation.](http://www.stat.columbia.edu/~gelman/research/published/standardizing7.pdf)
# 
# I chose this dataset because it's so small, and yet the presence of causal effects was really clear.  I found it really easy, although I'm not a health data expert by any means, to guess at the causal relationships between variables, which is the most important starting point in any analysis of this type. 
# 
# I should also add that I contributed these feature to pgmpy, which is why they're getting pulled from my repo on GitHub.  They'll hopefully be included in the 0.2.0 pgmpy release, but until then we have to pull directly from the development branch. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import statsmodels.api as sm
# I've added Daft to the stack so that I can visualize my graphical models beautifully
import daft
from daft import PGM

flatdata = pd.read_csv('../input/nhs-tobacco-flattened/flattened.csv').drop("Unnamed: 0", axis=1)


# For your reference, here's a preview of the data I've prepared.

# In[ ]:


flatdata.head(5)


# In[ ]:


flatdata.describe()


# In[ ]:


# This is was final version of my causal graph.  While initially taken a priori, it's validity is (somewhat) confirmed by the data. 
pgm = PGM(shape=[8, 5], origin=[0, 0])

pgm.add_node(daft.Node("Deaths", r"Deaths", 4, 0.5, aspect=1.75))
pgm.add_node(daft.Node("Admissions", r"Admissions", 2.5, 1.5, aspect=2.5))
pgm.add_node(daft.Node("Smokers", r"Smokers", 4, 2.5, aspect=2.25))
pgm.add_node(daft.Node("Sex", r"Sex", 7, 3.5, aspect=2))
pgm.add_node(daft.Node("Age", r"Age", 5.5, 3.5, aspect=2))
pgm.add_node(daft.Node("Drug Cost", r"Drug Cost", 5.75, 4.5, aspect=2.25))
pgm.add_node(daft.Node("In Treatment", r"In Treatment", 4, 3.5, aspect=2.75))
pgm.add_node(daft.Node("Year", r"Year", 1, 4.5, aspect=2.25)) # There isn't a great place for Year to go
pgm.add_node(daft.Node("Cost", r"Cost", 2.25, 3.5, aspect=2.25))
pgm.add_node(daft.Node("Income", r"Income", 0.75, 3.5, aspect=2.25))
pgm.add_node(daft.Node("Spent", r"Spend", 1.5, 2.5, aspect=2.25))

pgm.add_edge("Year", "In Treatment")
pgm.add_edge("Year", "Admissions")
pgm.add_edge("Year", "Deaths")
pgm.add_edge("Year", "Income")
pgm.add_edge("Year", "Cost")
pgm.add_edge("Year", "Drug Cost")
pgm.add_edge("Year", "Smokers")
pgm.add_edge("Age", "Smokers")
pgm.add_edge("Sex", "Smokers")
pgm.add_edge("Cost", "Smokers")
pgm.add_edge("Cost", "Spent")
pgm.add_edge("Drug Cost", "Smokers")
pgm.add_edge("Drug Cost", "In Treatment")
pgm.add_edge("In Treatment", "Smokers")
pgm.add_edge("Income", "Spent")
pgm.add_edge("Income", "Smokers")
pgm.add_edge("Smokers", "Admissions")
pgm.add_edge("Admissions", "Deaths")
pgm.add_edge("Smokers", "Deaths")

pgm.render()


# ## Modeling Causal Effects
# Our general goal is to measure average treatment effects (ATE). The ATE of X on Y is the *causal effect* of X on Y, and can be measured by carefully controlling for confounders in the data.  Central to understanding what is a confounder is the causal graph.
# 
# For the purposes of this notebook, I want to highlight a few interesting models that I come from my analysis of the causal graph above.  
# 
# ### Drug Cost $\rightarrow$ Smokers
# To measure the ATE of Drug Cost on Smokers we have to control for confounders.  In this case, controlling for Year is sufficient to deconfound Drug Cost and Smokers. The ATE is then simply given by the coefficient of Drug Cost.
# 
# The results are uncertain, with the confidence interval straddling 0, though it does lean negative.  There could be a small, but noisy effect present which a more precise model could understand. At this point we can't say anything for certain though. 

# In[ ]:


model = sm.OLS(endog=flatdata["Smokers"], exog=sm.add_constant(flatdata[['Drug Cost', 'Year']])).fit()
model.summary()


# ### Year $\rightarrow$ Smokers
# Measuring for the effect of Year on Smokers doesn't require controlling for anything.  This effect turns out to be quite decisively negative, which conforms to our expectations given the shift in social norms and view of smoking over time.

# In[ ]:


model = sm.OLS(endog=flatdata["Smokers"], exog=sm.add_constant(flatdata[['Year']])).fit()
model.summary()


# ### Smokers $\rightarrow$ Deaths
# This looks at the relationship between numbers of Smokers and numbers of deaths from diseases related to smoking.  The finding is actually a bit curious!  The numbers of smokers don't appear to be related to the numbers of deaths (explicitly related to smoking?!). We are left to speculate about the solution (because I haven't done the analysis myself yet), but I suspect there is still some confounding from the fact that deaths from smoking are really going to be cause by behaviors in smokers from the prior decades.  So we can't blame the people smoking today for the people who are also dying from smoking related causes.
# 
# Given this, you might be inclined to start to think about this in two parts.  First there is the model for how, in a single year, different measurements can give you a sense of how many smokers there will be.  The second model is about how across different years, the numbers of smokers can impact the numbers of admissions and deaths in the future.  

# In[ ]:


model = sm.OLS(endog=flatdata["Deaths"], exog=sm.add_constant(flatdata[['Smokers', 'Year']])).fit()
model.summary()


# In[ ]:




