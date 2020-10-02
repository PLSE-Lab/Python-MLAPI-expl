#!/usr/bin/env python
# coding: utf-8

# **Motivation**
# 
# Human trafficking is thought to be one of the fastest-growing activities of trans-national criminal organizations. It is defined as the trade of humans for the purpose of forced labour, sexual slavery, or commercial sexual exploitation for the trafficker or others. Human trafficking is condemned as a violation of human rights by international conventions. (Source: Wikipedia)
# 
# This notebook imports the human traficking data collected by UK National Crime Agency, and provides basic visualisation.

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **High level summary data**

# In[ ]:


decision_data=pd.read_csv("../input/2016_decision_data.csv")
print(decision_data.shape)
decision_data


# In[ ]:


plt.bar(decision_data['Year'], decision_data['Total Number of Referrals'])
plt.xticks(decision_data['Year'])
plt.ylabel('Total Number of Referrals')
plt.show()


# **Exploitation type data**

# In[ ]:


exploitation_type_16=pd.read_csv("../input/2016_exploitation_type.csv")
print(exploitation_type_16.shape)
exploitation_type_16


# In[ ]:


exploitation_type_13=pd.read_csv("../input/2013_exploitation_type.csv")
exploitation_type_14=pd.read_csv("../input/2014_exploitation_type.csv")
exploitation_type_15=pd.read_csv("../input/2015_exploitation_type.csv")


# In[ ]:


exploitation_type_13['Year'] = 2013
exploitation_type_14['Year'] = 2014
exploitation_type_15['Year'] = 2015
exploitation_type_16['Year'] = 2016


# In[ ]:


exploitation_type_13 = exploitation_type_13.rename(index=str, columns={"Claimed exploitation Type": "Claimed Exploitation Type"})
exploitation_type_14 = exploitation_type_14.rename(index=str, columns={"Claimed exploitation Type": "Claimed Exploitation Type"})
exploitation_type_15 = exploitation_type_15.rename(index=str, columns={"Total 2015": "Total", "Claimed exploitation Type": "Claimed Exploitation Type", "Transsexual": "Transgender"})
exploitation_type_16 = exploitation_type_16.rename(index=str, columns={"Total 2016": "Total", "Trans- gender": "Transgender"})
exploitation_type_13 = exploitation_type_13.drop(columns=['2012 - 2013% Change'])
exploitation_type_14 = exploitation_type_14.drop(columns=['2013 - 2014 % Change'])
exploitation_type_15 = exploitation_type_15.drop(columns=['2014 - 2015 % Change'])
exploitation_type_16 = exploitation_type_16.drop(columns=['2015 - 2016 % Change'])


# In[ ]:


exploitation_type = pd.concat([exploitation_type_13, exploitation_type_14, exploitation_type_15, exploitation_type_16], sort=True)
exploitation_type


# In[ ]:


def plotExploitation(exploitation_type, type_string):
    expl = exploitation_type.loc[exploitation_type['Claimed Exploitation Type'] == type_string]
    plt.bar(expl['Year'], expl['Total'])
    plt.xticks(expl['Year'])
    plt.ylabel(type_string)
    plt.show()


# In[ ]:


plotExploitation(exploitation_type, "Adult - Domestic Servitude")


# In[ ]:


plotExploitation(exploitation_type, "Minor - Domestic Servitude")


# In[ ]:


plotExploitation(exploitation_type, "Adult - Sexual Exploitation")


# In[ ]:


plotExploitation(exploitation_type, "Minor - Sexual Exploitation (non-UK national)")


# In[ ]:


plotExploitation(exploitation_type, "Minor - Sexual Exploitation (UK national)")


# In[ ]:


plotExploitation(exploitation_type, "Adult - Labour Exploitation")


# In[ ]:


plotExploitation(exploitation_type, "Minor - Labour Exploitation")


# Work in progress: import and join by year other csv files.
