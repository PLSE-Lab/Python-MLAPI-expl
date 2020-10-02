#!/usr/bin/env python
# coding: utf-8

# __How can the Indian state and central governments aid__ in mitigating the rising number of suicides among Indians? That is the question I am trying to address in this notebook.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/Suicides in India 2001-2012.csv")


# Marking Union territories and correcting typos.

# In[ ]:


data.replace('A & N Islands', 'A & N Islands (Ut)', inplace=True)
data.replace('Chandigarh', 'Chandigarh (Ut)', inplace=True)
data.replace('D & N Haveli', 'D & N Haveli (Ut)', inplace=True)
data.replace('Daman & Diu', 'Daman & Diu (Ut)', inplace=True)
data.replace('Lakshadweep', 'Lakshadweep (Ut)', inplace=True)
data.replace('Puducherry', 'Puducherry (Ut)', inplace=True)

data.replace('Bankruptcy or Sudden change in Economic', 'Bankruptcy or Sudden change in Economic Status', inplace=True)
data.replace('By Other means (please specify)', 'By Other means', inplace=True)
data.replace('Not having Children(Barrenness/Impotency', 'Not having Children (Barrenness/Impotency', inplace=True)


# In[ ]:


#Deleting 'Total' s
data = data.drop(data.loc[data.State.str.contains('Total ')].index)


# "It is better to teach a man to fish..." Isn't it?!
# ------------------

# In[ ]:


data1 = data.loc[(data.Type_code == 'Education_Status')].groupby(['Type',])['Total'].sum().reset_index().sort_values('Total',ascending=False).head(60)
data1.set_index(['Type',])
data1=data1.set_index(['Type',])
#sns.set_style('white')
plt.subplots(figsize=(15,10))
g = sns.barplot(y='Total',x=data1.index,data=data1,palette="Blues_d",).set_title('Impact of Education Status')
plt.xticks(rotation=30)
plt.xlabel('Total number of suicides')


# Except for 'No Education', the more educated a person is, the less likely would they be to commit suicide. The government can make sure people have access to schools and colleges that have adequate facilities and teaching personnel. 
# 
# If government could provide mandatory education, it could go a long way.

# Aggravated Agriculture Problem?
# ------------------

# They say farmers are the backbone of India as India is an agrarian society with a large fraction of the population living in the rural areas. 

# In[ ]:


data1 = data.groupby(['State',])['Total'].sum().reset_index().sort_values('Total',ascending=False)
data1.set_index(['State',])
data1=data1.set_index(['State',])
plt.subplots(figsize=(15,20))
g = sns.barplot(y=data1.index,x='Total',data=data1).set_title('State-wise Suicide of Indian Farmers')
plt.xlabel('Total number of suicides')


# Farmers from the southern and the central states seem to be suffering the most. This could due to a number of factors:
# 
#  - depletion of water table
#  - rivers drying up
#  - extreme monsoons
#  - poor water management
#  - improper farming techniques
#  - dependence on water guzzling crops
# 
# However, the government could provide some of the following relief measures:
# 
# - better water management
# - desalinized water
# - seed/fertilizer subsidiary
# - loan waiver
# - etc
# 

# Maha-Addiction problem?
# ------------------

# In[ ]:


data1 = data.loc[(data.Type == 'Drug Abuse/Addiction')].groupby(['State'])['Total'].sum().reset_index().sort_values('Total',ascending=False).head(60)
data1.set_index(['State'])
data1=data1.set_index(['State'])
plt.subplots(figsize=(15,20))
g = sns.barplot(x='Total',y=data1.index,data=data1,palette="Reds_d",).set_title('Drug Abuse/Addiction')
plt.xticks(rotation=30)
plt.xlabel('Total number of suicides')


# Maharashtra seen to have the highest drug abuse related suicides - more than **four times** the next highest states!
# 
# The government can approach this problem by cracking down on the supply of drugs and also by providing rehabilitation for the addicts to prevent suicide.

# 
