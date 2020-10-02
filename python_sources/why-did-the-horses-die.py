#!/usr/bin/env python
# coding: utf-8

# # Why did the horses die?
# 
# My goal is to get more insight into the horses that died or were euthanized.

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.svm import SVC


# In[ ]:


raw_data = pd.read_csv('../input/horse-colic/horse.csv')
raw_data.head()


# In[ ]:


raw_data.shape


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


sns.countplot(data=raw_data, x='outcome');


# Among the 299 cases, about 75 horses died and 40 were euthanized. We will study the condition of these horses in this kernel.

# # Surgery & Pain
# We will study the effects of surgery and pain. How many of the horses which died underwent surgery and what level of pain did they feel?

# In[ ]:


print(raw_data.outcome.value_counts())

sns.countplot(data=raw_data, x='outcome', hue='surgery');
plt.show()

sns.countplot(data=raw_data, x='outcome', hue='pain');
plt.show()

g = sns.catplot(data=raw_data, x='outcome', col='surgery', hue='pain', kind='count');
g.fig.suptitle('Horse deaths by Pain & Surgery');
plt.subplots_adjust(top=0.9)


# Points to note:  
# * Only 20 of the 77 horses that died underwent surgery. Were they neglected?
# * About 18 horses underwent surgery but were euthanized.
# * A large majority of the horses that died felt extreme or severe pain. While a majority of horses that were euthanized experienced severe or depressed pain.
# * Of all the horses that died and did not undergo surgery, most of them felt extreme pain.
# * Of all the horses that did not undergo surgery and were euthanized, most of them felt extreme pain.

# # Age & Pain

# In[ ]:


g = sns.catplot(data=raw_data, x='outcome', hue='pain', col='age', kind='count');
g.fig.suptitle('Horse deaths by Pain & Age');
plt.subplots_adjust(top=0.9)


# It seems like we don't have a substantial young population to compare.

# # Circulation

# In[ ]:


g = sns.FacetGrid(data=raw_data, col='outcome', margin_titles=True, height=6)
g.map(plt.hist, 'pulse')
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Outcome by Pulse');


# In[ ]:


g = sns.catplot(data=raw_data, x='peripheral_pulse', col='outcome', kind='count');
g.fig.suptitle('Outcome by Peripheral Pulse');
plt.subplots_adjust(top=0.9)


# In[ ]:


reduced_absent_pulse = raw_data[raw_data.outcome.isin(('died','euthanized')) & raw_data.peripheral_pulse.isin(('reduced','absent'))]

g = sns.catplot(data=reduced_absent_pulse, x='capillary_refill_time', col='outcome', kind='count');
g.fig.suptitle('Outcome by Capillary refill time');
plt.subplots_adjust(top=0.9)


# Points to note:
# * Most of the horses that died had a pulse approximately 80-100 bpm.
# * More than half the horses that died or were euthanized had a reduced peripheral pulse.
# * Of all the horses that died/euthanized and had a reduced/absent peripheral pulse, a majority had a capillary refill time greater than 3 seconds. That is the sign of a poor circulatory system.

# # Gut Health

# In[ ]:


g = sns.catplot(data=raw_data, x='abdominal_distention', col='outcome', hue='surgery', kind='count');
g.fig.suptitle('Outcome by Abdominal Distention & Surgery');
plt.subplots_adjust(top=0.9)


# In[ ]:


severe_died = raw_data[raw_data.outcome.isin(('died','euthanized')) & (raw_data.abdominal_distention=='severe') & (raw_data.surgery=='no')]

g = sns.countplot(data=severe_died, x='peristalsis').set_title('Horses with severe abdominal distention, did not undergo surgery and died');


# Points to note:
# * We were given the information that when there is Abdominal Distention, a surgery is necessary to alleviate the condition. A large proportion of horses with severe distention that died had not undergone surgery. 
# * Meanwhile, a large proportion of horses with severe distention that survived had undergone surgery.
# * Of the 9 horses with severe distention that did not undergo surgery and died, 7 had an absent peristalsis. That was the sign of a compromised gut.
