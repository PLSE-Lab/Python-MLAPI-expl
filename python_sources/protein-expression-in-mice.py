#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


mouse_data = pd.read_csv('../input/Data_Cortex_Nuclear.csv')
print(mouse_data.columns)

proteins = mouse_data.columns[1:78]
print(proteins)


# **BACKGROUND**
# ![](http://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0129126.g001)
# http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0129126
# 
# **QUESTION**
# Is the expression of a protein correlated with learning in trisomic mice?
# 
# **PROCEDURE**
# We compare protein levels in the normal and trisomic mice in the stimulated learning (CS) group. The boxplot for each protein is followed by a one-way ANOVA comparing the mean protein levels in the mice in the CS group. A p-value < 0.05 indicates that the difference between the means is statistically significant.
# 
# For example, a p-value of 0.0002 for DYRK1A_N indicates that if our null hypothesis, that there is no significant difference between the means, were true, there would be an 0.02% chance of observing our results. Thus, our results are statistically significant. We can see that DYRK1A_N levels in trisomic mice injected with memantine (rescued learning mice) are significantly higher than DYRK1A_N levels in trisomic mice injected with saline (failed learning mice). This suggests that higher levels of DYRK1A_N could be associated with learning in trisomic mice.
# 
# 
# 

# In[ ]:


for p in proteins:
    sns.boxplot(data = mouse_data, x = 'class', y = p)
    plt.show()
    
    ccs = mouse_data.loc[mouse_data['class'] == 'c-CS-s'][p]
    ccm = mouse_data.loc[mouse_data['class'] == 'c-CS-m'][p]
    tcs = mouse_data.loc[mouse_data['class'] == 't-CS-s'][p]
    tcm = mouse_data.loc[mouse_data['class'] == 't-CS-m'][p]
    css = mouse_data.loc[mouse_data['class'] == 'c-SC-s'][p]
    csm = mouse_data.loc[mouse_data['class'] == 'c-SC-m'][p]
    tss = mouse_data.loc[mouse_data['class'] == 't-SC-s'][p]
    tsm = mouse_data.loc[mouse_data['class'] == 't-SC-m'][p]
    print(stats.f_oneway(ccs, ccm, tcs, tcm))

