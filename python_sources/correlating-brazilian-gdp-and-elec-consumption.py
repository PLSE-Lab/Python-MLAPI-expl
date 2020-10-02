#!/usr/bin/env python
# coding: utf-8

# Reading the data :

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot
import seaborn as sns
from scipy.stats import stats

Dados = pd.read_csv("../input/dados_Brazil_GDP_Electricity.csv", sep = ",",index_col = False)
Dados = Dados[Dados.columns [1:]]


# Plotting using seaborn:

# In[ ]:


pyplot.style.use("ggplot")
ax = sns.JointGrid(x = "Tw/h", y = "GDP", data = Dados, xlim =(50,500), ylim = (-0.5,3) )
ax = ax.plot(sns.regplot, sns.distplot)
ax = ax.annotate(stats.pearsonr)

pyplot.xlabel("Terawatt hour - year")
pyplot.ylabel("GDP - Usd trillion")

