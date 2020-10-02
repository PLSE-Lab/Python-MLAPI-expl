#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


salary_data = pd.read_csv("../input/salary/Salary.csv")


# *A small Glimpse of the data*

# In[ ]:


salary_data.head()


# ***Line plot of years of experience and salary***

# In[ ]:


sns.lineplot(x = salary_data["YearsExperience"] , y = salary_data["Salary"])


# ***Regression plot of salary vs years of experience***

# In[ ]:


sns.regplot(x = salary_data["YearsExperience"] , y = salary_data["Salary"])


# ***Distribution of Data for Salaries vs Years of experience***

# In[ ]:


sns.scatterplot(x = salary_data["YearsExperience"] , y = salary_data["Salary"])

