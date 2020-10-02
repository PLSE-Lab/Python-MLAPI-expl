#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


student_data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
student_data.head()


# In[ ]:


#Check for any null data
student_data.isnull().sum()


# In[ ]:


student_data["race/ethnicity"].value_counts()


# In[ ]:


student_data["parental level of education"].value_counts()


# In[ ]:


student_data["lunch"].value_counts()


# In[ ]:


student_data["test preparation course"].value_counts()


# In[ ]:


student_data.describe()


# In[ ]:


student_data.corr()


# In[ ]:


sns.pairplot(student_data)
plt.show()
#Scores of subjects have linear trend with one another with a very good correlation for reading & writing


# In[ ]:


sns.distplot(student_data["math score"],hist_kws=dict(edgecolor="k", linewidth=1,color='grey'),color='red')
plt.show()


# In[ ]:


sns.distplot(student_data["reading score"],hist_kws=dict(edgecolor="k", linewidth=1,color='grey'),color='red')
plt.show()


# In[ ]:


sns.distplot(student_data["writing score"],hist_kws=dict(edgecolor="k", linewidth=1,color='grey'),color='red')
plt.show()


# In[ ]:


student_data.groupby(["test preparation course"]).mean().plot.bar()
plt.show()
#Average score is more for kids who took preparation course


# In[ ]:


student_data.groupby(["parental level of education"]).mean().plot.bar()
plt.show()


# In[ ]:


student_data.groupby(["gender"]).mean().plot.bar()
plt.show()


# In[ ]:


student_data.groupby(["race/ethnicity"]).mean().plot.bar()
plt.show()


# In[ ]:


student_data.groupby(["lunch"]).mean().plot.bar()
plt.show()


# **Correlation between various fields**

# In[ ]:


pivot = pd.pivot_table(data = student_data, index = ["parental level of education"], columns = ["race/ethnicity"], aggfunc = {'math score' : np.mean})
hm = sns.heatmap(data = pivot, annot = True, cmap = "Greens")
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[ ]:


pivot = pd.pivot_table(data = student_data, index = ["parental level of education"], columns = ["race/ethnicity"], aggfunc = {'writing score' : np.mean})
hm = sns.heatmap(data = pivot, annot = True, cmap = "Greens")
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[ ]:


pivot = pd.pivot_table(data = student_data, index = ["parental level of education"], columns = ["race/ethnicity"], aggfunc = {'reading score' : np.mean})
hm = sns.heatmap(data = pivot, annot = True, cmap = "Greens")
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[ ]:


pivot = pd.pivot_table(data = student_data, index = ["test preparation course"], columns = ["gender"], aggfunc = {'math score' : np.mean,
                                                                                                                 'reading score' : np.mean,
                                                                                                                 'writing score' : np.mean})
hm = sns.heatmap(data = pivot, annot = True, cmap = "Greens")
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[ ]:


pivot = pd.pivot_table(data = student_data, index = ["lunch"], columns = ["gender"], aggfunc = {'math score' : np.mean,
                                                                                                                 'reading score' : np.mean,
                                                                                                                 'writing score' : np.mean})
hm = sns.heatmap(data = pivot, annot = True, cmap = "Greens")
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

