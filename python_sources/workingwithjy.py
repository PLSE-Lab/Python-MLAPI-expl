#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


os.chdir("C:\\Downloads\\WA_Fn-UseC_-HR-Employee-Attrition (2).csv")


# In[ ]:


HRemployee= pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition (2).csv", header =0)


# In[ ]:


HRemployee.head()


# In[ ]:


sns.distplot(HRemployee['Age'])


# In[ ]:


sns.distplot(HRemployee['DailyRate'])


# In[ ]:


plt.show()


# In[ ]:


sns.countplot(HRemployee.DailyRate)


# In[ ]:


sns.barplot(x ='Education', y='EmployeeCount', data=HRemployee)


# In[ ]:


sns.barplot(x = 'Department',y='EmployeeNumber',data = HRemployee)


# In[ ]:


sns.boxplot(HRemployee['WorkLifeBalance'], HRemployee['YearsInCurrentRole'])


# In[ ]:


HRemployee['WorkLifeBalance'] = pd.cut(HRemployee['YearsInCurrentRole'], 3, labels=['low', 'middle', 'high'])


# In[ ]:


HRemployee.head()


# In[ ]:


sns.jointplot(HRemployee.Education,HRemployee.EmployeeCount, kind = "scatter") 


# In[ ]:


sns.jointplot(HRemployee.WorkLifeBalance,HRemployee.YearsInCurrentRole,kind="scatter") 


# In[ ]:


sns.jointplot(HRemployee.YearsSinceLastPromotion,HRemployee.YearsWithCurrManager, kind = "reg")


# In[ ]:


sns.jointplot(HRemployee.YearsSinceLastPromotion,HRemployee.YearsInCurrentRole, kind = "hex") 


# In[ ]:


cont_col= ['c', 'Attrition','DailyRate','DistanceFromHome']


# In[ ]:


sns.pairplot(HRemployee[cont_col], kind="reg",diag_kind="kde",hue='Attrition')


# In[ ]:


plt.show()


# In[ ]:


sns.factorplot(x='Age',y='Attrition',hue='DailyRate',col='DistanceFromHome',col_wrap=2,kind='box',
               data= HRemployee)

