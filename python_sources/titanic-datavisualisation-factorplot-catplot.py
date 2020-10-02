#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df= pd.read_csv('../input/train.csv')


# In[ ]:


df.head()


# In[ ]:


#1 axis x and y
sns.catplot(x='Pclass', y='Survived', data=df, size=5)
plt.show()


# ### 1. Next with kind barplot,  i will show u how factorplot works

# In[ ]:


#2 choose type of figure 'Bar'
sns.catplot(x='Pclass', y= 'Survived', kind='bar', data= df)


# In[ ]:


#3 choose type of figure 'Bar', non the confidence interval
sns.catplot(x='Pclass', y= 'Survived', kind='bar', data= df, ci=None)


# In[ ]:


#4 choose type of figure 'Bar', with Hue Sex male or femal
sns.catplot(x='Pclass', y= 'Survived', hue= 'Sex', kind='bar', data= df, ci=None)


# In[ ]:


#5 choose type of figure 'Bar', with Col Sex male or femal
sns.catplot(x='Pclass', y= 'Survived', col= 'Sex', kind='bar', data= df, ci=None)


# In[ ]:


#6 choose type of figure 'Bar', with hue (Sex male or femal) and  col (embarked)
sns.catplot(x='Pclass', y= 'Survived', hue='Sex', col= 'Embarked', kind='bar', data= df, ci=None)


# In[ ]:


#7 choose type of figure 'Bar', cross Sex and Embarked
sns.catplot(x='Pclass', y= 'Survived', row='Sex', col= 'Embarked', kind='bar', data= df, ci=None)


# ### 2. from the above exemples, i am sure that now u can know very well factorplot to show your data. for more kind plot, we can choose violinplot, boxplot. ect.

# In[ ]:


#8 choose type of figure 'violin'
sns.catplot(x='Survived', y= 'Age', hue='Sex',kind='violin', data= df, size=6)


# In[ ]:


#9 choose type of figure 'box'
sns.catplot(x='Sex', y= 'Age',kind='box', data= df, size=6)
plt.show()


# In[ ]:




