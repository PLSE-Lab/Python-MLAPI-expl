#!/usr/bin/env python
# coding: utf-8

# In[114]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("../input/train.csv")
df = df.drop('PassengerId', axis=1)
df.head()


# In[94]:


df.info()


# In[95]:


print(df.describe())


# In[ ]:


ax = df["Survived"].value_counts().plot.bar()
ax.set_title("Survivability")
ax.set(ylabel='', xlabel='Not survive/Survive')


# In[101]:


ax =df["Sex"].value_counts().plot.bar()
ax.set_title("Sex survivability")


# In[25]:


class_sur=df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
class_sur


# In[87]:


ax=sns.barplot(y="Survived", x="Pclass",data=class_sur)
ax.set(ylabel='Survived percent', xlabel='Pclass')
ax.set_title("Pclass survivability")


# In[28]:


sex_sur=df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sex_sur


# In[86]:


ax=sns.barplot(y='Survived', x='Sex', data=sex_sur)
ax.set(ylabel='Survived percent', xlabel='Sex')
ax.set_title("Sex survivability")


# In[115]:


g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

