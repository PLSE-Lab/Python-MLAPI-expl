#!/usr/bin/env python
# coding: utf-8

# # mary's data but in graph form
# ## also, guys have like,,, no imposter syndrome bruh

# In[ ]:


# shittily made by jacqs bc im procrastinating

import pandas as pd
import seaborn as sns

df = pd.read_csv("../input/maryexp/mary_exp.csv")

df = df.drop("score", axis=1)
df = df.drop("time", axis=1)


# In[ ]:


df.groupby('gender').mean()


# In[ ]:


colors = ["#2E2B5F", "#8B00FF"]
sns.set()
sns.set_palette(colors)
gender = sns.catplot(x="gender", y="school_int", kind="box", data=df)


# In[ ]:


colors = ["#2E2B5F", "#8B00FF"]
sns.set()
sns.set_palette(colors)
gender = sns.catplot(x="gender", y="rmp_int", kind="box", data=df)


# In[ ]:


rainbow = ["#2E2B5F", "#8B00FF"]
sns.set()
sns.set_palette(colors)
gender = sns.catplot(x="ta", y="rmp_int", kind="box", data=df)


# In[ ]:


rainbow = ["#FF0000","#00FF00"]
sns.set()
sns.set_palette(rainbow)
scores = sns.relplot(x="gpa", y="rmp_int", hue="gender", kind="line", data=df)


# In[ ]:


rainbow = ["#FF0000","#00FF00"]
sns.set()
sns.set_palette(rainbow)
gpa = sns.catplot(x="gender", y="gpa", hue="gender", kind="box", data=df)


# In[ ]:




