#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#you can see first kernel (called part1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = sns.load_dataset("tips").copy()
x = np.random.normal(size = 100)
x


# In[ ]:


sns.distplot(x);


# In[ ]:


sns.distplot(x, kde=False, rug=True);


# In[ ]:


sns.distplot(x, hist=False,rug = True);


# In[ ]:


sns.kdeplot(x);


# In[ ]:


sns.rugplot(x);


# In[ ]:


df.head(5)


# In[ ]:


sns.distplot(df.total_bill);


# In[ ]:


sns.distplot(df.tip, rug=True);


# In[ ]:


sns.kdeplot(df.tip, shade=True);


# In[ ]:


sns.kdeplot(x, color="g", shade =True)
sns.kdeplot(df.tip, color = "r", shade =True)
plt.legend();


# In[ ]:


dfMale = df[df["sex"] == "Male"].copy()


# In[ ]:


dfFemale = df[df["sex"] == "Female"].copy()


# In[ ]:


sns.kdeplot(dfMale.total_bill, color = "r", shade =True, alpha =.3)
sns.kdeplot(dfFemale.total_bill, color = "b", shade =True, alpha = .3);


# In[ ]:


sns.distplot(dfMale.total_bill, color = "r", kde=False)
sns.distplot(dfFemale.total_bill, color = "g", kde=False);


# In[ ]:


sns.distplot(dfFemale.total_bill, color = "b", kde=False, bins=15);


# In[ ]:


sns.jointplot(df.tip, df.total_bill, color="r");


# In[ ]:


sns.pairplot(df);


# In[ ]:


g = sns.PairGrid(df)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels = 6);


# In[ ]:


sns.barplot("size", "total_bill", data = df);


# In[ ]:


sns.barplot("size", "total_bill", hue = "sex", data = df); 


# In[ ]:


df.corr()


# In[ ]:


sns.regplot(x="total_bill", y = "tip", data = df);


# In[ ]:


sns.lmplot(x="total_bill", y = "tip", data = df);


# In[ ]:


sns.lmplot("size", "total_bill", data = df);


# In[ ]:


sns.lmplot(x="size", y="tip", data=df, x_jitter=.2);


# In[ ]:


sns.lmplot(x="size", y= "tip", data = df, x_estimator = np.mean);


# In[ ]:


sns.lmplot(x="size", y= "tip", data = df.query("sex == 'Female'"), ci = None,x_estimator = np.mean);


# In[ ]:


df["big_tip"] = (df.tip / df.total_bill) > .15
df["big_tip"].sample(5)


# In[ ]:


sns.lmplot("total_bill", "big_tip", data = df, y_jitter = .03);


# In[ ]:


sns.lmplot("total_bill", "big_tip", data = df, y_jitter = .03, logistic = True);


# In[ ]:


sns.lmplot("total_bill", "tip", hue = "smoker", data = df);


# In[ ]:


sns.lmplot(x="total_bill", y="tip", hue="smoker", data=df,
           markers=["o", "x"], palette="Set1");


# In[ ]:


sns.lmplot(x="total_bill", y="tip", hue="smoker", col="sex", data=df);


# In[ ]:


sns.lmplot(x = "total_bill", y = "tip", hue = "smoker", col = "sex", row = "time", data = df);


# In[ ]:


f, ax = plt.subplots(figsize=(8, 7))
sns.regplot(x="total_bill", y="tip", data=df, ax=ax);


# In[ ]:


sns.lmplot(x = "total_bill", y = "tip", col = "day", data = df, col_wrap = 2, height = 3);


# In[ ]:


sns.lmplot(x="total_bill", y="tip", col="day", data=df, aspect=.5);


# In[ ]:


sns.jointplot(x="total_bill", y="tip", data=df, kind="reg");


# In[ ]:


sns.pairplot(df, x_vars=["total_bill", "size"], y_vars=["tip"], height=5, aspect=.8, kind="reg");


# In[ ]:


sns.pairplot(df, x_vars=["total_bill", "size"], y_vars=["tip"], hue = "smoker", height=5, aspect=.8, kind = "reg");

