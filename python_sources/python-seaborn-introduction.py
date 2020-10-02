#!/usr/bin/env python
# coding: utf-8

# # PYTHON SEABORN
# 
# 
# This is my second kernel.If you like it Please upvote! 
# I prepared this kernel while i was studying Udemy course.
# https://www.udemy.com/course/veri-bilimi-ve-makine-ogrenmesi-icin-python/
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


e=pd.read_csv("../input/python-seaborn-datas/50ulke.csv")
e.pop("2023")
e.pop("Rankings")
e.pop("2. Rankings")


# # distplot ( )

# In[ ]:


sns.distplot(e["Growth"])


# In[ ]:


n=pd.read_csv("../input/python-seaborn-datas/dunyam.csv")
n.dropna(how="any",inplace=True)


# In[ ]:


sns.set(style="dark",palette="muted",font_scale=2)
sns.distplot(n["Birthrate"],bins=20,kde=False,color="y")
plt.tight_layout()


# In[ ]:


sns.set(style="darkgrid",palette="muted",font_scale=1.5)
sns.distplot(n["Annual rate of change"],hist=False,rug=True,color="r")
plt.tight_layout()


# In[ ]:


sns.set(style="white",palette="Blues",font_scale=1.5)
sns.distplot(n["Average age"],hist=False,color="g",kde_kws={"shade":True})
plt.tight_layout()


# In[ ]:


sns.set(style="whitegrid",palette="Blues",font_scale=1.5)
sns.distplot(n["Population"],color="m")
plt.tight_layout()


# In[ ]:


sns.set(style="darkgrid",font_scale=1.5)

f, axes = plt.subplots(2,2,figsize=(15,10))

sns.distplot(n["Birthrate"],bins=20,kde=False,color="y",ax=axes[0,0])

sns.distplot(n["Annual rate of change"],hist=False,rug=True,color="r",ax=axes[0,1])

sns.distplot(n["Average age"],hist=False,color="g",kde_kws={"shade":True},ax=axes[1,0])

sns.distplot(n["Population"],color="m",ax=axes[1,1])

plt.tight_layout()


# # jointplot ( )

# In[ ]:


sns.jointplot(n["Birthrate"],n["Average age"],data=n)


# In[ ]:


sns.jointplot(n["Annual rate of change"],n["Birthrate"],data=n)


# In[ ]:


sns.jointplot(n["Annual rate of change"],n["Birthrate"],data=n,kind="hex",color="r")


# In[ ]:


sns.jointplot(n["Annual rate of change"],n["Birthrate"],data=n,kind="reg",xlim=(-2,6),ylim=(0,10),color="r",size=10)


# In[ ]:


sns.jointplot(n["Annual rate of change"],n["Birthrate"],data=n,kind="kde",xlim=(-2,6),ylim=(0,10),color="r",size=8)


# # kdeplot  ( )
# 

# In[ ]:


sns.kdeplot(e["Per person"])


# In[ ]:


sns.kdeplot(e["Growth"])


# In[ ]:


sns.kdeplot(e["Per person"],e["Growth"],shade=True,cmap="Reds")


# In[ ]:


sns.kdeplot(e["Per person"],e["Growth"],shade=True,cmap="Blues")


# # pairplot ( )

# In[ ]:


sns.pairplot(n,palette="#95a5a6")


# In[ ]:


sns.pairplot(e,hue="Continent",palette="inferno")


# # rugplot () 

# In[ ]:


sns.rugplot(e["Growth"],color="y",height=0.2)
sns.kdeplot(e["Growth"],color="r")


# # boxplot ( )

# In[ ]:


sns.boxplot(x="Continent",y="Per person",data=e,width=0.5)
plt.tight_layout()


# In[ ]:


sns.boxplot(x="Continent",y="Growth",data=e,palette="Set3")


# In[ ]:


ev=pd.read_csv("../input/python-seaborn-datas/marriage.csv")


# In[ ]:


sns.boxplot(x="Month",y="Revenue",data=ev,hue="Marriage",palette="PRGn")


# # violinplot ( )

# In[ ]:


sns.set(style="whitegrid")
sns.violinplot(x="Month",y="Revenue",data=ev,hue="Marriage",palette="PRGn",split=True,inner="points")


# In[ ]:


sns.set(style="whitegrid")
sns.violinplot(x="Month",y="Revenue",data=ev,hue="Marriage",palette="PRGn")


# In[ ]:


sns.set(style="whitegrid")
sns.violinplot(x="Continent",y="Growth",data=e,palette="Set3",split=True)


# # barplot ( )

# In[ ]:


sns.barplot(x="Continent",y="Per person",data=e,palette="BuGn_d")


# In[ ]:


sns.barplot(x="Continent",y="Per person",data=e,palette="RdBu_r")


# In[ ]:


sns.barplot(x="Continent",y="Per person",data=e,palette="Set1")
sns.despine(left=True,bottom=True)


# # countplot()

# In[ ]:


sns.countplot(x="Continent",data=e)


# # stripplot ( )

# In[ ]:


sns.stripplot(x="Continent",y="Growth",data=e,color="red",jitter=True)


# # swarmplot()

# In[ ]:


sns.violinplot(x="Continent",y="Growth",data=e)
sns.swarmplot(x="Continent",y="Growth",data=e,color="red")


# # factorplot ( ) 

# In[ ]:


sns.factorplot(x="Continent",y="Growth",data=e,kind="bar")


# In[ ]:


sns.factorplot(x="Continent",y="Growth",data=e,kind="violin")


# In[ ]:


sns.factorplot(x="Per person",y="Growth",data=e,kind="point")


# In[ ]:


t=pd.read_csv("../input/python-seaborn-datas/titanic.csv")


# In[ ]:


sns.factorplot(x="Pclass",y="Survived",hue="Sex",size=6,data=t,kind="bar",palette="muted")


# In[ ]:


sns.factorplot(x="Pclass",y="Survived",hue="Sex",size=6,data=t,kind="violin",palette="muted")


# # heatmap ( )

# In[ ]:


enf=pd.read_csv("../input/python-seaborn-datas/tufe.csv")
enfl=enf.pivot_table(index="Month",columns="Year",values="inflation")


# In[ ]:


sns.heatmap(enfl,annot=True,linecolor="black",lw=0.5)


# # clustermap ( )

# In[ ]:


sns.clustermap(enfl,figsize=(6,6))


# # lmplot ( )

# In[ ]:


sns.lmplot(x="Per person",y="Growth",data=e)


# In[ ]:


sns.lmplot(x="Growth",y="Per person",data=e,hue="Continent")


# In[ ]:


sns.lmplot(x="Growth",y="Per person",data=e,col="Continent")


# # PairGrid ( )

# In[ ]:


m=sns.PairGrid(n)
m.map_diag(sns.distplot)
m.map_upper(plt.scatter)
m.map_lower(sns.kdeplot)


# In[ ]:


k=sns.PairGrid(e)
k.map_diag(sns.distplot)
k.map_upper(plt.scatter)
k.map_lower(sns.kdeplot)


# # FacetGrid

# In[ ]:


s=pd.read_csv("../input/python-seaborn-datas/marriage.csv")


# In[ ]:


ss=sns.FacetGrid(data=s,col="Month",row="Marriage")
ss.map(sns.distplot,"Revenue")


# In[ ]:


ss=sns.FacetGrid(data=s,col="Month",row="Marriage")
ss.map(plt.hist,"Revenue")

