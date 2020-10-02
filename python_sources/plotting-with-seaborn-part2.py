#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string as st # to get set of characters


# In[ ]:


col = [a for a in st.ascii_uppercase[:10]]
tmp = np.random.randint(1,30,1000).reshape(100,10)
df = pd.DataFrame(tmp,columns=col)


# In[ ]:


df.head(2)


# In[ ]:


df.describe()


# In[ ]:


df["categ"] = np.random.choice(col[:3],100)


# In[ ]:


date = pd.date_range("1/1/2018",periods=100)
df["date"] = date


# In[ ]:


df.set_index(date,inplace=True)


# In[ ]:


df.drop("date",axis=1,inplace=True)


# In[ ]:


df.head(2)


# # My random dataset is ready with some categorical value as well as time series info

# # FacetGrid

# In[ ]:


tip = pd.read_csv("../input/tips.csv")
tip.head(2)


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,row="smoker",col="time")
fg = fg.map(sns.scatterplot,"total_bill","size")


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,row="smoker",col="time")
fg = fg.map(plt.hist,"total_bill",color="g",bins=15)


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,row="smoker",col="time")
fg = fg.map(sns.regplot,"total_bill","tip")


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,col="time",hue="smoker")
fg = fg.map(sns.scatterplot,"total_bill","tip")
fg.add_legend()


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,row="smoker",col="time",hue="smoker")
fg = fg.map(sns.regplot,"total_bill","tip")
fg.add_legend()


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,row="smoker",col="time",hue="size")
fg = fg.map(sns.regplot,"total_bill","tip")
fg.add_legend()


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,row="smoker",col="time",hue="size")
fg = fg.map(sns.scatterplot,"total_bill","tip")
fg.add_legend()


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,col="size")
fg = fg.map(sns.boxplot,"time","total_bill")


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,col="size",size=10)
fg = fg.map(sns.boxplot,"time","total_bill")


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,col="size",size=10,aspect=.6)
fg = fg.map(sns.boxplot,"time","total_bill")


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,col="size",size=10,aspect=.6)
fg = fg.map(sns.boxplot,"time","total_bill",palette="husl")


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,col="size",size=10,aspect=.6)
fg = fg.map(sns.boxplot,"time","total_bill",color="r")


# In[ ]:


plt.figure(figsize=(16,4))
fg = sns.FacetGrid(tip,col="size",palette="husl")
fg = fg.map(sns.boxplot,"time","total_bill")


# # KDE (kernel density Estimation) plot

# In[ ]:


df.head(2)


# In[ ]:


plt.figure(figsize=(16,4))
sns.kdeplot(df.A)


# In[ ]:


plt.figure(figsize=(16,4))
sns.set_style("darkgrid")
sns.kdeplot(df.A,shade=True)


# In[ ]:


plt.figure(figsize=(16,4))
sns.set_style("darkgrid")
sns.kdeplot(df.A,shade=True,color="m")


# In[ ]:


# plt.figure(figsize=(16,4))
sns.set_style("darkgrid")
sns.kdeplot(df.A,df.C,shade=False,color="g",cbar=True)


# In[ ]:


# plt.figure(figsize=(16,4))
sns.set_style("darkgrid")
sns.kdeplot(df.A,df.C,shade=True,color="g",cbar=True)


# ## Possible color map:
#     Colormap gist_eart is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r

# In[ ]:


# plt.figure(figsize=(16,4))
sns.set_style("darkgrid")
sns.kdeplot(df.A,df.C,shade=False,color="g",cbar=True,cmap="bone_r",bw=1.8)


# In[ ]:


# plt.figure(figsize=(16,4))
sns.set_style("darkgrid")
sns.kdeplot(df.A,df.C,shade=False,color="g",cbar=True,bw=1.4)


# In[ ]:


sns.kdeplot(df.A,df.C,shade=False,color="g",cbar=True,bw=1.4,n_levels=20) # n_levels is to draw those many contour lines


# In[ ]:


# plt.figure(figsize=(16,4))
sns.set_style("darkgrid")
sns.kdeplot(df.A,shade=False,color="g",cbar=True,bw=1.6)


# In[ ]:


# plt.figure(figsize=(16,4))
# sns.set_style("darkgrid")
sns.kdeplot(df.A,bw=1.9,shade=True,color="r")
sns.kdeplot(df.C,bw=1.9,shade=True,color="g")

