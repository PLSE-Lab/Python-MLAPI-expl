#!/usr/bin/env python
# coding: utf-8

# # Geography of Human Freedom

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import random
import scipy.stats as stt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


data18 = pd.read_csv('../input/the-human-freedom-index/hfi_cc_2018.csv')
data18.shape


# In[ ]:


data18.head(3)


# In[ ]:


data18.info()


# In[ ]:


from mpl_toolkits.basemap import Basemap
concap = pd.read_csv('../input/world-capitals-gps/concap.csv')
concap.head(3)


# For visualization I use Basemap module - part of matplotlib. To visualize this map, I need capital's coordinates - latitude and longitude. I take this data from this site: http://techslides.com/list-of-countries-and-capitals. Then you just have to join the tables.

# In[ ]:


data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],         data18,left_on='CountryName',right_on='countries')


# ## Map visualization

# Let's start with main map - finall Human Freedom score:

# In[ ]:


def mapWorld():
    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,            llcrnrlon=-110,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90,91.,30.))
    m.drawmeridians(np.arange(-90,90.,60.))
    lat = data_full['CapitalLatitude'].values
    lon = data_full['CapitalLongitude'].values
    a_1 = data_full['hf_score'].values
    #a_2 = data_full['Economy (GDP per Capita)'].values
    #300*a_2
    m.scatter(lon, lat, latlon=True,c=a_1,s=500,linewidth=1,edgecolors='black',cmap='hot', alpha=1)
    
    #m.fillcontinents(color='#FFFFFF',lake_color='#FFFFFF',alpha=0.3)
    cbar = m.colorbar()
    cbar.set_label('Human Freedom',fontsize=30)
    #plt.clim(20000, 100000)
    plt.title("Human Freedom (score)", fontsize=30)
    plt.show()
plt.figure(figsize=(30,30))
mapWorld()


# Here you can see boxplots grouped by regions.

# In[ ]:


lst = ['pf_rol_procedural','pf_rol','pf_score','ef_legal','ef_trade','ef_score','hf_score']

def reg(x):
    if x=='Middle East & North Africa':
        res = 'Mid East & Nor Afr'
    elif x=='Latin America & the Caribbean':
        res = 'Lat Amer & Car'
    elif x=='Caucasus & Central Asia':
        res = 'Cauc & Cen Asia'
    elif x=='Sub-Saharan Africa':
            res = 'Sub-Sah Africa'
    else:
        res=x
    return res
data_bx = data18
data_bx['region'] = data_bx.region.apply(reg)

plt.figure(figsize=(30,10))
sns.set(style="white",font_scale=1.5)
sns.boxplot(x='region',y='hf_score',data=data_bx);
sns.swarmplot(x='region',y='hf_score',data=data_bx,color=".25");


# Choose the most influential features: (to the Human Freedom score)
# 
# pf_rol_procedural  - Procedural justice <br>
# pf_rol  - Rule of law <br>
# pf_score  - Personal Freedom (score) <br>
# ef_legal  - Legal system and property rights <br>
# ef_score  - Economic Freedom (score) <br>
# ef_trade  - Freedom to trade internationally <br>

# In[ ]:


data18.corr()[abs(data18.corr())>0.72]['hf_score'].dropna()[['pf_rol_procedural', 'pf_rol_civil', 'pf_rol_criminal', 'pf_rol',       'pf_ss', 'pf_expression_influence',       'pf_expression_control', 'pf_expression', 'pf_score',        'ef_legal', 'ef_trade', 'ef_score']]


# Consider map, where color means Human Freedom and size - Personal Freedom:

# In[ ]:


def mapWorld(col1,size2,title3,label4,metr=100,colmap='hot'):
    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,            llcrnrlon=-110,urcrnrlon=180,resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90,91.,30.))
    m.drawmeridians(np.arange(-90,90.,60.))
    
    #m.drawmapboundary(fill_color='#FFFFFF')
    lat = data_full['CapitalLatitude'].values
    lon = data_full['CapitalLongitude'].values
    a_1 = data_full[col1].values
    if size2:
        a_2 = data_full[size2].values
    else: a_2 = 1
    #300*a_2
    m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,linewidth=1,edgecolors='black',cmap=colmap, alpha=1)
    
    #m.fillcontinents(color='#FFFFFF',lake_color='#FFFFFF',alpha=0.3)
    cbar = m.colorbar()
    cbar.set_label(label4,fontsize=30)
    #plt.clim(20000, 100000)
    plt.title(title3, fontsize=30)
    plt.show()
plt.figure(figsize=(30,30))
mapWorld('hf_score','pf_score',"Human Freedom (score)", 'Human Freedom')


# Now color - Economic Freedom (score), size - Freedom to trade internationally:

# In[ ]:


plt.figure(figsize=(30,30))
mapWorld('ef_score','ef_trade',"Economic Freedom (score)", 'Economic Freedom',metr=100,colmap='viridis')


# ## Terrorism

# Here we can see most dangerous places because of terrorism:

# In[ ]:


plt.figure(figsize=(30,30))
mapWorld(col1='pf_ss_disappearances_injuries',size2=False,title3="Terrorism fatalities",         label4='Terrorism fatalities',metr=700,colmap='viridis')


# Here we can see boxplot graphic of Terrorism injuries. We can see that for this distribution, boxplot is extremely unfortunate.

# In[ ]:


plt.figure(figsize=(30,10))
sns.set(style="white",font_scale=1.5)
sns.boxplot(x='region',y='hf_score',data=data_bx);
sns.swarmplot(x='region',y='pf_ss_disappearances_injuries',data=data_bx,color=".25");


# For this map, color - Procedural justice, size - Rule of law to trade internationally:

# In[ ]:


plt.figure(figsize=(30,30))
mapWorld(col1='pf_rol_procedural',size2='pf_rol',title3="Procedural justice",         label4='Procedural justice',metr=200,colmap='viridis')


# ##  Business part

# In[ ]:


lstt = ['ef_regulation_labor','ef_regulation_business_adm','ef_regulation_business_bureaucracy','ef_regulation_business_start','ef_regulation_business_bribes','ef_regulation_business_licensing','ef_regulation_business_compliance','ef_regulation_business','ef_regulation']


# In[ ]:


data18.corr()[lstt][-3:]


# Here you can see description of business parameters:

# ef_regulation_business_bureaucracy - Bureaucracy costs <br>
# ef_regulation_business_start - Starting a business <br>
# ef_regulation_business_bribes - Extra payments/bribes/favoritism <br>
# ef_regulation_business - Business regulations <br>
# ef_regulation - Regulation <br>

# Now draw spider plot with business features for all regions. <br>
# Main model of this grephics I take from this kernel (https://www.kaggle.com/dczerniawko/fifa19-analysis). Many thanks to the author!

# In[ ]:


idx=1
def ff(x):
    return dict_val[x]
plt.figure(figsize=(20,50))
qqq = ['ef_regulation_business_bureaucracy','ef_regulation_business_start','ef_regulation_business_bribes','ef_regulation_business','ef_regulation']
my_data = data_full[qqq+['region']].dropna()
my_data.columns = ['Bureaucracy costs','Starting a business','Extra payments','Business regulations',             'Regulation','region']
qq_1 = ['Bureaucracy costs','Starting a business','Extra payments','Business regulations','Regulation','region']
for position_name, features in my_data.groupby(my_data['region'])[qq_1].median().iterrows():
    feat = dict(features)
    
    categs=feat.keys()
    N = len(categs)

    values = list(feat.values())
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(9, 3, idx, polar=True)

    plt.xticks(angles[:-1], categs, color='grey', size=15)
    ax.set_rlabel_position(0)
    plt.yticks([2,5,10], ["2","5","10"], color="grey", size=15)
    plt.ylim(0,10)
    
    plt.subplots_adjust(hspace = 0.5)
    ax.plot(angles, values, linewidth=3, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(position_name, size=15, y=1.1)
    
    idx += 1


# ## Women's security

# pf_ss_women_missing - Missing women <br>
# pf_ss_women_inheritance_widows - Inheritance rights for widows <br>
# pf_ss_women_inheritance_daughters - Inheritance rights for daughters <br>
# pf_ss_women_inheritance - Inheritance <br>
# pf_ss_women - Women's security <br>

# For this map, color - Women's security, size -  Missing women:

# In[ ]:


plt.figure(figsize=(30,30))
mapWorld(col1='pf_ss_women_missing',size2='pf_ss_women',title3="Missing women",         label4='Missing women',metr=100,colmap='viridis')


# In[ ]:


def ff(x):
    return dict_val[x]
plt.figure(figsize=(20,50))
qqq = ['pf_ss_women_missing','pf_ss_women_inheritance_widows','pf_ss_women_inheritance_daughters',       'pf_ss_women_inheritance','pf_ss_women','pf_ss_women_fgm']
my_data = data_full[qqq+['region']].dropna()
my_data.columns = ['Missing women','Inheritance rights for widows','Inheritance rights for daughters','Inheritance',"Women's security",'Female genital mutilation','region']
qq_1 = list(my_data.columns)
def spyder_plot(qq_1):
 idx=1
 for position_name, features in my_data.groupby(my_data['region'])[qq_1].median().iterrows():
    feat = dict(features)
    
    categs=feat.keys()
    N = len(categs)

    values = list(feat.values())
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(9, 3, idx, polar=True)
    plt.xticks(angles[:-1], categs, color='grey', size=15)
    ax.set_rlabel_position(0)
    plt.yticks([2,5,10], ["2","5","10"], color="grey", size=15)
    plt.ylim(0,10)
    
    plt.subplots_adjust(hspace = 0.5)
    ax.plot(angles, values, linewidth=3, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    plt.title(position_name, size=15, y=1.1)
    idx += 1
spyder_plot(qq_1)


# ## Freedom of association

# pf_association_association - Freedom of association <br>
# pf_association_assembly - Freedom of assembly <br>
# pf_association_political_establish - Freedom to establish political parties <br>
# pf_association_political_operate - Freedom to operate political parties <br>
# pf_association_political - Freedom to establish and operate political parties <br>

# For this map, color - Freedom of association:

# In[ ]:


plt.figure(figsize=(30,30))
mapWorld(col1='pf_association_association',size2=False,title3="Freedom of association",         label4='Freedom of association',metr=500,colmap='viridis')


# In[ ]:


idx=1
def ff(x):
    return dict_val[x]
plt.figure(figsize=(20,50))
qqq = ['pf_association_association','pf_association_assembly','pf_association_political_establish','pf_association_political_operate','pf_association_political']
my_data = data_full[qqq+['region']].dropna()
spyder_plot(qqq)


# ## Size of Government

# ef_government_consumption - Government consumption <br>
# ef_government_transfers - Transfers and subsidies <br>
# ef_government_enterprises - Government enterprises and investments <br>
# ef_government_tax_payroll - Top marginal income and payroll tax rate <br>
# ef_government - Size of government <br>

# For this map, color - Size of government 

# In[ ]:


plt.figure(figsize=(30,30))
mapWorld(col1='ef_government',size2=False,title3="",         label4='Size of government ',metr=500,colmap='viridis')


# In[ ]:


plt.figure(figsize=(20,50))
qqq = ['ef_government_consumption','ef_government_transfers','ef_government_enterprises','ef_government_tax_payroll','ef_government']
my_data = data_full[qqq+['region']].dropna()
spyder_plot(qqq)


# ## Money

# ef_money_growth - Money growth <br>
# ef_money_sd - Standard deviation of inflation <br>
# ef_money_inflation - Inflation: most recent year <br>
# ef_money_currency - Freedom to own foreign currency bank account <br>
# ef_money - Sound money <br>

# In[ ]:


plt.figure(figsize=(30,30))
mapWorld(col1='ef_money_inflation',size2=False,title3="Inflation: most recent year",         label4='',metr=500,colmap='viridis')


# In[ ]:


plt.figure(figsize=(20,50))
qqq = ['ef_money_growth','ef_money_sd','ef_money_inflation','ef_money_currency','ef_money']
my_data = data_full[qqq+['region']].dropna()
spyder_plot(qqq)


# Thank you for reading! I hope this kernel was helpful for you. <br>
# If you like same map visualization, you can see my other kernels:  <br>
# https://www.kaggle.com/nikitagrec/kernels
