#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from numpy.random import seed
from numpy.random import randn
from numpy import mean
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statistics
import seaborn as sns
import scipy
from math import sqrt
from scipy.stats import pearsonr
from scipy.stats import t
import plotly.graph_objects as go
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import shapiro
sns.set(style='white', font_scale=1.1)


data = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")

# Any results you write to the current directory are saved as output.


# In[ ]:


dataclean = data.dropna()


# ## data.dropna() digunakan untk menghapus row yang memiliki missing value dari objek data dan menyimpannya pada objek (dataclean).

# In[ ]:


dataclean.head(100)


# ### Di bawah ini mulai nomor 1

# In[ ]:


price_mean = dataclean['Price'].mean()
distance_mean = dataclean['Distance'].mean()
land_mean = dataclean['Landsize'].mean()
build_mean = dataclean['BuildingArea'].mean()

print ('Mean for Price : ' + str(price_mean))
print ('Mean for Distance : ' + str(distance_mean))
print ('Mean for Land Size : ' + str(land_mean))
print ('Mean for Building Area : ' + str(build_mean))


# In[ ]:


price_std = dataclean['Price'].std()
distance_std = dataclean['Distance'].std()
land_std = dataclean['Landsize'].std()
build_std = dataclean['BuildingArea'].std()

print ('Standard Deviation for Price : ' + str(price_std))
print ('Standard Deviation for Distance : ' + str(distance_std))
print ('Standard Deviation for Land Size : ' + str(land_std))
print ('Standard Deviation for Building Area : ' + str(build_std))


# In[ ]:


price_max = dataclean['Price'].max()
distance_max = dataclean['Distance'].max()
land_max = dataclean['Landsize'].max()
build_max = dataclean['BuildingArea'].max()

print ('Max Value for Price : ' + str(price_max))
print ('Max Value for Distance : ' + str(distance_max))
print ('Max Value for Land Size : ' + str(land_max))
print ('Max Value for Building Area : ' + str(build_max))


# In[ ]:


price_min = dataclean['Price'].min()
distance_min = dataclean['Distance'].min()
land_min = dataclean['Landsize'].min()
build_min = dataclean['BuildingArea'].min()

print ('Min Value for Price : ' + str(price_min))
print ('Min Value for Distance : ' + str(distance_min))
print ('Min Value for Land Size : ' + str(land_min))
print ('Min Value for Building Area : ' + str(build_min))


# In[ ]:


price_q1 = np.quantile(dataclean['Price'], .25)
distance_q1 = np.quantile(dataclean['Distance'], .25)
land_q1 = np.quantile(dataclean['Landsize'], .25)
build_q1 = np.quantile(dataclean['BuildingArea'], .25)


print ('Q1 Value for Price : ' + str(price_q1))
print ('Q1 Value for Distance : ' + str(distance_q1))
print ('Q1 Value for Land Area : ' + str(land_q1))
print ('Q1 Value for Building Area : ' + str(build_q1))


# In[ ]:


price_q2 = np.quantile(dataclean['Price'], .50)
distance_q2 = np.quantile(dataclean['Distance'], .50)
land_q2 = np.quantile(dataclean['Landsize'], .50)
build_q2 = np.quantile(dataclean['BuildingArea'], .50)


print ('Median Value for Price : ' + str(price_q2))
print ('Median Value for Distance : ' + str(distance_q2))
print ('Median Value for Land Area : ' + str(land_q2))
print ('Median Value for Building Area : ' + str(build_q2))


# In[ ]:


price_q3 = np.quantile(dataclean['Price'], .75)
distance_q3 = np.quantile(dataclean['Distance'], .75)
land_q3 = np.quantile(dataclean['Landsize'], .75)
build_q3 = np.quantile(dataclean['BuildingArea'], .75)


print ('Q3 Value for Price : ' + str(price_q3))
print ('Q3 Value for Distance : ' + str(distance_q3))
print ('Q3 Value for Land Area : ' + str(land_q3))
print ('Q3 Value for Building Area : ' + str(build_q3))


# In[ ]:


tipe_h = dataclean.loc[dataclean['Type'] == 'h']
tipe_u = dataclean.loc[dataclean['Type'] == 'u']
tipe_t = dataclean.loc[dataclean['Type'] == 't']


# di atas adalah deklarasi untuk tipe rumah.

# ## Nomer 2 di bawah ini

# In[ ]:


plt.boxplot(tipe_h.Price, showmeans=True)
plt.ylim(0, 4000000)
print ('Q1 : ' + str(np.quantile(tipe_h['Price'], .25)))
print ('Q2 : ' + str(np.quantile(tipe_h['Price'], .50)))
print ('Q3 : ' + str(np.quantile(tipe_h['Price'], .75)))
print ('Mean : ' + str(tipe_h['Price'].mean()))


# In[ ]:


plt.boxplot(tipe_h.BuildingArea, vert=False)
plt.xlim(0, 500)
print ('Q1 : ' + str(np.quantile(tipe_h['BuildingArea'], .25)))
print ('Q2 : ' + str(np.quantile(tipe_h['BuildingArea'], .50)))
print ('Q3 : ' + str(np.quantile(tipe_h['BuildingArea'], .75)))
print ('Mean : ' + str(tipe_h['BuildingArea'].mean()))


# In[ ]:


plt.boxplot(tipe_u.Price)
plt.ylim(0, 1500000)
print ('Q1 : ' + str(np.quantile(tipe_u['Price'], .25)))
print ('Q2 : ' + str(np.quantile(tipe_u['Price'], .50)))
print ('Q3 : ' + str(np.quantile(tipe_u['Price'], .75)))
print ('Mean : ' + str(tipe_u['Price'].mean()))


# In[ ]:


plt.boxplot(tipe_u.BuildingArea, vert=False)
plt.xlim(0, 250)
print ('Q1 : ' + str(np.quantile(tipe_u['BuildingArea'], .25)))
print ('Q2 : ' + str(np.quantile(tipe_u['BuildingArea'], .50)))
print ('Q3 : ' + str(np.quantile(tipe_u['BuildingArea'], .75)))
print ('Mean : ' + str(tipe_u['BuildingArea'].mean()))


# In[ ]:


plt.boxplot(tipe_t.Price)
plt.ylim(0, 3000000)
print ('Q1 : ' + str(np.quantile(tipe_t['Price'], .25)))
print ('Q2 : ' + str(np.quantile(tipe_t['Price'], .50)))
print ('Q3 : ' + str(np.quantile(tipe_t['Price'], .75)))
print ('Mean : ' + str(tipe_t['Price'].mean()))


# In[ ]:


plt.boxplot(tipe_t.BuildingArea, vert=False)
print ('Q1 : ' + str(np.quantile(tipe_t['BuildingArea'], .25)))
print ('Q2 : ' + str(np.quantile(tipe_t['BuildingArea'], .50)))
print ('Q3 : ' + str(np.quantile(tipe_t['BuildingArea'], .75)))
print ('Mean : ' + str(tipe_t['BuildingArea'].mean()))


#         **Nomor 3 sebagai berikut******

# In[ ]:


region_WM = dataclean.loc[dataclean['Regionname'] == 'Western Metropolitan']
region_EM = dataclean.loc[dataclean['Regionname'] == 'Eastern Metropolitan']
region_NM = dataclean.loc[dataclean['Regionname'] == 'Northern Metropolitan']
region_SM = dataclean.loc[dataclean['Regionname'] == 'Southern Metropolitan']
##Digunakan untuk membuat method yang dapat dipanggil


# In[ ]:


plt.boxplot(region_WM['Price']) ## Price Western Metro


# In[ ]:


plt.boxplot(region_WM['BuildingArea']) ##Building Area Western Metro


# In[ ]:


plt.boxplot(region_EM['Price']) ## Price Eastern Metro


# In[ ]:


plt.boxplot(region_EM['BuildingArea'])##Building Area Eastern Metro


# In[ ]:


plt.boxplot(region_NM['Price']) ## Price Northern Metro


# In[ ]:


plt.boxplot(region_NM['BuildingArea'])##Building Area Northern Metro


# In[ ]:


plt.boxplot(region_SM['Price']) ## Price Southern Metro


# In[ ]:


plt.boxplot(region_SM['BuildingArea']) ## Building Area Southern Metro


# ## Mulai dari sini pengerjaan soal nomor 4
# 

# In[ ]:


price = dataclean['Price']
dist = dataclean['Distance']

plt.scatter(price, dist, edgecolors = 'r')
plt.xlabel('Price')
plt.ylabel('Distance')
plt.title('Correlation between Price and Distance')
plt.show()
## Price VS Distance


# ## Mulai dari sini pengerjaan soal nomor 5

# In[ ]:


buildarea = dataclean['BuildingArea']
dist = dataclean['Distance']

plt.scatter(buildarea, dist, edgecolors = 'y')
plt.xlim(0,1000)
plt.xlabel('BuildingArea')
plt.ylabel('Distance')
plt.title('Correlation between BuildingAreaand Distance')
plt.show()


# ## Mulai dari sini pengerjaan soal nomor 6

# 

# In[ ]:


## Menghitung harga rata-rata tiap penjual
nels = dataclean.loc[dataclean['SellerG'] == 'Nelson']
nels_mean = nels['Price'].mean()
print ('Nelson Mean : ' + str(nels_mean))

barry = dataclean.loc[dataclean['SellerG'] == 'Barry']
barry_mean = barry['Price'].mean()
print ('Barry Mean : ' + str(barry_mean))

hock = dataclean.loc[dataclean['SellerG'] == 'hockingstuart']
hock_mean = hock['Price'].mean()
print ('hockingstuart Mean : ' + str(hock_mean))

ray = dataclean.loc[dataclean['SellerG'] == 'Ray']
ray_mean = ray['Price'].mean()
print ('Ray Mean : ' + str(ray_mean))


# In[ ]:




    nels_price = nels['Price']
    barry_price = barry['Price']
    hock_price = hock['Price']
    ray_price = ray['Price']


    length_nels = len(nels['Price'])
    length_barry = len(barry['Price'])
    length_hock = len(hock['Price'])
    length_ray = len(ray['Price'])

    std_nels = nels_price.std()
    std_barry = barry_price.std()
    std_hock = hock_price.std()
    std_ray = ray_price.std()


    se_nels = std_nels/sqrt(length_nels)
    se_barry = std_barry/sqrt(length_barry)
    se_hock = std_hock/sqrt(length_hock)
    se_ray = std_ray/sqrt(length_ray)


# In[ ]:


print ('N : ' + str(length_nels), 
       str(length_barry), str(length_hock), str(length_ray))


# In[ ]:


print ('Ray Std. Dev : ' + str(std_ray))
print ('Nelson Std. Dev : ' + str(std_nels))
print ('Barry Std. Dev : ' + str(std_barry))
print ('Hockingstuart Std. Dev : ' + str(std_hock))


# In[ ]:


print ('Ray Std. Error : ' + str(se_ray))
print ('Nelson Std. Error : ' + str(se_nels))
print ('Barry Std. Error : ' + str(se_barry))
print ('Hockingstuart Std. Error : ' + str(se_hock))


# In[ ]:


stat, p = shapiro(nels['Price'])
print ('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
else:
        print('Sample does not look Gaussian (reject H0)')


# In[ ]:


from scipy.stats import chi2

data2 = dataclean[dataclean.SellerG.str.contains("Nelson|Barry|hocking|Ray")]
data2 = data2[~dataclean.SellerG.str.contains("/")]
data2 = data2[dataclean.Regionname.str.contains("Western Metropolitan|Eastern Metropolitan|Northern Metropolitan|Southern Metropolitan")]
data2 = data2[~dataclean.Regionname.str.contains("/|South-Eastern Metropolitan")]
data2 = data2[['Regionname', 'SellerG']]

contingency_table=pd.crosstab(data2["Regionname"],data2["SellerG"])
print('contingency_table :-\n',contingency_table)

Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)

import scipy.stats
b=scipy.stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)

no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
df=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:",df)

alpha=0.05

chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:",chi_square_statistic)

critical_value=chi2.ppf(q=1-alpha,df=df)
print('critical_value:',critical_value)

p_value=1-chi2.cdf(x=chi_square_statistic,df=df)
print('p-value:',p_value)

print('Significance level: ',alpha)

if chi_square_statistic>=critical_value:
    print("There is a relationship between 2 categorical variables")
else:
    print("There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("There is a relationship between 2 categorical variables")
else:
    print("There is no relationship between 2 categorical variables")



