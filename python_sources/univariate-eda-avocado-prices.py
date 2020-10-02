#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as sp


# In[ ]:


#Read in data

avocadata = pd.read_csv("../input/avocado.csv")

#remove duplicate information
bools = np.any([(avocadata['region'] != "TotalUS")], axis=0)
avocadata = avocadata.loc[bools]

#select the price vector
arrayvocado = avocadata["AveragePrice"]


# __Data overview:__
# 
# The dataset was downloaded from [Kaggle](https://www.kaggle.com/neuromusic/avocado-prices) and was orginally compiled from data from the [Hass Avocado Board](http://www.hassavocadoboard.com/retail/volume-and-price-data) website. It represents retail scan data for the average price of a single avocado in cities and regions in the US between 2015 to 2018. There was a minor issue of duplication of information, as summary information for the entire US was included in the table, but this was easily removed and seems to be the only carpentry needed.

# In[ ]:


#Graphical EDA

plt.subplot(2,1,1)
sns.distplot(arrayvocado, axlabel = False, rug = True,
             kde_kws={"label": "Kernel Density", "color" : 'k'},
             hist_kws={"label": "Histogram"},
             rug_kws={"label": "Rug plot"})
plt.ylabel("Density")
plt.yticks([0,0.25,0.5,0.75,1])
plt.xticks([])

plt.subplot(2,1,2)
plt.boxplot(arrayvocado, vert = False)

plt.xlabel("Average avocado price ($)")
plt.xlim(0,3.5)
plt.yticks([])
plt.show()


# In[ ]:


#Data could be multi-modal, check with histogram of log-price.

sns.distplot((np.log(arrayvocado)))
plt.xlabel("Log of average avocado price (log($))")
plt.ylabel("Density")
plt.show()


# __Graphical EDA:__
# 
# Several plots were generated to visualise the data's distribution and they showed that it is slightly non-normal and right skewed. Both the initial histogram and the log-price histogram indicated that the data may be multimodal, but at this stage it is unclear. Multivariate analysis could help determine whether or not there are subpopulations in the dataset that are causing multiple modes in the price variable.

# In[ ]:


#Non-graphical EDA

print("Mean = " + str(np.mean(arrayvocado)))
print("Median = " + str(np.median(arrayvocado)))
print("Mode = " + str(sp.mode(arrayvocado)[0][0]))

#second central moment, variance
print("\nBiased variance = " + str(np.var(arrayvocado)))
print("Unbiased variance = " + str(np.var(arrayvocado, ddof=1)))

#third standardized moment, skew
sd = np.sqrt(np.var(arrayvocado))
skew = sp.stats.moment(arrayvocado,3)/(sd**3)
print("\nSkew = " + str(skew))

#fourth standardized moment, kurtosis
kurtosis = sp.stats.moment(arrayvocado,4)/(sd**4)
print("Excess kurtosis = " + str(kurtosis - 3))


# __Non-graphical EDA:__
# 
# The mean avocado price is \$1.41, slightly above the median of $1.37, and the variance of the data is ~ 0.16. Confirming suspicions of non-normality, the data was found to be slightly right-skewed (skew = 0.58) and leptokurtic (excess kurtosis = 0.31).
