#!/usr/bin/env python
# coding: utf-8

# **INVESTIGATING THE RELATIONSHIP BETWEEN POPULATION-BASED LUNG CANCER INCIDENCE RATES AND AIR QUALITY**
# 
# *by: Yasmine Hemmati*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# In[ ]:



data = pd.read_excel("../input/harvarddataset/Data_Set_Final_LTD_Slope_Intercept.xlsx")


# In[ ]:


data


# Link to dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HMOEJO 
# 
# Link to paper with description of Data Set (Lobdell 2011): https://www.researchgate.net/publication/51566166_Data_Sources_for_an_Environmental_Quality_Index_Availability_Quality_and_Utility

# - PM2.5 high (greater than 10.59 mg/m3) vs. low (less than 10.59 mg/m3)
# - Status Variable of 1 indicates the county has high PM2.5 levels and 0 indicates the county has low PM2.5 levels.

# In[ ]:


data.columns


# **Conduct a t-test**
# 
# Null Hypothesis : mean_deaths_lowPM2.5 = mean_deaths_highPM2.5
# 
# Assumptions of a t-test:
# 1. data follows a normal distribution -> we can make a qq plot
# 2. variance of 2 samples are the same -> use Chi-square test for variance to show this
# 3. independence between samples ->  from different counties so this holds
# 4. data is continuous -> upon inspection this holds
# 

# In[ ]:


# Status variable is 1 if PM2.5 levels are high and 0 if they are low    
df_pm25hi = data[data['Status Variable'] == 1]
df_pm25lo = data[data['Status Variable'] == 0]


# **Checking 1. holds**

# In[ ]:


import pylab 
import scipy.stats as stats

stats.probplot(df_pm25hi['Lung Cancer'], dist="norm", plot=pylab)
pylab.show() #there appears to be some outliers at the end but for the most part it follows a straight line


# In[ ]:



stats.probplot(df_pm25lo['Lung Cancer'], dist="norm", plot=pylab)
pylab.show()


# From the two plots we conclude that the data from both samples are follows the normal distribution

# **Checking 2. holds**

# In[ ]:



mean_hi = df_pm25hi['Lung Cancer'].mean()
mean_lo = df_pm25lo['Lung Cancer'].mean()
print(mean_hi, mean_lo)


# In[ ]:


diff_hi= df_pm25hi['Lung Cancer'] - mean_hi #note that subtracting a scalar from a vector works because of broadcasting
diff_lo = df_pm25lo['Lung Cancer'] - mean_lo
# next two lines is calculating var hat
var_hi = np.dot(diff_hi,diff_hi)/(df_pm25hi.shape[0]-1)
var_lo = np.dot(diff_lo,diff_lo)/(df_pm25lo.shape[0]-1)
F_stat = var_hi/var_lo # since under null hypothesis variance of two groups are equal so the variances would cancel out
print(var_hi, var_lo)


# Our test statistic is var_hi/var_lo under the null hypothesis now we need to calcuate the p value.

# In[ ]:


#calculate p-value
# to find degrees of freedom of F distribution 
dfhi = diff_hi.shape[0]-1 
dflo = diff_lo.shape[0]-1


# In[ ]:


import scipy.stats
p_value = scipy.stats.f.cdf(F_stat, dfhi, dflo)
p_value 


# The p-value is very large ( >> 0.05)  hence there is no evidence against the null hypothesis. We accept the null hypothesis that the variance of the two groups (counties with high PM2.5 levels and low PM2.5 levels) are the same.
# 

# **Now that we know the assumptions hold we can proceed with the t-test:**

# In[ ]:


from scipy.stats import ttest_ind
high = df_pm25hi['Lung Cancer']
low = df_pm25lo['Lung Cancer']

stats.ttest_ind(high, low)


# **Conclusion:**
# 
# **The p value is very small ( << 0.05) hence there is strong evidence against the Null hypothesis. Thus, the mean population based lung cancer incidence rates between counties with high PM2.5 and low PM2.5 are not same.**
# 
# Moreover, recall that the mean Lung Cancer incident rate of counties with high PM2.5 levels is 75.77 and the mean Lung Cancer incident rate of counties with low PM2.5 levels is 62.42. Hence it seems reasonable to assume that counties with a high PM2.5 have a higher lung cancer incident rates that those with low PM2.5 levels.

# **According to Google the Most Common air pollutants**:
# 
# Particulate matter (PM10 and PM2.5)
# 
# Ozone (O3)
# 
# Nitrogen dioxide (NO2)
# 
# Carbon monoxide (CO)
# 
# Sulphur dioxide (SO2)
# 
# So I will plot these against the Population Lung Cancer incidence rates to see if there is a positive correlation
# 

# *Plot 'Lung Cancer' values vs 'PM2.5' values*

# In[ ]:


import matplotlib.pyplot as plt
df1 = pd.DataFrame({'LungCancer': data['Lung Cancer'],
                   'PM2.5': data["PM2.5"]})

plt.scatter(data["PM2.5"],data["Lung Cancer"] )


# There appears to be a  moderate positive correlation between PM2.5 Levels and population lung cancer incidence rates

# In[ ]:


corr = np.corrcoef(data["PM2.5"],data["Lung Cancer"])[0,1]
corr


# In[ ]:


r_square = corr**2
r_square


# Therefore the pearsons correlation coefficient is 0.46 and the R^2 value is 0.2107 -> 21.07% of the variation in population based lung cancer incidence rates can be explained by variations in PM2.5 levels.

# *Plot 'Lung Cancer' values vs 'PM10' values*

# In[ ]:


plt.scatter(data["PM10"],data["Lung Cancer"] )


# In[ ]:


np.corrcoef(data["PM10"],data["Lung Cancer"])


# *Plot 'Lung Cancer' values vs 'SO2' values*

# In[ ]:


plt.scatter(data['SO2'],data["Lung Cancer"] )


# In[ ]:


np.corrcoef(data["SO2"],data["Lung Cancer"])[0,1]


# *Plot 'Lung Cancer' values vs 'NO2' values*

# In[ ]:


plt.scatter(data['NO2'],data["Lung Cancer"] )


# In[ ]:


np.corrcoef(data["NO2"],data["Lung Cancer"])[0,1] 


# *Plot 'Lung Cancer' values vs 'O3' values*

# In[ ]:


plt.scatter(data['O3'],data["Lung Cancer"] )


# In[ ]:


np.corrcoef(data["O3"],data["Lung Cancer"])[0,1]


# *Plot 'Lung Cancer' values vs 'PM10' values*

# In[ ]:


plt.scatter(data[ 'CO'],data["Lung Cancer"] )


# In[ ]:


np.corrcoef(data["O3"],data["Lung Cancer"])[0,1]


# For every plot of "Lung Cancer" vs and an air pollutant, there appears to be a very low correlation between lung cancer and that air pollutant except in the case of PM2.5. Moroeover, there is a statistical difference between lung cancer rates for counties with low PM2.5 and counties with high PM2.5.

# **FOR MASSACHUSSETES ONLY**

# In[ ]:


df_ma = data[data['State'] == 'MA']
df_ma


# In[ ]:


df_ma_lo = df_ma[df_ma['Status Variable'] == 0]
df_ma_hi = df_ma[df_ma['Status Variable'] == 1]


# Checking Normality of data

# In[ ]:


stats.probplot(df_ma_hi['Lung Cancer'], dist="norm", plot=pylab)
pylab.show() 


# In[ ]:


stats.probplot(df_ma_lo['Lung Cancer'], dist="norm", plot=pylab)
pylab.show()


# 2. Checking Variance

# In[ ]:


mean_hi = df_ma_hi['Lung Cancer'].mean()
mean_lo = df_ma_hi['Lung Cancer'].mean()
diff_hi= df_ma_hi['Lung Cancer'] - mean_hi #note that subtracting a scalar from a vector works because of broadcasting
diff_lo = df_ma_lo['Lung Cancer'] - mean_lo
# next two lines is S squared
var_hi = np.dot(diff_hi,diff_hi)/(df_ma_hi.shape[0]-1)
var_lo = np.dot(diff_lo,diff_lo)/(df_ma_lo.shape[0]-1)
dfhi = diff_hi.shape[0]-1 
dflo = diff_lo.shape[0]-1
F_stat = var_hi/var_lo # since under null hypothesis variance of two groups are equal so the variances would cancel out


# In[ ]:


p_value = scipy.stats.f.cdf(F_stat, dfhi, dflo)
p_value #greater than 0.05 so we accept the null hypothesis


# In[ ]:


ma_high = df_ma_hi['Lung Cancer']
ma_low = df_ma_lo['Lung Cancer']

stats.ttest_ind(ma_high, ma_low)


# p-value is greater than 0.05 so we accept the null hypothesis that the mean population based lung cancer incidence rates between counties with high PM2.5 and low PM2.5 in MA are the same.

# In[ ]:


plt.scatter(df_ma["PM2.5"],df_ma["Lung Cancer"] )


# In[ ]:


np.corrcoef(df_ma["PM2.5"],df_ma["Lung Cancer"])[0,1]


# Notes regarding the MA analysis: Due to the smaller number of counties, especially counties with a low PM2.5 levels we cannot confidently say that the data follows the normal distribution. Moreover, the F-test used to check equality of variances is dependent upon the data being normal so if this assumption does not hold that test may not be valid. With all the counties our data we could also appeal to the central limit thereom due to the large sample size.
