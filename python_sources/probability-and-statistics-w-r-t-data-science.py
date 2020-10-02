#!/usr/bin/env python
# coding: utf-8

# # **SOME PROBABILITY AND STATISTICS W.R.T DATA SCIENCE**

# # **1. CONFIDENCE INTERVAL USING BOOTSTRAPPING**

# IMPORT NECESSARY LIBRARIES

# In[ ]:


import numpy
from sklearn.utils import resample
from matplotlib import pyplot


# NEXT I HAVE X AS A SAMPLE FROM POPULATION WITH 10 VALUES (n=10)
# I AM GOING TO FORM 500 BOOTSTRAP SAMPLES (boot_samples=500)

# In[ ]:


x=numpy.array([60,76,54,66,72,80,93,48,67,55])
boot_samples=1000
n=int (len(x))


# NOW WE FORM 500 BOOTSTRAP SAMPLES OF SIZE 10 USING sklearn.utils.resample
# NEXT WE CALCULATE MEDIAN OF EACH OF THESE SAMPLES AND APPEND IT IN THE medians LIST

# In[ ]:


medians=list()
for i in range(boot_samples):
    s=resample(x,n_samples=n)
    med=numpy.median(s)
    medians.append(med)


# NOW PLOTTING HISTOGRAM

# In[ ]:


pyplot.hist(medians)


# NOW CALCULATING CONFIDENCE INTERVAL WITH conf 0.95 i.e WITH 95% CONFIDENCE

# In[ ]:


conf=0.95
L=((1-conf)/2)*100
lower=numpy.percentile(medians,L)
U=(conf+(1-conf)/2)*100
upper=numpy.percentile(medians,U)


# FINALLY WE HAVE THE CONFIDENCE INTERVAL OF MEDIAN

# In[ ]:


print ("The confidence interval is : [",lower,",",upper,"]")


# # 2. KS TEST FOR SIMILARITY OF TWO DISTRIBUTIONS

# IMPORT LIBRARIES

# In[ ]:


import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


# NEXT WE CREATE A NORMAL RANDOM VARIABLE X AND PLOT ITS P.D.F USING K.D.E

# In[ ]:


x=stats.norm.rvs(size=1000)
sns.set_style('whitegrid')
sns.kdeplot(np.array(x),bw=0.5)


# 1. NOW WE USE KSTEST TO TEST WHETHER THE RANDOM VARIABLE ACTUALLY FOLLOWS NORMAL DISTRIBUTION OR NOT
# 2. THE OUTPUT IS THE TEST STATISTIC AND THE P VALUE
# 3. SINCE P VALUE IS HIGH (and greater than test statitic), THE DISTRIBUTION IS NORMAL

# In[ ]:


stats.kstest(x,'norm')


# NEXT WE CREATE A UNIFORM RANDOM VARIABLE AND PLOT ITS PDF

# In[ ]:


y=np.random.uniform(0,1,10000)
sns.kdeplot(np.array(y),bw=0.1)


# 1. NOW WE USE KSTEST TO TEST WHETHER THE RANDOM VARIABLE FOLLOWS NORMAL DISTRIBUTION OR NOT
# 2. THE OUTPUT IS THE TEST STATISTIC AND THE P VALUE
# 3. SINCE P VALUE IS ZERO, THE DISTRIBUTION OF Y IS NOT NORMAL

# In[ ]:


stats.kstest(y,'norm')

