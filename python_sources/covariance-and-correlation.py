#!/usr/bin/env python
# coding: utf-8

# ## Covariance, Pearson correlation between two columns And Chi-Square test.

# In[ ]:


#import libraries
import numpy as np
import pandas as pd #for DataFrame
from scipy import stats #for pearson correlation


# In[ ]:


list_1 = {'H': pd.Series([34,76,21,98,12,98,55,76,22,43]),
          'W': pd.Series([65,32,77,87,54,22,90,78,65,33]),
          'Marks':pd.Series([56,76,34,78,56,76,98,76,87,46]),
         }
list_1 = pd.DataFrame(list_1)
list_1


# In[ ]:


covariance = list_1.cov(min_periods=None)
covariance


# Covariance provides insight into how two variables are related to one another.

# More precisely, covariance refers to the measure of how two random variables in a data set will change together. 

# A positive covariance means that the two variables at hand are positively related, and they move in the same direction. 
# 
# A negative covariance means that the variables are inversely related, or that they move in opposite directions.

# In[ ]:


list_of_H = list_1['H']


# In[ ]:


list_of_W = list_1['W']
list_of_Marks = list_1['Marks']


# In[ ]:


corr = stats.pearsonr(list_of_H,list_of_W)


# In[ ]:


print('Correlation between H and W column: ',corr)


# The Pearson correlation coefficient measures the linear relationship between two datasets. The calculation of the p-value relies on the assumption that each dataset is normally distributed. 

# Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply an exact linear relationship. Positive correlations imply that as x increases, so does y. Negative correlations imply that as x increases, y decreases.

# The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one computed from these datasets.

# In[ ]:


corr_1 = stats.pearsonr(list_of_H,list_of_Marks)
print('Correlation between H and Marks column: ',corr_1)


# In[ ]:


corr_2 = stats.pearsonr(list_of_W,list_of_Marks)
print('Correlation between W and Marks column: ',corr_2)


# $$ x^2 = \sum_{all cells} \frac{(observed - expected)^2}{expected} $$

# $$ expected count = \frac{rowTotal * colTotal}{overallTotal} $$

# $$ Degree of freedom = ((NumberOfRows) - 1)*((NumberOfColumns) - 1)  $$

# In[ ]:


list_1


# In[ ]:


chi_square_test = stats.chisquare(list_of_H, 
                                  list_of_W,
                                  axis=0)


# In[ ]:


print('chi-square test of given dataset is: ',chi_square_test)


# In[ ]:


obs = np.array([[71.0,154.0,398.0],[4992.0,2808.0,2737.0]])
exp = np.array([[282.6,165.4,175.0],[4780.4,2796.6,2960.0]])


# In[ ]:


print('chi-square of given set matrix: ',stats.chisquare(f_obs=obs,f_exp=exp,axis=None))


# In[ ]:




