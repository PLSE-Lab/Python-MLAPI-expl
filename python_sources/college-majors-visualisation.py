#!/usr/bin/env python
# coding: utf-8

# #### Importing Libraries 
# 

# In[ ]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# #### Importing The Data

# In[ ]:


recent_grads = pd.read_csv('../input/recent-graduates.csv')
recent_grads.iloc[0]


# In[ ]:


recent_grads.head(5)


# In[ ]:


recent_grads.tail(3)


# #### Summary Statistics

# In[ ]:


recent_grads.describe()


# #### Cleaning Missing Values

# In[ ]:


raw_data_count = recent_grads.shape
raw_data_count


# In[ ]:


recent_grads = recent_grads.dropna()


# In[ ]:


cleaned_data_count = recent_grads.shape
cleaned_data_count


# In[ ]:


cleaned_data_count


# #### Visual Analysis Of Different Columns

# In[ ]:


X = recent_grads['Sample_size']
Y = recent_grads['Median']

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

a, b = best_fit(X, Y)

plt.scatter(X, Y, c='b')
yfit = [a + b * xi for xi in X]
plt.title('Sample size vs. Median')
plt.xlabel('Sample size')
plt.ylabel('Median')
plt.plot(X, yfit)


# ##### Do students in more popular majors make more money?

# From the data, we can see there is a weak negative linear correlation between the majors which are more popular and median income. Therefore students studying more popular majors dont make more money.

# In[ ]:


recent_grads.plot(x='Sample_size', y='Unemployment_rate', kind='scatter',title="Sample size vs. Unemployment rate")


# In[ ]:


X = recent_grads['Full_time']
Y = recent_grads['Median']

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

a, b = best_fit(X, Y)

plt.scatter(X, Y, c='b')
yfit = [a + b * xi for xi in X]
plt.title('Full_time vs. Median')
plt.xlabel('Full_time')
plt.ylabel('Median')
plt.plot(X, yfit)


# ##### Is there any link between the number of full-time employees and median salary?

# There is a weak negative linear correlation between full-time employees and median income. As the number of 
# full-time employees increases the median salary decreases.

# In[ ]:


recent_grads.plot(x='ShareWomen', y='Unemployment_rate', kind='scatter',title="ShareWomen vs. Unemployment_rate")


# In[ ]:


recent_grads.plot(x='Men', y='Median', kind='scatter',title="Men vs. Median")


# In[ ]:


X = recent_grads['Women']
Y = recent_grads['Median']

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

a, b = best_fit(X, Y)

plt.scatter(X, Y, c='b')
yfit = [a + b * xi for xi in X]
plt.title('Women vs. Median')
plt.xlabel('Women')
plt.ylabel('Median')
plt.plot(X, yfit)


# ##### Do students that majored in subjects that were majority female make more money?

# From the regression line we can see there is a moderately negative linear correlation between students that majored in 
# subjects with a mojarity of female students and median income.

# #### Visualising Data Using Histograms

# In[ ]:


recent_grads['Sample_size'].hist(bins=25, range=(0,5000), color='m')


# In[ ]:


recent_grads['Median'].hist(bins=25, color='b')


# ##### What's the most common median salary range?

# From the data the most common median salary range is around $30,000.

# In[ ]:


recent_grads['Employed'].hist(bins=25, range=(0,5000), color='black')


# In[ ]:


recent_grads['Full_time'].hist(bins=25, range=(0,5000), color='g')


# In[ ]:


recent_grads['ShareWomen'].hist(bins=25, color='violet')


# ##### What percent of majors are predominantly male? Predominantly female?

# In[ ]:


mostly_female = recent_grads['ShareWomen'] > 0.5
mostly_female.value_counts()


# In[ ]:


76/96 * 100


# From the data we can see approximately 80% of majors are predominately female, the remaining 20% are predominately male.

# In[ ]:


recent_grads['Unemployment_rate'].hist(bins=25, color='yellow')


# In[ ]:


recent_grads['Men'].hist(bins=25, range=(0,5000), color='c')


# In[ ]:


recent_grads['Women'].hist(bins=25, range=(0,5000), color='pink')


# #### Scatter Matrices 

# In[ ]:


from pandas.plotting import scatter_matrix

scatter_matrix(recent_grads[['Sample_size', 'Median']], figsize=(20,20), c='black')


# In[ ]:


scatter_matrix(recent_grads[['Sample_size', 'Median', 'ShareWomen']], figsize=(20,20), c='black')


# ##### Do students that majored in subjects that were majority female make more money?

# From the scatter plot matrix ShareWomen-Median is moderately negatively linearly correlated. Therefore Majors that are 
# more popular with women tend to pay less.

# #### Finding The Most Popular Majors

# In[ ]:


recent_grads[:10].plot.barh(x='Major',y='ShareWomen', color='c');
plt.title('Proportion Of Women In The Ten Highest Paying Majors')
plt.xlabel('Percentage')
plt.ylabel('Major')


# In[ ]:


end = len(recent_grads) - 10
recent_grads[end:].plot.barh(x='Major',y='ShareWomen', color='c');
plt.title('Proportion Of Women In The Ten Lowest Paying Majors')
plt.xlabel('Percentage')
plt.ylabel('Major')


# Women are under represented in the the highest paying majors, and over represented in the lowest paying majors.

# In[ ]:


recent_grads[:10].plot.barh(x='Major',y='Unemployment_rate');
plt.title('Proportion Of Unemployed In The Ten Highest Paying Majors')
plt.xlabel('Percentage')
plt.ylabel('Major')


# In[ ]:


end = len(recent_grads) - 10
recent_grads[end:].plot.barh(x='Major',y='ShareWomen');
plt.title('Proportion Of Unemployed In The Ten Lowest Paying Majors')
plt.xlabel('Percentage')
plt.ylabel('Major')

