#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Let's look at the limits of human performance over time

# In[ ]:


# loading data 
lifts = pd.read_csv('../input/openpowerlifting.csv')


# In[ ]:


# i'll take a quick look
lifts.head()


# In[ ]:


# Let's limit this to Men, for now
liftsM = lifts.query("Sex == 'M'")


# In[ ]:


# Let's apply a simple regression to see how age effects lifts
tmp = abs(liftsM[['Age', 'TotalKg']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['TotalKg'])+20)
sns.regplot(tmp['Age'], (tmp['TotalKg']), order=4, scatter_kws={"alpha": 0.05}, line_kws={"color": 'black'})
plt.title('Total KG lifted (10K samples)');


# No surprise here, of course men between 20-30 have the most peak strength. What's most interesting here is the flattening around 50-60 after a steady dip in overall lifts.
# 
# Also, people at 40 can lift as much as people at 21. That was unexpected.

# The bench press. Probably the most familiar lift to the general public. You load a barbell, lie on your back, and lift up. Mostly a chest exercise, though front shoulders and traps are also acting as major supporters.

# In[ ]:


# Looking at bench press
tmp = abs(liftsM[['Age', 'BestBenchKg']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['BestBenchKg'])+20)
sns.regplot(tmp['Age'], (tmp['BestBenchKg']), order=4, scatter_kws={'color': "C2", "alpha": 0.05}, line_kws={"color": 'black'})
plt.title("Bench Press", size=16)
plt.ylabel("Measured in kg");


# The deadlift - king of all lifts. It's a compound movement that requires lifting a weighted barbell from the ground to about hip-level. Great workout for the back and posterior chain as well as grip strength.

# In[ ]:


tmp = abs(liftsM[['Age', 'BestDeadliftKg']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['BestDeadliftKg'])+20)
sns.regplot(tmp['Age'], (tmp['BestDeadliftKg']), order=4, scatter_kws={"alpha": 0.05}, line_kws={"color": 'black'})
plt.title("Deadlifts", size=16)
plt.ylabel("Measured in kg");


# Squats are another good compound movement that requires lifters to load a barbell onto their upper back, squat to a seated position and back up again. It works out core stability as well as legs and posterior chain.

# In[ ]:


tmp = abs(liftsM[['Age', 'BestSquatKg']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['BestSquatKg'])+20)
sns.regplot(tmp['Age'], (tmp['BestSquatKg']), order=4, scatter_kws={'color': "C1","alpha": 0.05}, line_kws={"color": 'black'})
plt.title("Squats", size=16)
plt.ylabel("Measured in kg");


# [Wilks Coefficient](https://en.wikipedia.org/wiki/Wilks_Coefficient) - a devised score of measuring relative strength
# 
# $$\textit{Coeff} = \frac{500}{a + bx + cx^2 + dx^3 + ex^4 + fx^5}$$
# 
# where:
# a=-216.0475144
# / b=16.2606339
# / c=-0.002388645
# / d=-0.00113732
# / e=7.01863E-06
# / f=-1.291E-08
# 
# x = body weight of lifter in kg
# 
# The Coeff is then multiplied by their total lifts (bench + deadlift + squat)  to give a Wilks Score

# In[ ]:


tmp = abs(liftsM[['Age', 'Wilks']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['Wilks'])+20)
sns.regplot(tmp['Age'], (tmp['Wilks']), order=4, scatter_kws={"alpha": 0.05}, line_kws={"color": 'black'})
plt.title("Wilks", size=16)
plt.ylabel("Wilks Score ");


# What did we learn? Overall and maximal strength seem to peak in the late 20s. The outliers who lifted the most weight in pure numbers as well Wilks score are here.
# 
# But more interestingly, the notion of "old man strength" seems to hold up here, as men generally seem to be able to hold their own in their 40s and 50s with younger men in their early 20s across the three lifts. Men in their 50s not uncommonly bench press over 250kg / 550lbs. This is an impressive number regardless of age.

# ## Comparing Lifts

# In[ ]:


plt.figure(figsize=(12,6)), plt.xlim(18,80), plt.title('Regression Line for Powerlifts by Age')
tmp = abs(liftsM[['Age', 'BestDeadliftKg']].dropna().sample(5000))
sns.regplot(tmp['Age'], tmp['BestDeadliftKg'], order=4, scatter=False, label='Deadlift')
tmp = abs(liftsM[['Age', 'BestSquatKg']].dropna().sample(5000))
sns.regplot(tmp['Age'], tmp['BestSquatKg'], order=4, scatter=False, label='Squat')
tmp = abs(liftsM[['Age', 'BestBenchKg']].dropna().sample(5000))
sns.regplot(tmp['Age'], tmp['BestBenchKg'], order=4, scatter=False, label='Bench')
plt.ylabel("Best fit regression for lift, in kg"), plt.legend()
# tmp = abs(liftsM[['Age', 'BestDeadliftKg']].dropna().sample(5000))
# sns.regplot(tmp['Age'], tmp['BestDeadliftKg'], order=4, scatter_kws={ "alpha": 0.1})


# It's when we compare the three lifts side by side that it gets a little more interesting. 
# 
# The trajectory of decline of the regression line is pretty steady across the three lifts. For men getting older, this is a sad reality.
# 
# What's interesting here, however is that the bench press seems to stave off Father Time just a little longer, peaking around 40 - about 5 years later than squats and the deadlift. While these are all compound movements, the bench is probably the most isolated of the three, focusing primarily on the chest. Compared with the squat and the DL, which use many more muscle groups including the largest in the body, the glutes.

# ## Conclusion

# 
# It's possible that the isolation of the muscle is what keeps older men's strength reaching its peak later in life. The suggestion here is that as men get older, a bodybuilding approach (more isolation, working for hypertrophy) might be a better approach to retain strength over time than compound movements.

# In[ ]:




