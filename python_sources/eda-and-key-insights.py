#!/usr/bin/env python
# coding: utf-8

# Please upvote if you found the EDA insightful. I am trying to become an expert :). 
# 
# 
# Getting recruited from campuses is becoming increasingly competitive. As a student, you want to maximize your chances of getting placed. Using this dataset, let's explore some of the factors that can help increase your chances of getting placed.
# 
# Let me just list all of the findings first, and if you are interested in the details, you can read the kernel.
# 1. Study Hard in 10th, 12th and college. Make sure you get those scores.
# 2. Choose the right specialization, both in 12th, undergrad and MBA. It makes a difference.
# 3. Go for internships. Work Experience helps.
# 4. Try to go for a non-central board, but if you can't make it, don't fret about it.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# So, let's start by importing the data.

# In[ ]:


data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# Let us print the first 5 rows, to get a feel for the kind of values and columns present.

# In[ ]:


data.head()


# So, first let us convert status to a 1/0 format, rather than a string. This will be easier to work with. 

# In[ ]:


data['binary_status'] = data['status']
data['binary_status'] = data['status'] == 'Placed'
data["binary_status"] = data["binary_status"].astype(int)


# First Let us explore gender. Let us see the application imbalance first.

# In[ ]:


plt.hist(data['gender'])


# So, we can see that about twice as many males apply as compared to females. This is very interesting a majority of students in college are women, yet very few apply. I expected a higher number for the females category.

# In[ ]:


ax = sns.barplot(x="gender", y="binary_status", data=data)


# Even though very few women apply, the percentage of women and men accepted is almost the same, however men have a slightly higher chance of getting accepted. So this wraps up gender. 
# 
# Let us move onto the board of secondary education.

# In[ ]:


plt.hist(data['ssc_b'])


# So, most people who apply are from the central board, but there is a sizeable group that applies from other boards. Now let's see if the board studied makes a difference.

# In[ ]:


ax = sns.barplot(x="ssc_b", y="binary_status", data=data)


# It turns out, that people who go a non-central board have a slightly higher chance. This might be because non-central boards are more difficult to complete, and hence, companies slightly prefer such students.
# 
# Now let's see is secondary scores play a role.

# In[ ]:


ax = sns.barplot(x="binary_status", y="ssc_p", data=data)


# Clear Indicator. As you can see, people who get placed have a higher average score in secondary school. This means that academic rigour is extremely important, more so that the board. So start studying from 10th grade to get a good job!!! Now we can try to understand why other boards have a higher chance of acceptance. Maybe because they have a higher score?
# 
# 

# In[ ]:


ax = sns.barplot(x="ssc_b", y="ssc_p", data=data)


# Yes, they do. Only slightly, which resembles what we saw above.
# 
# Now let's look at higher education.

# In[ ]:


ax = sns.barplot(x="hsc_b", y="binary_status", data=data)


# Again, people who go to a non-central board for 12th grade have a slightly higher advantage, but not that much. Let's look at scores again.

# In[ ]:


ax = sns.barplot(x="binary_status", y="hsc_p", data=data)


# Ok, so first thing that we can see is that the scores in 12th grade are very similar to scores in 10th grade. So, they are very predictive. Apart from that, even 12th grade scores are extremely important. So, keep studying Hard.

# In[ ]:


ax = sns.barplot(x="hsc_b", y="hsc_p", data=data)


# So, in 12th grade, both non-central and central board have similar scores. So, people must become more serious during this time.
# 
# Now let's look at specialization.

# In[ ]:


ax = sns.barplot(x="hsc_s", y="binary_status", data=data)


# So, sadly, if you want a good job, you should pursue science ro commerce. Taking arts in like betting on an all or nothing. So, focus on science or commerce to increase your chances. Let's also look at whether specialization affects scores.

# In[ ]:


ax = sns.barplot(x="hsc_s", y="hsc_p", data=data)


# So, we can see that science and arts students have similar scores, meaning that companies don't value an art specialization that much, and it is purely the fact that you are taking science that can get you in.
# 
# Now let's look at undergraduate degree.

# In[ ]:


ax = sns.barplot(x="degree_t", y="binary_status", data=data)


# Again, colleges prefer Commerce and Science more as compared to other degrees for jobs. This is similar to what we saw in 12th grade specializations. Let's look at scores.

# In[ ]:


ax = sns.barplot(x="binary_status", y="degree_p", data=data)


# Again we see, that you need to get good scores in college. So keep working hard!!
# 
# Let's also look at degree chosen and scores.

# In[ ]:


ax = sns.barplot(x="degree_t", y="degree_p", data=data)


# So this might explain the college degree chosen. People who take science and commerce have better scores, so it looks like college scores are CRUCIAL to getting a good job. 
# 
# Now let's see if you should have any work experience.

# In[ ]:


ax = sns.barplot(x="workex", y="binary_status", data=data)


# So, companies REALLY like students with work experience. Maybe it's because they will fit into the company more easily and start working diligently more quickly than those with no experience. So, go for at least one internship, cause it will count.
# 
# Now, let's look at the employability test.

# In[ ]:


ax = sns.barplot(x="binary_status", y="etest_p", data=data)


# So, the employability test doesn't make too much of a difference, since it is easy for everyone to get a decently high score. So just make sure you get a score that is not too worrying. 
# 
# Now let's see what you should do for post-grad

# In[ ]:


ax = sns.barplot(x="specialisation", y="binary_status", data=data)


# So, you should go for marketing and finance rather than marketing and HR. Maybe it is because companies don't need as many HR workers as Finance workers.
# Now, let's look at scores again.

# In[ ]:


ax = sns.barplot(x="binary_status", y="mba_p", data=data)


# So, getting a good score isn't too hard in MBA, so don't work extra hard to get a perfect score. An average one is good enough. 

# So, looking back, you should:
# 1. Study Hard in 10th, 12th and college. Make sure you get those scores.
# 2. Choose the right specialization, both in 12th, undergrad and MBA. It makes a difference.
# 3. Go for internships. Work Experience helps.
# 4. Try to go for a non-central board, but if you can't make it, don't fret about it.
