#!/usr/bin/env python
# coding: utf-8

# #  Analyzing San Fransisco City Salaries

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

salaries = pd.read_csv('../input/SF_salary_data_gender.csv')
print(salaries.head())


# ##Analysis
# 
# **First, I want to see the overall trend in salaries for all jobs for 2011 - 2014**
# 
#  - **Below,we see that there generally was an overall increase in salaries, with a dip in 2014**

# In[ ]:


def bar(df, name):
    plt.bar(df.Year, df.TotalPay, color = 'darkorange')
    plt.title('{} Salary Trends'.format(name))
    plt.xticks(np.arange(min(df.Year), max(df.Year)+1, 1.0))
    plt.ylim(min(df.TotalPay)*.99, max(df.TotalPay)*1.01)
    plt.show()
    plt.clf()
    


# In[ ]:


all_groups = salaries.groupby(['Year'],as_index=False).mean()
bar(all_groups, 'All Jobs')


# In[ ]:


years = np.arange(2011,2015,1)

def boxplots(df):
    fig = plt.figure(figsize=(17,8))
    ax1 = fig.add_subplot(1,4,1)
    ax2 = fig.add_subplot(1,4,2)
    ax3 = fig.add_subplot(1,4,3)
    ax4 = fig.add_subplot(1,4,4)
    axs = [ax1,ax2,ax3, ax4]
    salaries = {}
    male_salaries = []
    female_salaries = []
    for year in years:
        salaries[year] = df['TotalPay'][(df.Year == year)&(df.gender == 'f')],        df['TotalPay'][(df.Year == year)&(df.gender == 'm')]
        male_salaries.append(salaries[year][1])
        female_salaries.append(salaries[year][0])
    for i, ax in enumerate(axs):
        ax.boxplot([female_salaries[i], male_salaries[i]])
        ax.set(xlabel = 'Female vs. Male', ylabel = 'Salary', title = years[i], xlim = (0,3), ylim = (0,250000))
    if df is salaries:
        fig.suptitle('All Jobs: Female vs Male Salary Distributions')
    else:
        fig.suptitle('{}: Female vs Male Salary Distributions'.format(df.reset_index().JobTitle[0]))
    plt.show()
    plt.clf()
    
    print('Counts by Gender:')
    for i, salary in enumerate(female_salaries):
        print('{}: female: {}, male: {}'.format(years[i], salary.shape[0], male_salaries[i].shape[0]))


# In[ ]:


boxplots(salaries)


# **Above, we can see that men, have been averaging a higher salary overall, every year, from 2011 to 2014.**
# 
# **Next I want to look at certain jobs, to see if the above results hold true across many professions.**
# 
# **In order to do that, I have created a dictionary, with these professions, and the number of people employed in that particular profession.**

# In[ ]:


job_dict ={}
job_titles = salaries['JobTitle']
for job in job_titles:
    if job in job_dict:
        job_dict[job] += 1
    else:
        job_dict[job] = 1

job_dict2 = {k: v for k, v in job_dict.items() if v >= 1000}
print(sorted(job_dict2.items(), key=lambda x:x[1], reverse = True))


# The above dictionary contains the counts of people by job title. Based on the above dictionary, **Transit Operators** represent a good amount of people in this dataset. I will also look at **Registered Nurses** , and **Firefighters**.

# In[ ]:


transitjob = salaries[salaries.JobTitle == 'transit operator']


# In[ ]:


transit_group = transitjob.groupby(['Year'],as_index=False).mean()


# In[ ]:


bar(transit_group, 'Transit Operator')


# **Here we see a very similar trend for Transit Operators, where there has been an increase from 2011 - 2013, and a drop in 2014**

# In[ ]:


boxplots(transitjob)


# **Among Transit Operators, we can see that the same holds true here as far as Female vs Male total salaries. Men are generally better paid, but we can see an intersting fact here:**
# 
#  - **In 2013 and 2014, the highest paid person in this job was a woman.**

# In[ ]:


nursejob = salaries[salaries.JobTitle == 'registered nurse']
nurse_group = nursejob.groupby(['Year'],as_index=False).mean()
bar(nurse_group, 'Registered Nurse')


# **For Registered Nurses, salaries are higher than the average. Also, the same trend holds true as we saw earlier. Salaries in 2014 decreased, but the decrease was not so severe in this profession, as in other professions**

# In[ ]:


boxplots(nursejob)


# **Also We can see that this is a very female driven profession. While men are still averaging higher salaries, the difference between the two is alot smaller than other professions, and in every year, the highest paid person is a woman.**

# In[ ]:


firefighter = salaries[salaries.JobTitle == 'firefighter']
firefighter_group = firefighter.groupby(['Year'],as_index=False).mean()
bar(firefighter_group, 'Firefighter')


# **Firefighters are also a very well paid group**

# In[ ]:


boxplots(firefighter)


# **For Firefighters, salaries are well above average. Amongst Firefighters, women make up only about 20% of all Firefighters, and the same holds true once again. Women are averaging a lower salary than men.**

# # Conclusion 
# 
# ##Is there gender discrimination based on San Francisco City data?
# 
# Nothing definitive can be said, just yet. We looked at overall salary trends, and salary for different professions, and it is clear the men are earning more on average, but our dataset was very limited.
# 
# **Important information that we do not know:**
# 
#  - How many years of schooling does each person have?
#  - How many hours on average to they work?
#  - How many years of experience do they have?
#  - Are they married? / Have children?
#  - How good/bad is their health?
# 
# **While this is a good initial look into the salary trends, there really is no conclusion that can be made, just yet.**

# In[ ]:




