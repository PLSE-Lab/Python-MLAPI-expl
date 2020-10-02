#!/usr/bin/env python
# coding: utf-8

# # Donors Choose: Metro Types and Project Categories
# 
# ## Contents
# 
# 1. Introduction
# 2.  Packages
# 3. Free Lunch Across Metro Types
# 4. Are Free Lunch Distributions Normal?
# 5. Are Free Lunch Distributions Significantly Difference?
# 6. Are Grade Level and Successful Campaigning Independent?
# 7. Which Project Categories Have the Best Success?
# 8. Conclusion
# 9. References

# ## 1. Introduction
# 
# Digging a little more into the various datasets we have with the DonorsChoose competition, we can start looking at the projects dataset.  This is the big one that links all the others together.  After some initial EDA, a few questions came up and I decided to do some quick tests to try and answer these questions for me.  I hope you enjoy and, as always, let me know what you think and where you think I can improve.

# ## 2. Packages

# In[106]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
import os


# In[107]:


teachers = pd.read_csv('../input/Teachers.csv')
projects = pd.read_csv('../input/Projects.csv')
schools = pd.read_csv('../input/Schools.csv')
donations = pd.read_csv('../input/Donations.csv')


# ## 3. Free Lunches Across Metro Types
# 
# Here, we want to see if the distribution of percentage of free lunches for schools is different for the various metro types.

# In[108]:


plt.figure(figsize=(15,7))
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="rural"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='b', alpha=0.5,label='Rural')
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="urban"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='g', alpha=0.5,label='Urban')
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="suburban"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='orange',alpha=0.5, label='Suburban')
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="unknown"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='yellow', alpha=0.5,label='Unknown')
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="town"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='cyan', alpha=0.5,label='Town')
plt.title("Free Lunch Distributions")
plt.xlabel("Percentage Free Lunch")
plt.ylabel("Percentage")
plt.legend(loc='upper left')
plt.show()


# *Not a huge visual difference between the metro types and their percentage of students that get free lunch.   But, it looks like suburban metro types may have a lower incidence of free lunches, while urban has high incidence of free lunches.  We should do some testing on this.*

# In[109]:


schools_perc = schools[schools['School Percentage Free Lunch']<=100]
x1=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="rural"])
x2=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="urban"])
x3=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="suburban"])
x4=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="unknown"])
x5=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="town"])


# ## 4. Are Free Lunch Distributions Normal?
# 
# Here, we can take a look at whether the individual distributions are normally distributed or not.  This will determine whether or not ANOVA is appropriate or not.

# In[110]:


x_dict = {'rural':stats.normaltest(x1).pvalue,
              'urban':stats.normaltest(x2).pvalue,
              'suburban':stats.normaltest(x3).pvalue,
              'unknown':stats.normaltest(x4).pvalue,
              'town':stats.normaltest(x5).pvalue}


# In[111]:


for keys,values in x_dict.items():
    print(keys)
    print("p-value:"+str(values))


# *Well, we reject the hypothesis that these distributions comes from normal distributions.  This is not surprising from the visual inspection of the above graph.  So, we can a non-parametric alternative called the Kruskal-Wallis test:*

# ## 5. Are Free Lunch Distributions Significantly Difference?

# In[112]:


kruskaltest=stats.kruskal(x1,x2,x3,x4,x5)
kruskaltest


# *Definitely looks like there is a difference between the medians of these distributions.  Most likely the difference is between urban and suburban, so lets check that out in a boxplot:*

# In[113]:


plt.figure(figsize=(10,7))
ax = sns.boxplot(y=schools['School Percentage Free Lunch'],x=schools['School Metro Type'])
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x) for x in vals],fontsize=15)
ax.set_ylabel('School Percentage Free Lunch',fontsize=16)
ax.set_xlabel('School Metro Type',fontsize=16)
plt.title('Distribution of Free Lunch Across School Metro Type',fontsize=18)
plt.show()


# *Yes, the incidence of free lunches are higher among urban school types, and the difference is significant.*

# ## 6. Are Grade Level and Successful Campaigning Independent?

# In[114]:


funded_df=pd.get_dummies(projects['Project Current Status'])
fund_df=pd.concat([projects['Project Grade Level Category'], funded_df], axis=1)


# In[115]:


project_grades = fund_df.groupby(['Project Grade Level Category']).agg([np.sum])
project_grades.columns=list(['Expired','Fully_Funded','Live'])
project_grades=project_grades.reset_index(level=0)
project_grades=project_grades.drop(4)


# *After grouping together the number of Expired, Fully Funded, and Live projects by each grade level, we are left with a nice contingency table like this.  This table is good for testing whether or not that 'Project Grade Level Category' is and independent variable from whether the project succeeded or not.*

# In[117]:


project_grades


# In[120]:


chi2_contingency(project_grades.drop(['Project Grade Level Category'],axis=1))
expec = chi2_contingency(project_grades.drop(['Project Grade Level Category'],axis=1))[3]
observed=np.array(project_grades.drop(['Project Grade Level Category'],axis=1))


# **Chi-square test result:**

# In[121]:


chi2_contingency(project_grades.drop(['Project Grade Level Category'],axis=1))


# *We reject the null hypothesis here (p-value is the 5.6*10-96 number), so there is some relationship between these variables.  Now, I'm going to see which one of these combinations have the most observed successes over the expected number of success.*

# In[122]:


results=pd.concat([project_grades['Project Grade Level Category'],pd.DataFrame(observed-expec)],axis=1)
#results.columns(['Expired','Fully Funded','Live'])
results.columns = ['Project Grade Level Category','Expired','Fully Funded','Live']
results


# *Well, by taking the difference (observed-expected), we can see that Grades PreK-2 has a much high amount of Fully Funded successes and Live successes than the other grade levels.  This grade level also has a much lower number of Expired projects.  You could say that projects for this grade level succeed much more often than other grade levels.*

# ## 7. Which Project Categories Have the Best Success?
# 
# There are 50 project categories, but we can take a look at which project categories have a high proportion of successful programs.  Here, I consider a project successful if it is Fully Funded or Live.

# In[126]:


cat_df = pd.concat([projects['Project Subject Category Tree'], funded_df], axis=1)
project_cats = cat_df.groupby(['Project Subject Category Tree']).agg([np.sum])
project_cats.columns=list(['Expired','Fully_Funded','Live'])
project_cats=project_cats.reset_index(level=0)


# In[127]:


project_cats['Proportion Expired'] = project_cats['Expired']/(project_cats['Fully_Funded']+project_cats['Live']+project_cats['Expired'])
project_cats['Proportion Success'] = 1-project_cats['Proportion Expired']
project_cats_sorted = project_cats.sort_values(by='Proportion Success',ascending=True).reset_index()


# In[128]:


plt.figure(figsize=(10,20))
ax=project_cats_sorted.drop('index',axis=1).plot(kind='barh',y=['Proportion Success'],figsize=(10,20),color='teal',xlim=[0,1],position=0)
for i,v in enumerate(project_cats_sorted['Project Subject Category Tree']):
    plt.text(0,i,str(v)+','+str("{:.2f}%".format(project_cats_sorted['Proportion Success'][i]*100)),fontsize=16)
plt.title('Proportion of Successful Campaigns by Subject',fontsize=20)
plt.xlabel('Proportion',fontsize=14)
plt.show()


# *Warmth, Care & Hunger are all over the ranking, but appear a lot in the most successful subject types.  Overall, all of the subjects are successful.  However, it is hard to determine which individual subjects are good or bad for successful campaigns when they are tied together.  My future work will include separating the Warmth, Care & Hunger categories and the Health  & Sports categories from the rest of the subjects to test for academic versus health.*

# ## 9. Conclusion
# 
# Free lunch percentages are different among metro types and programs have historically been more successfull for certain grade levels. 

# ## 8. References
# 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
# 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html

# In[ ]:




