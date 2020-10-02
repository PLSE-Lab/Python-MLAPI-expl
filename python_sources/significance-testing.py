#!/usr/bin/env python
# coding: utf-8

# Hey,
# 
# So this is a dataset that I uploaded recently studying the brain activity of Alcoholics.
# 
# You can do a lot of stuff with it
# 
# Here, I will use Hypothesis Testing to find differences in Alcoholics and Control across different stimulus

# In[1]:


#importing header files
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors


# In[2]:


#Checking a sample file
F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data1.csv")
#Dropping an unnecessary column
F=F.drop('Unnamed: 0',axis=1)
#Checking the sample data
print(F.head(5))
print(F.dtypes)


# Let us now separate the control and alcoholics group along with the different stimuli(And also visualise them)

# 1) Subject: 'Alcoholic(a)' Stimulus: S1

# In[3]:


n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S1 obj' and F.loc[5,'subject identifier']=='a':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X1=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)

s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X1)


# 2) Subject: 'Control(c)'  Stimulus: S1

# In[4]:


n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S1 obj' and F.loc[5,'subject identifier']=='c':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X2=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X2)


# 3) Subject: 'Alcoholic(a)' Stimulus: S2 match

# In[5]:


n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S2 match' and F.loc[5,'subject identifier']=='a':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X3=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X3)


# 4) Subject: 'Control(c)'  Stimulus: S2 match

# In[6]:


n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S2 match' and F.loc[5,'subject identifier']=='c':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X4=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X4)


# 5) Subject: 'Alcoholic(a)' Stimulus: S2 no match

# In[7]:


n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S2 nomatch,' and F.loc[5,'subject identifier']=='a':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X5=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X5)


# 6) Subject: 'Control(c)'  Stimulus: S2 no match

# In[8]:


n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S2 nomatch,' and F.loc[5,'subject identifier']=='c':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X6=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X6)


# Now, let's combine them according to stimulus

# In[9]:


S1=pd.DataFrame({'Channel':X1['channel'], 'Alcohol':X1['sensor value'],'Control':X2['sensor value']})
S2Match=pd.DataFrame({'Channel':X3['channel'], 'Alcohol':X3['sensor value'],'Control':X4['sensor value']})
S2NoMatch=pd.DataFrame({'Channel':X5['channel'], 'Alcohol':X5['sensor value'],'Control':X6['sensor value']})


# Now let's define a function that would help us implement hypothesis testing
# (Code Snippet/Description taken from EDX course- DAT203.2x: Principles of Machine Learning)

# Applying t test(Hypothesis Testing)
# 
# Now that we have examined some of the relationships between the variables in these data, we will now apply formal hypothesis testing. In hypothesis testing the a null hypothesis is tested against a statistic. The null hypothesis is simply that the difference is not significant. Depending on the value of the test statistic, you can accept or reject the null hypthesis.
# In this case, we will use the two-sided t-test to determine if the difference in means of two variables are significantly different. The null hypothesis is that there is no significant difference between the means. There are multiple criteria which are used to interpret the test results. You will determine if you can reject the null hyposesis based on the following criteria:
# 
# 1. Selecting a confidence level of 5% or 0.05.
# 2. Determine if the t-statistic for the degrees of freedom is greater than the critical value. The difference in means of Normally distributed variables follows a t-distribution. The large t-statistic indicates the probility that the difference in means is unlikely to be by chance alone.
# 3. Determine if the P-value is less than the confidence level. A small P-value indicates the probability of the difference of the means being more extreme by chance alone is the small.
# 4. The confidence interval around the difference of the means does not overlap with 0. If the confidence interval is far from 0 this indicates that the difference in means is unlikely to include 0.
# 
# Based on these criteria we will accept of reject the null hypothesis. However, rejecting the null-hypothesis should not be confused with accepting the alternative. It simply means the null is not a good hypothesis.
# The brain_test function in the cell below uses the CompareMeans function from the weightstats package to compute the two-sided t statistics. The brain_family_conf function calls the family_test function and plots the results.

# In[10]:


def brain_test(df, col1, col2, alpha):
    from scipy import stats
    import scipy.stats as ss
    import pandas as pd
    import statsmodels.stats.weightstats as ws
    
    n, _, diff, var, _, _ = stats.describe(df[col1] - df[col2])
    degfree = n - 1

    temp1 = df[col1].as_matrix()
    temp2 = df[col2].as_matrix()
    res = ss.ttest_rel(temp1, temp2)
      
    means = ws.CompareMeans(ws.DescrStatsW(temp1), ws.DescrStatsW(temp2))
    confint = means.tconfint_diff(alpha=alpha, alternative='two-sided', usevar='unequal') 
    degfree = means.dof_satt()

    index = ['DegFreedom', 'Difference', 'Statistic', 'PValue', 'Low95CI', 'High95CI']
    return pd.Series([degfree, diff, res[0], res[1], confint[0], confint[1]], index = index)   
    
def hist_brain_conf(df, col1, col2, num_bins = 30, alpha =0.05):
    import matplotlib.pyplot as plt
    ## Setup for ploting two charts one over the other
    fig, ax = plt.subplots(2, 1, figsize = (12,8))
    
    mins = min([df[col1].min(), df[col2].min()])
    maxs = max([df[col1].max(), df[col2].max()])
    
    mean1 = df[col1].mean()
    mean2 = df[col2].mean()
    
    tStat = brain_test(df, col1, col2, alpha)
    pv1 = mean2 + tStat[4]    
    pv2 = mean2 + tStat[5]
    
    ## Plot the histogram   
    temp = df[col1].as_matrix()
    ax[1].hist(temp, bins = 30, alpha = 0.7)
    ax[1].set_xlim([mins, maxs])
    ax[1].axvline(x=mean1, color = 'red', linewidth = 4)    
    ax[1].axvline(x=pv1, color = 'red', linestyle='--', linewidth = 4)
    ax[1].axvline(x=pv2, color = 'red', linestyle='--', linewidth = 4)
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel(col1)
    
    ## Plot the histogram   
    temp = df[col2].as_matrix()
    ax[0].hist(temp, bins = 30, alpha = 0.7)
    ax[0].set_xlim([mins, maxs])
    ax[0].axvline(x=mean2, color = 'red', linewidth = 4)
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel(col2)
    
    return tStat


# 1. For Stimulus :S1

# In[11]:


hist_brain_conf(S1, 'Control','Alcohol')


# Let's examine the printed table of results and the charts noting the following:
# 
# The difference of the means is  0.32 inches. You can see this difference graphically by comparing the positions of the solid red lines showning the means of the two distirbutions.
# 
# The critical value of the two-sided t-statistic at 480 degrees of freedom is 1.96. The t-ststistic of 2.5 is larger than this critical value.
# 
# The P-value is effectively 0.01, which is smaller than the confidence level of 0.05.
# 
# The 95% confidence interval of the difference in means is from -0.08 to 0.73, which does overlap 0. 
# You can see the confidence interval plotted as the two dashed red lines in the lower chart shown above. This confidence interval around the mean of the alcohol's sensor value does overlap with the mean of the control's sensor value .
# 
# Overall, these statistics indicate you can reject the null hypothesis, or that there difference in the means is not 0.
# 

# In[12]:


hist_brain_conf(S2Match, 'Alcohol','Control')


# Let's examine the printed table of results and the charts noting the following:
# 
# The difference of the means is  0.07 inches. You can see this difference graphically by comparing the positions of the solid red lines showning the means of the two distirbutions.
# 
# The critical value of the two-sided t-statistic at 480 degrees of freedom is 1.96. The t-stastistic of 0.92 is smaller than this critical value.
# 
# The P-value is effectively 0.39, which is larger than the confidence level of 0.05.
# 
# The 95% confidence interval of the difference in means is from -0.59 to 0.74, which does overlap 0. 
# You can see the confidence interval plotted as the two dashed red lines in the lower chart shown above. This confidence interval around the mean of the alcohol's sensor value does overlap with the mean of the control's sensor value .
# 
# Overall, these statistics indicate you cannot reject the null hypothesis, or that there difference in the means is 0.

# In[ ]:


hist_brain_conf(S2NoMatch, 'Control','Alcohol')


# Let's examine the printed table of results and the charts noting the following:
# 
# The difference of the means is  0.5 inches. You can see this difference graphically by comparing the positions of the solid red lines showning the means of the two distirbutions.
# 
# The critical value of the two-sided t-statistic at 480 degrees of freedom is 1.96. The t-stastistic of 6.09 is larger than this critical value.
# 
# The P-value is effectively 0.00000007395597, which is smaller than the confidence level of 0.05.
# 
# The 95% confidence interval of the difference in means is from -0.59 to 0.74, which does overlap 0. 
# You can see the confidence interval plotted as the two dashed red lines in the lower chart shown above. This confidence interval around the mean of the alcohol's sensor value does overlap with the mean of the control's sensor value .
# 
# Overall, these statistics indicate you can reject the null hypothesis, or that there difference in the means is not 0.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Now, let's divide the data by drinking category i.e. Alcoholic and Control Patients

# 

# In[ ]:


n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'subject identifier']=='a':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop1")

Alc=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)

n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'subject identifier']=='c':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop2")
Cont=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)


# Let's apply hypothesis testing on the above splitting of dataset
# 

# In[ ]:


Hypo=pd.DataFrame({'Channel':Alc['channel'], 'Alcohol':Alc['sensor value'],'Control':Cont['sensor value']})
hist_brain_conf(Hypo, 'Control','Alcohol')


# Let's examine the printed table of results and the charts noting the following:
# 
# The difference of the means is  0.25 inches. You can see this difference graphically by comparing the positions of the solid red lines showning the means of the two distirbutions.
# 
# The critical value of the two-sided t-statistic at 480 degrees of freedom is 1.96. The t-stastistic of 3.6 is larger than this critical value.
# 
# The P-value is effectively 0.0006, which is smaller than the confidence level of 0.05.
# 
# The 95% confidence interval of the difference in means is from -0.21 to 0.72, which does overlap 0. 
# You can see the confidence interval plotted as the two dashed red lines in the lower chart shown above. This confidence interval around the mean of the alcohol's sensor value does overlap with the mean of the control's sensor value .
# 
# Overall, these statistics indicate you can reject the null hypothesis, or that there difference in the means is not 0.

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# After the four different hypothesis/Significance Testing we can conclude that the brain signals of Alcoholic is in fact, significantly different from Normal Subjects
