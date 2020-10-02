#!/usr/bin/env python
# coding: utf-8

# ![https://imgur.com/iwoT9aD](http://)
# [https://imgur.com/iwoT9aD](http://)
# ![imgur.com/iwoT9aD](http://)
# 
# **If you found value in this project then please upvote and support :)
# **
# Introducing my very own Brain Computer Interface (BCI) Reseach:
# 
# When doing **Neurofeedback (NF)** research, one of the most important first steps is to start by categorizing  our participants to learners and non-learners. 
# 
# In this project we will try out two different approaches of classifying NF learners and non-learners as well as measure their **heart-rate-variability (HRV)** and attempting to make sense of the data, to find correlations, maybe even make predictions based on our data.
# 
# **The research design is as follows:**
# 
# 1) We measure the baseline: 1 recording of brain signals during free thought.
# 
# 2) 5 training trails: which are averaged and used as practice data
# 
# 3) Test trail: which is used to measure how well did the participant learn/ didn't learn.
# 
# 
# **As mentioned, there are two approaches for classifiying;**
# 
# 1) Global baseline : measuring the brain signals of the particpant before we start the training trails and using it as a measure from progress.
# 
# 2) Baseline Per. Block: With each training trail, a new Baseline is calculated.
# 
# 
# 
# **Note: one major downfall of this project is that it has a very small dataset of 22 participants.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Lock And Load!**

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# these are the results of all subjects. included in them are those who were classified later on 
# as learner/responders and non-learners/non-responders
data = pd.read_csv('../input/BCI_input.csv')

print(data)
plt.show()


# okay, let me tell you a bit about our data:
# 
# **subjects**: the subject id
# 
# **'HRV-Vagal-Practice'** - Heart rate variablity measured through vagus nerve during practice trails.
# 
# **'HRV-HF-Practice'** - Heart rate variability based on high frequency bands during practice trails.
# 
# **'Average of NF-BB'** - Average NF trails based on Baseline per Block approach.
# 
# **'Average of NF_GL'** - Average NF trails based on Global Baseline.
# 
# **'HRV_Vagal-test'** - Heart rate variablity measured through vagus nerve during test trail.
# 
# **'HRV_HF-test'** - Heart rate variability based on high frequency bands during test trail.
# 
# **'NF(GL)-Test'** - NF score for test trail using Global baseline approach.
# 
# **'NF(BB)-Test'** - NF score for test trail using Baseline per Block approach.
# 
# 

# In[ ]:


sns.pairplot(data)
plt.show()


# Just by looking at the last row (and the last column) of the pairplot, we can see a potential outlier (let's leave it for now). 

# In[ ]:


t = data[data['HRV_HF-test']>0.001]
t


# Let's rename the columns, it would be easier to work with better name:

# In[ ]:


data.rename(index=str, columns={'HRV-Vagal-Practice':'Vagal',
                                'HRV-HF-Practice':'HF',
                                'Average of NF-BB':'BLpractice',
                                'Average of NF_GL':'GLpractice',
                                "HRV_Vagal-test": "Vagaltest", 
                                "HRV_HF-test": "HFtest",
                                "NF(GL)-Test": "GLtest", 
                                "NF(BB)-Test": "BLtest"}
            ,inplace=True)


# Out of consideration of not diving too deep into this research, i will not code the entire process of classifying the participants to learner and non-learners but i can describe the process for one sample (participant 106) for those who are interested:
# 
# ![https://www.deviantart.com/silverbullet227/art/Samp106-769762068](http://)
# 
# The above data is for the participant 116 using the Baseline Per. Block approach:
# 
# Trial number 0 i for measure of th baseline.
# 
# The next 5 trials to follow are the training trails, each has baseline_mean and feedback_mean. subtracting these two values gives us an estimate of the participant's progress for the trail, which is refered to as nf_bl_diff. After that we perform a t-test to classify if the current trial was considered successful or not (in other words whether the participant managed to learn or not, it wasn't mentioned before but each participant is performing a task while doing each trail, in this research the task was to lower the volume of a paino tune playing in the background). 
# 
# After the training we performed, another baseline was measured (without the task) and followed by a test trail (with the task) to see whether the participant was successful with regulating their brain signals. it the participant was successful the success_Ttest would suggest so. classifying the participant as a learner (116 was indeed a learner).
# 
# I already did this prcoess in excel, so i'll just list the learners and non learners below:

# In[ ]:


nonLearnersLabels = [102,103,104,112,107,108,109,115,121]
# learnersLabels = [101,105,106,110,111,113,114,116,117,118,119,120,122]
learnersLabels = [ x for x in range(101,123) if x not in nonLearnersLabels]

# isin function here helps us filter the participants based on their subject ID
learners = data[data.subject.isin(learnersLabels)]
nonLearners = data[data.subject.isin(nonLearnersLabels)]

# let's show the learners dataframe as an example
learners


# Let's look at the relationship between NF practice trails and NF test trail for learners and non-learners:

# In[ ]:


from scipy.stats import pearsonr

plt.figure(num=None, figsize=(10, 4))
ax = plt.subplot(1,2,1)
ax.set_title("Learners")
sns.regplot(x=learners.GLpractice,y=learners.GLtest)
ax = plt.subplot(1,2,2)
ax.set_title("Non-Learners")
sns.regplot(x=nonLearners.GLpractice,y=nonLearners.GLtest)

r1,p1 = pearsonr(learners.GLpractice,learners.GLtest)
r2,p2 = pearsonr(nonLearners.GLpractice,nonLearners.GLtest)
# corrLearn = np.corrcoef(learners.GLpractice,learners.GLtest)[1][0]
# corrNonLearn = np.corrcoef(nonLearners.GLpractice,nonLearners.GLtest)[1][0]
print("learners: r={}, p-val={}, \nnon-learners: r={}, p-val={}".format(r1,p1,r2,p2))


# It seem like we have significant results (significant results means p-val<0.05) for the non-learners. however, let us not for get that this is unreliable data since we have a very small dataset especially for the non-learners (14 samples for learners, 8 samples for non-learners..)
# 
# Now, let's take a lot at the second approach (Baseline Per Block).

# In[ ]:


from scipy.stats import pearsonr

plt.figure(num=None, figsize=(10, 4))
ax = plt.subplot(1,2,1)
ax.set_title("Learners")
sns.regplot(x=learners.BLpractice,y=learners.BLtest)
ax = plt.subplot(1,2,2)
ax.set_title("Non-Learners")
sns.regplot(x=nonLearners.BLpractice,y=nonLearners.BLtest)

r1,p1 = pearsonr(learners.BLpractice,learners.BLtest)
r2,p2 = pearsonr(nonLearners.BLpractice,nonLearners.BLtest)
# corrLearn = np.corrcoef(learners.GLpractice,learners.GLtest)[1][0]
# corrNonLearn = np.corrcoef(nonLearners.GLpractice,nonLearners.GLtest)[1][0]
print("learners: r={}, p-val={}, \nnon-learners: r={}, p-val={}".format(r1,p1,r2,p2))


# No significant results for this approach.
# 
# Now what about Heart Rate Variablity:

# In[ ]:


print(learners[['HF','HFtest']])
plt.scatter(learners.HF,learners.HFtest)


# Let's make the data more comfortable to wrok with (by trnasforming the data using log):

# In[ ]:


learners = learners.assign(logHFpractice=np.log(learners.HF))
learners = learners.assign(logHFtest=np.log(learners.HFtest))

nonLearners = nonLearners.assign(logHFpractice=np.log(nonLearners.HF))
nonLearners = nonLearners.assign(logHFtest=np.log(nonLearners.HFtest))

# display the learners to check the data
learners


# Now let's try again;

# In[ ]:


print(learners[['logHFpractice','logHFtest']])
plt.scatter(learners.logHFpractice,learners.logHFtest)


# okay, now that our data is more readable and comfortable to work with, we can see how the average HF Heart Rate Variablity across the 5 training trials correlates with the test trail.

# In[ ]:


from scipy.stats import pearsonr

plt.figure(num=None, figsize=(10, 4))
ax = plt.subplot(1,2,1)
ax.set_title("Learners")
sns.regplot(x=learners.logHFpractice,y=learners.logHFtest)
ax = plt.subplot(1,2,2)
ax.set_title("Non-Learners")
sns.regplot(x=nonLearners.logHFpractice,y=nonLearners.logHFtest)

r1,p1 = pearsonr(learners.logHFpractice,learners.logHFtest)
r2,p2 = pearsonr(nonLearners.logHFpractice,nonLearners.logHFtest)
# corrLearn = np.corrcoef(learners.GLpractice,learners.GLtest)[1][0]
# corrNonLearn = np.corrcoef(nonLearners.GLpractice,nonLearners.GLtest)[1][0]
print("learners: r={}, p-val={}, \nnon-learners: r={}, p-val={}".format(r1,p1,r2,p2))


# It seems that we have a very high significant correlation r=0.96 for the non-learners, this might simply mean that those who were classfied as non-learners did not display any difference in HRV during the practice trails and the test trail (in other words, it hand no affect on them what so ever). however, those who were classified as learners did in fact display a change in HRV between the practice trails and the test trails (research tends to support the idea that those who are successful with NF manage to  be in a calm/meditative state after NF practice with more calm breathing patterns)
# 
# Now what about the Vagal HRV (practice vs. test):

# In[ ]:


from scipy.stats import pearsonr

plt.figure(num=None, figsize=(10, 4))
ax = plt.subplot(1,2,1)
ax.set_title("Learners")
sns.regplot(x=learners.Vagal,y=learners.Vagaltest)
ax = plt.subplot(1,2,2)
ax.set_title("Non-Learners")
sns.regplot(x=nonLearners.Vagal,y=nonLearners.Vagaltest)

r1,p1 = pearsonr(learners.Vagal,learners.Vagaltest)
r2,p2 = pearsonr(nonLearners.Vagal,nonLearners.Vagaltest)
# corrLearn = np.corrcoef(learners.GLpractice,learners.GLtest)[1][0]
# corrNonLearn = np.corrcoef(nonLearners.GLpractice,nonLearners.GLtest)[1][0]
print("learners: r={}, p-val={}, \nnon-learners: r={}, p-val={}".format(r1,p1,r2,p2))


# In[ ]:





# In[ ]:




