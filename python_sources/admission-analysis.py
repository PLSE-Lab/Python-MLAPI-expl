#!/usr/bin/env python
# coding: utf-8

# ### if you like it please upvote

# This Kernel provides Basic analysis on the graduate admissions dataset and also correlation between features in the dataset

# In[20]:


#Add All the imports here
import pandas as pd

import os
print(os.listdir("../input"))


# In[21]:


#load the data
df = pd.read_csv('../input/Admission_Predict.csv',index_col = 'Serial No.')
df.head()


# we have 9 features where the dependent feature is chance of admit which is the probability of admission of a student into a particular university. we take serialNo as index.

# Let us have a look at the summary over the dataset

# In[22]:


df.info()


# #### summary
# 1. we have 400 rows and 8 columns(serial no excluded). 
# 2. all the features are numeric.
# 3. no missing values

# In[23]:


sum(df.duplicated())


# It seems there are no duplicate values in the dataset

# In[24]:


df.describe()


# #### summary
# 1. for all the columns mean and 50%(median) values are near so no large outliers
# 2. min, 25%, 50%, 75% and max tell that distribution is normal (except for research as it has either 0 or 1 values) but we can't say it for sure to check distribution we need to plot them.
# 3. count is equal to total no of rows so no missing values

# As all the columns are numeric lets check the distribution of values using hist plot

# In[25]:


df.hist(figsize =(10,10));


# ### understanding
# 1. university ratiing is between(1 to 5) so it looks discrete but it looks well distributed.
# 2. resaerch is either 0 or 1.
# 3. cgpa,gre,tofel look pretty well distributed (so algorithms have good scope for understanding these features well)
# 4. chance of admit looks right skewed most of the values lie in range 60% to 100% (which is a good news for students) but still there might be chance that the alogrithm we use for prediction might not correctly predict the chances below 50%.
# 5. lor and sop doesn't contain any value between 2.5 to 3. sop looks ok even though values not present(doesn't look completely right skewed) but looks it is right skewed.

# In[26]:


pd.plotting.scatter_matrix(df,figsize =(17,17));


# from chance of admit point of view there are positive correleations for gre,tofel and cgpa. these can be critical features for predicting chance of admit. sop and lor seems to also effect positively chance of admit but not effectively as the above three.

# lets check all the three graphs seperately.

# In[27]:


df.plot(x = 'GRE Score', y = 'Chance of Admit ', kind='scatter');


# In[28]:


df.plot(x = 'TOEFL Score', y = 'Chance of Admit ', kind='scatter');


# In[29]:


df.plot(x = 'CGPA', y = 'Chance of Admit ', kind='scatter');


# from all the three graphs we observe in commom is that 
# 1. when their values are less the probability range(chance of admit range) is large and the probability values are small. 
# 2. when their values are large the probability range is less and probability values are large.
# 
# so better the score of gre,tofel and cgpa higher the probability of admit.

# In[30]:


df['GRE Score'].plot(kind = 'box');


# In[31]:


df['TOEFL Score'].plot(kind = 'box');


# In[32]:


df['University Rating'].plot(kind = 'box');


# In[33]:


df['SOP'].plot(kind = 'box');


# In[34]:


df['LOR '].plot(kind = 'box');


# In[35]:


df['CGPA'].plot(kind = 'box');


# In[36]:


df['Research'].plot(kind = 'box');


# from the boxplots we can find the outliers. outliers are those which seem not in range. outlier will be those who have value which are different from normal values in that feature.

# main disadvantage of outliers is that they effect the model which trains from the training data. so because of outliers model can learn it differently than it actually has to learn.

# ### conclusion
# from the analysis we made we can conclude that
# 1. The dataset collected is clean and requires no other changes.
# 2. There are very few outliers which can be removed while training the model.
# 3. GRE Score,TOEFL Score, CGPA show positive correlation to chance of admit. And these three are normally distributed so model can learn effectively from these features.
# 4. LOR and SOP also show a tiny bit of positive correlation so these might incerase the accuracy of the model a bit.
