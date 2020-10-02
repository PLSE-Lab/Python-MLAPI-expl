#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy .stats import norm 
df=pd.read_csv('/kaggle/input/data_1.csv')
df


# This is the dataset of height of individuals, we will try to remove or define outliers using z score and std deviation.Now we will plot the histogram of the distribution of heights

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(df.Height,bins=20,rwidth=0.8)
plt.xlabel('height(inches)')
plt.ylabel('count')
plt.show()


# Now we will plot the normal distribution ofthe same
# for more information https://www.mathsisfun.com/data/standard-normal-distribution.html
# try changing the bin size value

# now we will plot a bell curve using a scipy module for visualisation

# In[ ]:


from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
plt.hist(df.Height,bins=20,rwidth=0.8)
plt.xlabel('height(inches)')
plt.ylabel('count')
rng=np.arange(df.Height.min(),df.Height.max(),0.1)
plt.plot(rng,norm.pdf(rng,df.Height.mean(),df.Height.std()))
plt.show()
#matplotlib. rcParams['figure.figsize']=(10,6)

#i dont know why the bell curve isnt plotting in Kaggle(was plotting in JN),Trouble shoot and let me know


# In[ ]:


#max height
df.Height.max()


# In[ ]:


#mean height
df.Height.mean()


# In[ ]:


#std. deviation of height
df.Height.std()


# ![std.png](attachment:std.png)

# Standard deviation is the measure of how far your values from mean
# Normal practise is to remove the more than 3sigma.
# We should use our sense of judgement to zero in on outliers it may br 2 sigma for smaller datasets
# 

# In[ ]:


#so my upper limit will be my mean value plus 3 sigma
upper_limit=df.Height.mean()+3*df.Height.std()
upper_limit


# In[ ]:


#my lowar limit will be my mean - 3 sigma
lowar_limit=df.Height.mean()-3*df.Height.std()
lowar_limit


# In[ ]:


#now that my outliers are defined, i want to see what are my outliers
df[(df.Height>upper_limit)|(df.Height<lowar_limit)]


# In[ ]:


#now we will visualise the good data
new_data=df[(df.Height<upper_limit)& (df.Height>lowar_limit)]
new_data


# In[ ]:


#shape of our new data
new_data.shape


# In[ ]:


#shape of our outliers
df.shape[0]-new_data.shape[0]


# # now we will try to remove the outliers by z scores
# # z score tells how many standard deviations  away a data point is
# # in our case mean is 66.36 and std deviation is 3.84
# # so our Z SCORE for datapoint 70 is 70-66.36(mean)/3.84(std)=0.94
# ![z.png](attachment:z.png)
# 

# In[ ]:


#now we will calculate the z score of all our datapoints and display in a dataframe
df['zscore']=(df.Height-df.Height.mean())/df.Height.std()
df


# In[ ]:


#figuring out all the datapoints more than 3
df[df['zscore']>3]


# In[ ]:


#figuring out all the datapoints less than 3
df[df['zscore']<-3]


# In[ ]:


#displaying the outliers with respect to the zscores
df[(df.zscore<-3)|(df.zscore>3)]


# In[ ]:


new_data_1=df[(df.zscore>-3)& (df.zscore<3)]
new_data_1


# # Now we will try to find the outliers using Inter Quartile Range

# In[ ]:


df


# Now we will drop the ZSCORE from the dataframe

# In[ ]:


df=df.drop(['zscore'],axis=1)


# In[ ]:


df


# In[ ]:


df.describe()


#  Visit for more details on quartiles https://www.mathsisfun.com/data/quartiles.html

# In[ ]:


Q1=df.Height.quantile(0.25)
Q3=df.Height.quantile(0.75)
Q1,Q3
#WHICH MEANS THAT Q1 CORRESPONDS TO 25% OF ALL THE HEIGHT DISTRIBUTION IS BELOW 63.50
#Q3 CORRESPONDS TO 75% OF ALL THE HEIGHT DISTRIBUTION IS BELOW 69.174


# In[ ]:


#NOW WE WILL CALCULATE THE IQR
IQR=Q3-Q1
IQR


# In[ ]:


#NOW WE WILL DEFINE THE UPPER LIMITS AND LOWAR LIMITS
LOWAR_LIMIT=Q1-1.5*IQR
UPPER_LIMIT=Q3+1.5*IQR
LOWAR_LIMIT,UPPER_LIMIT


# In[ ]:


#NOW WE SHALL DISPLY THE OUTLIERS HEIGHTS
df[(df.Height<LOWAR_LIMIT)|(df.Height>UPPER_LIMIT)]


# In[ ]:


#NOW WE WILL DISPLAY THE REMAINING SAMPLES ARE WITHIN THE RANGE
df[(df.Height>LOWAR_LIMIT)&(df.Height<UPPER_LIMIT)]


# # Upvote if you like this
