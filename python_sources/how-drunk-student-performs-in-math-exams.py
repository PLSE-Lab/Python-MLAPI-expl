#!/usr/bin/env python
# coding: utf-8

# # Student Alcohol Consumption
# 
# ## Context:
# The data were obtained in a survey of students math and portuguese language courses in secondary school. It contains a lot of interesting social, gender and study information about students. You can use it for some EDA or try to predict students final grade.
# 
# ![Figure1. Does Alcohol really affect student performance? Does drunk student perform worse?](http://www.quickmeme.com/img/a6/a617241ae190e2db9ef75e5ffc73afa79ad6c46778fc697668541f7b81b501ab.jpg)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualize
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


#Core data manipulation
import numpy as np
import pandas as pd

#Matlplotlib & Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# others
import os
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#import data of student's math grade
path = "../input/student-mat.csv"
student_df = pd.read_csv(path)

student_df.head(7)


# In[ ]:


#statistical summary of floating point data
student_df.describe()


# In[ ]:


#statistical summary of categorical/ordinal data:
student_df.describe(include=['O'])


# Seem's good. Notice we have 395 rows x 33 cols and the data is clean. Meaning data that there is no Nan data point. Thus, some data are represented as strings.
# 
# If you look at the data description provided by [UCI Machine Learning](https://www.kaggle.com/uciml/student-alcohol-consumption/home) on Kaggle,  those string/object type features are, most of the time, binary features. It means that those the string-type data only have two values. Take a look at *sex* feature for example, only has either 'M' or 'F' values. The same goes for *school*, *Pstatus*, *romantic*, *internet*, *higher* features. 
# 
# Notice that we can use Dalc+Walc (Weekly alcohol consumption index) to measure individual alcohol consumption rate. Now let's  dive head first into our data. But first, we need to tidy up a few things before we start:

# In[ ]:


#Let's get the average grade of three exams:
student_df["G_avg"] = (student_df["G1"]+student_df["G2"]+student_df["G3"])/3

#Then combine workday and weekend alcohol consumption
student_df["Dalc"] = student_df["Dalc"]+student_df["Walc"]


# ## Univariate plotting
# 

# Looking to each individual data, let's do some plotting. The first is you might wanna know how student is distributed over alcoholic consumption rate. 

# In[ ]:


# Pie plot & Bar plot

#setup pie plot
labels= student_df["Dalc"].unique()
sizes = student_df["Dalc"].value_counts()[labels]
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)  # explode 1st slice

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

#plotting pie plot
ax1.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True);
ax1.axis('equal')
ax1.set_title("Pie Plot")

#plotting bar plot
sns.countplot(x="Dalc", data=student_df, ax=ax2).set_title("Count/Bar plot");


# **Conclusion:**
# tbd

# # How alcohol affect student's performance during Mathematics exam?
# 
# That should be the big question of everyone, isn't it? However the question seems to abstract and not well defined. Ask yourself, how do we quantify good performance (eg. by each student's highest score, or by average score over G1-G3). Aside alcohol consumption, other factors play a significant role in affecting exams performance. 
# 
# So how about dividing the big question into several detailed questions? First, let us agree that student performance can be measure using the average of three mathematic exams (G1, G2, G3). Now, these questions will guide us to get insights from different perspectives to answer the big one.
# 

# 
# ## Bivariate plot
# ### How alcohol consumption affect student's average grade over different gender? Over relationship status?
# 
# Let's go!

# In[ ]:


#define bivariate bar plotting function
def bivariatte_barplot(df, x="Dalc", y="G_avg", hue=None, ax=None, color_set=1):
  pivtab_ser = df.groupby([x, hue])[y].mean().reset_index()
  #plotting
  sns.barplot(x=x, y=y, hue=hue,
               data=pivtab_ser, ax=ax, palette="Set%s"%(color_set+1)).set_ylabel("average grade")


# In[ ]:


# categorical features to plot
hues = ['sex', 'school', 'romantic']

#plotting
fig, axes = plt.subplots(1, len(hues), figsize=(17,6), sharey=True)
for idx, hue in enumerate(hues):
  bivariatte_barplot(student_df, hue=hue, ax=axes[idx], color_set=idx)


# ## Distribution plot
# To understand further  how alcohol consumption affect grades, let's compare their grades over the average scores. Let's see how each drunk students performs against each other.

# In[ ]:


#calculate average students grade over the entire batch
#then mark if student performs below or above average 
avg_batch = student_df["G_avg"].mean()
student_df['is_abv_avg'] = student_df["G_avg"] > avg_batch


# In[ ]:


# swarm plot
fig, ax = plt.subplots(1,1,figsize=(10,7))
g = sns.swarmplot(x="Dalc", y="G_avg", 
                  hue="is_abv_avg", data=student_df, ax=ax, palette="Set1")


# **Conclusion:**
# 
# tbd

# ## Multivariatte Plotting
# 
# When we want to see how student of multiple category performs in exams, we can use factorplot to combine multiple categorical features.

# In[ ]:


# make a new columns to mark
student_df["is_healthy"] = student_df["health"]>=3

student_df.head()


# In[ ]:


# define function for factor plotting
def multivariatte_factplot(x="Dalc", y="G_avg", hue="sex", col="is_healthy", df=pd.DataFrame(), cs=1):
  piv_tab = df.groupby([x, hue, col])[y].mean().reset_index()
  #plotting
  sns.factorplot(x=x, y=y, hue=hue, col=col, 
                     data=piv_tab, kind='bar', palette="Set%s"%cs, size=7);
  
# execute factor plotting
multivariatte_factplot(df=student_df);


# **Conclusion:**
# 
# Surprisingly, unhealthy males with drunk level (alcoholic consumption index)=7 turns out to perform better in school then any other category. The opposite happens for the sex counter-part, female with drunk level = 7 score the lowest among all groups. What confuse me then is the fact that healthy male with drunk level=6-7 perform much worse as well as the unhealthy counterpart. It seems that alcoholic consumption, health, and grades has a peculiar relationship.
# 
# In addition, unhealthy female with drunk level=4 performs the best among other female groups. Idk how heavy drinker are you to have drunk level >=4 but having a good pubtime with a right dosis seems to enhance performance. Happy mind happy brain... 
# 
# Another interesting thing to note is that health and alcohol consumption might do harm to students' performance on the extreme level. 

# In[ ]:


# another example
multivariatte_factplot(hue="school", col="address", df=student_df, cs=2);


# **Concolusion:**
# 
# Interesting things happen in the rural area. The non-frequent drinkers in that area eventually scores the highest, so perhaps those students manage their study well and only go for fun at the weekend. This story might make sense since student in rural area need to go to town to buy drinks, hence they only do it on weekend. Good boy!
# 
# On the other side, interesting things happend in the urban area. Those who has alcoholoic consumption index =7 in urban area do score the highest.

# In[ ]:




