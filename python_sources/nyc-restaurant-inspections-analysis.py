#!/usr/bin/env python
# coding: utf-8

# <center><img src="http://www.marea-nyc.com/files/7172260197fc3b8e8368bc1cc0fc5e9c_full_size.jpg"></center>

# # Introduction

# The data presented is for restaraunt inspections for permitted food establishments in NYC.  The restaraunts are graded from A-F by inspectors from the department of health.
# 
# The data has many factors and covers all of NYC and is from Jan 1, 2010 to Aug 29, 2017.
# 
# The plan is to:
# 
# Find out:
#   Are certain violations more prominent in certain neighborhoods? By cuisine?
#   Who gets worse grades--chain restaurants or independent establishments?
# 
# If there are any suggestions/changes you would like to see in the Kernel please let me know. 
# 
# Please leave any comments about further improvements to the notebook! Any feedback or constructive criticism is greatly appreciated!. **If you like it or it helps you , you can upvote and/or leave a comment**
# 
# **This is my first go around of using python for any analysis so any feedback would be greatly appreciated**

# ## Load Libraries

# In[4]:


# Let us import what we need to the analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import datetime 
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Data

# Now that we have all that we need let us get down to business. Firstly let us get the data imported and take a look at a few observations

# In[5]:


data = pd.read_csv('../input/DOHMH_New_York_City_Restaurant_Inspection_Results.csv')


# In[ ]:


# Lets us look at a few records
data.head()


# Let us find out the column names for what we just pulled in and looked at.

# In[ ]:


# Get the coulmns in the dataset
data.columns


# # Data Analysis

# ## Check for Missings

# In[ ]:


# checking missing data in data 
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# **Half the records do not have a Grade** maybe do not include this in a model.
# **Violation(s) variables, and Score are mostly filled**, can be included in a model.
# **The Score variable can be imputed for he missings**

# Let us start looking at the variables in the dataset

# ## Score Analysis 

# How about getting an idea of what the scores look like.  The score is based on the results of the inspection

# In[ ]:


data['SCORE'].describe()


# So we know that the **mean score is 18.910181**, with 376,704 records having a score assigned.

# Let us have a graphical representation of the Scores

# In[ ]:


# Plot a histogram
data.SCORE.hist(figsize=(10,4))
plt.title("Boxplot for the Scores", fontsize=15)
plt.xlabel('Score', fontsize = 12)


# The histogram shows a bunching of scores to the left

# In[ ]:


# Have a look at a distribution plot of the Score
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
sns.distplot(data.SCORE.dropna())
plt.title("Distribution Plot of the Scores", fontsize=15)


# The distribution has a very long tail

# In[ ]:


# Let us look at a violin plot of the scores
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
sns.violinplot(data.SCORE.dropna())
plt.title("Violin plot of the Scores", fontsize=15)


# We can see that most of the scores are close to the mean.  There are two main groupings of the data.  There are a few outliers in the data

# ## Grades Analysis

# How are the grades looking?

# In[ ]:


data.GRADE.value_counts()


# **Most of the restaurants have A's**.  There are a few restaurants that are still waiting to get their grades

# In[ ]:


# A look at the histogram of the Grades.
data.GRADE.hist(figsize = (15, 4))
plt.title("Histogram of the Grades", fontsize=15)
plt.xlabel('Grades', fontsize = 12)


# **Most of the grades are A's**

# In[ ]:


# Lets look at scores by grades
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
sns.boxplot(data.SCORE.dropna(), data.GRADE)
plt.title('Boxplot by Grade', fontsize = 15)


# Now looking at the plot above it seems as if you **want to get the lowest score** as possible as this means you have an A for your inspection.

# ## Boroughs Analysis

# What borough has the most inspections?

# In[ ]:


# Look at whih Boroughs have the highest number of inspections
data.BORO.value_counts()


# **Manhattan has the most number of inspections by almost 60,000 inspections to second placed Brooklyn.**  There are 9 records in the dataset that have not been assigned a borough.

# In[ ]:


# Here is a look at a histogram of the numbers we just saw above.
data.BORO.hist(figsize = (15, 4))
plt.title('Boxplot of the count of inspections per Borough', fontsize = 15)
plt.xlabel('Borough', fontsize = 12)


# There aren't a lot of inspections in Staten Island.  I wonder how many restaurants are in Staten Island.  It would be interesting to see how many of the restaurants are inspected.

# In[ ]:


# Breakdown scores by borough
fig, ax = plt.subplots()
fig.set_size_inches(15, 4)
sns.boxplot(data.SCORE.dropna(), data.BORO)
plt.title('Boxplot by Borough', fontsize = 15)


# Brooklyn, Manhattan and Queens have outliers, with the highest one in Brooklyn.  **The boroughs are very similar in terms of the score distributions**

# In[ ]:


# Contingency table for Grade and Borough
boro_grade = pd.crosstab(data.GRADE, data.BORO, margins = True)
boro_grade


# Lookig at the contingency table above, we can see the breakdown of the grades across the boroughs

# In[ ]:


# Plot of grade by borough
pd.crosstab(data.BORO, data.GRADE).plot(kind="bar", figsize=(15,8), stacked=True)
plt.title('Grade Distribution by Borough', fontsize = 15)


# **The majority of the inspections resulted in A's across all the boroughs**

# In[ ]:


# Test if the grades are independent of the borough
boro_grade.columns = ["BRONX","BROOKLYN","MANHATTAN", "QUEENS", "STATEN ISLAND" ,"All"]

boro_grade.index = ["A","B","C","Not Yet Graded","P", "Z", "All"]

observed = boro_grade.ix[0:6,0:5]   # Get table without totals for later use

expected =  np.outer(boro_grade["All"][0:6],
                     boro_grade.ix["All"][0:5]) / 1000

expected = pd.DataFrame(expected)

expected.columns = ["BRONX","BROOKLYN","MANHATTAN", "QUEENS", "STATEN ISLAND"]
expected.index = ["A","B","C","Not Yet Graded","P", "Z"]

chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()

print("Chi Squared Stat")
print(chi_squared_stat)

crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 20)   # (5-1) * (6-1)

print("Critical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=20)
print("P value")
print(p_value)

stats.chi2_contingency(observed= observed)


# **As expected, given the small p-value, the test result detects a significant relationship between the variables**

# ## Cuisine Analysis

# What cuisine is the most popular when it comes to inspections?

# In[ ]:


data['CUISINE DESCRIPTION'].value_counts()


# It seems as if **restaurants that serve American cuisine have the most number of inspections**, and this makes sense as the data is from an American city.

# In[ ]:


# Let us look at the scores by cuisine
score_cuisine = pd.concat([data['CUISINE DESCRIPTION'], data['SCORE']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x = 'CUISINE DESCRIPTION', y="SCORE", data = score_cuisine)
plt.xticks(rotation=90);


# There is a lot of infomation to be gleaned from the plot above.  One of the things to notice is the **variablity in the score across the many cuisine descriptions provided.**

# ## Action Taken Analysis

# Which action was chosen the most?

# In[ ]:


data.ACTION.value_counts()


# **The most common action taken was citing the violation**

# In[ ]:


# Histogram of the Action taken
data.ACTION.hist(figsize = (15,8))
plt.title('Histogram of the Action taken', fontsize = 15)
plt.xlabel('Action', fontsize = 12)
plt.xticks(rotation=90)


# ## Critical Flag Analysis

# Which flag was the most popular?

# In[ ]:


data['CRITICAL FLAG'].value_counts()


# **Critical violations are those most likely to contribute to foodborne illness and most cases were critical**

# In[ ]:


# Graphical representation of the critical flag
data['CRITICAL FLAG'].hist(figsize=(15,4))
plt.title('Histogram of the Critical Flag', fontsize = 15)
plt.xlabel('Flag', fontsize = 12)


# In[ ]:


# Critical Flag by Borough
pd.crosstab(data.BORO, data['CRITICAL FLAG']).plot(kind="bar", figsize=(15,8), stacked=True)
plt.title('Critical Flag by Borough', fontsize = 15)


# **The majority of flags for each borough were critical flags.**

# In[ ]:


# Critical Flag by Cuisine
pd.crosstab(data['CUISINE DESCRIPTION'], data['CRITICAL FLAG']).plot(kind="bar", figsize=(18,18), stacked=True)
plt.title('Critical Flag by Cuisine', fontsize = 15)


# **Again American cuisine has the most critical flags**, and as before this makes sense since the data is from an American city

# In[ ]:


# Let us look at the scores by critical flag
score_flag = pd.concat([data['CRITICAL FLAG'], data['SCORE']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x = 'CRITICAL FLAG', y="SCORE", data = score_flag)
plt.title('Score by Critical Flag', fontsize = 15)
plt.xticks(rotation=90);


# **Looking at the plot above it seems as if the establishmens with a higher score are usually flagged as critical**

# ## Inspection

# Which inspection type was the most popular?

# In[ ]:


data['INSPECTION TYPE'].value_counts()


# ## Violation Description WordCloud

# In[6]:


import re
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()# lowercase text  
    text = REPLACE_BY_SPACE_RE.sub(' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text    
    text = BAD_SYMBOLS_RE.sub('',text)# delete symbols which are in BAD_SYMBOLS_RE from text    
    temp = [s.strip() for s in text.split() if s not in STOPWORDS]# delete stopwords from text
    new_text = ''
    for i in temp:
        new_text +=i+' '
    text = new_text
    return text.strip()


# In[7]:


# Let us create a word cloud for the violation description
temp_data = data.dropna(subset=['VIOLATION DESCRIPTION'])
# converting into lowercase
temp_data['VIOLATION DESCRIPTION'] = temp_data['VIOLATION DESCRIPTION'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['VIOLATION DESCRIPTION'] = temp_data['VIOLATION DESCRIPTION'].map(text_prepare)


from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['VIOLATION DESCRIPTION'].values))
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.title("Top Words Used for the Violation Descriptions", fontsize=25)
plt.axis("off")
plt.show() 


# Looking at the cloud above it seems as if most of the violaions are to do with nonfood contact and the surfaces the food is in contact with.

# # Conclusions
# 
# From looking at the variables above we can see some differences across the boroughs and also the cuisine.  We can see that the action taken is also different depending on what the score of the inspection is, so if we were to build a prediction model to determine what action was to be taken it wold be important to include the score variable in that model I would think.  There is more work to be done with this data.

# 
