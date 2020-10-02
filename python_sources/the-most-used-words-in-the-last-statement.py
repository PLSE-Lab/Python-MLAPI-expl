#!/usr/bin/env python
# coding: utf-8

# The purpose of this script is to get a quick visualization of the most used words in the last statement, grouping them by the race of the offender. What we pretend is to see if the words used are different depending on the race, maybe due to educational, culture or other external agents. 
# 
# 
# Sorry for the possible mistakes (it's my first contribution) and sorry for my English (I'm trying to improve it)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt  
import matplotlib
from wordcloud import WordCloud, STOPWORDS #wordcloud's generator and english STOPWORDS list
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

pd.options.mode.chained_assignment = None #Just disabling SettingWithCopyWarning
# Any results you write to the current directory are saved as output.


# First of all, we need to know how many different races are there in the dataset.

# In[ ]:


df = pd.read_csv("../input/offenders.csv", encoding = "latin1")
races = df["Race"].unique()
print (races)


# We found we have four different races: White, Hispanic, Black and Other.
# 
# As we can see, we have a little problem with the first two. It seems that, in some rows, a blank space is added, so we have that "White " is different from "White", and "Hispanic" from "Hispanic ".
# We could use "regex" or another module to fix it, but in this case I find easier to just replace the words.

# In[ ]:


for i in range(0,len(df)):
	race = df["Race"].iloc[i]
	if race == "White ":
		 df["Race"].iloc[i] = "White"
	elif race == "Hispanic ":
		 df["Race"].iloc[i] = "Hispanic"
	else:
		pass
    
races = df["Race"].unique()
print (races)


# Now the problem is fixed, we can get a different dataframe for each race and start getting wordclouds. However,
# before that, we need to know the number of rows by race.

# In[ ]:


sns.factorplot('Race',data=df,kind='count')
plt.title("Number of offenders by race")
plt.show()


# It seems that the rows of offenders with "other" race are not enough to make a wordcloud, but we can work with the rest.
# Let's get the different dataframes and plot the wordclouds!

# In[ ]:


white = df[df["Race"]=="White"]
hispanic = df[df["Race"]=="Hispanic"]
black = df[df["Race"]=="Black"]

#Wordcloud of white offender's last statements:
wordcloud = WordCloud(
                         stopwords=STOPWORDS,
                         background_color='white',
                         width=1200,
                         height=1000
                        ).generate(" ".join(white['Last Statement']))

plt.imshow(wordcloud)
plt.title("White offenders")
plt.axis('off')
plt.show()

#Wordcloud of hispanic offender's last statements:
wordcloud = WordCloud(
                         stopwords=STOPWORDS,
                         background_color='white',
                         width=1200,
                         height=1000
                        ).generate(" ".join(hispanic['Last Statement']))

plt.imshow(wordcloud)
plt.title("Hispanic offenders")
plt.axis('off')
plt.show()

#Wordcloud of black offender's last statements:
wordcloud = WordCloud(
                         stopwords=STOPWORDS,
                         background_color='white',
                         width=1200,
                         height=1000
                        ).generate(" ".join(black['Last Statement']))
plt.imshow(wordcloud)
plt.title("Black offenders")
plt.axis('off')
plt.show()


# With this quick visualization, we can find some interesting **conclussions**:
# 
#  - It seems there are not too many differences between the words used by each race.
#  - Generally, the offenders of this dataset (no matter their race), frequently used the words "love" and "family". "Thank" and "forgive" were frequent too.
#  - It appears that saying "sorry" in the last statement was also important for them, but specially for hispanic offenders.
# 
