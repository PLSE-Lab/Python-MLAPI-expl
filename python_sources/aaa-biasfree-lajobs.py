#!/usr/bin/env python
# coding: utf-8

# # Nelson Submission - AAA Bias Free LA Jobs
# 
# #### First, thank you to the City of Los Angeles and Kaggle for hosting this interesting competition.  I'm already using these tools at work to improve our documents.  
# 
# ### Challenge:
# 
# The City of Los Angeles requested tools to check Job Bulletins for biased words and a CSV file containing Job Bulletin data plus Career Paths in DAG format.  A DAG is a special data structure that looks like a family-tree.  A Career Path DAG contains the path a City employee would take for promotions.
# 
# ### Solution:
# 
# I offer this two-part strategy:
# 
# **Part 1** a prototype app that finds bias words using a Natural Language Processing (NLP) 
# 
# **Part 2** proof of concept scripts that create a CSV file with text gathered from the Job Bulletins and Career Paths.  I included Career Path data in DAG format and added a NEXT_PROMOTION field.  
# 
# 

# In[ ]:


from IPython.display import Image
Image("../input/jobbulletinresults/machine.png")


# #### Part 1:  Finding Bias Words using Natural Language Processing (NLP)
# 
# RESULTS:  The City of Los Angeles has taken care to write Job Bulletins that are free from bias.  Only 2% of the words in Job Bulletins may be biased.  Only 12% of the time biased words are used.
# 
# Having led projects in 27 countries (so far), I study how language affects team productivity.  Teams thrive in collaborative environments with clear goals.  When staffing teams, I write job descriptions using short, simple words so that all candidates can read them, even "non-native" English speakers who didn't learn English as their first language.  
# 
# Additionally, I realize that some team members respond negatively to gender-bias words.  Some job candidates told me that they don't like a "competitive" work environment.  Steering clear of bias can be difficult.  Fortunately, Gaucher, Friesen, and Kay researched gender bias in job advertisements and made a list of gender-coded words. They published this list in "Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality" in the Journal of Personality and Social Psychology, July 2011, Vol 101(1), p109-28.  I used this list to evaluate the City of Los Angeles Job Bulletins.  
# 
# Text analysis can be fast and accurate because the original Job Bulletins are processed directly.  There is no "middle-man" converting them to a database first.  I was able to go beyond proof of concept and conduct an automated review looking for biased words that might discourage job applicants.  The City of Los Angeles scored very well.  Check out this kernel:
# 
# [BiasAnalyzer kernel](https://www.kaggle.com/nelsondata/BiasAnalyzer)
# 
# BiasAnalyzer is modular so you can plug-in other tools from this competition or from other sources.  Here are two modules that I included:
# 
# [WordCounts kernel](https://www.kaggle.com/nelsondata/wordcounts)
# 
# [Big Huge Thesaurus API kernel](https://www.kaggle.com/nelsondata/bighugethesaurusapi-jobads)
# 
# The following diagram shows how it works:

# In[ ]:


from IPython.display import Image
Image("../input/jobbulletinresults/Bias.png")


# ### Part 2:  Gathering data from Job Bulletins and Career Paths
# 
# First, I used the [Text2CSV kernel](https://www.kaggle.com/nelsondata/text2csv) to gather data from Job Bulletins.  Then, I used [Chart2DAG](https://www.kaggle.com/nelsondata/chart2dag) to generate DAGs.  I created a CSV file containing the data requested in the competition guidelines plus data from the Career Paths.
# 
# #### Text2CSV
# [Text2CSV kernel](https://www.kaggle.com/nelsondata/text2csv) gathers data from Job Bulletins and creates JobBulletin.csv
# 
# #### Chart2DAG
# [Chart2DAG](https://www.kaggle.com/nelsondata/chart2dag) reads the Career Path charts and generate DAGs.  Updates JobBulletins.csv.  I added the NEXT_PROMOTION field so that you can quickly see the next step in your career path. 
# 
# 

# In[ ]:


from IPython.display import Image
Image("../input/jobbulletinresults/Text2CSV.png")


# 
# #### List of Kernels
# 
# AAA-BiasFree-LAJob - https://www.kaggle.com/nelsondata/aaa-biasfree-lajobs (this kernel)
# 
# BiasAnalyzer kernel - https://www.kaggle.com/nelsondata/BiasAnalyzer
# 
# BigHugeThesaurusAPI-JobAds - https://www.kaggle.com/nelsondata/bighugethesaurusapi-jobads
# 
# Chart2DAG - https://www.kaggle.com/nelsondata/chart2dag
# 
# Text2CSV - https://www.kaggle.com/nelsondata/text2csv
# 
# WordCounts - https://www.kaggle.com/nelsondata/wordcounts
# 
# #### List of Datasets
# 
# CareerPathData - https://www.kaggle.com/nelsondata/careerpathdata
# 
# JobBulletinData - https://www.kaggle.com/nelsondata/jobbulletindata
# 
# JobBulletinResults -  https://www.kaggle.com/nelsondata/jobbulletinresults

# In[ ]:


'''
Script:  Stats.py
Purpose: Provide new insights about Job Bulletins and Career Paths
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# dnlp:  get statistics from JBR_NLP.py
dnlp = pd.read_csv('../input/jobbulletinresults/JBR_Output/fileStats.csv')
pd.options.display.max_columns=len(dnlp)

# dbias and dbw:  get statistics from biasVerifier.py
dbias = pd.read_csv('../input/jobbulletinresults/JBR_Output/BiasedWords.csv')
dbw = dbias[['FILE_NAME','WORD']]


# In[ ]:


# dbw2:  dataframe containing filename and the number of biased words found in it
dbw2 = pd.DataFrame({'TOT_BIAS_WORDS':dbw.groupby(['FILE_NAME'])['WORD'].count()}).reset_index()

# dstats: dataframe with statistics from word analysis in JBR_NLP and bias analysis in biasVerifier
dstats = pd.merge(dnlp,dbw2, how="left", on='FILE_NAME')
dstats["TOT_BIAS_WORDS"] = dstats["TOT_BIAS_WORDS"].fillna(0.0)
dstats["TOT_BIAS_WORDS"] = dstats["TOT_BIAS_WORDS"].astype('int64')


# In[ ]:


# djbstats:  merge statistics with Job Bulletin data
jb = pd.read_csv("../input/jobbulletinresults/JBR_Output/JobBulletin.csv")
djbstats = pd.merge(jb,dstats,how="left",on='FILE_NAME')
# df and df2: slice the dataframe to gather only numeric and categories useful for analysis
df = djbstats[['DRIVER_LICENSE_REQ', 'DRIV_LIC_TYPE', 'EDUCATION_YEARS','EXPERIENCE_LENGTH', 'FULL_TIME_PART_TIME', 'HIGH_SALARY', 'LOW_SALARY','SCHOOL_TYPE', 'TOT_DIF_WORDS', 'TOT_LONG_WORDS','TOT_WORDS', 'TOT_BIAS_WORDS']]
df = df.fillna(0)
df2 = pd.get_dummies(df)


# In[ ]:


# CHARTS and REPORTS

dnlp['TOT_WORDS'].plot(figsize=(10,5), kind='hist', color='black', title='Frequency of words in Job Bulletins')
plt.show()    

print("\nLongest Job Bulletin: ")
longest = dnlp['TOT_WORDS'].max()
longs = dnlp['FILE_NAME'][dnlp['TOT_WORDS'] == longest]
for long in longs: print(longest, " words in ", long)

print("\nShortest Job Bulletin:")
shortest = dnlp['TOT_WORDS'].min()
shorts = dnlp['FILE_NAME'][dnlp['TOT_WORDS'] == shortest]
for short in shorts: print(shortest, " words in ", short)

print("\nAverage Job Bulletin length:")
avgJB = dnlp['TOT_WORDS'].mean()
print(int(round(avgJB,0)), " words")


# In[ ]:


dnlp['TOT_DIF_WORDS'].plot(figsize=(10,5), kind='hist', color='navy', title='Frequency of different words in Job Bulletins')
plt.show()    
print("\nJob Bulletin with most different words: ")
longest = dnlp['TOT_DIF_WORDS'].max()
longs = dnlp['FILE_NAME'][dnlp['TOT_DIF_WORDS'] == longest]
for long in longs: print(longest, " words in ", long)

print("\nJob Bulletin with the least number of different words:")
shortest = dnlp['TOT_DIF_WORDS'].min()
shorts = dnlp['FILE_NAME'][dnlp['TOT_DIF_WORDS'] == shortest]
for short in shorts: print(shortest, " words in ", short)

print("\nAverage Job Bulletin length based on different words:")
avgJB = dnlp['TOT_DIF_WORDS'].mean()
print(int(round(avgJB,0)), " words")


# In[ ]:


dnlp['TOT_LONG_WORDS'].plot(figsize=(10,5), kind='hist', color='lightblue', title='Frequency of complex words found in Job Bulletins')
plt.show()    
print("\nJob Bulletin with the most complex words: ")
longest = dnlp['TOT_LONG_WORDS'].max()
longs = dnlp['FILE_NAME'][dnlp['TOT_LONG_WORDS'] == longest]
for long in longs: print(longest, " words in ", long)

print("\nJob Bulletin with the least number of complex words:")
shortest = dnlp['TOT_LONG_WORDS'].min()
shorts = dnlp['FILE_NAME'][dnlp['TOT_LONG_WORDS'] == shortest]
for short in shorts: print(shortest, " words in ", short)

print("\nAverage Job Bulletin complexity:")
avgJB = dnlp['TOT_LONG_WORDS'].mean()
print(int(round(avgJB,0)), " words")


# In[ ]:


print("Job Bulletins ranked from longest to shortest:\n")
dtot = dnlp[['FILE_NAME', 'TOT_WORDS']]
print(dtot.sort_values('TOT_WORDS', ascending=False))


# In[ ]:


print("Job Bulletins ranked from longest to shortest based on number of different words used:\n")
ddif = dnlp[['FILE_NAME', 'TOT_DIF_WORDS']]
print(ddif.sort_values('TOT_DIF_WORDS', ascending=False))


# In[ ]:


print("Job Bulletin ranked from most complex words used to least:\n")
dlong = dnlp[['FILE_NAME', 'TOT_LONG_WORDS']]
print(dlong.sort_values('TOT_LONG_WORDS', ascending=False))


# In[ ]:


# Feature correlation
# Features are columns of data.  Feature correlation shows which columns affect each other.  
# A heatmap will highlight correlated data so we can investigate it 

print("\nFeatures (interesting data that can be analyzed): ")
cols = df2.columns
for col in cols:  print(col)

#plt.matshow(df2.corr())
#plt.show()

def seabornCM(corr):
    import seaborn as sns
    sns.heatmap(corr, cmap="Blues")

seabornCM(df2.corr())
plt.show()


# In[ ]:


print("\nQUESTIONS POSED FROM ANALYZING FEATURES")

df2.plot(figsize=(10,5), x="LOW_SALARY", y="TOT_BIAS_WORDS", kind="scatter", title="Do jobs with higher salaries have more biased words?")
plt.show()

print("\nDo jobs with higher salaries have more biased words?")
print("Yes, jobs with higher salaries tend to have more biased words.")


# In[ ]:


print("\nI'm a Public Relations Specialist, what is the next step in my career path?")
nextPromotions = jb['NEXT_PROMOTION'][jb['JOB_CLASS_TITLE'] == 'PUBLIC RELATIONS SPECIALIST']
for nextPromotion in nextPromotions: print(nextPromotion)
careerPathImg = "../input/jobbulletinresults/"+jb['DAG_FILE'][jb['JOB_CLASS_TITLE'] == 'PUBLIC RELATIONS SPECIALIST']

img = mpimg.imread(careerPathImg.item())
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


df2["EXPERIENCE_LENGTH"] = df2["EXPERIENCE_LENGTH"].fillna(0.0)
df2.plot(figsize=(10,5), x="EXPERIENCE_LENGTH", y="TOT_BIAS_WORDS", kind="scatter",  color="grey", title="Do jobs requiring more experience have more biased words?")
plt.show()

print("\nDo jobs requiring more experience have more biased words?")
print("Inconclusive because jobs paying $100k to 150k tend to be more scientific so these Job Bulletins have more complex words.  I plan to investigate only the gender-based words as a future project")


# In[ ]:


df2["EDUCATION_YEARS"] = df2["EDUCATION_YEARS"].fillna(0.0)
df2.plot(figsize=(10,5), x="EDUCATION_YEARS", y="TOT_BIAS_WORDS", kind="scatter",  color="teal", title="Do jobs requiring more years of education have more biased words?")
plt.show()

print("\nDo jobs requiring more years of education have more biased words?")
print("Inconclusive for non-native speakers bias because Job Bulletins for jobs requiring more education often include words related to a specific field of study, such as oceanography...  If a professional is skilled in that field, they should know these words even if they are not a native speaker.")


# In[ ]:


df2["EXPERIENCE_LENGTH"] = df2["EXPERIENCE_LENGTH"].fillna(0.0)
df2.plot(figsize=(10,5), x="EXPERIENCE_LENGTH", y="LOW_SALARY", kind="scatter", color="green", title="Do jobs requiring more experience pay a higher salary?")
plt.show()

print("\nDo jobs requiring more experience pay a higher salary?")
print("Inconclusive because EXPERIENCE_LENGTH needs to be double-checked.  It appears that some Job Bulletins only request the last 2-3 years of experience rather than a complete history while others ask for the history")


# In[ ]:


word_in_file = dbw.groupby(['FILE_NAME'])['WORD'].count()
print("\nSTATISTICS FOR BIASED WORDS USED IN EACH JOB BULLETIN:")
print("\nMost biased words used in one Job Bulletin: ", word_in_file.max())
print("Least biased words used in one Job Bulletin: ", word_in_file.min())
print("Average biased words per Job Bulletin: ", int(round(word_in_file.mean(),0)))

print("\nThe number of times a biased word was used in each file:\n")
print(word_in_file.sort_values(ascending=False))
print("\nThe number of times each biased word was used in each file:")
print(dbw.groupby(['FILE_NAME','WORD'])['WORD'].count())


# In[ ]:


dstats['TOT_BIAS_WORDS'].plot(figsize=(10,5), kind='hist', color='teal', title='Frequency of potentially biased words found in Job Bulletins')
plt.show()   


# In[ ]:


img = mpimg.imread("../input/jobbulletinresults/results.png")
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


locateBias = dbias[['FILE_NAME','WORD_LOCATION','SENTENCE']]
locateBias = locateBias.drop_duplicates()
print("Number of times a biased word was found after dropping duplicates: ", len(locateBias))


# In[ ]:


print("Job Bulletins containing 'open competitive candidates'")
res = locateBias[locateBias["SENTENCE"].str.contains("open competitive candidates", na=False)]
print(res)

# "Competition attracts male candidates.  Could a gender-neutral term be used instead?
# Perhaps "open-hire candidates" or "new-hire candidates"?  
# If "open competitive candidates" is the required term, then the City can place this word on the non-bias list
# and it will not appear in future analysis


# In[ ]:


img = mpimg.imread("../input/jobbulletinresults/male2.png")
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


print('Job Bulletins containing "COMPETITIVE BASIS"\n')
res = locateBias[locateBias["SENTENCE"].str.contains(">>> COMPETITIVE <<<  BASIS", na=False)]
print(res)

#Competitive basis" is a term that refers to the examination process.  Could a gender-neutral term be used such as "skills basis"
# If not, the City can add this term to the non-bias list 


# In[ ]:


print('Job Bulletins containing "determines"\n')

res = locateBias[locateBias["SENTENCE"].str.contains("determines", na=False)]
print(res)

# "determines" attracts male candidates.  Could a gender-neutral word be used in these sentences?
# For example, "determines procedures and methods" could be replaced with "finds procedures and methods"


# In[ ]:


print('Job Bulletins containing "competencies"\n')
res = locateBias[locateBias["SENTENCE"].str.contains("competencies", na=False)]
print(res)

# "competencies" is a complex word rarely used in daily speech.
# "competencies" could be replaced with "skills" because "skills" is a more commonly used word


# In[ ]:


print("Other words were found that are biased towards women and non-native speakers")


# In[ ]:


img = mpimg.imread('../input/jobbulletinresults/female.png')
imgplot = plt.imshow(img)
plt.show()

print('Job Bulletins containing the most used word biased towards women:  "responsibilities"\n')
res = locateBias[locateBias["SENTENCE"].str.contains("responsibilities", na=False)]
print(res)


# In[ ]:


img = mpimg.imread('../input/jobbulletinresults/female2.png')
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


img = mpimg.imread('../input/jobbulletinresults/nonnative.png')
imgplot = plt.imshow(img)
plt.show()


# In[ ]:


img = mpimg.imread('../input/jobbulletinresults/nonnative-2.png')
imgplot = plt.imshow(img)
plt.show()

print('Job Bulletins containing the most used word biased towards non-native speakers: "examination"\n')
res = locateBias[locateBias["SENTENCE"].str.contains("examination", na=False)]
print(res)

# Can "examination" be replaced with "exam" or "test"?

