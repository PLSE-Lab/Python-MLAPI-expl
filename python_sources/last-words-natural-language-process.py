#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# <br>
# <p>Death penalty is a government-sanctioned practice whereby a person is killed by the state as a punishment for a crime. Capital punishment is a matter of active controversy in several countries and states, and positions can vary within a single political ideology or cultural region. The Council of Europe, which has 47 member states, has sought to abolish the use of the death penalty by its members absolutely, through Protocol 13 of the European Convention on Human Rights. However, this only affects those member states which have signed and ratified it, and they do not include Armenia, Russia, and Azerbaijan. Although most nations have abolished capital punishment, over 60% of the world's population live in countries where the death penalty is retained, such as China, India, the United States, Indonesia, Pakistan, Bangladesh, Nigeria, Egypt, Iran, among all mostly Islamic countries, as is maintained in Japan and Sri Lanka. <a href="https://en.wikipedia.org/wiki/Capital_punishment">Learn more</a></p>
# <p>This dataset includes information on criminals executed by Texas Department of Criminal Justice from 1982 to November 8th, 2017
# <br/><br/>The dataset consists of 545 observations with 21 variables. They are: <br/>
#     - <b>Execution</b>: The order of execution, numeric. <br/>
#     - <b>LastName</b>: Last name of the offender, character. <br/>
#     - <b>FirstName</b>: First name of the offender, character. <br/>
#     - <b>TDCJNumber</b>: TDCJ Number of the offender, numeric. <br/>
#     - <b>Age</b>: Age of the offender, numeric. <br/>
#     - <b>Race</b>: Race of the offender, categorical : Black, Hispanic, White, Other. <br/>
#     - <b>CountyOfConviction</b>: County of conviction, character. <br/>
#     - <b>AgeWhenReceived</b>: Age of offender when received, numeric. <br/>
#     - <b>EducationLevel</b>: Education level of offender, numeric. <br/>
#     - <b>Native County</b>: Native county of offender, categorical : 0 = Within Texas, 1= Outside Texas. <br/>
#     - <b>PreviousCrime</b> : Whether the offender committed any crime before, categorical: 0= No, 1= Yes. <br/>
#     - <b>Codefendants</b>: Number of co-defendants, numeric. <br/>
#     - <b>NumberVictim</b>: Number of victims, numeric. <br/>
#     - <b>WhiteVictim, HispanicVictim, BlackVictim, VictimOtherRace. FemaleVictim, MaleVictim</b>: Number of victims with specified demographic features, numeric. <br/>
#     - <b>LastStatement</b>: Last statement of offender, character.</p>
#     
# 
# <br><br>
# <img src="https://fee.org/media/29658/deathpenaltyelectricchair.jpg?center=0.42737430167597767,0.48&mode=crop&width=1920&rnd=131783744600000000" width="700px" />
# 
# 
# <br>
# <br>
# <br>
# 
# **We will try to predict if an offender is male or female by using his/her last statement and age.**

# ## <font color = "blue">**Reading and Cleaning Data**</font>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/Texas Last Statement - CSV.csv", encoding="latin1")


# In[ ]:


df.head()


# In[ ]:


df["Gender"] = 0
genderList = [1 if i == 1.0 else 0 for i in df.MaleVictim]
df.Gender = genderList


# In[ ]:


df.drop(["Execution","TDCJNumber","PreviousCrime","Codefendants","NumberVictim","WhiteVictim","HispanicVictim","BlackVictim","VictimOther Races",],axis=1, inplace=True)


# In[ ]:


df.drop(["FemaleVictim","MaleVictim"],axis=1,inplace=True)

df.head()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.AgeWhenReceived = df.AgeWhenReceived.astype(int)
df.EducationLevel = df.EducationLevel.astype(int)
df.head()


# ## <font color = "blue">**Explore Data**</font>

# ### Ages

# In[ ]:


plt.figure(figsize=(12,7))
sns.boxplot(y=df.Age)
plt.show()

print("Minimum Age is {}".format(df.Age.min()))
print("Maximum Age is {}".format(df.Age.max()))
print("Average Age is {:.1f}".format(df.Age.mean()))


# In[ ]:


df.corr()


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), linewidths=2, annot=True)
plt.show()


# Not suprisingly Age and AgeWhenReceived correlated with each other. But other features not saying much. Let's continue.

# In[ ]:


educationList= sorted(list(zip(df.EducationLevel.value_counts().index, df.EducationLevel.value_counts().values)))
eduYear, eduCount = zip(*educationList)
eduYear, eduCount = list(eduYear), list(eduCount)

plt.figure(figsize=(16,10))
plt.xlabel("Education Level")
plt.ylabel("Number of Offender")
plt.title("Number of Offender According To Their Education Level")
sns.barplot(x=eduYear,y=eduCount)
plt.show()


# We can see that most of the offenders were graduated from high school or they were high school drop out.

# In[ ]:


plt.figure(figsize=(16,10))
plt.xlabel("Last Name")
plt.ylabel("Frequency")
plt.title("Most 10 Frequent Last Names")
sns.barplot(x=df.LastName.value_counts()[:11].index,y=df.LastName.value_counts()[:11].values, palette="cubehelix")
plt.show()


# In[ ]:


plt.figure(figsize=(16,10))
plt.xlabel("First Name")
plt.ylabel("Frequency")
plt.title("Most 10 Frequent First Names")
sns.barplot(x=df.FirstName.value_counts()[:11].index,y=df.FirstName.value_counts()[:11].values, palette="gist_ncar_r")
plt.show()


# In[ ]:


import squarify

plt.figure(figsize=(15,8))
squarify.plot(sizes=df.Race.value_counts().values, label=df.Race.value_counts().index, color=["#17A096","#CBC015","#E4595D", "#979797"], alpha=.8 )
plt.axis('off')
plt.show()


# In[ ]:


print(df.Race.value_counts())


# Majority of offenders were White with 220 persons. 188 of offenders were Black and 90 of them were Hispanic.

# ## <font color="blue">**Natural Language Process**</font>

# Now we have to do some operations to prepare data but before applying that operations whole statements I will show you what actually we will do with one example step by step.

# In[ ]:


# We use regular expression to delete non-alphabetic characters on data.
import re

first_lastStatement = df.LastStatement[0]
lastStatement = re.sub("[^a-zA-Z]"," ",first_lastStatement)


# In[ ]:


print(lastStatement)


# In[ ]:


# Since upper and lower characters are (e.g a - A) evaluated like they are different each other by computer we make turn whole characters into lowercase.

lastStatement = lastStatement.lower()
print(lastStatement)


# In[ ]:


import nltk  # Natural Language Tool Kit

nltk.download("stopwords")  # If you dont't have that module this line will download it.
nltk.download('punkt') # It's necessary to import the module

from nltk.corpus import stopwords # We are importing 'stopwords'

lastStatement = nltk.word_tokenize(lastStatement) # We tokenized the statement

print(lastStatement)


# In[ ]:


# We will remove words like 'the', 'or', 'and', 'is' etc.

lastStatement = [i for i in lastStatement if not i in set(stopwords.words("english"))]
print(lastStatement)


# #### Lematization

# In[ ]:


# e.g: loved => love

nltk.download('wordnet') # It can be necessary
import nltk as nlp

lemmatization = nlp.WordNetLemmatizer()
lastStatement = [lemmatization.lemmatize(i) for i in lastStatement]

print(lastStatement)


# In[ ]:


# Now we turn our lastStatement list into sentence again

lastStatement = " ".join(lastStatement)

print(lastStatement)


# ## Preparing Entire Data

# In[ ]:


statementList = list()

for statement in df.LastStatement:
    statement = re.sub("[^a-zA-Z]"," ",statement)
    statement = statement.lower()
    statement = nltk.word_tokenize(statement)
    statement = [i for i in statement if not i in set(stopwords.words("english"))]
    statement = [lemmatization.lemmatize(i)for i in statement]
    statement = " ".join(statement)
    statementList.append(statement)


# In[ ]:


statementList


# ### Bag of Words

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

max_features = 600
count_vectorizer = CountVectorizer(max_features=max_features) 
sparce_matrix = count_vectorizer.fit_transform(statementList)
sparce_matrix = sparce_matrix.toarray()

print("Most Frequent {} Words: {}".format(max_features, count_vectorizer.get_feature_names()))


# In[ ]:


y = df.iloc[:,9].values # gender column
x = sparce_matrix


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 42)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

# Prediction
y_pred = nb.predict(x_test)


# In[ ]:


print("Accuracy: {:.2f}%".format(nb.score(y_pred.reshape(-1,1), y_test)*100))


# Let's try this algorithm by adding age feature beside last statement.

# In[ ]:


df2 = pd.DataFrame(sparce_matrix)
df2["Age"] = df.Age
df2["Age"].fillna((df2["Age"].mean()),inplace=True)

# Normalization
x2 = (df2 - np.min(df2)) / (np.max(df2) - np.min(df2)).values


# In[ ]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y, test_size = 0.05, random_state = 42)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train2, y_train2)

# Prediction
y_pred2= nb.predict(x_test2)


# In[ ]:


print("Accuracy: {:.2f}%".format(nb.score(y_pred2.reshape(-1,1), y_test2)*100))


# ## Our model works with <font color="red">**64% accuracy**</font> which is okay.

# **Thank you! Please upvote and make me comment your feedback to help me improve myself**

# In[ ]:




