#!/usr/bin/env python
# coding: utf-8

# # **What is PowerLifting?**
# 
# Hey, thanks for viewing my Kernel! 
# 
# **If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)**
# 
# Powerlifting is a strength sport that consists of three attempts at maximal weight on three lifts: squat, bench press, and deadlift. As in the sport of Olympic weightlifting, it involves the athlete attempting a maximal weight single lift of a barbell loaded with weight plates. Powerlifting evolved from a sport known as "odd lifts", which followed the same three-attempt format but used a wider variety of events, akin to strongman competition. Eventually odd lifts became standardized to the current three.
# 
# In competition, lifts may be performed equipped or un-equipped (typically referred to as 'raw' lifting or 'classic' in the IPF specifically). Equipment in this context refers to a supportive bench shirt or squat/deadlift suit or briefs. In some federations, knee wraps are permitted in the equipped but not un-equipped division; in others, they may be used in both equipped and un-equipped lifting. Weight belts, knee sleeves, wrist wraps and special footwear may also be used, but are not considered when distinguishing equipped from un-equipped lifting.
# 
# Competitions take place across the world. Powerlifting has been a Paralympic sport (bench press only) since 1984 and, under the IPF, is also a World Games sport. Local, national and international competitions have also been sanctioned by other federations operating independently of the IPF.
# 
# Source: [Wikipedia](https://en.wikipedia.org/wiki/Powerlifting)

# ![](http://www.beyondlimitstraining.net/wp-content/uploads/2017/11/FullSizeRender.jpg)

# # **About the Dataset**
# 
# **Context**
# This dataset is a snapshot of the [OpenPowerlifting](https://www.openpowerlifting.org/) database as of February 2018. OpenPowerlifting is an organization which tracks meets and competitor results in the sport of powerlifting, in which competitors complete to lift the most weight for their class in three separate weightlifting categories.
# 
# **Content**
# This dataset includes two files. meets.csv is a record of all meets (competitions) included in the OpenPowerlifting database. competitors.csv is a record of all competitors who attended those meets, and the stats and lifts that they recorded at them.
# 
# For more on how this dataset was collected, see the [OpenPowerlifting FAQ](https://www.openpowerlifting.org/faq).
# 
# **Acknowledgements**
# This dataset is republished as-is from the [OpenPowerlifting source](https://www.openpowerlifting.org/data).
# 
# **Inspiration**
# How much influence does overall weight have on lifting capacity?
# How big of a difference does gender make? What is demographic of lifters more generally?

# # Index
# 
# 1. Importing the modules
# 2. Reading the Data
# 3. Collecting information about the two dataset
# 4. Equipment Analysis
# 5. Gender Analysis
# 6. Weight analysis
# 7. Winners Analysis
# 8. Variation of athletes age over time
# 9. Number of Athletes per year
# 10. Some machine learning

# # 1. Importing the modules

# In[ ]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# # 2. Reading the Data

# In[ ]:


meets = pd.read_csv('../input/meets.csv')
openpl = pd.read_csv('../input/openpowerlifting.csv')


# # 3. Collecting information about the two dataset

# We are going to:
# 
# * Review the first lines of the data;
# * Use the describe and info functions to collect statistical information, datatypes, column names and other information.

# In[ ]:


meets.head(10)


# In[ ]:


openpl.head(10)


# We can now join the two dataframes using as key the NOC column with the Pandas 'Merge' function.

# In[ ]:


merged = pd.merge(meets, openpl, on='MeetID', how='left')


# Let's see the result:

# In[ ]:


merged.head(5)


# And now, some columns and statistical information:

# In[ ]:


merged.info()


# In[ ]:


merged.describe()


# # 4. Equipment Analysis

# As explained in the intro, in competitions, lifts may be performed equipped or un-equipped (typically referred to as 'raw' lifting or 'classic' in the IPF specifically). Equipment in this context refers to a supportive bench shirt or squat/deadlift suit or briefs. 
# 
# Let's make an analysis of the equipments!

# In[ ]:


print(merged['Equipment'].value_counts())


# Straps and wraps are the same, so we can include straps inside wraps

# In[ ]:


def strapsInWraps(x):
    if x == 'Straps':
        return 'Wraps'
    return x


# In[ ]:


merged['Equipment'] = merged['Equipment'].apply(strapsInWraps)


# In[ ]:


print(merged['Equipment'].value_counts())


# Done! Let's move on.

# # 5. Gender Analysis

# We can now divide our athletes according to the gender.
# 
# Below we will build a graph and a recap of the numbers to display more information.

# In[ ]:


plt.figure(figsize=(10,7))
merged['Sex'].value_counts().plot(kind='bar')
plt.title('Gender division in the dataframe',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
print('Percentage of Male lifters: {}%\n'.format(round(len(merged[merged['Sex']=='M'])/len(merged)*100),4))
print(merged['Sex'].value_counts())


# # 6. Weight analysis

# Another important topic that can be analyzed is the bodyweight of the athletes: with the graph below we can compare male and female using the colours.

# In[ ]:


g = sns.FacetGrid(merged,hue='Sex',size=6,aspect=2,legend_out=True)
g = g.map(plt.hist,'BodyweightKg',bins=50,alpha=.6)
plt.title('Bodyweight Kg',fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.xlabel('BodyweightKg',fontsize=15)
plt.legend(loc=1)
plt.show()


# # 7. Winners Analysis

# First of all, we have to create a new "AgeCategory" column. 
# 
# Why? Because otherwise when creating a plot with the age the column number will be huge (imagine to have 21, 22, 23,24 and so on for each age range).
# 
# That is why we will use the function below to split the ages in ranges.

# In[ ]:


def age_calculate(x):
    if(x < 10.0):
        return "05-10"
    if(x >= 10.0 and x < 20.0):
        return "10-20"
    if(x >= 20.0 and x < 30.0):
        return "20-30"
    if(x >= 30.0 and x < 40.0):
        return "30-40"
    if(x >= 40.0 and x < 50.0):
        return "40-50"
    if(x >= 50.0 and x < 60.0):
        return "50-60"
    if(x >= 60.0 and x < 70.0):
        return "60-70"
    if(x >= 70.0 and x < 80.0):
        return "70-80"
    if(x >= 80.0 and x < 90.0):
        return "80-90"
    else:
        return "90-100"
    
merged['ageCategory'] = pd.DataFrame(merged.Age.apply(lambda x : age_calculate(x)))


# Done.
# 
# At this point, the first thing that we have to do to analyze the winners is to create a new dataframe that will include only first place athletes.
# 
# We will do so assigning the filtered dataframe to a new variable, *firstPlace*.

# In[ ]:


firstPlace = merged[(merged.Place == '1')]


# Now, we can split the data to have two dataframes according to the gender of the athletes.

# In[ ]:


firstPlaceMale = firstPlace[(firstPlace.Sex == 'M')]
firstPlaceFemale = firstPlace[(firstPlace.Sex == 'F')]


# In[ ]:


firstPlaceMale.head(5)


# In[ ]:


firstPlaceFemale.head(5)


# To make a graph about the age, we have to consider only the values in which the Age is not null.

# In[ ]:


firstPlaceMale.isnull().any()


# Ok, we have Null values: to consider only the not nulls we can use *np.isfinite* as below:

# In[ ]:


firstPlaceMale = firstPlaceMale[np.isfinite(firstPlaceMale['Age'])]


# Let's analyze the unique Age values remained to decide which graph is better to use:

# In[ ]:


uniqueAgeValuesMale = firstPlaceMale.Age.unique() 


# In[ ]:


uniqueAgeValuesMale


# Well, the first thing that I noticed is that for me it is really strange to see kids in powerlifting competitions: I would like to make a cross check and see all the records in the dataframe with an age equal to 9.5.

# In[ ]:


firstPlaceMale[firstPlaceMale['Age'] == 9.5]


# Okay, it is effectively correct.
# 
# At this point we can try to plot a countplot to see the distribution of the age using the column 'ageCategory' previously created:

# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(firstPlaceMale['ageCategory'], palette="muted")
plt.title('Distribution of Age for Male Athletes (winners)')


# Let's do the same for female athletes.

# In[ ]:


firstPlaceFemale = firstPlaceFemale[np.isfinite(firstPlaceFemale['Age'])]


# In[ ]:


plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(firstPlaceFemale['ageCategory'], palette="Set1")
plt.title('Distribution of Age for Female Athletes (winners)')


# **Best lifts per age category**

# Let's review the best squat, bench, deadlift for all the athletes:

# In[ ]:


plt.figure(figsize=(40,20))
plt.tight_layout()
sns.catplot(x="ageCategory", y="BestSquatKg", kind="box", data=firstPlace, palette="Set1")
plt.title('Best Squat per Age Category')


# Okay, the first noticeable thing is that we have a negative lift (???).
# 
# Let's convert the column values to their absolute value.

# In[ ]:


firstPlace['BestSquatKg'] = firstPlace['BestSquatKg'].abs()
firstPlace['BestBenchKg'] = firstPlace['BestBenchKg'].abs()
firstPlace['BestDeadliftKg'] = firstPlace['BestDeadliftKg'].abs()


# Okay, at this point we can proceed with the plots:

# In[ ]:


plt.figure(figsize=(40,20))
plt.tight_layout()
sns.catplot(x="ageCategory", y="BestSquatKg", kind="box", data=firstPlace, palette="Set1")
plt.title('Best Squat per Age Category')

plt.figure(figsize=(40,20))
plt.tight_layout()
sns.catplot(x="ageCategory", y="BestBenchKg", kind="box", data=firstPlace, palette="Set1")
plt.title('Best Bench per Age Category')

plt.figure(figsize=(40,20))
plt.tight_layout()
sns.catplot(x="ageCategory", y="BestDeadliftKg", kind="box", data=firstPlace, palette="Set1")
plt.title('Best Deadlift per Age Category')


# # 8. Variation of athletes age over time

# To work with time, let's create a column 'Year' starting from the column 'Date'.

# In[ ]:


merged['Year'] = pd.DatetimeIndex(merged['Date']).year  


# Now let's take a look at the result:

# In[ ]:


merged.head()


# Okay, we can now create a lineplot:

# In[ ]:


plt.figure(figsize=(20, 10))
sns.set(style="ticks", rc={"lines.linewidth": 5})
sns.lineplot('Year', 'Age', data=merged)
plt.title('Variation of Age for Athletes over time')


# # 9. Number of Athletes per year

# Let's now plot the data of the number of the athletes every year thanks to our new column 'Year':

# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(30, 10))
sns.countplot(x='Year', data=merged, palette='Set2')
plt.title('Variation of the number of athletes over time')


# # 10. Some machine learning

# Let's try to predict if an athlete won a competion, starting with a review of the data:

# In[ ]:


merged.head(5)


# Let's now create a new column, *isWinner*: it will be 1 if the place is 1, in other cases it will be 0.
# 
# It is important because our algorithm needs a categorical variable to make predictions.
# 
# To proceed, let's first check the dtype of the column Place:

# In[ ]:


merged['Place'].dtype


# Okay, let's replace  values that are not INT with 0 in the column Place to have the possibility to use it to create our "isWinner" column:

# In[ ]:


merged['Place'] = pd.to_numeric(merged.Place, errors='coerce').fillna(0, downcast='infer')


# Perfect, now let's create our column:

# In[ ]:


def is_Winner(x):
    if(x == 1):
        return 1
    else:
        return 0
    
merged['isWinner'] = pd.DataFrame(merged.Place.apply(lambda x : is_Winner(x)))


# Now let's drop all the non-categorical, not useful or strings columns:

# In[ ]:


final_data = merged.drop(['MeetID', 'MeetPath', 'Federation', 
                          'MeetName', 'Date', 'MeetName', 
                          'Name', 'WeightClassKg', 'Division', 'Squat4Kg', 
                          'Bench4Kg', 'Deadlift4Kg', 'Place', 'ageCategory'], axis=1)


# In[ ]:


final_data.head(5)


# **Categorical features/NaN analysis**

# We will use get_dummies to create categorical features automatically on target columns:

# In[ ]:


catData = pd.get_dummies(final_data, columns=['Sex', 'MeetCountry', 'MeetState', 'MeetTown', 'Equipment'])


# In[ ]:


catData.head(5)


# Let's check data types again:

# In[ ]:


catData.dtypes


# [](http://)Okay, let's now remove NaN's (not a number) values replacing them with the mean of the value of the column:

# In[ ]:


catData['Age'] = catData['Age'].fillna(catData['Age'].mean())
catData['BodyweightKg'] = catData['BodyweightKg'].fillna(catData['BodyweightKg'].mean())
catData['BestSquatKg'] = catData['BestSquatKg'].fillna(catData['BestSquatKg'].mean())
catData['BestBenchKg'] = catData['BestBenchKg'].fillna(catData['BestBenchKg'].mean())
catData['BestDeadliftKg'] = catData['BestDeadliftKg'].fillna(catData['BestDeadliftKg'].mean())
catData['TotalKg'] = catData['TotalKg'].fillna(catData['TotalKg'].mean())
catData['Wilks'] = catData['Wilks'].fillna(catData['Wilks'].mean())
catData['Year'] = catData['Year'].fillna(catData['Year'].mean())


# **Model**

# Let's try with a decision tree.

# In[ ]:


from sklearn.model_selection import train_test_split
X = catData.drop('isWinner',axis=1)
y = catData['isWinner']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train, y_train)


# In[ ]:


predictions = dtree.predict(X_test)


# Done, let's now see the metrics:

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


# # 11. Who is the best lifter?

# I will try to answer this question using different approaches.
# 
# In our analysis, best lifters will be:
# 
# * Boys and Girls that lifts a lot;
# * Professional athletes that lifts a lot according to age category and bodyweight.
# 
# Let's start reviewing the head of the data as always.

# In[ ]:


merged.head(5)


# # Work in progress...
