#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

guns = pd.read_csv("../input/guns.csv", index_col=0)


# Let peak into our data

# In[ ]:


print(guns.shape)
guns.head()


# So we have 10 columns and over 10k datapoints. Let's give a description of the columns:
# 1. Year: year when incident occured
# 2. Month: month when incident occured
# 3. Intent: intent of the crime via the perpetrator, can be:
#     + Suicide
#     + Accidental
#     + NA
#     + Homicide
#     + Undetermined
# 4. Police: whether the police were involved in the shooting or not
# 5. Sex: victim's gender
# 6. Age: victim's age
# 7. Race: victim's race, can be:
#     + Asian/Pacific Islander 
#     + Native American/Native Alaskan 
#     + Black 
#     + Hispanic 
#     + White
# 8. Hispanic: hispanic origin code from CDC, can find lookup here:
#     + https://www.cdc.gov/nchs/data/dvs/Appendix_D_Accessible_Hispanic_Origin_Code_List_Update_2011.pdf
# 9. Place: where shooting occured
# 10. Education: victim's education, can be:
#     + 1- Less than High School
#     + 2- Graduated from High School or equivalent
#     + 3- Some College 
#     + 4- At least graduated from College
#     + 5- Not available

# Let's check for how complete the data is

# In[ ]:


guns.notnull().sum() * (100/guns.shape[0])


# Since our data is extremely large and we aren't missing many datapoints, we will just remove points that have anything null in it.  Will only lose ~1% of data

# In[ ]:


guns = guns.dropna(axis=0, how="any")


# Just looking at the columns, the most interesting column to predict will probably be intent.  So that is what we will aim for.  That being said we can remove any columns that have intent as undetermined

# In[ ]:


guns = guns[guns.intent != "Undetermined"]


# Month and Year are separate, not sure why.. let's take a look at them to see if anything stands out

# In[ ]:


print(guns.month.value_counts(sort=False))
print(guns.year.value_counts(sort=False))


# The only thing that stands out is there is less deaths in February, but I am nearly positive that is because of it being a shorter month.  Because the rest is relatively evenly distributed at first glance, we will combine month and year

# In[ ]:


from datetime import datetime
guns["date"] = guns.apply(lambda row: datetime(row.year, row.month, 1), axis=1)
del guns["year"]
del guns["month"]

# while I'm here just gonna make police into a boolean for readability 
guns.police = guns.police.astype(bool)


# In[ ]:


guns.head()


# I'm gonna start visualizing some of the data on things that I think might be interesting and tell us some stuff; starting with Intent and whether or not the police were involved or not

# In[ ]:


intentAndPolice = guns.groupby([guns.intent, guns.police]).intent.count().unstack('police')
plot = intentAndPolice.plot(kind="bar", stacked=True)
plot.legend(labels=["Police Involved", "Police Not Involved"])
plot.set_xlabel("Intent")
plot.set_ylabel("Count")
plt.show()


# In[ ]:


intentAndPolice


# So our bar graph didn't tell us much because police were only involved with 19 shootings, BUT all 19 dealt with homocides.  This will come in useful when making your model

# Let's look at the age distributions and intent

# In[ ]:


plt.hist(guns.age, range(0, 100))
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Death Distribution by Age")
plt.show()


# Looking at this two things stand out, the bump in the 20s and the bump between about 50-60.  I will take a guess that the 20s would be homicides and the 50-60 bump would be suicides.  I would also guess that accidental will be that small number below the 20s

# In[ ]:


plt.hist(guns.age[guns.intent == "Homicide"], range(0, 100))
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Homicide")
plt.show()


# In[ ]:


plt.hist(guns.age[guns.intent == "Suicide"], range(0, 100))
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Suicide")
plt.show()


# In[ ]:


plt.hist(guns.age[guns.intent == "Accidental"], range(0, 100))
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Accidental")
plt.show()


# So it looks like I was correct about homicide and suicide given the age groups, but I was wrong about accidental deaths, which for the most part seems to be rather scattered

# Now let's look at Intent vs Sex

# In[ ]:


intentAndSex = guns.groupby([guns.intent, guns.sex]).intent.count().unstack('sex')
plot = intentAndSex.plot(kind="bar", stacked=True)
plot.legend(labels=["Female", "Male"])
plot.set_xlabel("Intent")
plot.set_ylabel("Count")
plt.show()


# In[ ]:


intentAndSex


# As expected, males dramatically outweight women in all incidents.  There is no large distinction between determining if a women was involved with suicide or homicide.  Hopefully we can find this elsewhere

# Let's now look at the victim's race and see if we can find any information from this 

# In[ ]:


intentAndRace = guns.groupby([guns.intent, guns.race]).intent.count().unstack('race')
plot = intentAndRace.plot(kind="bar", stacked=True)
plot.set_xlabel("Intent")
plot.set_ylabel("Count")
plt.show()


# In[ ]:


intentAndRace


# Since accidental is rather disproportionate to homicide and suicide we should look at this in percentages 

# In[ ]:


intentAndRace = intentAndRace.div(intentAndRace.sum(1).astype(float), axis=0)
plot = intentAndRace.plot(kind="bar", stacked=True)
plot.set_xlabel("Intent")
plot.set_ylabel("Count")
plt.legend(bbox_to_anchor=(1.1,0.9))
plt.show()


# That's better, now we can see that whites are the main victims of suicide by a long shot; same goes for accidental deaths, but not as much.  We can also see that blacks are suffer significantly more than any race in homocides

# Looking at the race distibutions, the hispanic field might not be worth really looking at, let's take a quick look at the value counts

# In[ ]:


guns.hispanic.value_counts() * (100/guns.hispanic.shape[0])


# Whelp that feature is pretty worthless, > 90% of the victims weren't even hispanic! So we will just remove that column

# In[ ]:


del guns["hispanic"]


# In[ ]:


guns.head()


# Let's take a look at Intent vs Place

# In[ ]:


intentAndPlace = guns.groupby([guns.intent, guns.place]).intent.count().unstack('place')
plot = intentAndPlace.plot(kind="bar", stacked=True)
plot.set_xlabel("Intent")
plot.set_ylabel("Count")
plt.show()


# Again, pretty disproportionate, so lets make them percentages again.  But it is worthy to note that suicide seems to occur in the home most of the time

# In[ ]:


intentAndPlace = intentAndPlace.div(intentAndPlace.sum(1).astype(float), axis=0)
plot = intentAndPlace.plot(kind="bar", stacked=True)
plot.set_xlabel("Intent")
plot.set_ylabel("Count")
plt.legend(bbox_to_anchor=(1.1,0.9))
plt.show()


# So we can see that there is a lot of data here that looks like it might distract our model.  There are only 5 options that really stand out: Home, Other specified, Other unspecified, Street, and Trade/service area.  And among those, Other specified and Other unspecified tell us absolutely nothing and could potentially throw off our model.  Trade/service area also is pretty evenly distributed and doesn't tell us too much, so I think what will be best is to just keep Home and Street then replace the rest with 'Other'

# In[ ]:


indexOfOthers = guns[(guns.place != "Home") & (guns.place != "Street")].index
guns.loc[indexOfOthers, "place"] = "Other"


# Now let's look again

# In[ ]:


intentAndPlace = guns.groupby([guns.intent, guns.place]).intent.count().unstack('place')
plot = intentAndPlace.plot(kind="bar", stacked=True)
plot.set_xlabel("Intent")
plot.set_ylabel("Count")
plt.show()


# In[ ]:


intentAndPlace = intentAndPlace.div(intentAndPlace.sum(1).astype(float), axis=0)
plot = intentAndPlace.plot(kind="bar", stacked=True)
plot.set_xlabel("Intent")
plot.set_ylabel("Count")
plt.legend(bbox_to_anchor=(1.1,0.9))
plt.show()


# Much better.  This still doesn't tell us a whole lot but we can derive that there are strong possibilities of 2 things: if incedent occured at home then there is a high chance it was a suicide and if incedent occured in the streets then there is a high chance of it being a homicide 

# And finally, let's look at education

# In[ ]:


intentAndEducation = guns.groupby([guns.intent, guns.education]).intent.count().unstack('intent')
plot = intentAndEducation.plot(kind="bar", stacked=True)
plot.set_xlabel("Education")
plot.set_ylabel("Count")
plt.show()


# The only thing I can gather from this is that is seems that there seems to be a lot of suicides when the victim has graduated from high school (2.0).  It is not signifcant but it is something

# I want to look at one more thing before building my model.  I want to see if there is anything with Month vs Intent.  Since I previously combined month and year, I will just reload the dataframe.  I will also get rid of year because that doesn't seem to tell us anything at all.
# 

# In[ ]:


guns = pd.read_csv("../input/guns.csv", index_col=0)

intentAndMonth = guns.groupby([guns.intent, guns.month]).intent.count().unstack('intent')
plot = intentAndMonth.plot(kind="bar", stacked=True)
plt.legend(bbox_to_anchor=(1.1,0.9))
plot.set_xlabel("Month")
plot.set_ylabel("Count")
plt.show()


# Ok, so there doesn't seem to be anything of substance here.  The only thing that could be said is in July, there are slightly more homicides and suicides. But nothing substantial.

# So before building my model there are a few things that I will need to do to prep my data given my analysis:
# 1. Month and year features can be removed 
#     + Removing month because even though July has a little bit higher levels, February has significantly lower levels which
# 2. Get rid of hispanic feature
#     + Over 90% rows weren't even hispanic, so this feature is worthless in my opinion
# 3. Get rid of data not that have any null data
#     + We have a large dataset and it is mostly complete, so we can afford to clean this up
# 4. Remove "Undetermined" intents
#     + Since we are predicting intent, this doesn't help us
#  could throw off the model.  I am almost positive this is because of the shorter month so better safe than sorry and remove
# 5. For places, should only keep Home and Street; the rest can be combined in "Other"

# Now we can build a model:

# In[ ]:


import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


guns = pd.read_csv("../input/guns.csv", index_col=0)

# Prep the data (See Analysis.ipynb)
del guns["year"]
del guns["month"]
del guns["hispanic"]
guns = guns.dropna(axis=0, how="any")
guns = guns[guns.intent != "Undetermined"]
indexOfOthers = guns[(guns.place != "Home") & (guns.place != "Street")].index
guns.loc[indexOfOthers, "place"] = "Other"

guns = guns.apply(LabelEncoder().fit_transform)
        
X = guns.iloc[:, 1:]
y = guns.iloc[:, 0]

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = .20)


# In[ ]:


# Just gonna try a simple knn classifier, will be good base to compare to
# After testing, n = 10 seems to be about the best we will get
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
knn.fit(XTrain, yTrain)
accuracy = knn.score(XTest, yTest)
print("KNN: {0} %".format(accuracy * 100))


# In[ ]:


# So since this is a classification problem, we can't use linear regression,
# it will give too much weight to data far from the decision frontier 
# I still want to use a linear approach, so I will choose logistic regression 
logReg = LogisticRegression()
logReg.fit(XTrain, yTrain)
accuracy = logReg.score(XTest, yTest)
print("Logistic Regression: {0} %".format(accuracy * 100))


# In[ ]:


# Might as well try a decision tree
decisionTree = DecisionTreeClassifier()
decisionTree.fit(XTrain, yTrain)
accuracy = decisionTree.score(XTest, yTest)
print("Decision Tree: {0} %".format(accuracy * 100))

print("Limiting Linear SVC to 5000 points or it will take ages")
time.sleep(1) # just so the message above will show


# In[ ]:


# Using the ultimate machine learning cheat sheet, it says given
# the parameters, I should choose linear svc
svc = LinearSVC()
svc.fit(XTrain[:5000], yTrain[:5000])
svc.score(XTest, yTest)
accuracy = svc.score(XTest, yTest)
print("SVC: {0} %".format(accuracy * 100))


# In[ ]:




