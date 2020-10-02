#!/usr/bin/env python
# coding: utf-8

# # Predicting Titanic Surviors
# 
# For more information regarding this dataset, visit: 
# https://www.kaggle.com/c/titanic/data.
# 
# This particular script will contain an initial exploration of the data, followed by some necessary preprocessing, 
# feature enineering and selection, and finally the models and results.
# 
# ## Some general information regarding the Titanic
# The Titanic is the world famous ship that didn't even manage to complete 1 trip. On the 10th of April 1912 it left Southhampton, England, to first pick up passengers in Cheroux, France and Queenstown, Ireland and make its way to New York. Unfortunately, in the night of the 14th/15th of April, it was steered into an iceberg. This wrecked the ship. Since there were not enough lifeboats, people were stuck on the sinking ship. As a result, the majority of the passengers drowned or froze to death in the icy waters south of Newfoundland, Canada.
# 
# 
# ##### SAD
# Now that's out of the way, let's start by importing the libraries used in the exploratory part:

# In[ ]:


#call libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
import seaborn as sns

# Set default matplot figure size
pylab.rcParams['figure.figsize'] = (6.5, 5.0)

#Turn off pandas warning for changing variables & future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


# In[ ]:


#set random seed
np.random.seed(123)


# In[ ]:


#Import dataset and look at what information we are dealing with
titanic = pd.read_csv("../input/train.csv", header = 0)
titanic.head(20)


# ### Nice!
# First I'll check if there are any duplicates by finding duplicates in the names column.
# 
# I already see some columns that i'm pretty sure of that removing them do not cause any problems, namely PassengerID and Name. It seems extremely unlikely that someone's name or their "ID" has anything to do with surviving a boat crash. Someone's ticket ID also seems useless since everyone's ID is likely to be unique, hence has no predicting power. <br />For now I'll leave Cabin ID in the dataset since its numbers might entail information whether someones cabin was in proximity of a lifeboat 
# 
# Dropping stuff makes the data cleaner, so i'll start with removing these columns before going further with the analysis.

# In[ ]:


#column with names all passangers
names = titanic["Name"]

#Check whether there are duplicates in the name list
duplicates = names[names.duplicated()]
print(duplicates)


# In[ ]:


titanic = titanic.drop(columns = "PassengerId")
titanic = titanic.drop(columns = "Ticket")
titanic = titanic.drop(columns = "Name")
titanic.head()


# ### Column Names
# Next let's change some names of the columns. This will make future tables easier for people that have not read about what the variables exactly mean. 

# In[ ]:


titanic.columns = (['Survived', 'Class', 'Sex', 'Age', 
                    'n_Siblings_Spouse', 'n_Parents_Chidren', 
                    'Fare_Price', 'Cabin_ID', 'Embarked'])


# ### Summary statistics and NaN's
# Now that's out of the way let's look at what the available columns exactly entail. 
# 
# It was observable that there are multiple missing vaues in a few columns. Let's see which columns have these missing values and 

# In[ ]:


print("Summary statistics of the numerical columns:")
print()
print(titanic.describe())
print()
print("Missing values per column:")
print()
print(titanic.isna().sum())


# ### What can we make up of this?
# 
# Okay so lets list a few interesting facts that are distinghuisable from these previous statistics:
# 
# - There are a total of 891 observations
# - +- 38% survived (342 passengers)
# - There were 3 classes
# - There were many youngsters/ adolescents on the boat half of the people were below the age of 28)
# - The youngest person had an age of 0.42. This is either a very weird measurement, or there is an error and he should actually be 42 years old.
# - Most people were not traveling with their siblings/spouse/parents/children.
# - There is one(?) observation that had 8 siblings on board. Jesus that's a big family.
# - There were fare prices of 0,-. That's one cheap ride.
# 
# 
# - For 177 observations the age was missing. Thats +- 20% of the data. 
# - 687 observations have no Cabin_ID. Maybe they did not have a cabin?
# - For 2 observations it is unknown where they had embarked.
# 
# This information will come in handy in a later stage when deciding which features will be used to model with and/or when features are being engineered. Let's first make sure some of the numerical columns become categories and take a look at how the data is distributed!

# In[ ]:


# Turn sex, class, survived and embarked in categorical variable
titanic["Sex"] = titanic.Sex.astype('category')
titanic["Class"] = titanic.Class.astype('category')
titanic["Survived"] = titanic.Survived.astype('category')
titanic["Embarked"] = titanic.Embarked.astype('category')

#Rename Embarked cities
titanic["Embarked"] = titanic.Embarked.cat.rename_categories({"S" :"Southampton",
                                                              "C" : "Cherbourg",
                                                              "Q" : "Queenstown"})
#Plot barplots for the independent categorical variables 
sns.countplot(titanic.Sex).set_title('Distribution of sexes')
plt.show()
sns.countplot(titanic.Class).set_title('Distribution of classes')
plt.show()
sns.countplot(titanic.Embarked).set_title('Distribution of where people embarked')
plt.show()


# ### About the plots
# 
# Alright so: 
# - There were almost twice as many men as there were females on the ship. 
# - Majority was in third class, 1st and second class have almost the same amount of passengers
# - Majority of the people embarked in Southampton, followed by Queenstown and Cherbourg
# 
# Now lets see the distribution of ages and ticket prices, followed by how the different categories compare depending on their survival rate!

# In[ ]:


#Plot the distribution of the ticket price
sns.distplot(titanic.Fare_Price, bins  = 50).set_title('Distribution of ticket prices')
plt.show()

#Plot the distribution of ages
#Since there are Na's in the age distribution, we specify we only want to see the distribution of the available age data
sns.distplot(titanic.Age[-titanic.Age.isnull()], bins = 40).set_title('Distribution of ages')
plt.show()


# ### Prices and age
# 
# Most ticket prices seem fairly low. There are a few exeptions of very high prices. Later on i'll check the average price per class.
# 
# The age seems fairly normally distributed, although it is skewed to the right and there are many values of zero's. Are these babies or is there something weird with the data regarding the ages of the passengers? I'll have a closer look at what these children's values entail.
# 

# In[ ]:


babies = titanic[titanic.Age < 2]
babies


# #### Babies
# It seems that the children that were not born yet were given floats depending on how close it was to their birthday(?) Im annoyed by the fact that this causes all the ages to be catagorized as 'floats' so i'm gonna change these values into 0's (since that is their actual age.

# In[ ]:


#specify the ages that should be turned into 0 and change types to integer
titanic.Age[titanic.Age < 1] = 0
titanic.Age[-titanic.Age.isnull()] = titanic.Age[-titanic.Age.isnull()].astype('int')


# #### The rich kids
# Now lets have a closer look at the more wealthy people, the ones that purchased a more expensive boat ticket, and see what kind of people these were.

# In[ ]:


#show observations where fare price > 100,-
titanic[titanic.Fare_Price > 100]


# #### Wealthy French women
# When looking at the table presented above, interestingly enough it appears that there are many women that have bought the more expensive tickets, even though the passengerlist is predominately filled with men (65%). This means that the lower tickts were relatively sold even more to men than to women. Perhaps poor men were more attracted in the 1910's to migrate and find their luck in the USA than the women in their social class. <br />Also interesting to note is that many of the observations seem to have embarked in Cherbourg, whereas in general most people embarked in Southhampton (only 19% of the total passengers embarked in Cherbourg). Must be a fancy place. 
# 
# I'll estimate the specific percentages to see whether these observations are actually true.

# In[ ]:


#Create the dataframe for ticket prices that were above 100,-
the_wealthy = titanic[titanic.Fare_Price > 100]

#calculate the percentage that was either female and the percentage that embarked in Cherbourg within this dataframe. 
print("Of the people that had a ticket price of more than 100,",
      (len(the_wealthy[the_wealthy.Sex == 'female'])) / (len(the_wealthy))*100, 
      "% was female.")
print()
print("Of the people that had a ticket price of more than 100,",
      (len(the_wealthy[the_wealthy.Embarked == 'Cherbourg'])) / (len(the_wealthy))*100,
     "% embarked in Cherbourg.")


# #### Okay 
# So that quick hypothesis appeared to be correct. Distributions seem to shift depending on specific customer segments. 
# #### The missing ages
# The 177 missing observations regarding the age of passengers is currently still unsolved. I think i will impute these values, but let's first see whether there is a specific pattern observable for the customers where the age is missing. 

# In[ ]:


NaN_ages = titanic[-(titanic.Age > -2)]

#Plot barplots for the independent categorical variables 
sns.countplot(NaN_ages.Sex).set_title('Distribution of sexes ageless')
plt.show()
sns.countplot(NaN_ages.Class).set_title('Distribution of classes ageless')
plt.show()
sns.countplot(NaN_ages.Embarked).set_title('Distribution of where people embarked ageless')
plt.show()
sns.countplot(NaN_ages.Survived).set_title('Distribution of whether people survived ageless')
plt.show()


# #### Regarding the Ageless...
# Allright so it appears that the people where no age was noted have a similiar distribution of males and females when compared to the full dataset. They overwhelmingly were in third class, and by the looks of it, almost everyone that embarked in Queenstown misses lacks their respective age. The amount of survivors amongst these ageless people appears to be similarly distributed when compared with the full dataset.
# 
# let's extend this analysis to see whether spouse/children might have caused the data to go missing....

# In[ ]:


#full dataset children/parent distribution
sns.countplot(titanic.n_Parents_Chidren).set_title('Children/parents amongst titanic passengers')
plt.show()
#ageless children/parent distribution
sns.countplot(NaN_ages.n_Parents_Chidren).set_title('Children/parents amongst ageless titanic passengers ')
plt.show()

#full dataset sibling/spouse distribution
sns.countplot(titanic.n_Siblings_Spouse).set_title('Siblings/spouse amongst titanic passengers')
plt.show()
#ageless sibling/spouse distribution
sns.countplot(NaN_ages.n_Siblings_Spouse).set_title('Siblings/spouse amongst ageless titanic passengers')
plt.show()


# #### No major differences
# People seem to be slightly more on themselves compared to the full dataset. The difference does not appear to be striking though.
# 
# ### Dead or alive???
# Let's have a closer look at the thing we are interested in the most when researching this dataset: What did the people that survived the disaster have relatively a lot in common. Did they all embark in the same city? Were they in the same age group? Did the men leave the women to die while rowing out of the danger zone quickly? The following few plots and statistics should provide a more in-depth 

# In[ ]:


#distinguish survival by sex
sns.catplot('Sex', data = titanic, hue = 'Survived', kind='count', aspect=1.5)
plt.show()


# #### The ladies live
# Well this plot already shows that sex was of a huge huge influence when it comes to surviving. About 1/6th of the men appear to have survived compared to 3/4th of the women. Well done ladies!

# In[ ]:


#distinguish survival by class
sns.catplot('Class', data = titanic, hue = 'Survived', kind='count', aspect=1.5)
plt.show()


# #### Show some class
# Which class your ticket the people were residing in was also of major influence on whether they survived. This ofcourse can be related to sex as well, since there were a lot more men in third class than female. Anyhow, Third class citizens died by the numbers, where the mahjority of the first classers survived.  

# In[ ]:


#distinguish survival by where people were embarked
sns.catplot('Embarked', data = titanic, hue = 'Survived', kind='count', aspect=1.5)
plt.show()


# #### Sacre Bleu!
# There are some differences distinguishable between where people embarked and whether they surived or not. The French had a bigger chance of survival than the English had. Again, this seems more than a correlation than as a causation, considering that the French were outnumbering the others in the first-class suites. 

# In[ ]:


#make a categorical variable for the ages
titanic.loc[(titanic.Age < 15), "AgeCat"] = "Kids"
titanic.loc[(titanic.Age >= 15) & (titanic.Age <= 30), "AgeCat"] = "Adolescents"
titanic.loc[(titanic.Age >= 31) & (titanic.Age <= 60), "AgeCat"] = "Adults"
titanic.loc[(titanic.Age >= 61), "AgeCat"] = "Elderly"


# In[ ]:


#distinguish survival by agecategories

sns.catplot('AgeCat', data = titanic, hue = 'Survived', kind='count', aspect=1.5)
plt.show()


# #### Forever Young
# Alright so I took the age categories very broad but it was mainly to get some idea of what's going on. These categories therefore probably give an incomplete image. Anyhow, the kids were the luckiest ones. More Survived than died, which is nice. The elderly seem to have experienced the worst survival ratio. about 1/4th survived. Ice Cold water and the ability to run towards the lifeboats might have played a role here. Adolescents and adults both were not very lucky, but someone has to take the blow amiright?

# In[ ]:


#create dataframe for only the kid category
save_the_kids = titanic[titanic.AgeCat == "Kids"]
#plot the kids' sex and survival
sns.catplot('Sex', data = save_the_kids, hue = 'Survived', kind='count', aspect=1.5)
plt.show()
#plot the kids' ticket class and survival
sns.catplot('Class', data = save_the_kids, hue = 'Survived', kind='count', aspect=1.5)
plt.show()
#plot the how many parents were with the children and their survival
sns.catplot('n_Parents_Chidren', data = save_the_kids, hue = 'Survived', kind='count', aspect=1.5)
plt.show()


# #### All about the kids
# When taking a look at the passengers up untill the age of 14, we see that even for the young ones wthe women were more lucky than the man. Also, sad to see is that class mattered even for the children. Kids in third class had lower survivability rate than in dhe higher classes (only 1 didn't survive in all of class 1 and 2)
# 
# Also remarkable is that children that traveled with 2 parents had a much lower survival rate than children that traveled with 1 parent or no parents. Could be a coincidence, or there might be an interesting cause to this.

# ### Lets make a quick model!
# 
# Another way that can help to learn more about the data is to create a quick first model without any specific feature engineering. Let's also run a random forest to see what variables currently have to most predictive power. For simplicty lets start by dropping rows where the age is unknown and getting rid of cabinID.

# In[ ]:


#convert all categories into numerical variables so they can be proberly used to model with
titanic.Sex = pd.CategoricalIndex(titanic.Sex)
titanic.Class = pd.CategoricalIndex(titanic.Class)
titanic.Embarked = pd.CategoricalIndex(titanic.Embarked)


titanic['Sex'] = titanic.Sex.cat.codes
titanic['Class'] = titanic.Class.cat.codes
titanic['Embarked'] = titanic.Embarked.cat.codes

titanic = titanic.drop(["Cabin_ID", "AgeCat"], axis = 1)


# In[ ]:


titanic2 = titanic.dropna()


# In[ ]:


#Split the data in a train and testset
titanic_dep = titanic2.Survived
titanic_indep = titanic2.drop(['Survived'], axis=1)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(titanic_indep, titanic_dep, test_size=0.3)

#Tree packages for checking the feature importance
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier() 

# Build a forest and compute the feature importances
## Fit the model on your training data.
rf.fit(X_train, y_train) 
## And score it on your testing data.
rf.score(X_test, y_test)


# In[ ]:


feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',
                                                                        ascending=False)
print(feature_importances)


# #### Feature imporances.
# 
# So the sex, age and the fare price are considered the most important features in the current state of the data. Class a bit less, but class and fare price are ofcourse highly intertwined.

# In[ ]:


#oerform logistic regression and KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Do a Kfold cross validation on the training data for k = 3

knn = KNeighborsClassifier(n_neighbors = 3)
CVscores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = "accuracy")
print("knn score:", CVscores.mean())

# Do a Kfold cross validation on the training data for a logistic regression

logregression = LogisticRegression(solver='liblinear')
CVscores = cross_val_score(logregression, X_train, y_train, cv = 10, scoring = "accuracy")
print("logistic regression score:", CVscores.mean())


# ### Let's take a random guess
# Currently our model is not performing very well. Knn is performing just as well as a random guess. What does this mean? That its time for some:
# ## FEATURE ENGINEERING

# Since we want to achieve a much higher accuracy (100000% correct please), the dataset requires some hard mangling and engineering. I'll give a step, by step explanation of the things that first come to mind, and apply them. Afterwards i'll test the model again to see whether my predictions have led to an improvement.

# ### What's in the name!?
# On the current passengers list, everyone has a title. Common titles are Mr. (for men), Mrs. (for married women), Miss. (for unmarried women), Master. (for young boys). Some passangers did not go by either of these the titles These were people with special titles. Lets use this informtion by making a variables for each specific title.

# In[ ]:


#Import dataset and look at what information we are dealing with
titanic = pd.read_csv("../input/train.csv", header = 0)


# In[ ]:


#Create dummy variable for married by looping over whether passangers names' contain Mr. or Mrs.
titanic["Mr."] = 0
for i in range(0,len(titanic["Name"])):
    if "Mr." in titanic.loc[i]["Name"]:
        titanic.at[i, "Mr."] = 1

titanic["Mrs."] = 0
for i in range(0,len(titanic["Name"])):
    if "Mrs." in titanic.loc[i]["Name"]:
        titanic.at[i, "Mrs."] = 1
        
titanic["Miss."] = 0
for i in range(0,len(titanic["Name"])):
    if "Miss." in titanic.loc[i]["Name"]:
        titanic.at[i, "Miss."] = 1        

                
titanic["Master."] = 0
for i in range(0,len(titanic["Name"])):
    if "Master." in titanic.loc[i]["Name"]:
        titanic.at[i, "Master."] = 1
        
titanic["Other_Title"] = 1 - (titanic["Master."] + titanic["Miss."] + titanic["Mrs."] + titanic["Mr."])


# ### What's in the ticket?!
# 
# Also, there might be some interesting information observable in people's ticket number as well. Lets see what kind of information we can extract from people's ticket numbers.

# In[ ]:


#Sort by ticket number and see if we can find any interesting patterns
titanic.sort_values(by = "Ticket")


# ### What? No Munny?
# Also when scrolling through the tickets and looking at their respective fares, I saw a few people that had a ticket fare of 0. Let's have a closer look at these people to see if there is a pattern observable for these uncommon ticket prices.

# In[ ]:


#Display the passangers that didn't pay for their tickets
titanic.loc[(titanic.Fare == 0)]


# #### Cheap but fatal
# A few things can be noted here. First: most of the people that had a zero ticket fare died. For a lot of them, their age is missing. They are all men. Looking up their names, it seems that these passengers were either working for the titanics main company (White Star Lines) or were working for partners of the company. Therefore most of them probably helped people get off the boat before going themselves, which has led to their death. This is noteworthy, hence i'll create a variable that notes this.
# 
# A few of the passangers in this list has the ticket number "LINE". I looked for information on the people traveling on these tickets and found that they were travelling together and were rebooked from a different ship. Their actual ticket number number is 370160, so lets change that.

# In[ ]:


#generate a zero ticket fare variable
titanic["Zero_ticket_fare"] = 0
for i in range(0,len(titanic["Fare"])):
    if titanic.loc[i]["Fare"] == 0:
        titanic.at[i, "Zero_ticket_fare"] = 1


#locate and change ticket nr.
titanic.loc[(titanic.Ticket == "LINE"), "Ticket"] = str(370160)


# ### Traveling with the bunch
# Another thing that was observable when looking at the ticket numbers is that people were traveling on the same ticket number and that the costs that accompanied the ticket are always the same amount. There are 2 interesting things that can be obtained from this knowledge: 1 is with how many persons the passengers were traveling by counting the amount of duplicates per ticket), and the price per person for their trip (by dividing the total fare price by the passengers on 1 ticket). 
# 
# The following lines of code will provide these 2 variables.

# In[ ]:


#Create a seperate dataframe with count data for how often each ticket occurs
ticket_counts = titanic['Ticket'].value_counts()
ticket_counts = pd.Series.to_frame(ticket_counts)
ticket_counts["Ticket_nr"] = ticket_counts.index
ticket_counts.index = range(0,len(ticket_counts))
ticket_counts.columns = ["Ticket_group_size", "Ticket"]

#Add this column to the full dataframe 
titanic = pd.merge(titanic, ticket_counts, how='outer', on='Ticket')


# In[ ]:


#Now calculate the actual ticket value
titanic["Price_per_person"] = (titanic["Fare"] / titanic["Ticket_group_size"])


# ### Dummies for the cats
# 
# To use some of the categorical values in our model, we will have to dummify them. in the following frames I will create dummies for passenger's class and their location of embarkment  

# In[ ]:


#generate dummies for the class variable
class_dummies = pd.get_dummies(titanic.Pclass)
class_dummies.columns = ["First_class", "Second_class", "Third_class"]
titanic = pd.concat([titanic, class_dummies], axis=1, sort=False)


# In[ ]:


#generate dummies for where the ship embarked
embarked_dummies = pd.get_dummies(titanic.Embarked)
embarked_dummies.columns = ["Southampton", "Cherbourg", "Queenstown"]
titanic = pd.concat([titanic, embarked_dummies], axis=1, sort=False)


# In[ ]:


#Drop columns that wont be used in the analysis
titanic = titanic.drop(columns = "PassengerId")
titanic = titanic.drop(columns = "Ticket")
titanic = titanic.drop(columns = "Name")
titanic = titanic.drop(columns = "Cabin")
titanic = titanic.drop(columns = "Fare")
titanic = titanic.drop(columns = "Pclass")
titanic = titanic.drop(columns = "Embarked")


# ### Dep & Indep
# Seperate the columns for the dependent variable (survived) and the independent variables, so that the independent variable columns can be further preprocessed (normalization etc.)

# In[ ]:


#Seperate columns
titanic_indep = titanic.drop(columns = "Survived")
titanic_dep = titanic["Survived"]


# In[ ]:


#Change the type of all dummies to categories 
titanic_indep["Sex"] = titanic_indep.Sex.astype('category')
titanic_indep["First_class"] = titanic_indep.First_class.astype('category')
titanic_indep["Second_class"] = titanic_indep.Second_class.astype('category')
titanic_indep["Third_class"] = titanic_indep.Third_class.astype('category')
titanic_indep["Southampton"] = titanic_indep.Southampton.astype('category')
titanic_indep["Cherbourg"] = titanic_indep.Cherbourg.astype('category')
titanic_indep["Queenstown"] = titanic_indep.Queenstown.astype('category')

#convert all categories into numerical variables so they can be properly used to model with
titanic_indep.Sex = pd.CategoricalIndex(titanic_indep.Sex)
titanic_indep.First_class = pd.CategoricalIndex(titanic_indep.First_class)
titanic_indep.Second_class = pd.CategoricalIndex(titanic_indep.Second_class)
titanic_indep.Third_class = pd.CategoricalIndex(titanic_indep.Third_class)
titanic_indep.Southampton = pd.CategoricalIndex(titanic_indep.Southampton)
titanic_indep.Cherbourg = pd.CategoricalIndex(titanic_indep.Cherbourg)

#Turn sex variable into a dummy
titanic_indep['Sex'] = titanic_indep.Sex.cat.codes


# ### Normalizing the independent variables
# Since KNN will be used to impede missing values and eventually as an algorithm to predict survivability, it is necessary to normalize the data so the variables values comparable to the model. However, the missing values in the age column make it not possible to normalize the complete dataframe at once. Therefore, I'll do the normalization column by columns, where I first take care of the ages. 

# In[ ]:


# Take the age column seperate and split them in the section with Na's and without Na's
agecolumn = titanic_indep["Age"]
Ages_noNA = agecolumn[agecolumn > -1]
Ages_yesNa = agecolumn[agecolumn.isna()]

# Normalize the section without Na's and add both sections back together
Ages_noNA = (Ages_noNA - Ages_noNA.mean()) / (Ages_noNA.max() - Ages_noNA.min())
agecolumn = Ages_noNA.append(Ages_yesNa, ignore_index=False)


# In[ ]:


# Take all columns except age
restcolumns = titanic_indep.loc[:, titanic_indep.columns != "Age"]
restcolumns = restcolumns.apply(pd.to_numeric)

# Apply normalization to each column of this dataframe
for i in (range(0, len(list(restcolumns)))):
    restcolumns.iloc[:,[i]] = ((restcolumns.iloc[:,[i]] - restcolumns.iloc[:,[i]].mean()) / 
                               (restcolumns.iloc[:,[i]].max() - restcolumns.iloc[:,[i]].min()))


# In[ ]:


#Add the normalized age columns to the dataframe
restcolumns["Age"] = agecolumn
titanic_indep = restcolumns


# ### Impute missing ages
# Since there are many ages missing, I'll impute them using a machine learning technique instead of using the mean. This means that the ages will be imputed based on the other variables

# In[ ]:


from fancyimpute import KNN

#We use the train dataframe from Titanic dataset
#fancy impute removes column names, so let's save them.
titanic_cols = list(titanic_indep)

# Use Knn to fill in each value and add the column names back to the dataframe
titanic_indep = pd.DataFrame(KNN(k = 9).fit_transform(titanic_indep))
titanic_indep.columns = titanic_cols
titanic_indep["Age"] = round(titanic_indep["Age"])


# ### Create the models
# Allright so here the moddeling finally starts. For now i'm not going to get into the actual parameter engineering, but I might do this in the near future. I'll start with generating a Random forest model, knn model and Logistic regression model using a seperate training set, and see how acurate they are on the test set. 

# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(titanic_indep, titanic_dep, test_size=0.3)

#Tree packages for checking the feature importance
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier() 

# Build a forest and compute the feature importances
## Fit the model on your training data.
rf.fit(X_train, y_train) 

rf_predictions = rf.predict(X_test)
## And score it on your testing data.
rf.score(X_test, y_test)


# In[ ]:


#perform logistic regression and KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Do a Kfold cross validation on the training data for k = 3

knn = KNeighborsClassifier(n_neighbors = 7)
CVscores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = "accuracy")
print("knn score:", CVscores.mean())

knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# Do a Kfold cross validation on the training data for a logistic regression

logregression = LogisticRegression(solver='liblinear')
CVscores = cross_val_score(logregression, X_train, y_train, cv = 10, scoring = "accuracy")
print("logistic regression score:", CVscores.mean())

        
logregression.fit(X_train, y_train)        
logreg_predictions = logregression.predict(X_test)


# In[ ]:


#Print the accuracy scores for each model.
from sklearn.metrics import accuracy_score

print("Random forest fit score",accuracy_score(rf_predictions, y_test))
print("Knn fit score", accuracy_score(knn_predictions, y_test))
print("Logistic regression fit score", accuracy_score(logreg_predictions, y_test))


# #### Just some fun stuff 
# Let's see (because we can) whether the average of the 3 predictions give a more accurate estimation of the survivability. By adding each predicion together and dividing it by 3. 

# In[ ]:


#Let's see what the accuracy of the combined score would be:
combined_predictions = (rf_predictions + knn_predictions + logreg_predictions)/3
combined_predictions[combined_predictions < 0.5] = 0
combined_predictions[combined_predictions > 0.5] = 1
print("Combined predictions fit score", accuracy_score(combined_predictions, y_test))


# ## Logistic regression wins
# In the end, the logistic regression model was estimated on average to be the most accurate. Therefore I'll make a model using the full dataset (so without splitting it to test and train). Then, I'll use this model on the competition testset to check the results. **EXCITING**  

# In[ ]:


Log_regression_model = logregression.fit(titanic_indep, titanic_dep)
import pickle

# save the model to disk
filename = 'Logistic_reg_model.sav'
pickle.dump(Log_regression_model, open(filename, 'wb'))


# In[ ]:




