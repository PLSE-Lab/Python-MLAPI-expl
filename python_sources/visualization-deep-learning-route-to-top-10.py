#!/usr/bin/env python
# coding: utf-8

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# The guided approach explained here will help you to understand how you should design and approach Data Science problems. Though there are many ways to do the same analysis, I have used the codes which I found more efficient and helpful.
# 
# The idea is just to show you the path, try your own ways and share the same with others.

# **What would be the workflow?**
# 
# **1. Problem Identification**
# 
# **2. What data do we have?**
# 
# **3. Exploratory data analysis**
# 
# **4. Feature engineering**
# 
# **5. Creating a model using Keras**
# 
# **6. Model evaluation**
# 
# **7. Conclusions**
# 
# That's all you need to solve a data science problem.

# # Problem Identification
# ![Titanic.jpg](attachment:Titanic.jpg)
# 
# **Best Practice -** The most important part of any project is correct problem identification. Before you jump to "How to do this" part like typical Data Scientists, understand "What/Why" part.  
# Understand the problem first and draft a rough strategy on a piece of paper to start with. Write down things like what are you expected to do & what data you might need or let's say what all algorithms you plan to use. 
# 
# Now the [Titanic challenge](https://www.kaggle.com/c/titanic/) hosted by Kaggle is a competition in which the goal is to **predict the survival or the death of a given passenger based on a set of variables describing  age, sex, or passenger's class on the boat**.
# 
# ![](http://www.tyro.com/content/uploads/2016/04/blog-twenty-one-business-icebergs-sink-business-280416.jpg)
# 
# So it is a classification problem and you are expected to predict Survived as 1 and Died as 0.

# # What data do we have?
# ![Data.jpg](attachment:Data.jpg)
# 
# Let's import necessary libraries & bring in the datasets in Python environment first. Once we have the datasets in Python environment we can slice & dice the data to understand what we have and what is missing.

# In[ ]:


# Import the python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# load the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
IDtest = pd.DataFrame(test["PassengerId"])


# In[ ]:


test.info() # We have 891 observations & 12 columns. See the mix of variable types.


# In[ ]:


test.info() # We have 417 observations & 11 columns (no response 'Survived' column).


# # Exploratory data analysis
# ![analysis.png](attachment:analysis.png)
# 
# One important aspect of machine learning is to ensure that the variables show almost the same trend across train & test data. If not, it would lead to overfitting because model is representing a relationship which is not applicable in the test dataset. 
# 
# I will give you one example here. As we do variable analysis, try to replicate (wherever applicable) the code for test data and see if there is any major difference in data distribution. 
# 
# **Example** - Let's start with finding the number of missing values. If you compare the output you will see that missing value percentages do not vary much across train & test datasets.
# 
# Use the groupby/univariate/bivariate analysis method to compare the distribution across Train & Test data

# In[ ]:


train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
miss_train = pd.DataFrame({'Train Missing Ratio' :train_na})
miss_train.head()


# In[ ]:


test_na = (test.isnull().sum() / len(test)) * 100
test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)[:30]
miss_test = pd.DataFrame({'Test Missing Ratio' :test_na})
miss_test.head()


# In[ ]:


# Fill empty and NaNs values with NaN
train = train.fillna(np.nan)
test = test.fillna(np.nan)


# **PassengerId**
# 
# Not relevant from modeling perspective so we will drop this variable later

# **Pclass**
# 
# Pclass is categorical variable. Let's look at the distribution.

# In[ ]:


# Analyze the count of survivors by Pclass

ax = sns.countplot(x="Pclass", hue="Survived", data=train)
train[['Pclass', 'Survived']].groupby(['Pclass']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze the Survival Probability by Pclass

g = sns.barplot(x="Pclass",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)


# Approximately 62% of Pclass = 1 passenger survived followed by 47% of Pclass2.

# **Name**
# 
# Not relevant from analysis & modeling perspective. We will drop this feature later after creating a new variable as Title.

# **Sex**
# 
# Based on analysis below, female had better chances of survival. 
# 
# ![](https://www.ajc.com/rf/image_large/Pub/p9/AJC/2018/07/12/Images/newsEngin.22048809_071418-titanic_Titanic-Image-7--2-.jpg)

# In[ ]:


# Count the number of passengers by gender
ax = sns.countplot(x="Sex", hue="Survived", data=train)

# Analyze survival count by gender
train[["Sex", "Survived"]].groupby(['Sex']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze the Survival Probability by Gender

g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[["Sex", "Survived"]].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)


# Based on data above, female passengers had better chances of survival than male passengers

# **Age**
# 
# The insight below connects back to "Ladies and Kids First" scene of the movie. It shows that a good number of babies & young kids survived.

# In[ ]:


# Let's explore the distribution of age by response variable (Survived)
fig = plt.figure(figsize=(10,8),)
axis = sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='Survived')
axis = sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='b',shade=True,label='Did Not Survived')
plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 20)
plt.xlabel("Passenger Age", fontsize = 12)
plt.ylabel('Frequency', fontsize = 12);


# In[ ]:


sns.lmplot('Age','Survived',data=train)

# We can also say that the older the passenger the lesser the chance of survival


# **SibSP**
# 
# This variable refers to number of siblings/spouse onboard. SibSP = 1 and SibSP = 2 shows higher chances of survival.

# In[ ]:


# Analyze the count of survivors by SibSP

ax = sns.countplot(x="SibSp", hue="Survived", data=train)
train[['SibSp', 'Survived']].groupby(['SibSp']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze probability of survival by SibSP

g  = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train[["SibSp", "Survived"]].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)


# **Parch**
# 
# Parch indicates number of parents / children aboard the Titanic. Note that Parch = 3 and Parch = 1 shows higher survival probabilities. 

# In[ ]:


# Analyze the count of survivors by Parch

ax = sns.countplot(x="Parch", hue="Survived", data=train)
train[['Parch', 'Survived']].groupby(['Parch']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze the Survival Probability by Parch

g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Survival Probability")
train[["Parch", "Survived"]].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)


# **Ticket**
# 
# This variable has alphanumeric value which might not be related to Survival directly but we can use this variable to create some additional features.

# In[ ]:


train['Ticket'].head()


# **Fare**
# 
# Let's check the distribution first.

# In[ ]:


from scipy import stats
from scipy.stats import norm, skew #for some statistics
sns.distplot(train['Fare'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['Fare'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('Fare distribution')


# The Fare variable is right skewed. 
# So either we can transform this variable using log function and make it more normally distributed or we can create bins. We will do this during feature engineering process & decide what works best.

# **Cabin**
# 
# Alphanumeric variable. 
# 
# 687 missing values in train & 327 missing values in test data - which needs to be treated. We can create more features using this Cabin variable. 

# In[ ]:


# Let's check the unique values
train['Cabin'].unique()


# **Embarked**
# 
# C = Cherbourg, Q = Queenstown, S = Southampton
# 
# Let's explore the variable with Survival rate. Embarked represents port of embarkation. As the analysis output below suggests Emabrked C shows high probabilities of survival.

# In[ ]:


# Analyze the count of survivors by Embarked variable

ax = sns.countplot(x="Embarked", hue="Survived", data=train)
train[['Embarked', 'Survived']].groupby(['Embarked']).count().sort_values(by='Survived', ascending=False)


# In[ ]:


# Analyze the Survival Probability by Embarked

g  = sns.factorplot(x="Embarked",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train[["Embarked", "Survived"]].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False)


# **Additional analysis**
# 
# Let's create few additional charts to see how different variables are related.

# In[ ]:


# Age, Pclass & Survival
sns.lmplot('Age','Survived',data=train,hue='Pclass')


# In[ ]:


# Age, Embarked, Sex, Pclass
g = sns.catplot(x="Age", y="Embarked",  hue="Sex", row="Pclass",   data=train[train.Embarked.notnull()], 
orient="h", height=2, aspect=3, palette="Set3",  kind="violin", dodge=True, cut=0, bw=.2)


# In[ ]:


# Relation among Pclass, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="Pclass", data=train, saturation=.5, 
                kind="bar", ci=None, aspect=.6)


# In[ ]:


# Relation among SibSP, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="SibSp", data=train, saturation=.5, 
                kind="bar", ci=None, aspect=.6)


# In[ ]:


# Relation among Parch, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="Parch", data=train, saturation=.5, 
                kind="bar", ci=None, aspect=.6)


# # Feature engineering
# ![FE.png](attachment:FE.png)
# 
# This kernel is based on classic "**LESS IS MORE**" approach so we will try some iterations of feature addition and deletion and will try to keep the ones which give the best output.

# 
# 
# What we need to do to process following variables  - 
# 
# **PassengerID** - No action required
# 
# **PClass** - Have only 3 numerical values. We will use it as it is.
# 
# **Name** - Can be used to create new variable Title by extracting the salutation from name.
# 
# **Sex** - Create dummy variables
# 
# **Age** - Missing value treatment, followed by creating bins for this feature
# 
# **SibSP** - Drop the variable after using it to create few additional features
# 
# **Parch** - Drop the variable after using it to create few additional features
# 
# **Ticket** - Create dummy variables post feature engineering
# 
# **Fare** - Create bins for this feature
# 
# **Cabin** - Drop the variable after testing the importance
# 
# **Embarked** - Drop the variable after testing the importance

# In[ ]:


# Let's combining train & test for quick feature engineering. 
# Variable source is a kind of tag which indicates data source in combined data
train['source']='train'
test['source']='test'
combdata = pd.concat([train, test],ignore_index=True)
print (train.shape, test.shape, combdata.shape)


# In[ ]:


# Let's check the data
combdata.head()


# **PassengerID**

# In[ ]:


# PassengerID - Drop PassengerID
combdata.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# **Pclass**

# In[ ]:


# Pclass - Use as it is
combdata['Pclass'].unique()


# **Name**

# In[ ]:


combdata['Title'] = combdata.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())

# inspect the amount of people for each title
combdata['Title'].value_counts()


# In[ ]:


# Name - Create 4 major categories & analyze the survival rate

combdata['Title'] = combdata['Title'].replace('Mlle', 'Miss')
combdata['Title'] = combdata['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
combdata.Title.loc[ (combdata.Title !=  'Master') & (combdata.Title !=  'Mr') & 
                   (combdata.Title !=  'Miss')  & (combdata.Title !=  'Mrs')] = 'Others'

# inspect the correlation between Title and Survived
combdata[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


# Create dummy variable & drop variable Name

combdata = pd.get_dummies(combdata, columns = ["Title"])


# In[ ]:


# Drop the variable Name
combdata = combdata.drop(labels=['Name'], axis=1)


# **Sex**

# In[ ]:


# Sex - Create dummy variables
combdata["Sex"] = combdata["Sex"].map({"male": 0, "female":1}) 
# combdata = pd.get_dummies(combdata, columns = ["Sex"])


# **Creating Family Size variable using SibSp & Parch**

# In[ ]:


# Create a variable representing family size from SibSp and Parch
combdata["Fsize"] = combdata["SibSp"] + combdata["Parch"] + 1

# Analyze the correlation between Family and Survived
combdata[['Fsize', 'Survived']].groupby(['Fsize'], as_index=False).mean()


# In[ ]:


# Check the count
combdata["Fsize"].value_counts()


# Survival rate improves with family size but not beyond family size 4 so we can combine the family size > 4 together

# In[ ]:


# Analyze the Survival Probability by Fsize

combdata.Fsize = combdata.Fsize.map(lambda x: 0 if x > 4 else x)
g  = sns.factorplot(x="Fsize",y="Survived",data=combdata,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Survival Probability")
combdata[["Fsize", "Survived"]].groupby(['Fsize']).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Check the count by Family Size now
combdata['Fsize'].value_counts()


# **Parch**

# In[ ]:


# Drop the variable Parch
combdata = combdata.drop(labels='Parch', axis=1)


# **Ticket**

# In[ ]:


# Ticket - Extracting the ticket prefix. This might be a representation of class/compartment.
combdata["Ticket"].head(10)


# Tickets are of 2 types here. 
# 
# Type 1 has only number and 
# Type 2 is a combination of some code followed  by the number. Let's extract the first digit and compare it with survival probability.

# In[ ]:


combdata.Ticket = combdata.Ticket.map(lambda x: x[0])

# inspect the correlation between Ticket and Survived
combdata[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()

#combdata[["Ticket", "Survived"]].groupby(['Ticket']).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Let's look at the number of people for each type of tickets
combdata['Ticket'].value_counts()


# Most of these tickets belong to category 1, 2, 3, S, P, C. Based on value counts and average survival, we can put all other ticket categories into a new category '4'.

# In[ ]:


combdata['Ticket'] = combdata['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')

# check the correlation again
combdata[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()


# In[ ]:


# Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["Ticket"], prefix="T")


# **Fare**

# In[ ]:


# Fare - Check the number of missing value
combdata["Fare"].isnull().sum()

# Only 1 value is missing so we will fill the same with median
combdata["Fare"] = combdata["Fare"].fillna(combdata["Fare"].median())


# In[ ]:


# bin Fare into five intervals with equal amount of people
combdata['Fare-bin'] = pd.qcut(combdata.Fare,5,labels=[1,2,3,4,5]).astype(int)

# inspect the correlation between Fare-bin and Survived
combdata[['Fare-bin', 'Survived']].groupby(['Fare-bin'], as_index=False).mean()


# **Cabin**

# In[ ]:


# Cabin - Replace the missing Cabin number by the type of cabin unknown 'U'
combdata["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'U' for i in combdata['Cabin'] ])


# In[ ]:


# Let's plot the survival probability by Cabin
g  = sns.factorplot(x="Cabin",y="Survived",data=combdata,kind="bar", size = 7 ,
                    palette = "muted",order=['A','B','C','D','E','F','G','T','U'])
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[ ]:


combdata = combdata.drop(labels='Cabin', axis=1)


# **Embarked**

# In[ ]:


combdata = combdata.drop(labels='Embarked', axis=1)


# **Age**
# 
# There are 2 ways of handling the missing age values.
# 1. Fill the age with median age of similar rows according to Sex, Pclass, Parch & SibSP
# 2. or use a quick machine learning algorithm to predict the age values based on Age, Title, Fare & SibSP
# 
# I used both of them to test which one works better. One of the code will be markdown to avoid confusion.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
combdata_temp = combdata[['Age','Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Others','Fare-bin','SibSp']]

X  = combdata_temp.dropna().drop('Age', axis=1)
Y  = combdata['Age'].dropna()
holdout = combdata_temp.loc[np.isnan(combdata.Age)].drop('Age', axis=1)

regressor = RandomForestRegressor(n_estimators = 300)
#regressor = GradientBoostingRegressor(n_estimators = 500)
regressor.fit(X, Y)
y_pred = np.round(regressor.predict(holdout),1)
combdata.Age.loc[combdata.Age.isnull()] = y_pred

combdata.Age.isnull().sum(axis=0) 


# In[ ]:


bins = [ 0, 4, 12, 18, 30, 50, 65, 100] # This is somewhat arbitrary...
age_index = (1,2,3,4,5,6,7)

combdata['Age-bin'] = pd.cut(combdata.Age, bins, labels=age_index).astype(int)
combdata[['Age-bin', 'Survived']].groupby(['Age-bin'],as_index=False).mean()


# **SibSP**

# In[ ]:


# Drop the variables we don't need

combdata =combdata.drop(labels=['Age', 'Fare', 'SibSp'],axis = 1)


# # Build a model
# ![NN%20Model.png](attachment:NN%20Model.png)

# In[ ]:


## Separate train dataset and test dataset using the index variable 'source'

train_df = combdata.loc[combdata['source']=="train"]
test_df = combdata.loc[combdata['source']=="test"]
test_df.drop(labels=["Survived"],axis = 1,inplace=True)

train_df.drop(labels=["source"],axis = 1,inplace=True)
test_df.drop(labels=["source"],axis = 1,inplace=True)

test_df.info()


# In[ ]:


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train_df.drop(labels = ["Survived"],axis = 1)

X_train.info()


# In[ ]:


import keras 
from keras.models import Sequential # intitialize the ANN
from keras.layers import Dense      # create layers

# Initialising the NN
model = Sequential()

# layers
model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_train, Y_train, batch_size = 32, epochs = 200)


# # Model evaluation

# In[ ]:


scores = model.evaluate(X_train, Y_train, batch_size=30)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# # Final submission

# In[ ]:


y_pred = model.predict(test_df)
y_final = (y_pred > 0.5).astype(int).reshape(test_df.shape[0])

output = pd.DataFrame({'PassengerId': IDtest['PassengerId'], 'Survived': y_final})

#output = pd.concat([IDtest,y_final],axis=1)
output.to_csv('Neural Network Prediction.csv', index=False)


# # Conclusion
# ![Conclusion.png](attachment:Conclusion.png)
# 
# Title, Sex_Female, Fare & PClass seems to be common features preferred for classification.
# 
# While Title & Age feature represents the Age category of passengers the features like Fare, PClass, Cabin etc. represents the economic status. Based on our findings we can conclude that Age, Gender & features representing social/economic status were primary factors affecting the survival of passenger.
# 

# **If you like this notebook or find this notebook helpful, Please upvote and/or leave a comment**
# ![Good%20Bye.png](attachment:Good%20Bye.png)
