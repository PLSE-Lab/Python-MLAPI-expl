#!/usr/bin/env python
# coding: utf-8

# # Kaggle Titanic Challenge
# ****
# ### **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**

# In[ ]:


import pandas as pd

titanic_df = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_df.head()


# ### Variable Notes
#    - **PassengerId** Unique ID of the passenger
# 
#    - **Survived** Survived (1) or died (0)
# 
#    - **Pclass** Passenger's class (1st, 2nd, or 3rd)
# 
#    - **Name** Passenger's name
# 
#    - **Sex** Passenger's sex
# 
#    - **Age** Passenger's age
# 
#    - **SibSp** Number of siblings/spouses aboard the Titanic
# 
#    - **Parch** Number of parents/children aboard the Titanic
# 
#    - **Ticket** Ticket number
# 
#    - **Fare** Fare paid for ticket
# 
#    - **Cabin** Cabin number
# 
#    - **Embarked** Where the passenger got on the ship (C - Cherbourg, S - Southampton, Q = Queenstown)

# In[ ]:


# Exploring the data using pandas methods : 'shape', 'info', 'describe', 'dtype', 'mean()', ...
print(f"DataFrame shape : {titanic_df.shape}\n=================================")
print(f"DataFrame info : {titanic_df.info()}\n=================================")
print(f"DataFrame columns : {titanic_df.columns}\n=================================")
print(f"The type of each column : {titanic_df.dtypes}\n=================================")
print(f"How much missing value in every column : {titanic_df.isna().sum()}\n=================================")


# All good data analysis projects begin with trying to answer questions. Now that we know what column category data we have let's think of some questions or insights we would like to obtain from the data. So here's a list of questions we'll try to answer using our data analysis skills!
# 
# First some basic questions:
# 
#     1.) Who were the passengers on the Titanic? (Ages, Gender, Class,..etc)
#     2.) What deck were the passengers on and how does that relate to their class?
#     3.) Where did the passengers come from?
#     4.) Who was alone and who was with family?
#     
# Then we'll dig deeper, with a broader question:
# 
#     5.) What factors helped someone survive the sinking?

# # 1. Who were the passengers on the titanic?

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks")
plt.style.use("fivethirtyeight")


# In[ ]:


# Let's first check gender
# 'catplot()': Figure-level interface for drawing categorical plots onto a FacetGrid.
sns.catplot('Sex', data=titanic_df, kind='count')


# In[ ]:


# Now let separate the gender by classes passing 'Sex' to the 'hue' parameter
sns.catplot('Pclass', data=titanic_df, hue='Sex', kind='count')


# Wow, quite a few more males in the 3rd class than females, an interesting find. However, it might be useful to know the split between males, females, and children. How can we go about this?

# In[ ]:


# Create a new column 'Person' in which every person under 16 is child.

titanic_df['Person'] = titanic_df.Sex
titanic_df.loc[titanic_df['Age'] < 16, 'Person'] = 'Child'


# In[ ]:


# Checking the distribution
print(f"Person categories : {titanic_df.Person.unique()}\n=================================")
print(f"Distribution of person : {titanic_df.Person.value_counts()}\n=================================")
print(f"Mean age : {titanic_df.Age.mean()}\n=================================")


# Excellent! Now we have seperated the passengers between female, male, and child. This will be important later on beacuse of the famous **"Women and children first policy"**!

# In[ ]:


sns.catplot('Pclass', data=titanic_df, hue='Person', kind='count')


# Interesting, quite a bit of children in 3rd class and not so many in 1st! How about we create a distribution of the ages to get a more precise picture of the who the passengers were.

# In[ ]:


# visualizing age distribution
titanic_df.Age.hist(bins=80)


# In[ ]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(titanic_df, hue="Sex", aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))

fig.add_legend()


# In[ ]:


# We could have done the same thing for the 'person' column to include children:

fig = sns.FacetGrid(titanic_df, hue="Person",aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))

fig.add_legend()


# In[ ]:


# Let's do the same for class by changing the hue argument:

fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0, oldest))

fig.add_legend()


# We've gotten a pretty good picture of who the passengers were based on Sex, Age, and Class. So let's move on to our 2nd question: What deck were the passengers on and how does that relate to their class?

# # 2. What deck were the passengers on and how does that relate to their class?

# In[ ]:


# visualizing the dataset again
titanic_df.head()


# So we can see that the Cabin column has information on the deck, but it has several NaN values, so we'll have to drop them.

# In[ ]:


# First we'll drop the NaN values and create a new object, deck
deck = titanic_df['Cabin'].dropna()
deck


# Notice we only need the first letter of the deck to classify its level (e.g. A, B, C, D, E, F, G)

# In[ ]:


# let's grab that letter for the deck level with a simple for loop
levels = []
for level in deck:
    levels.append(level[0])

cabin_df = pd.DataFrame(levels)
cabin_df.columns = ['Cabin']
cabin_df.sort_values(by='Cabin', inplace=True)
sns.catplot('Cabin', data=cabin_df, kind='count', palette='winter_d')


# Interesting to note we have a 'T' deck value there which doesn't make sense, we  can drop it out with the following code:

# In[ ]:


cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot('Cabin', data=cabin_df, kind='count', palette='summer')


# Great now that we've analyzed the distribution by decks, let's go ahead and answer our third question.

# # 3. Where did the passengers come from?

# In[ ]:


titanic_df.head()


# Note here that the Embarked column has C, Q, and S values. Reading about the project on Kaggle you'll note that these stand for Cherbourg, Queenstown, Southhampton.

# In[ ]:


# Now we can make a quick factorplot to check out the results, note the 
# order argument, used to deal with NaN values

sns.catplot('Embarked', data=titanic_df, hue='Pclass', kind='count', order=['C', 'Q', 'S'])


# An interesting find here is that in Queenstown, almost all the passengers that boarded there were 3rd class. It would be intersting to look at the economics of that town in that time period for further investigation.

# # 4. Who was alone and who was with family?

# In[ ]:


titanic_df.head()


# In[ ]:


# Let's start by adding a new column to define alone
# We'll add the parent/child column with the sibsp column

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df.Alone


# Now we know that if the Alone column is anything but 0, then the passenger had family aboard and wasn't alone. So let's change the column now so that if the value is greater than 0, we know the passenger was with his/her family, otherwise they were alone.

# In[ ]:


# Look for > 0 or == 0 to set alone status
titanic_df.loc[titanic_df['Alone'] > 0, 'Alone'] = 'with Family'
titanic_df.loc[titanic_df['Alone'] == 0, 'Alone'] = 'Alone'


# In[ ]:


# Let's check to make sure it worked
titanic_df.head()


# In[ ]:


# Now let's get a simple visualization!
sns.catplot('Alone', data=titanic_df, kind='count', palette='Blues', 
            order=['Alone', 'with Family'])


# Great work! Now that we've throughly analyzed the data let's go ahead and take a look at the most interesting (and open-ended) question: *What factors helped someone survive the sinking?*

# # 5. What factors helped someone survive the sinking?

# In[ ]:


# Let's start by creating a new column for legibility purposes through mapping
titanic_df['Survivor'] = titanic_df.Survived.map({0:'No', 1:'Yes'})

# Let's just get a quick overall view of survied vs died. 
sns.catplot('Survivor', data=titanic_df, kind='count')


# So quite a few more people died than those who survived. Let's see if the class of the passengers had an effect on their survival rate, since the movie Titanic popularized the notion that the 3rd class passengers did not do as well as their 1st and 2nd class counterparts.

# In[ ]:


# Let's use a factor plot again, but now considering class
sns.catplot('Pclass', 'Survived', data=titanic_df, kind='point')


# Look like survival rates for the 3rd class are substantially lower! But maybe this effect is being caused by the large amount of men in the 3rd class in combination with the women and children first policy. Let's use 'hue' to get a clearer picture on this.

# In[ ]:


# Let's use a factor plot again, but now considering class and gender
sns.catplot('Pclass', 'Survived', data=titanic_df, hue='Person', kind='point')


# From this data it looks like being a male or being in 3rd class were both not favourable for survival. Even regardless of class the result of being a male in any class dramatically decreases your chances of survival.
# 
# But what about age? Did being younger or older have an effect on survival rate?

# In[ ]:


# Let's use a linear plot on age versus survival
sns.lmplot('Age', 'Survived', data=titanic_df)


# Looks like there is a general trend that the older the passenger was, the less likely they survived. Let's go ahead and use hue to take a look at the effect of class and age.

# In[ ]:


# Let's use a linear plot on age versus survival using hue for class seperation
sns.lmplot('Age', 'Survived',hue='Pclass', data=titanic_df)


# We can also use the x_bin argument to clean up this figure and grab the data and bin it by age with a std attached!

# In[ ]:


# Let's use a linear plot on age versus survival using hue for class seperation
generations = [10, 20, 40, 60, 80]
sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_df, palette='winter', x_bins=generations)


# Interesting find on the older 1st class passengers! What about if we relate gender and age with the survival set?

# In[ ]:


sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df, palette='winter', x_bins=generations)


# Awesome! we've gotten some really great insights on how gender,age, and class all related to a passengers chance of survival. Now you take control: Answer the following questions using pandas and seaborn:
# 
#     1.) Did the deck have an effect on the passengers survival rate? Did this answer match up with your intuition?
#     2.) Did having a family member increase the odds of surviving the crash?

# ## References:
# - [Jose Portilla Udemy Course: Learning Python for Data Analysis and Visualization](https://www.udemy.com/course/learning-python-for-data-analysis-and-visualization/)
