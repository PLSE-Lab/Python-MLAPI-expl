#!/usr/bin/env python
# coding: utf-8

# 

# # Content
# 
# * [1. Intro](#intro)
# * [2. Imports](#imports)
# * [3. Reading and Describing Data](#readinganddescribingdata)
# * [4. Exploratory Data Analysis](#exploratorydataanalysis)
# * [5. Feature Engineering](#featureengineering)
# * [6. Model Building](#modelbuilding)

# # 1. Intro
# <a id="intro"></a> 
# 
# In this Kernel first I will apply some data exploration analysis so we can gain understand and maybe get some insights of the data, next I will try to extract some features from the columns dataset have and finally I will build a simple model trying to predict the animal's outcomes. I hope you can learn something from this kernel, its my first one so lets do it.

# # 2. Imports
# <a id="imports"></a> 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style="white", color_codes=True)


# # 3. Reading and Describing Data
# <a id="readinganddescribingdata"></a> 

# In[2]:


data = pd.read_csv('../input/aac_shelter_outcomes.csv')


# In[3]:


data_cols = data.columns
data.info()


# In[4]:


print(data.iloc[0])


# It seems that we are mainly dealing with a mix of categorical data and numeric data, in the case of numeric data we will need to extract it from some columns so they can be used, such as age and date_of_birth. It will be done later in the Feature Extraction section.

# # 4. Exploratory Data Analysis
# <a id="exploratorydataanalysis"></a> 

# In[5]:


data.nunique()


# There seems to be a lot of breeds and names, maybe if we do some clean up on this columns we can reduce this number (e.g. getting the lowercase)

# In[6]:


data.groupby(['outcome_type','outcome_subtype']).size()


# ### outcome_type

# In[7]:


# Lets start taking a loot at the possible outcomes
plt.figure(figsize=(12,4))
sns.countplot(y=data['outcome_type'], 
              palette='rainbow',
              order=data['outcome_type'].value_counts().index)
plt.show()


# We can see that most of the animals have a good outcome, beeing adoption the biggest one and cases like death and disposal are very rare. In this case we can see that the classes are not really balanced and the fact that don't have much features to work with, it seems that correctly predicting death or any of those unlikely outcomes will be a challenge.

# ### sex_upon_outcome vs outcome_type

# In[26]:


# plt.figure(figsize=(12,10))
# sns.countplot(y=data['sex_upon_outcome'], 
#                   palette='rainbow',
#                   hue=data['outcome_type'])
# plt.show()
#data['sex_upon_outcome'].value_counts()
g = sns.FacetGrid(data, row='sex_upon_outcome', aspect=5)
g.map(sns.countplot, 'outcome_type', palette='rainbow')
#x = data[['sex_upon_outcome', 'outcome_type']].groupby(by=['sex_upon_outcome', 'outcome_type']).head()
#x.value_counts()


# This plot is interesting because it shows a clear difference in the distributions of outcome when looking at the animal's sex, for example neutered males seems much more likely to be transfered than intact males. This tells us that this should be a strong variable when trying to build our prediction model.

# In[30]:


plt.figure(figsize=(12,6))
sns.countplot(data=data,
              x='animal_type',
              hue='outcome_type')
plt.legend(loc='upper right')
plt.show()
# g = sns.FacetGrid(data, row='animal_type', aspect=4)
# g.map(sns.countplot, 'outcome_type', palette='rainbow')


# It seems that the distribution of outcomes also differs from animal types, we can clearly see that dogs are more likely to be returned to the owner than cats,  that animals from other category are more likely to be euthanised that any other outcome. With this we can see that this would be a very strong variable for the later model that will be built.
# 
# We will continue doing some exploration while making new features

# # 5. Feature Engineering
# <a id="featureengineering"></a> 
# 

# There are many categorical data that would gives us a lot more information and become easier to use if we do some wrangling around. Let's take a look at them

# In[ ]:


age_types = data['age_upon_outcome'].dropna().unique().tolist()
print(age_types)


# Things don't look good with the age column, but do we really need it? The format seems pretty imprecise, and if we take a look at the columns we have there are two very interest columns: 'datetime' and 'date_of_birth', from some observation we can see that 'datetime' is the time of the outcome, so why don't we calculate the animal's age ourselfs?

# In[ ]:


# Let's format our datetime and date_of_birth columns to datetime format
data['date_of_birth'] = pd.to_datetime(data['date_of_birth'], format='%Y-%m-%d')
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')


# In[ ]:


data['days_of_age'] =  (data['datetime'] - data['date_of_birth']).dt.days


# Now let's use our new columns for more exploration

# In[ ]:


data['days_of_age'].describe()


# Whops, seems that this column is not the most realiable one, you can see that there is some negative values for days_of_age, what shoudn't happen in real life. Its not so unusual to see things like this, so lets just do a quick fix to this column.

# In[ ]:


data[data['days_of_age'] < 0] = 0 


# Tha should fix it, now let's use our new column for some plots.

# In[ ]:


plt.subplots(figsize=(12, 6))
g = sns.boxplot(data=data, x='outcome_type', y='days_of_age', hue='animal_type')
labels = g.get_xticklabels()
g.set_xticklabels(labels,rotation=50)
plt.show(g)


# Now we can see new things! It seems that different outcomes happen at different ages of the animal, but we can go more in-depth because different types of animals would certainly have different age of outcome distributions and this is a good thing because the more we can differentiate it just by looking at some plots, a good model would definitely be abre to capture this information.
# 
# A we have this days column why not use it to drop some kde plots, just because we can? Lets do it for animal_types.

# In[ ]:


g = sns.FacetGrid(data, hue="animal_type", size=12)
g.map(sns.kdeplot, "days_of_age") 
g.add_legend()
g.set(xlim=(0,5000), xticks=range(0,5000,365))
plt.show(g)


# That's interesting, we can see a trend here, if we look closely we can see that those peaks happen in what would be when the animal completed another year of age. It make sense if we think that animals shelters would make cutoffs for the age when deciding what to do with an animal. For example when an animal completes 4 years and they transfer it.
# Or maybe they don't known exacly the age of the animal, so the column date of birth that we derived ours dates from are just an approximation.
# 
# Let's take a closer look at the cats.

# In[ ]:


g = sns.FacetGrid(data[(data['animal_type']=='Cat') & (data['days_of_age']<2000)], 
                  hue="outcome_type", 
                  size=12)
g.map(sns.kdeplot, "days_of_age")
g.add_legend()
g.set(xlim=(0,1200), xticks=range(0,1200,365))
plt.show(g)


# It seems that most cats are adopted within the first months, we can also see the yearly trend happening here and that there is a lot of disposal outcome happening on the first year. Many things to think around here.
# 
# Let's do some feature engineering in the other variables so that we can build our model.

# In[ ]:


data.head()


# In[ ]:


# Lets remove the age_upon_outcome because we haver our days_of_age column already
data.drop('age_upon_outcome', axis=1, inplace=True)
# Lets also drop the date of birth, datetime and monthyear of our dataset for the sake of simplicity
data.drop(['date_of_birth','datetime', 'monthyear'], axis=1, inplace=True)
# The animal_id column is made of unique values that add nothing to our model, lets drop it
data.drop('animal_id', axis=1, inplace=True)


# In[ ]:


# Lets transform all string columns into lowercase strings 
string_columns = ['animal_type', 'breed', 'color', 'name', 'outcome_subtype', 'outcome_type', 'sex_upon_outcome']
for col in string_columns:
    data[col] = data[col].str.lower()


# In[ ]:


data.head()


# From looking at the name column just at the head of the dataset we see something not quite right, at index 2 the name is "*johny", lets clean this up too, lets remove alll ponctuation from the string columns.

# In[ ]:


def text_process(text):
    '''
    takes in a string and return
    a string with no punctuations
    but puts spaces in slash place
    '''
    text = text.replace('/', ' ')
    return ''.join([char for char in text if char not in string.punctuation])


# In[ ]:


for col in string_columns:
    data[col] = data[col].apply(lambda x:text_process(str(x)))


# In[ ]:


data.head()


# # 6. Model Building
# <a id="modelbuilding"></a> 

# Now the action starts. We will try many types of model but first let's do something about the categorical variables because we can't usem them as is and also do some modifications that seems fit.

# In[ ]:


data2 = data # data2 = placeholder


# In[ ]:


# dropping outcome_subtype as we won't try to predict it (yet?)
data.drop('outcome_subtype', axis=1, inplace=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


columns = ['animal_type', 'breed', 'color', 'name',
       'sex_upon_outcome', 'outcome_type']

def encoder(df):
    for col in columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col])
    return df


# In[ ]:


# feature = 'outcome_type'
# label_encoder = LabelEncoder()
# label_encoder.fit(data[feature])
# data[feature] = label_encoder.transform(data[feature])
data = encoder(data)


# In[ ]:


data.head()


# Now it's time to split up the data so we can later test the accuracy of our model in the test split. Just remember that we have a imbalanced target value, so we will tell the split function that we whant a stratified split.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = data.drop('outcome_type', axis=1)
y = data['outcome_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0, stratify=y)


# Let's try starting with a simple Decisions Tree classifier.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)


# In[ ]:


predictions = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, predictions))


# In[ ]:


list(data['outcome_type'].value_counts() / data['outcome_type'].count())[0]


# It seems that with this classifier we got a 61% f1 score, to use some baseline we can look at the most baseline model which would be just predicting the most common class, that would be 42%. Compared to it our model looks like it's actually doing something, but can we improve it? 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfclf = RandomForestClassifier()
rfclf.fit(X_train, y_train)


# In[ ]:


rfclf_predictions = rfclf.predict(X_test)


# In[ ]:


print(classification_report(y_test, rfclf_predictions))


# Looks like our predictions improved a bit, now how about taking a look at the feature importance of our model.

# In[ ]:


feat_importance = pd.DataFrame({'Feature':data.columns[:-1],'Importance':rfclf.feature_importances_.tolist()})

plt.subplots(figsize=(8, 6))
g = sns.barplot(data=feat_importance, x='Feature', y='Importance')
labels = g.get_xticklabels()
g.set_xticklabels(labels,rotation=50)
plt.show(g)


# Oh my, that's interesting, it seems that all columns have a significant importance, but sex_upon_outcome is a surprise, maybe because the sex of the animal make one or another outcome much more probable? It seems that the name column also have a great impact, maybe having ou having not a name is a big sign also, should they give a name to all new animals? Well, maybe, but that's it for this notebook.

# I appreciate if you took your time to read it until the end, it is my first public kernel and I hope it's the first of many.

# In[ ]:




