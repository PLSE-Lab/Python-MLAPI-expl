#!/usr/bin/env python
# coding: utf-8

# Titanic for dummies (Pandas,Logistic Regression)
# =======

# This kernel is sort of first step beyond the "**gender submission**" and a **first approch** to **Pyhton** and **ML**.  It introduces the most simple classificator, **logistic regression**. 
# 
# The notebook is dedicated to any beginner but especially to my students at [MaCSIS](http://http://www.colpodiscienza.it/). I am **not** a data scientist, they aren't too, but we are **discovering together** the fantastic world of **data science** and it's very exciting!

# Preliminary Operations
# ------

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


train_filepath='../input/train.csv'
test_filepath='../input/test.csv'


# In[ ]:


import numpy as np
import pandas as pd
df_train0 = pd.read_csv(train_filepath,decimal=",")
df_test0 = pd.read_csv(test_filepath,decimal=",")


# In[ ]:


df_train0.head()


# In[ ]:


df_test0.head()


# The output variable (Survived) is a binary variable. It doesn't make sense using commands like df_train0['Survived'].describe(), .skew(), .kurt() which would be very useful if it was a continue one. In this case, the best point to start from is an histogram or, even more simply, a count of the unique values and their occurrences.

# In[ ]:


df_train0['Survived'].value_counts(normalize=True)


# So we have about 38% if survived. It's useful concatenating the two dataframes, train and test. In the test part wel'll add a Survived column, set to -1. We give the command reindex to preserve the columns sort between the first and the second dataframe.

# In[ ]:


df_test0['Survived']=-1
df_train0.columns
df_test0=df_test0.reindex(columns=df_train0.columns)
df_test0.head()


# And now, the concatenation:

# In[ ]:


df_all=pd.concat([df_train0,df_test0]).copy()
df_all.head()


# In[ ]:


df_all.dtypes


# Missing Data
# -----
# 
# To deal with the missing data you can use the isnull() function. There is also an isna() but in this case it's easy to verify that it gives the same result.

# In[ ]:


df_all.isnull().sum()


# Now let's treat the various columns/variables one by one, starting from Age.

# In[ ]:


df_all['Age'].describe()


# Previously we filled the missing cells of the Age column with a unique default/mean value. This is ok, but we can also go beyond filling the empty spaces with values depending on other higly related cells. So let's try to find what are the most important variables for Age.
# 
# Probably Pclass and Sex (both categorial can play a role), so we'll use for the exploration the boxplot diagram, after the conversion in numeric of Age, which is now a string variable.

# In[ ]:


dg=df_all[['Sex','Pclass','Age']].dropna().copy()


# In[ ]:


dg.head()


# In[ ]:


dg['Age']=pd.to_numeric(dg['Age'])


# In[ ]:


import seaborn as sns


# In[ ]:


sns.boxplot(x='Pclass', y='Age', data=dg)


# In[ ]:


sns.boxplot(x='Sex', y='Age', data=dg)


# To sum up we can say that different Sexes don't imply too different Ages. On the contrary, PClass and Age can be considered as highly related. Therefore we'll assign default values not as constants but as functions of the PClass variable. We can do that using a simple function like this:

# In[ ]:


def input_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1: return 38
        elif Pclass==2: return 30
        else: return 25
    else: return Age


# In[ ]:


df_all['Age']=df_all[['Age','Pclass']].apply(input_age,axis=1)


# In[ ]:


df_all['Age']=df_all['Age'].astype(float)


# Now let's consider the Fare column:

# In[ ]:


df_all['Fare'].describe()


# The case is quite similar but we have only one NaN, therefore using the basic mean it's enough.

# In[ ]:


bool_val_not_null=np.invert(df_all['Fare'].isnull())
fares_num=pd.to_numeric(df_all['Fare'][bool_val_not_null])
np.average(fares_num)


# This is the command:

# In[ ]:


df_all.loc[df_all['Fare'].isnull(),'Fare']=33.3


# .. and this is the conversion in a floating point variable:

# In[ ]:


df_all['Fare']=df_all['Fare'].astype(float)


# Embarked is a categorial variable so we cannot calculate its mean. But we can find the frequencies of the distinct values and that is exactly what we are going to do:

# In[ ]:


df_all['Embarked'].value_counts(normalize=True)


# We can simpy assing the 'S' to the missing values.

# In[ ]:


df_all.loc[df_all['Embarked'].isnull(),'Embarked']='S'


# Now it's the turn of Cabin and Ticket which have too many distinct values to convert them in dummy varables. This is our dataset:

# In[ ]:


df_all.head(10)


# We choose to ignore ticket and focus only on the Cabin column, which has these distinc values:

# In[ ]:


df_all['Cabin'].unique()


# The idea is to delete nulls and then take only the first char, mainly to cut off multiple tickets:

# In[ ]:


df_all.loc[df_all['Cabin'].isnull(),'Cabin']='XX'


# Let's check that we have no more null Cabin values ...

# In[ ]:


df_all.isnull().sum()


# ... and then extract the first letter of that string variable:

# In[ ]:


df_all['Cabin']=df_all['Cabin'].str.extract('(.)')


# In[ ]:


df_all.head()


# The Name can be interesting mainly because it contains the title, which is reasonably related to the economic status of the person and to a probably better accomodation. We'll use a regular expression to do that in this way: 

# In[ ]:


n=df_all['Name'].str.extract('.+?,\s(.+?).\s')[0]


# We have the names in the form "XXX, [Title] YYY" so our regexp simply says: "find the comma, ignore XXX and put in the extracted value what you find between the two spaces". These are the different values found in this way:

# In[ ]:


n.value_counts()


# Now we have to deal with some remaining "strange" cases. Let's see them one by one.

# In[ ]:


df_all[n=='Jonkheer']


# In[ ]:


df_all[n=='th']


# In[ ]:


df_all[n=='Master'].head()


# Probably "Master" is the older male son in the richest families. 
# 
# The following are considered synonims of noble people:

# In[ ]:


n[n=='Jonkheer']='Nh'


# In[ ]:


n[n=='Sir']='Nh'


# In[ ]:


n[n=='th']='Nh'


# Other translations (from different classifications and/or languages):

# In[ ]:


n[n=='Mlle']='Miss'


# In[ ]:


n[n=='Mme']='Mrs'


# In[ ]:


n[n=='Lady']='Miss'


# In[ ]:


n[n=='Dona']='Miss'


# In[ ]:


n[n=='Don']='Rev'


# In[ ]:


n[n=='Ms']='Miss'


# In[ ]:


n[n=='Major']='Nh'


# In[ ]:


n[n=='Capt']='Nh'


# In[ ]:


n[n=='Col']='Nh'


# In[ ]:


n.value_counts()


# In[ ]:


df_all['Title']=n.copy()


# We can say that SibSp and Parch are both related to family and we think that the only importat thing is this: if you have a familiar on boad, probably he/she and you will collaborate to survive, so surviving chances of both of you, reasonably increase. Then let's create a new feature called "Fam" (for number of FAMily members):

# In[ ]:


df_all['Parch'].value_counts()


# In[ ]:


df_all['Fam']=df_all['SibSp']+df_all['Parch']


# To sum up this is the head of our dataset, after data preparation:

# In[ ]:


df_all.head()


# First Modeling
# -----

# First of all, we'll work on a backup copy of the cleaned dataframe (just to be protected in case of wrong updates, or to have a checkpoint to restart from):

# In[ ]:


df=df_all.copy()
df.head()


# We save it into file, too.

# In[ ]:


df.to_csv('titanic_full.csv',index=False)


# Now let's proceed with the ML optimization. At first, we delete some useless columns:

# In[ ]:


df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.drop('SibSp', axis=1, inplace=True)
df.drop('Parch', axis=1, inplace=True)
df.head()


# Another backup in the filesystem:

# In[ ]:


df.to_csv('titanic_clean.csv',index=False)


# ### Dummy Variables

# Dummy variable, i.e. one column with binary values for all the different values of the source column. We use drop_first just because, for example in the Sex case, if we keep two column Male=0/1 and Female=0/1, we introduce a relationship in the dataset (Male=Not(Female)) which can crash our previsional model (based on the hypotesis of indipendent input variables).

# In[ ]:


sex=pd.get_dummies(df['Sex'],drop_first=True)


# In[ ]:


embarked=pd.get_dummies(df['Embarked'],drop_first=True)


# In[ ]:


title=pd.get_dummies(df['Title'],drop_first=True)


# In[ ]:


cabin=pd.get_dummies(df['Cabin'],drop_first=True)


# In[ ]:


df=pd.concat([df,sex,embarked,title,cabin], axis=1)
df.drop('Embarked', axis=1, inplace=True)
df.drop('Sex', axis=1, inplace=True)
df.drop('Title', axis=1, inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df.head()


# ### Test/Train separation and first training

# Let's split df into two new sets, one for the train and one for the validation. Why now, after a previous join of the same set? We previously concatenated them to be sure to have applied to both the **same transformation**.

# In[ ]:


train=df[df['Survived']>=0].copy()
test=df[df['Survived']==-1].copy()


# In[ ]:


test=test.drop('Survived',axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


y = train['Survived']
X = train.drop('Survived',axis=1)


# The **test set above defined** is the one to use to do the **submission to Kaggle**. We haven't a Survived values for this one (calculating it is our task!). We have, instead, the real Survived value for all the train set. So we'll split it into two further dataframes: 90% for the training and 10% for the validation. The Scikit-Learn package offers us an affordable function which we now import and use:

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)


# And now finally, we **import and call our classifier**, chosen in the family "Logistic Regression":

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lrm = LogisticRegression()


# In[ ]:


lrm.fit(X_train,y_train)


# In[ ]:


y_test_pred=lrm.predict(X_test)


# It's a classification problem, so to understand if our model performs well or not, an usefult tool is the **confusion matrix**: how many real Survived=1 are (wrongly) predicted as Survived=0 and so on?

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


C = confusion_matrix(y_test,y_test_pred)
print(C)
accuracy=(C[0,0]+C[1,1])/(C[0,0]+C[1,0]+C[0,1]+C[1,1])
print(accuracy)


# With this tool we reached accuracy=84% starting from the accuracy=78% of the gender submission.

# ### Final Fitting and Submission

# In[ ]:


lrm.fit(X,y)


# In[ ]:


y_pred=lrm.predict(test)


# In[ ]:


subm=pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
subm.head()


# In[ ]:


subm.to_csv('titanic_third.csv', index=False)


# The prediction is different int 35 records out of 418. In the leatherboard we reach the score of 77.5%. We improve of about 3000 positions. Other suggestions to improve further:
# * more sophisticated and/or optimtimizable classificators (such as Random Forest, Xgboost...)
# * better use of Ticket and Cabin columns
# * more intense featuring engineering
# * ...
