#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Load the trainig data csv file and make the data frame out of it
train_df = pd.read_csv('../input/train.csv')
#Load the test data csv file and make the data frame out of it
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


#display the first five rows of train_df
train_df.head()


# In[ ]:


print("The train data has {} rows and {} columns".format(train_df.shape[0],train_df.shape[1]))


# In[ ]:


#display the first five rows of test_df
test_df.head()


# In[ ]:


print("The test data has {} rows and {} columns".format(test_df.shape[0],test_df.shape[1]))


# As we can see that in test dataframe only 11 columns are there.(Survival column is missing). so our taget column is survival and we are going to predict the survival via Logistic Regression.

# In[ ]:


#check column wise null and missing values
train_df.apply(lambda x: sum(x.isnull()))


# from above we can see that Age,Cabin and Embarked column has null values.

# In[ ]:


#display information of train dataframe
train_df.info()


# In[ ]:


#display 5 number summary of train dataframe
train_df.describe()


# In[ ]:


print("Percent of missing Age records is {}%".format((177/891)*100))


# so from above we can see that only 19.86% Age records are missing.

# In[ ]:


#display distribution of age column
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# from above we can see that Age is right skewed because long tail is at the right side(i.e.mean>median). then replacing null values with mean is not good. we will fill null values with median.

# In[ ]:


print("Percent of missing Cabin records is {}%".format((687/891)*100))


# so from above we can see that 77% Cabin records are missing.so imputing values and using this column is not useful. so, we will drop this column.

# In[ ]:


print("Percent of missing Embarked records is {}%".format((2/891)*100))


# so from above we can see that only 0.22%(2) Embarked records are missing. so we can just impute with most occurring value in this column.

# In[ ]:


#from data set we know C==Cherbourg,Q=Queens,S=Southampton
print("Boarded passengers are grouped by port of Embark(C==Cherbourg,Q=Queens,S=Southampton)")
print(train_df['Embarked'].value_counts())


# In[ ]:


sns.countplot(train_df['Embarked'])


# so from above we can see most people embarked from Southampton port. so we can fill null values of Embarked Column with S (Southampton)

# Based on my assessment of the missing values in the dataset, I'll make the following changes to the data:
# 
# If "Age" is missing for a given row, I'll impute with 28 (median age).
# If "Embarked" is missing for a riven row, I'll impute with "S" (the most common boarding port).
# I'll ignore "Cabin" as a variable. There are too many missing values for imputation. Based on the information available, it appears that this value is associated with the passenger's class and fare paid.

# Make another dataframe from train dataframe and make above changes to new dataframe.

# In[ ]:


train_data = train_df.copy()


# Now we have new data frame called train_data. we will make changes to this dataframe.

# In[ ]:


train_data['Age'].fillna(train_df['Age'].median(),inplace=True)
train_data['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(),inplace=True)
train_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


#check now null values are there in new dataframe
train_data.apply(lambda x: sum(x.isnull()))


# so from above we can see that now no null values are there.

# In[ ]:


#comaprison of Age distibution before and after adjustment
plt.figure(figsize=(15,10))
ax = train_df['Age'].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
train_df['Age'].plot(kind='density', color='teal')
ax = train_data['Age'].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)
train_data['Age'].plot(kind='density', color='orange')
ax.legend(['Age Before Adjustment','Age After Adjustment'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# So from above graph we can compare Age distribution before and after adjustment.After adjustment Age looks like normally distributed.

# According to the dataset, both SibSp and Parch relate to traveling with family. For simplicity's sake (and to account for possible multicollinearity), I'll combine the effect of these variables into one categorical predictor: whether or not that individual was traveling alone

# In[ ]:


train_data['TravelAlone'] = np.where((train_data['SibSp']+train_data['Parch'])>0, 0, 1)
train_data.drop('SibSp',axis=1,inplace=True)
train_data.drop('Parch',axis=1,inplace=True)


# I'll also create categorical variables for Passenger Class ("Pclass"), Gender ("Sex"), and Port Embarked ("Embarked").

# In[ ]:


training = pd.get_dummies(train_data,columns=["Pclass","Embarked","Sex"])


# In[ ]:


training.drop('Sex_female', axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop('Ticket', axis=1, inplace=True)
final_train = training
final_train.head()


# now check the null values in test dataframe

# In[ ]:


test_df.apply(lambda x : sum(x.isnull()))


# so from above we can see that Age,Fare and Cabin column are having missing values.

# In[ ]:


#display the 5 number summary of test dataframe
test_df.describe()


# So, we will apply the same changes to test dataframe what we apply to train dataframe.

# In[ ]:


test_data = test_df.copy()
test_data['Age'].fillna(train_df['Age'].median(),inplace=True)
test_data['Fare'].fillna(train_df['Fare'].median(),inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)


# In[ ]:


#now check null values are there in test dataframe
test_data.apply(lambda x : sum(x.isnull()))


# In[ ]:


test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)
test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)


# In[ ]:


testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)


# In[ ]:


final_test = testing
final_test.head()


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.kdeplot(final_train['Age'][final_train['Survived']==1],color='Teal',shade=True)
sns.kdeplot(final_train['Age'][final_train['Survived']==0],color='lightcoral',shade=True)
plt.legend(['Survived','Died'])
plt.title("Density plot of Age for surviving people and died people")
ax.set(xlabel='Age')
plt.xlim(-10,85)


# The age distribution for survivors and died people is actually very similar. One notable difference is that, of the survivors, a larger proportion were children.means belowe 16 year age people are survived more than died

# In[ ]:


final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)

final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)


# In[ ]:


plt.figure(figsize=(15,10))
avg_survival_by_age = final_train[['Age','Survived']].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age',y='Survived',data=avg_survival_by_age)


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.kdeplot(final_train['Fare'][final_train['Survived']==1],color='Teal',shade=True)
sns.kdeplot(final_train['Fare'][final_train['Survived']==0],color='lightcoral',shade=True)
plt.legend(['Survived','Died'])
plt.title("Density plot of Fare for surviving people and died people")
ax.set(xlabel='Fare')
plt.xlim(-20,200)


# As the distributions are clearly different for the fares of survivors vs. deceased, it's likely that this would be a significant predictor in our final model. Passengers who paid lower fare appear to have been less likely to survive. 

# In[ ]:


sns.barplot('Pclass','Survived',data=train_df)


# from above it looks like first class passengers has highest survival rate.

# In[ ]:


sns.barplot('Embarked','Survived',data=train_df)


# people who embarked from Cherbourg port appear to have highest survival rate.

# In[ ]:


sns.barplot('TravelAlone','Survived',data=final_train)


# people who are travelling alone has less survival than who are travelling with family.

# In[ ]:


sns.barplot('Sex','Survived',data=train_df)


# females survival is more than males

# In[ ]:


final_train.columns


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(final_train.corr(),annot=True)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
X = final_train.drop('Survived',axis=1)
y = final_train['Survived']


# In[ ]:


# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the scores change a lot, this is why testing scores is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[ ]:


# check classification scores of logistic regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))


# In[ ]:


final_test.info()


# In[ ]:


test_pred = logreg.predict(final_test)


# In[ ]:


test_pred.shape

