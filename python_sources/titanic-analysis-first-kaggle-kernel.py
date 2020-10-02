#!/usr/bin/env python
# coding: utf-8

# New to data science, just finished several online courses, and want to put knowledge to practice.
# I know that I'm using a very simplistic approach to dealing with null values, but just want to keep it simple for the first run through.

# In[ ]:


# Import packages
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

# Open dataset
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
combined_data = [train_data, test_data]


# In[ ]:


# Look at first rows in training set
train_data.head()


# In[ ]:


# Look at first rows in test set
test_data.head()


# In[ ]:


# Get summary data for training set
train_data.describe()


# There are 891 rows in the training dataset.

# In[ ]:


# Now look at the test set
test_data.describe()


# There are 418 rows in the test dataset.

# In[ ]:


# Look at how many null values there are for the different columns
train_data.info()


# The Cabin columns has more nulls than values, so I shouldn't rely on this for data analysis.  I will drop this column.

# In[ ]:


train_data = train_data.drop(['Cabin'], axis='columns')
test_data  = test_data.drop(['Cabin'], axis='columns')

train_data.info()
print()
test_data.info()


# In[ ]:


# For simplicity, I will set the variabes with null values to the mean and 
# median of the non-null values.
train_data.Age  = train_data.Age.fillna(train_data.Age.median() )
train_data.Fare = train_data.Fare.fillna(train_data.Fare.mean() )

test_data.Age  = test_data.Age.fillna(test_data.Age.median() )
test_data.Fare = test_data.Fare.fillna(test_data.Fare.mean() )


# In[ ]:


train_data.info()
print()
test_data.info()


# In[ ]:


# There are still 2 null values for the column Embarked.  Since we can't apply a mean/median 
# for this value, we will put in the most used value in the dataset
train_data.Embarked.value_counts()


# In[ ]:


# I will set Embarked = "S" where it is null
train_data.Embarked  = train_data.Embarked.fillna("S")


# The datasets are now without any null values.

# To practice what I have learned in courses, I used Gretl to determine which independent variables were more likely to have an influence on the dependent variable (Survival).
# 
# I used the Backward eliminination method with multiple logistic regression.  
# The model iterations showed the following:
# - Model 1 - p-value was highest for Parch, so removed it and reran model
# - Model 2 - p-value was highest for Embarked, so removed it and reran model
# - Model 3 - p-value was highest for Fare, so removed it and reran model  
# - Model 4 - p-value was highest for Ticket, so removed it and reran model
# - Model 5 - all remaining independent variables are showing as significant with this model: 
# Pclass, Sex, Age, SibSp
# Model 5 shows an accuracy percentage of 80.8%.
# 
# Will continue on here with Python, using these remaining 4 variables.

# Look at distribution of male versus female, and how many survived.

# In[ ]:


# Use Seaborn category plot 
gender_plot = sns.catplot(x="Sex", col="Survived", data=train_data, kind="count", 
              height=4, aspect=.75)
(gender_plot.set_axis_labels("", "Count")
            .set_xticklabels(["Men", "Women"]) 
            .set_titles("{col_name} {col_var}") )


# This shows that females were more likely to survive.

# In[ ]:


# Look at distribution of survivors by Pclass and gender
class_plot = sns.catplot(x="Pclass", col="Survived", hue="Sex", data=train_data,  
             kind="count", height=4, aspect=.75)


# This shows again that females were more likely to survive, and that Pclass is also a factor in survival.

# In[ ]:


# Look at distribution of survivors by Age, classified into 10 groups
age_plot = plt.hist(train_data[train_data.Survived == 1].Age, bins=10)


# This shows that the majority of the surviving passengers are in the age bracket 20-40.

# In[ ]:


# Look at distribution of survivors by number of siblings or spouse
plcass_plot = plt.hist(train_data[train_data.Survived == 1].SibSp)


# This shows that passengers with no sibling/spouse, or just one sibling/spouse were more likely to survive.

# All of the plots above confirm possible linearity from the four independent variables, as per the analysis done in Gretl.

# In[ ]:


# Convert the gender to 0 (male) and (female)
train_data.Sex = [0 if i=="male" else 1 for i in train_data.Sex]
test_data.Sex  = [0 if i=="male" else 1 for i in test_data.Sex]

# Drop all of the columns that were not statistically significant
# (Cabin has already been removed)
new_train_x = train_data.drop(["PassengerId", "Survived", "Name", "Parch", "Ticket", 
                               "Fare", "Embarked"], axis=1)

new_train_y = train_data["Survived"]

# Drop the variables from the test data as well
new_test_x =  test_data.drop(["PassengerId", "Name", "Parch", "Ticket", 
                               "Fare", "Embarked"], axis=1)


# Now I will split up the training file into train versus test, to evaluate different models.

# In[ ]:


# X is the training set, y is the prediction 
y = new_train_y 
X = new_train_x

# Split the training set in two, with 20% going to test set
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = .20, random_state = 0)


# In[ ]:


# Try Logistic Regression model
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(train_x, train_y)
print("Accuracy - Logistic Regression - train:", round(reg.score(train_x, train_y), 3),
      "test:", round(reg.score(test_x, test_y), 3) )


# The accuracy of this model is 79.9% .

# In[ ]:


# Try gradient boosting model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

GBC = GradientBoostingClassifier()
GBC.fit(train_x, train_y)

# Predicting the test set results
pred_y = GBC.predict(test_x)

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, pred_y)

print ("Accuracy - GradientBoosting - train:", round(GBC.score(train_x , train_y), 3),
       "test:", round(GBC.score(test_x , test_y), 3) )


# The accuracy of this model is 82.7%, which is better.

# In[ ]:


# Try other algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Random Forest
RFC = RandomForestClassifier(n_estimators=100)
RFC.fit(train_x, train_y)
print ("Accuracy - RFC - train:", round(RFC.score(train_x , train_y), 3),
       "test:", round(RFC.score(test_x , test_y), 3) )


# In[ ]:


# Support Vector Classification
svc = SVC()
svc.fit(train_x, train_y)
print ("Accuracy - SVC - train:", round(svc.score(train_x , train_y), 3),
       "test:", round(svc.score(test_x , test_y), 3) )


# In[ ]:


# k-nearest neighbors
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(train_x, train_y)
print ("Accuracy - KNN - train:", round(knn.score(train_x , train_y), 3),
       "test:", round(knn.score(test_x , test_y), 3) )


# In[ ]:





# Of the 5 models that I tried, GradientBoosting had the highest accuracy rating with 82.7%.  I will submit the results from this model.

# In[ ]:


# Create the submission file, with PassengerId and Survival prediction
psgr_id = test_data["PassengerId"]

# Rerun the model on the entire training set
GBC = GradientBoostingClassifier()
GBC.fit(new_train_x, new_train_y)

# Apply the algorithm to the test data file
prediction = GBC.predict(new_test_x)

# Save the results to a csv file
submission = pd.DataFrame( {"PassengerId" : psgr_id, "Survived": prediction} )
submission.to_csv("submission.csv", index=False)

