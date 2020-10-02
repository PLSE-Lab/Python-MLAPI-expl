#!/usr/bin/env python
# coding: utf-8

# # Titanic - EDA, Logistic Regression, Random Forest
# 
# We'll be approaching the Titanic survival classification competition using two well-known modeling techniques and look at the pros and cons of choosing either one of the models over the other.
# 
# # 1 - Introduction
# 
# The titanic classification competition is the starter competition here in Kaggle for new members. I've chosen Logistic Regression as my primary choice for the model. Afterwards, I'll test out Random Forest as well and check the difference in results between the two models.
# 
# ## 1.1 - Why Logistic Regression
# Logistic Regression is a popular traditional machine learning algorithm. Since our model predicts whether the passenger survives or not (binary classifier), then a logistic regression model would be a perfect approach for this problem as it is effective and also very simple. Training time is also very fast!
# 
# Another reason as to why Logistic Regression was chosen is because its loss function is convex, meaning that finding the global minimum is certain as long as your learning rate is not too high.
# 
# #### Note: Logistic Regression is usable as well for multi-class problems by doing Softmax Regression.
# 
# ## 1.2 - Why Random Forest
# Random Forest is another popular algorithm for creating powerful classifiers by utilizing an ensemble of decision trees. Although it is very powerful, it is not my primary choice as it trades off interpretability for the sake of accuracy. Another sacrifice that would be made here as well is computational power. Random Forest models take a longer time to train, hyperparameter tuning can become a problem because of this.
# 
# # 2 - Exploratory Data Analysis
# 
# We'll first do an exploratory data analysis so we can get a feel on how the data looks like and have an idea as to what features to select. The primary steps that we'll have would be:
# 
# - Get a feel of the dataset by examining the variables
# - Data Manipulation and Feature Selection
# 
# ## 2.1 - Get a feel of the dataset
# 
# Let's first import the dependencies that we'll be using then load the data.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
train


# Let's check how many missing values there are for each column.

# In[ ]:


train.isnull().sum()


# There's plenty of missing values for the Age column and Cabin column. We'll check later if it's worth imputing values into those columns.
# Let's then check the relationship between the categorical variables and the dependent variable.
# 
# Let's check the Sex variable which indicates the passenger's sex.

# In[ ]:


from scipy import stats

survived = train[train['Survived']==1]
did_not_survive = train[train['Survived']==0]

# libraries
import numpy as np
import matplotlib.pyplot as plt
 
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams.update({'font.size': 13})
# set width of bar
barWidth = 0.25


male = [len(survived[survived['Sex']=='male']),         len(did_not_survive[did_not_survive['Sex']=='male'])]
female = [len(survived[survived['Sex']=='female']),          len(did_not_survive[did_not_survive['Sex']=='female'])]

# set height of bar
bars1 = [len(survived[survived['Sex']=='male']),          len(did_not_survive[did_not_survive['Sex']=='male'])]
bars2 = [len(survived[survived['Sex']=='female']),          len(did_not_survive[did_not_survive['Sex']=='female'])]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survived')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Did not survive')
 
# Add xticks on the middle of the group bars
plt.xlabel('Sex', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Male', 'Female'])
 
# Create legend & Show graphic
plt.legend()
plt.show()

p_value = stats.chi2_contingency([male, female])[1]
print("Chi-Square P-Value: " + str(p_value))


# Based on the charts above, we can see that there's a noticeable difference between survival depending on the sex of the passenger. We can observe that females were more likely to survive than males. This is backed up by the Chi-Square's P-value, which indicates that we reject the Null Hypothesis that knowing Sex does not help in predicting Survival.
# 
# Let's then check the class variable which indicates the seat class of a passenger. (Class 1 being the best seat)

# In[ ]:



# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25

class1 = [len(survived[survived['Pclass']==1]),         len(survived[survived['Pclass']==2]),         len(survived[survived['Pclass']==3]),]
class2 = [len(did_not_survive[did_not_survive['Pclass']==1]),         len(did_not_survive[did_not_survive['Pclass']==2]),         len(did_not_survive[did_not_survive['Pclass']==3])]
 
# Set position of bar on X axis
r1 = np.arange(len(class1))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, class1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survived')
plt.bar(r2, class2, color='#557f2d', width=barWidth, edgecolor='white', label='Did not survive')
 
# Add xticks on the middle of the group bars
plt.xlabel('Class', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(class1))], ['Class 1', 'Class 2', 'Class 3'])
 
# Create legend & Show graphic
plt.legend()
plt.show()


class1 = [len(survived[survived['Pclass']==1]),         len(did_not_survive[did_not_survive['Pclass']==1])]
class2 = [len(survived[survived['Pclass']==2]),         len(did_not_survive[did_not_survive['Pclass']==2])]
class3 = [len(survived[survived['Pclass']==3]),         len(did_not_survive[did_not_survive['Pclass']==3])]

p_value = stats.chi2_contingency([class1, class2, class3])[1]
print("Chi-Square P-Value: " + str(p_value))


# We can see that passengers who belonged to Class 3 were less likely to survive than Class 1 and 2 passengers. Class 1 passengers had the highest chance of survival as compared to the other two classes. This may indicate that Class 1 passengers were given more priority in rescue operations than Class 3 passengers. The Chi-square P-value also tells us that the Null Hypothesis(knowing the class variable does not help in predicting survival) is rejected.
# 
# Let's now take a look at the Embarked variable which tells us where the port of embarkation is for the passengers.

# In[ ]:



# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25

class1 = [len(survived[survived['Embarked']=='Q']),         len(survived[survived['Embarked']=='C']),         len(survived[survived['Embarked']=='S']),]
class2 = [len(did_not_survive[did_not_survive['Embarked']=='Q']),         len(did_not_survive[did_not_survive['Embarked']=='C']),         len(did_not_survive[did_not_survive['Embarked']=='S'])]
 
# Set position of bar on X axis
r1 = np.arange(len(class1))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, class1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Survived')
plt.bar(r2, class2, color='#557f2d', width=barWidth, edgecolor='white', label='Did not survive')
 
# Add xticks on the middle of the group bars
plt.xlabel('Embarked', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(class1))], ['Queenstown', 'Cherbourg', 'Southampton'])
 
# Create legend & Show graphic
plt.legend()
plt.show()

Q_embark = [len(survived[survived['Embarked']=='Q']),         len(did_not_survive[did_not_survive['Embarked']=='Q'])]
C_embark = [len(survived[survived['Embarked']=='C']),         len(did_not_survive[did_not_survive['Embarked']=='C'])]
S_embark = [len(survived[survived['Embarked']=='S']),         len(did_not_survive[did_not_survive['Embarked']=='S'])]

p_value = stats.chi2_contingency([Q_embark, C_embark, S_embark])[1]
print("Chi-Square P-Value: " + str(p_value))


# Embarked seems to also tell us something interesting. Passengers that have embarked from Southampton had a noticeable difference in survival rate which leans more towards not surviving. Passengers coming from Cherbourg were a bit more likely to survive and passengers embarking from Queenstown is less likely to survive. Doing another Chi-square test tells us to reject the Null Hypothesis as well.
# 
# Let's now take a look at the numerical variables by using a barplot.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import random

plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams.update({'font.size': 13})

data = {}
data['Age'] = {
    'Survived': list(survived.dropna()['Age']),
    'Did not Survive': list(did_not_survive.dropna()['Age']),
}
data['SibSp'] = {
    'Survived': list(survived.dropna()['SibSp']),
    'Did not Survive': list(did_not_survive.dropna()['SibSp']),
}
data['Parch'] = {
    'Survived': list(survived.dropna()['Parch']),
    'Did not Survive': list(did_not_survive.dropna()['Parch']),
}
data['Fare'] = {
    'Survived': list(survived.dropna()['Fare']),
    'Did not Survive': list(did_not_survive.dropna()['Fare']),
}

fig, axes = plt.subplots(ncols=4)
fig.subplots_adjust(wspace=0)

for ax, name in zip(axes, ['Age', 'SibSp', 'Parch', 'Fare']):
    ax.boxplot([data[name][item] for item in ['Survived', 'Did not Survive']])
    ax.set(xticklabels=['Survived', 'Did not Survive'], xlabel=name)
    ax.margins(0.05) # Optional

plt.show()


# The median age of the passengers have a bit of a difference, with the median age being higher for passengers who did not survive. SibSp and Parch does not have a noticeable difference. Fare for the passengers that did not survive has a lower median as compared to passengers that survived. Passengers that survived had noticeably more outliers that passenger that did not survive.

# ## 2.2 - Data Manipulation and Feature Selection
# 
# Now that we've taken a look at the data, let's now start to manipulate the data for modeling and choose the features from our dataset. First, let's one-hot our categorical variables.

# In[ ]:


train = pd.get_dummies(train, columns=['Sex', 'Pclass', 'Embarked'])


# Let's take a look at the correlation table to check what variables correlate to "Survived".

# In[ ]:


train.corr()


# The variables that noticeably correlated with Survived were Fare, Sex, Pclass and Embarked. Knowing that, we'll choose those variables as our predictors for our model.
# 
# Since one of our predictors is Embarked, and we have observed earlier that there were 2 missing values for that. Let's create a simple imputer that will impute the missing values.

# In[ ]:


from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

X = train[['Fare', 'Sex_male', 'Pclass_1', 'Pclass_3', 'Embarked_C', 'Embarked_S']]
y = train['Survived']
X = my_imputer.fit_transform(X)


# # 3 - Modeling
# Let's start the modeling process. We'll go for the primary model of choice, the Logistic Regression model.
# 
# ## 3.1 - Logistic Regression
# Logistic Regression is a common choice for binary classification. It is also much more simple and interpretable when compared to most models. First, we'll split our dataset into test and train. Then we'll add a standard scaler to standardize our data so our model would converge faster. We'll then use grid search to find the optimal hyperparameters for our model.

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

clf = LogisticRegression(solver='liblinear')

# Create regularization hyperparameter space
C = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 10, 100]
penalty = ['l1','l2']

hyperparameters = dict(C=C, penalty=penalty)

grid_clf_acc = GridSearchCV(clf, param_grid=hyperparameters, cv=5, verbose=0)
grid_clf_acc.fit(X_train, y_train)


# Let's take a look at the grid search results.

# In[ ]:


grid_clf_acc.best_params_


# Let's create the model using our grid search results as the hyperparameters. We'll then fit the model with the training data and test it out on the test set.

# In[ ]:


clf = LogisticRegression(solver='liblinear', **grid_clf_acc.best_params_)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

clf.fit(X_train, y_train)

X_test = scaler.transform(X_test)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))


# We achieved an accuracy of ~79% using the Logistic Regression model. It's satisfactory. However, accuracy isn't always the best way to assess the model's performence. Let's look at the confusion matrix.

# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Survived', 'Did not survive']))


# We can see that the model performs better when predicting passengers that Survived than passengers that didn't survive. Let's train the Logistic Regression model using the entirety of our data.

# In[ ]:


lr = clf.fit(X, y)


# ## 3.2 - Random Forest
# This time we'll be using training a Random Forest model with out dataset. A random forest model utilizes an ensemble of trees to create a prediction. Since we're using a tree-based algorithm, we won't have to standardize our data. This model is well-known due to its power, but we are trading interpretability for it (when compared to our Logistic Regression model). Let's see if this model performs better than our Logistic Regression model.
# 
# Let's repeat the same steps that we've done previously.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

clf = RandomForestClassifier()

hyperparameters = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, 110],
    'n_estimators': [100, 200, 300, 1000]
}

grid_clf_acc = GridSearchCV(clf, param_grid=hyperparameters, cv=5, verbose=0)
grid_clf_acc.fit(X_train, y_train)


# In[ ]:


clf = RandomForestClassifier(**grid_clf_acc.best_params_)


# In[ ]:


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))


# We've achieved roughly ~81% accuracy using our random forest model, which is a 2% bump from our previous Logistic Regression model. Let's check the confusion matrix as well.

# In[ ]:


print(classification_report(y_test, y_pred, target_names=['Survived', 'Did not survive']))


# We can see that the model actually does better for both classes when we compare it to our Logistic Regression model from before. It still does better when it predicts the class of passengers that survived.
# 
# 
# ## 3.3 - So which one's better?
# So in the end, can we conclude that a random forest model is much better than a logistic regression model? Well, that depends on what you mean by better. Sometimes when resources are constrained and less computational power is better, the Logistic Regression model would win on that scenario. A Logistic Regression model is much simpler and is easily understandable by many practitioners. However, its simplicity comes with the sacrifice of a bit of accuracy. If that extra increase in predictive power is of utmost importance, like in competitions. Then our random forest model would be the better choice. So in the end, it depends on the scenario.
# 
# Let's train our random forest model as well on the entire dataset.

# In[ ]:


rf = clf.fit(X, y)


# ## 3.3 - Making the Submission
# We'll be using our random forest model for submitting our results. Since this is a competition, we'd be choosing the more powerful model that performed better despite the sacrifice in computational power and in simplicity.

# In[ ]:


test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


test = pd.get_dummies(test, columns=['Sex', 'Pclass', 'Embarked'])
X = test[['Fare', 'Sex_male', 'Pclass_1', 'Pclass_3', 'Embarked_C', 'Embarked_S']]
X = my_imputer.fit_transform(X)


# In[ ]:


pred = rf.predict(X)


# In[ ]:


submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': pred})


# In[ ]:


submission.to_csv('/kaggle/working/titanic',index=False)


# # 4 - Last Remarks
# Thanks for allowing me to present this work to you guys! If you liked the notebook, please give it an upvote!
