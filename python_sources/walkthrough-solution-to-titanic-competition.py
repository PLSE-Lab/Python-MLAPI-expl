#!/usr/bin/env python
# coding: utf-8

# # Titanic Disaster Survival: A Supervised Learning Classification Problem

# So, the old and traditional Titanic Competition once again... If you are not familiar with it no worries, I'll give you a quick explanation about it and mention the core concepts tackled in this notebook. 
# * This is a very popular Data Science challange with a dataset inspired on the sinking of RMS Titanic on April 14 1912. From the 2,224 passenger onboard, there are registers for 1,237 of them including a collection of personal and demographical information such as passenger's names, ticket class, fares and most importantly where the passanger has survived or not to this disaster. This last piece of information is only available for a fraction of the dataset though, 891 people more specifically. What happens to the rest of them is our job to predict!

# The proposal is to show a straight-forward approach for this Classification Problem, giving special attention to insight creation, explanation and further hypothesis discussions. Main concepts handled:
# 
# * Exploratory Analysis
# * Feature Engineering
# * Data Visualization
# * Predictive Model Training and Evaluation 
# 
# With no further ado, let's dig into it!

# ### Data Toolkit

# Let's start importing the necessary packages.

# In[ ]:


# Make sure to have the library versions below for interactive plotting
from IPython.display import clear_output
get_ipython().system('pip install cufflinks')
clear_output() # Clears out huge shell output from cufflinks installation...


# In[ ]:


# Data Wrangling
import os
import re
import sys
import itertools
import numpy as np
import pandas as pd
import random as rnd

# Visualizations - Regular plotting
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,6)
get_ipython().run_line_magic('matplotlib', 'inline')

# Visualizations - Interactive plotting
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=False)
import cufflinks as cf
cf.go_offline()

# Feature Engineering
import scipy.stats as st
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Model Evaluation
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Warning Handling
import warnings
warnings.filterwarnings(action='ignore')

print(f'Python environment: {sys.version}')


# ### Dataset

# Let's load our train and hold-out (test) datasets and combine them. <br>
# That's a good practice because we can preprocess the whole dataset at once.

# In[ ]:


# Load from .csv
train_df = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

# Extract survival information from train_df and combine the  rest
has_survided = train_df['Survived']
train_df.drop('Survived', axis=1, inplace=True)
df = pd.concat([test_df, train_df])

# Let's not forget to save train and test index for splitting it later
train_index, test_index = train_df.index, test_df.index 

# As we no longer need these dataframes, let's clear some memory
del train_df, test_df

# And them let's have a look at what we've got
df.head()


# ### Feature engineering

# Everything's looking good so far. Let's take a closer look to each variable, or feature...

# In[ ]:


df.info()


# It looks like there are some missing values... Age, Fare, Cabin and Embarked. We're gonna handle it in a while!

# In[ ]:


# Let's create some features that might be insightful and easier to grasp during our exploratory analysis
df['FamilySize'] = 1 + df.SibSp + df.Parch
df['NameLength'] = df.Name.apply(len)
df['TravelsAlone'] = df.FamilySize.apply(lambda x: 1 if x == 1 else 0)

# I noticed every name has a prefixed title, maybe that's predictive in some way... Let's extract it and find it out!
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
df.Title.value_counts(dropna=False).iplot('bar', title='Captured Titles from Names')


# Apparently there are too many titles with the same meaning, let's simplify things...

# In[ ]:


title_dict = {
    'Mrs': 'Mrs', 'Lady': 'Mrs', 'Countess': 'Mrs',
    'Jonkheer': 'Other', 'Col': 'Other', 'Rev': 'Other',
    'Miss': 'Miss', 'Mlle': 'Miss', 'Mme': 'Miss', 'Ms': 'Miss', 'Dona': 'Miss',
    'Mr': 'Mr', 'Dr': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Master': 'Mr'
}

df.Title = df.Title.map(title_dict)
print('Title count')
df.Title.value_counts(dropna=False)


# Much better now! Let's move back to the missing values now. But what could be good criteria for each feature? Let's have a look...

# In[ ]:


# Maybe de distribuition of age might help us
df[['Title', 'Age']].pivot(columns='Title', values='Age').iplot(kind='box', title='Age Distribution across Titles')


# Well indeed it looks like title is a good proxy for age, we are going to use it to fill the missing ages...

# In[ ]:


# What about Fares and Pclass
df[['Pclass', 'Fare']].pivot(columns='Pclass', values='Fare').iplot(kind='box', title='Fare Distribution across Ticket Classes')


# And Pclass stacks fare groups fairly well... So let's start filling it all up!

# In[ ]:


# Average Age per Title is the best we can get in a quick-fix
for title in df.Title.unique():
    df.loc[(df.Age.isnull())&(df.Title==title), 'Age'] = df.Age[df.Title==title].mean()

# Average Fare per Pclass as well... The boxplots were indeed insightful
for pclass in df.Pclass.unique():
    df.loc[(df.Fare.isnull())&(df.Pclass==pclass), 'Fare'] = df.Fare[df.Pclass==pclass].mean()

# Let's just take the most common value (mode) to fill Embarked... It was just one missing piece after all
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])


# In[ ]:


# Now, let's turn our categorical features into numerical ones so we can plug them into our Machine Learning Models
df['Sex'], int2class_sex           = pd.factorize(df.Sex)
df['Title'], int2class_title       = pd.factorize(df.Title)
df['Embarked'], int2class_embarked = pd.factorize(df.Embarked)

# To easily navigate from numeric to categorical variables, let's re-write those lists as dictionaries
int2class_sex      = {key: value for key, value in enumerate(int2class_sex)}
int2class_title    = {key: value for key, value in enumerate(int2class_title)}
int2class_embarked = {key: value for key, value in enumerate(int2class_embarked)}

# These features are no longer necessary, let's drop them
df.drop(['Ticket', 'Cabin', 'Name'], axis=1, inplace=True)

# Let's store the categorial and continuous features in lists is good practice, it might come in handy in the future
categorical_features = ["Pclass","Sex","TravelsAlone","Title", "Embarked"]
continuous_features = ['Fare','Age','NameLength']

# Let's check if we forgot to fill up any missing value
print('Missing value percentage')
(df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)


# Great! Our dataset is looking much tidier now... <br> 
# Now that all features were turned into numeric values, we can plot some histograms. <br>
# Tha's a good practice so we can get a feeling of the distributions for each feature. <br>
# It helps also to spot possible mistakes we might have made with so much data wrangling...

# In[ ]:


_ = pd.concat([df, has_survided], axis=1).hist()
plt.tight_layout(pad=1)


# Everything look right... Some intermidiate insights:
# * The majority of passengers were between 25 and 50 years old
# * 60% of them were travelling alone
# * 40/60 female-male ratio
# * 50% travelling third class
# * 80% Embarked at 1... Wait, what was that once again?

# In[ ]:


int2class_embarked


# Yeap, dictionaries are also awesome for quick peeks like this... So, 1 refers to 'S' which it's supposed to be a harbor or something.

# Moving on, a good practice with numerical features is to use StandardScaler, a built in feature fron scikit-learn. It standartizes all values from a given feature, so that its mean and standard deviation turn to 0 and 1, respectively.

# In[ ]:


# Feature distributions prior to StandardScaler
_ = df[continuous_features].hist()
plt.tight_layout(pad=1)


# In[ ]:


# Applying StandardScaler
for col in continuous_features:
    transf = df[col].values.reshape(-1,1)
    scaler = StandardScaler().fit(transf)
    df[col] = scaler.transform(transf)


# In[ ]:


# Feature distributions prior to StandardScaler
_ = df[continuous_features].hist()
plt.tight_layout(pad=1)


# Awesome, the distribuition silhouette hasn't changed, only the mean and standard deviation values... Normalization properly accomplished!

# Prior to moving to our Machine Learning Models, let's have a loot at how the features correlate with each other... Seaborn and Pandas have great implementations for that.

# In[ ]:


_, ax = plt.subplots(figsize=(12,8))
_ = sns.heatmap(pd.concat([df, has_survided], axis=1).corr(), annot=True, fmt=".1f", cbar_kws={'label': 'Percentage %'}, cmap="coolwarm", ax=ax)
_ = ax.set_title("Feature Correlation Matrix")

# Try out the interactive plot as alternative below! 
# pd.concat([df, has_survided], axis=1).corr().iplot(kind='heatmap', colorscale="RdBu", title="Feature Correlation Matrix") 


# Here are some insights:
# * Title and Sex are strongly correlated as expected. After all, they are linearly correlated by definition.
# * Same thing goes for FamiliSize and SibSp, Parch and Travels alone. 
# * Our target feature, Survived, shows stronger correlation with Sex, Title Fare and strangely NameLength. Let's dig in deeper...

# ### Predictive Modelling

# Now things start getting fun... Let's train some Machine Learning models to make predictions for us! <br>
# Let's start splitting our dataset into test and train. Then, declaring our target (y) and independent (X) variables.

# In[ ]:


train_df = df.loc[train_index, :]
train_df['Survived'] = has_survided
test_df = df.loc[test_index, :]

del df


# In[ ]:


X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']


# Let's have a look at how our target variable is distributed

# In[ ]:


print("Target Variable Distribution - Survived")
print(y.value_counts(normalize=True))


# It's slightly imbalanced, which may lead to biased classification towards the majoritary class... There are some techniques to handle this issue such as data augmentation, resampling, or stratifing train and test sets. For the purposes of this notebook, such techniques will be left as future work.<br><br>
# Now, let's split the dataset into train and test for training...

# In[ ]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In order to facilite performance comparison among models, let's wrap it into a method. It basically fits the model to the data and return a dictionary containing classification report statistics, such as accuracy and f1-score.

# In[ ]:


def eval_model(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    output_dict =  classification_report(y_test, pred, output_dict=True)
    fpr, tpr, _ = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
    output_dict['auc'] = skplt.metrics.auc(fpr, tpr)
    output_dict['classifier'] = model
    return output_dict


# Let's apply it in three different classifiers for demonstration purposes. In a real project I would recommend picking one of these and work on hyperparameter tunning which usually yields better results than trial and error with a bunch of poorly parametrized classifiers.<br><br>
# I chose the following algorithms:
# * K Nearest Neighbors (KNN)
# * Logistic Regression
# * Random Forest

# In[ ]:


# I usually store every model and corresponding metric in dictionaries for quick later access
models = {}
models['KNN']                = eval_model(KNeighborsClassifier())
models['LogisticRegression'] = eval_model(LogisticRegression())
models['RandomForest']       = eval_model(RandomForestClassifier())

models = pd.DataFrame.from_dict(models)
for metric in ['precision', 'recall', 'f1-score']:
    models = models.append(
        models.loc['macro avg'].apply(
            lambda x: dict(x)[metric]).rename(f'{metric} avg'))


# ### Model Evaluation

# Let's visualize it so it's easier to grasp which model has the best performance...

# In[ ]:


models.loc[['accuracy', 'auc', 'precision avg', 'recall avg', 'f1-score avg']].T.sort_values(by='accuracy', ascending=False).iplot('bar', title='Model Performance Comparison - Four Metrics', yrange=[.7,.85])


# So far the LogistRegression excels in all metrics, being the accuracy a good proxy in this case... Let's take a closer look to some other metrics.

# In[ ]:


model = models['LogisticRegression']['classifier']
skplt.metrics.plot_confusion_matrix(y_test, model.predict(X_test), title='Logistic Regression Confusion Matrix')


# Analysing the Confusion Matrix we can notice the false negatives are proporcionally higher than the rest - see the 29% missclassified as negative at bottom left quadrant. That's a issue we can leave for improvement on next steps.<br>
# Now let's have a look at the ROC Curve.

# In[ ]:


model = models['LogisticRegression']['classifier']
skplt.metrics.plot_roc_curve(y_test, model.predict_proba(X_test))


# A core concept of the ROC Curve is that the diagonal line represents random guessing and the ideal model (which would always predict correctly) being at True and False Positive Rates axis. We can measure this objectively via AUC (Area Under Curve) which represents here 85%, not bad for starting...

# Finally let's have a look at feature importance. It means which features have the largest impact on the models predictions.

# In[ ]:


model = models['RandomForest']['classifier']
feats = {feature: importance for feature, importance in zip(X.columns, model.feature_importances_)}
pd.DataFrame.from_dict(feats,orient='index', columns=['importance']).sort_values('importance').iplot('barh', title='Feature Importances - RandomForest')


# Here we can se that NameLength, Fare, Age and Sex were the most important features to predict wether a person would survive or not to the Titatic sinking. That's a great way to understand why your model performs like it does, and thus find room for improvement. NameLength is for me an unexpected feature to be on the top of this list. That investigation is work for next steps!

# ### Conclusions

# We were able to tackle just a tiny fraction of the Data Science pipeline with this Classification Challenge, but nevertheless it was a lot! Through Exploratory analysis, Feature Engineering, Model Training and Performance Evaluation we put into practice some of the core concepts of Machine Learning and Data Science in general. There's still much to do though, like Hyperparameter Tunning via  Grid Search Cross Validation for instance, that's an assisted method to check several model presets and pick the best performer from them. Or check for better performing algorithms such as Neural Netwokrs of Support Vectors. <br><br>
# The methods and techniques applied here are far from optimal, but put into perspective how to handle a classification problem. Specially due to a still high False Negative Rate such a model would not yet be ready for production as many improvements listed above are recommended.
