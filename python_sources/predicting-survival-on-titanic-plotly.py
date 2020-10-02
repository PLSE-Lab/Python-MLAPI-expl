#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 1. Introduction
# 2. Useful Functions
# 3. Loading Modules & Datasets
# 4. Dataset Fundamentals
# 5. Data Exploration
# >5.1. Pclass<br/>
# >5.2. Sex<br/>
# >5.3. Age<br/>
# >5.4 Fare<br/>
# >5.5 Embarkation Point<br/>
# >5.6 Heatmap<br/>
# 6. Feature Engineering
# >6.1. SibSp & Parch<br/>
# >6.2. Name<br/>
# >6.3. Cabin<br/>
# >6.4 Sex<br/>
# >6.5 Age<br/>
# >6.6 Embarkation Point<br/>
# >6.7 Fare<br/>
# 7. Machine Learning
# >7.1 Final Datasets<br/>
# >7.2 Setting Parameters & Fitting Models<br/>
# >7.3 Feature Importances<br/>
# >7.4 Correlation Between Models<br/>
# >7.5 Parameter Tuning Using GridSearch<br/>
# >7.6 Ensemble Model<br/>
# 8. Submission File

# # 1. Introduction
# This notebook aims to build a binary classification model that predicts survival for the Titanic dataset. The primary purpose of this kernel is to improve my knowledge of data visualization techniques, feature engineering, and classification algorithms. A description of the dataset can be found [here](https://www.kaggle.com/c/titanic).
# 
# I hope that my work in this notebook can help other beginners in data science develop their own skills in the aformentioned areas. So, I've included comments and my observations as much as possible within the notebook. 
# 
# That being said, my approach is by no means perfect - hence, any recommendations and suggestions for improvement is always recommended. 

# # 2. Useful Functions

# This section includes some functions that help me further on in the notebook.

# In[ ]:


# Flattens stacked grouped columns
def flatten(dataframe):
    dataframe.columns = [' '.join(col).strip() for col in dataframe.columns.values]
    return dataframe

# Object to extend the functionality of the ML models
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        self.__name__ = clf.__name__

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def cv_score(self,x_train,y_train):
        array = cross_val_score(self.clf,x_train,y_train, cv = 5,scoring = 'accuracy')
        return array.mean()
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_


# # 3. Loading Modules & Datasets

# This section loads all necessary modules and raw datasets.

# In[ ]:


# Importing datatable modules
import numpy as np
import pandas as pd

# Importing Graphing Modules
import plotly_express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# Importing ML models/metrics
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Importing other useful libraries
import os
from collections import defaultdict

# Printing list of files in input folder
print(os.listdir("../input"))

# Initializing plotly offline
init_notebook_mode(connected=True)


# In[ ]:


# Loading datasets 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
full_dataset = [train,test]


# # 4. Dataset Fundamentals

# This section explores the basic structure, features, and limitations of our dataset.

# In[ ]:


# Printing dataset information
print("Training Dataset Information:")
print(train.info())
print("\nTest Dataset Information:")
print(test.info())


# In[ ]:


# Characterizing null values
print("Training Null Values (%):")
print(train.isnull().sum()*100/train.shape[0])
print("\n")
print("Test Null Values (%):")
print(test.isnull().sum()*100/test.shape[0])


# ### Observations:
# - There's a large proportion of missing age and cabin data. We will need to figure out a robust method to impute these values
# - Fares and embarked have very few missing values; we could possibly replace them with medians/modes

# # 5. Data Exploration

# In this section, we will explore a few numerical/discrete features of the data (Pclass, Sex, Age, Fare & Embarkation Point) through graphical representations using [Plotly Express/Plotly](https://plot.ly/python/).

# ## 5.1 Pclass

# In[ ]:


# Grouping by survival rate and standard deviation and plotting results
pclass_grouped = train[['Pclass','Survived']].groupby('Pclass', as_index=False).agg({'Survived':['mean','std']})
pclass_grouped = flatten(pclass_grouped)
fig = px.bar(pclass_grouped,x = "Pclass", y = "Survived mean", color = "Pclass", error_y = "Survived std")
fig.update_traces(error_y_color = "black")


# ### Observations:
# - Lower class passengers seem to have had a lower chance of survival on average. This is probably because lower class passengers were given decks at lower levels in the ship or rooms with poor evacuation routes.

# ## 5.2 Sex

# In[ ]:


# Looking at survival by gender
sex_grouped = train[['Sex','Survived']].groupby('Sex',as_index=False).agg({'Survived':['mean','std']})
sex_grouped = flatten(sex_grouped)
fig = px.bar(sex_grouped,x = "Sex",y = "Survived mean", error_y = "Survived std", color = "Sex")
fig.update_traces(error_y_color = "black")


# ### Observations:
# - Females had a much higher chance of survival on average. This is probaly due to the fact that evacuation procedures prioritized women.

# ## 5.3 Age

# In[ ]:


# Age purely versus survival
px.histogram(train,x = "Age", opacity = 0.7, color = "Survived")


# In[ ]:


# Looking at age distribution and Pclass relationship
px.histogram(train, x = "Age", y = "Name", color = "Survived", facet_row = "Pclass", labels = dict(Name = "People"), opacity = 0.7)


# ### Observations:
# > **Note:** I'm unsure why the Pclasses are in an awkward order...
# 
# - There's a higher proportion of younger people (particularly those between the ages of 20 and 30) in the lower classes (Pclass 2 and 3)
# - Pclass 1 has the 'flattest' distribution of people
# - There's a higher survival chance the higher your socioeconomic status (similar observation as above). Pclass 1 has pink bars that are higher in more age brackets compared to Pclass 2 or 3
# - Extremely young children seem to have a high likliehood of survival (refer to graph of Pclass 2)
# - It **seems** that age is not as important as class. We will explore this concept in later sections

# ## 5.4 Fare

# In[ ]:


# Plotting distribution of fare on a log graph
px.histogram(train, x = "Fare", log_y = True, color = "Survived", opacity = 0.7)


# In[ ]:


# Plotting distribution of fares by class
px.histogram(train, x = "Fare", log_y = True, facet_col = "Pclass", color = "Pclass")


# ### Observations:
# - People who paid higher fares tend to be more likely to survive; the pink bars are more prominent and are usually larger as you move toward higher fares
# - The reasons most likely have to do with Pclass. Higher classed people were more likely to pay higher fares that could give them better rooms with better evacuation routes, etc

# ## 5.5 Embarkation Point

# In[ ]:


# Plotting embarkation point by survival rate
emb_grouped = train[['Embarked','Survived']].groupby('Embarked',as_index=False).agg({'Survived':['mean','std']})
emb_grouped = flatten(emb_grouped)
fig = px.bar(emb_grouped,x = "Embarked",y = "Survived mean", error_y = "Survived std", color = "Embarked")
fig.update_traces(error_y_color = "black")


# ### Observations:
# - People who embarked on the ship at Cherbourg had a higher chance of survival compared to those embarking from Queenstown or Southampton
# - Unsure why this trend exists. Should follow up by analyzing the distribution of classes and genders between cities

# ## 5.6 Heatmap

# In[ ]:


# Checking correlation between features to understand relative trends/comparisons before feature engineering
z = train.corr()
trace = go.Heatmap(
    z = z,
    x = z.columns,
    y = z.columns
)
iplot([trace])


# ### Observations:
# - Of the features, Sex, Passenger Class, and Fare have small to medium correlations with survival and will most likely be the most important features. This follows well from the observations earlier
# - In general, most of the features seem uncorrelated with each other which suggests that each feature will probably play an important role
#     - Siblings and parents are quite correlated; probably due to the fact that families travelled together
#     - Age, Parents/Children, and Siblings/Spouses are also somewhat correlated; probably because the younger you are, the more likely you are to travel with siblings/parents. Likewise, the older you are, the more likely you are to have children/family members that you travel with    
#     - Age and Pclass are quite correlated; this supports the plots in Section 5.3
#     - Fare and Pclass are very correlated; this supports the plots in Section 5.4
# 

# ### Notes Before Feature Engineering:
# - I will ignore Ticket Number and Passenger ID for the time-being because they are unique values and probably do not have any bearing on survival (not that I know of, at least)

# # 6. Feature Engineering

# This section will extend the data exploration section. We will also clump together similar features and/or categorize discrete features.

# ## 6.1 SibSp & Parch

# Based on the correlation between siblings and parents, we will combine them into one feature.

# In[ ]:


# Combining SibSp and Parch in one column since they're highly correlated
for dataset in full_dataset:
    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[ ]:


# Grouping by family size and plotting
grouped_family_size = train[['Family_Size','Survived','Name']].groupby(['Family_Size','Survived']).count().reset_index()
grouped_family_size.columns = ["Family_Size","Survived","Name"]
px.bar(grouped_family_size, x = "Family_Size", y = "Name", color = "Survived", barmode = "group", labels = dict(Name = "Count"))


# ### Observations:
# - If travelling alone, there's a higher chance of the person not surviving than if he/she travels in families of 2 to 4
# - If travelling in families of 2 to 4, theres a higher chance of the person surviving - probably because these families consist of both children and parents who get priority in evacuation procedures
# - Larger than 4, there's a very high chance of dying; probably because locating all people in your family during an evacuation is difficult

# We will categorize the 'Family_Size' feature to represent the observations above:
# - 1: Travelling alone
# - 2: Families of 2-4 people
# - 3: Families of 5 and above

# In[ ]:


# Cutting data
family_bins = [0, 1, 4, 20]
family_labels = [1, 2, 3]
for dataset in full_dataset:
    dataset['Family_Cat'] = pd.cut(dataset['Family_Size'], bins = family_bins, labels = family_labels, include_lowest = True)


# In[ ]:


# Re-visualizing the grouping
grouped_family_size = train[['Family_Cat','Survived','Name']].groupby(['Family_Cat','Survived']).count().reset_index()
grouped_family_size.columns = ["Family_Cat","Survived","Name"]
px.bar(grouped_family_size, x = "Family_Cat", y = "Name", color = "Survived", barmode = "group", labels = dict(Name = "Count"))


# ## 6.2 Name

# Here we explore a non-numerical feature that was not considered in the Data Exploration section.

# In[ ]:


# Visualizing a few name variables
train['Name'].head(10)


# ### Observations:
# These name variables contain Mr., Mrs., etc. Considering the previous relationship we observed with gender, it may be useful to extract this information. These titles might also give an indication of whether the person is single or married.
# 
# Let's extract it.

# In[ ]:


# Extracting name information and storing it in the 'Title' column
for dataset in full_dataset:
    dataset['Title'] = dataset.Name.str.extract(r"([A-Za-z]+)\.", expand = False)


# In[ ]:


# Printing all unique titles
set(train['Title'].unique()) | set(test['Title'].unique())


# In[ ]:


# Replacing values with the following mappings
title_map = {'Rare': [ 'Capt', 'Col','Countess','Dr','Jonkheer','Lady','Major','Rev','Sir'],
             'Mr': ['Mr','Don'],
             'Mrs':['Mme','Mrs','Dona'],
             'Miss':['Ms', 'Miss','Mlle'],
             'Master':['Master']}

for dataset in full_dataset:
    for key, value in title_map.items():
        dataset['Title'] = dataset['Title'].replace(value,key)


# In[ ]:


# Printing new set of unique values to make sure that we didn't miss anything
unique_titles = list(set(train['Title'].unique()) | set(test['Title'].unique()))
print("Unique Titles:",unique_titles)


# In[ ]:


# Categorizing the titles into buckets and printing the mapping
title_mapping = dict(zip(unique_titles,list(range(1,6))))
print("Title Mapping:",title_mapping)
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


# Visualizing survival and death rates by title
grouped_title = train[['Title','Survived']].groupby('Title',as_index=False).agg({"Survived":['mean','std']})
grouped_title = flatten(grouped_title).sort_values(by="Survived mean", ascending = False)
fig = px.bar(grouped_title, x = "Title", y = "Survived mean", error_y = "Survived std", color = "Title")
fig.update_traces(error_y_color = "black")


# ### Observations:
# - Mrs and Miss (2 and 3) have the best chance of surviving because of female priority in evacuation
# - Males (Master (1), Mr (4)) have lower chance of survival 
# - Rares weren't prioritized as much over women but they still have higher survival likliehood than Mr

# ## 6.3 Cabin

# Cabin has a lot of null values. However, cabins can allow us to infer a lot about a person's position on the ship and their proximity to evacuation routes or the collision. 
# 
# From the references below, I couldn't find information to link the cabin number to location. However, I could extract cabin deck. For the time-being, I will fill null values with a U until I figure out a better solution.
# 
# Based on the references, it seems like most of the first class cabins/amenities were in decks A-D and other sections were for lower class passengers. Hence, it'll be interesting to see the relationship between passenger class and deck
# 
# **References**:
# - Cutout of the Titanic: https://upload.wikimedia.org/wikipedia/commons/8/84/Titanic_cutaway_diagram.png <br/>
# - More Information: https://www.dummies.com/education/history/titanic-facts-the-layout-of-the-ship/

# In[ ]:


# Filling in null cabin values with U
for dataset in full_dataset:
    dataset['Cabin'].fillna("U",inplace=True)


# In[ ]:


# Extracting only deck letter from cabin
for dataset in full_dataset:
    dataset['Cabin_Deck'] = dataset['Cabin'].apply(lambda x: x.strip()[0])


# In[ ]:


# Grouping decks by class 
grouped_cabinclass = train.groupby(['Cabin_Deck','Pclass']).agg({'Name':'nunique'})
grouped_cabindeck = train.groupby(['Cabin_Deck']).agg({'Name':'nunique'})

# Normalizing the grouping
grouped_cabin = (grouped_cabinclass/grouped_cabindeck).reset_index()
fig2 = px.bar(grouped_cabin, x = "Cabin_Deck",y = "Name", color = "Pclass", labels = dict(Name = "% of Total"), title = "Normalized Distribution of Classes By Deck")
iplot(fig2)


# In[ ]:


# Grouping by cabin and plotting bar graph
grouped_cabindeck = train[['Cabin_Deck','Survived','Fare']].groupby('Cabin_Deck',as_index=False).agg({'Survived':['mean','std'],"Fare":['mean','std']})
grouped_cabindeck = flatten(grouped_cabindeck)
fig = px.bar(grouped_cabindeck, x = "Cabin_Deck", y = "Survived mean", error_y = "Survived std", color = "Cabin_Deck")
fig.update_traces(error_y_color = "black")


# ### Observations:
# > **Note:** There's a lot of "U" (missing) data - Refer to section 4
# 
# - First class people were predominantly assigned to A, B, C, D, T, E. Other decks have mostly second and third class passengers. This supports the reference shared earlier
# - Most unassigned passengers were third class
# - People unassigned cabins have a much lower chance of survival than those assigned cabins (probably because they were mostly third class passengers)
# - Those assigned cabins have ~50%+ chance of survival
# - Due to the large amount of missing data, I am very skeptical of imputing "U" using the trends observed above. I think that I will treat each cabin type (including the missing category) separately. Please share your feedback on this approach if you have any

# In[ ]:


# Generating map for cabin deck
possible_decks = list(set(test['Cabin_Deck'].unique()) | set(train['Cabin_Deck'].unique()))
cabin_mappings = dict(zip(possible_decks,list(range(1,len(possible_decks)+1))))
print(cabin_mappings)


# In[ ]:


# Mapping cabin decks
for dataset in full_dataset:
    dataset['Cabin_Deck'] = dataset['Cabin_Deck'].map(cabin_mappings)


# ## 6.4 Sex

# In[ ]:


# Converting sex to binary mapping
for dataset in full_dataset:
    dataset['Gender'] = dataset['Sex'].map({'male':1,'female':2})


# ## 6.5 Age

# In[ ]:


# Filling missing values for age with random integers within 1 standard deviation of the mean for combined train/test datasets
concatenated = pd.concat([train.drop('Survived', axis = 1),test])
age_avg = concatenated['Age'].mean()
age_std  = concatenated['Age'].std()

for dataset in full_dataset:
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    # Setting NaN 
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = age_null_random_list
    dataset["Age"] = age_slice
    dataset["Age"] = dataset["Age"].astype(int)


# In[ ]:


# Creating bins to cut ages into. These bins were determined by splitting the age data into 10 quantiles of approximately equal 
# number of people. The work was done in a separate notebook 
# Please comment if you'd like to see how this was done

age_bins = [0,16,19,22,25,28,31,35,40,47,100]
ages = ["0-16","16-19","19-22","22-25","25-28","28-31","31-35","35-40",'40-47',"47+"]
age_labels = list(range(1,11))
print("Age Mappings:",dict(zip(ages,age_labels)) )


# In[ ]:


# Cutting the data by the bins
for dataset in full_dataset:
    dataset['Age_Cat'] = pd.cut(dataset['Age'], bins = age_bins, include_lowest = True, labels = age_labels)


# In[ ]:


# Plotting the cut data
grouped_agecats = train.groupby('Age_Cat', as_index=False).agg({'Survived':['mean','std']})
grouped_agecats = flatten(grouped_agecats)
fig = px.bar(grouped_agecats,x = "Age_Cat", y = "Survived mean", error_y = "Survived std", range_y = [0, 1], color = "Age_Cat")
fig.update_traces(error_y_color = "black")


# ### Observations:
# - The youngest people have a higher chance of survival than older people
# - People within younger age brackets (teens to early 20s) have a lower chance of survival
# - In general, age doesn't seem to be a very huge factor in survival. This supports our earlier observations

# ## 6.6 Embarkation Point

# There's only two people whose embarkation point is unknown. 
# 
# Both these people boarded in Southampton.
# 
# **Reference:**
# - https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html

# In[ ]:


# Filling missing values for train set
train[train['Embarked'].isnull()]


# In[ ]:


# Filling in missing values accordingly
train.loc[[61,829],"Embarked"] = "S"


# In[ ]:


# Converting Embarked to numerical
embarked_mapping = {"S":1,"C":2,"Q":3}
print("Embarked Mapping:",embarked_mapping)
for dataset in full_dataset:
    dataset['Embarked_Cat'] = dataset['Embarked'].map(embarked_mapping)


# ## 6.7 Fare

# Since there's only 1 missing fare value, I'll simply fill it with the median.

# In[ ]:


# Filling missing values for test set with median
test['Fare'].fillna(test['Fare'].median(),inplace=True)


# In[ ]:


# Creating bins to cut fares into. These bins were generated by analyzing quantiles of data in a separate jupyter notebook (similar 
# process as Section 6.5)
fare_bins = [0, 7.8, 10.5, 21.7, 39.7, 550]
fares = ["0-7.8","7.8-10.5","10.5-21.7","21.7-39.7","39.7+"]
fare_labels = list(range(1,6))
print("Fare Mappings:",dict(zip(fares,fare_labels)))


# In[ ]:


# Cutting the dataset according to the bins above
for dataset in full_dataset:
    dataset['Fare_Cats'] = pd.cut(dataset['Fare'], bins = fare_bins, labels = fare_labels, include_lowest = True)


# In[ ]:


grouped_fares = train[['Fare_Cats', "Survived"]].groupby("Fare_Cats",as_index=False).mean()
px.bar(grouped_fares,x = "Fare_Cats",y="Survived", color = "Fare_Cats")


# ### Observations:
# - Higher fares have higher chances of survival. This ties in with the Pclass observations made earlier

# # 7. Machine Learning

# In this section I will establish my final test and training datasets, test a variety of sophisticated classification algorithms, tune the best performing algorithms, and combine them in an ensemble ML model.

# ## 7.1 Final Datasets

# In this section, I select by engineered features and create test and training datasets.

# In[ ]:


# Finalized training and test models
features = ['Pclass','Title','Gender','Age_Cat','Family_Cat','Fare_Cats','Cabin_Deck','Embarked_Cat']
X_train = train[features].values
Y_train = np.array(train[['Survived']]).ravel()
X_test = test[features].values
print("X_train")
print(X_train[0:10])
print("\nX_test")
print(X_test[0:10])


# ## 7.2 Setting Parameters & Fitting Models

# In this section, I define general parameters for the following models:
# - Extra Trees
# - Support Vector Machine
# - Random Forest
# - XG Boost
# - Gradient Boost
# - ADA Boost
# 
# I then fit each model to the training data and quantify the best performing models by comparing cross validation scores on the training set.

# In[ ]:


# Random seed
SEED = 0

# Declaring parameters for each of the models
# Extra Trees Classifier
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# SVC
svc_params = {
    'kernel' : 'rbf',
    'C' : 1,
    'gamma': 'auto'
}

# RandomForestClassifier
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'warm_start': True, 
    'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# XGB
xgb_params = {
    "learning_rate": 0.02,
    "n_estimators": 2000,
    "max_depth": 4,
    "min_child_weight": 2,
    "gamma":1,                        
    "subsample":0.8,
    "colsample_bytree":0.8,
    "objective": 'binary:logistic',
    "nthread": -1,
    "scale_pos_weight": 1
}

# Gradient Boosting
gb_params = {
    'n_estimators': 500,
    'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}


# In[ ]:


# Initializing models using the SklearnHelper function defined in Section 2
et = SklearnHelper(clf = ExtraTreesClassifier, seed = SEED, params = et_params)
svc = SklearnHelper(clf = SVC, seed = SEED, params = svc_params)
rf = SklearnHelper(clf = RandomForestClassifier, seed = SEED, params = rf_params)
xgb = SklearnHelper(clf = XGBClassifier, seed = SEED, params = xgb_params)
gb = SklearnHelper(clf = GradientBoostingClassifier, seed = SEED, params = gb_params)
ada = SklearnHelper(clf = AdaBoostClassifier, seed = SEED, params = ada_params)
models = [ada,et,gb, rf,svc,xgb]


# In[ ]:


# Looping through scores and getting dataframe of cross validation scores
score_dict = defaultdict(list)
for model in models:
    score_dict['Model'].append(model.__name__)
    score_dict['CV_Score'].append(model.cv_score(X_train,Y_train))
scores = pd.DataFrame(score_dict).sort_values(by = "CV_Score", ascending = False)
print(scores)


# In[ ]:


# Plotting cross validation scores as a function of model
px.bar(scores, y = "Model", x = "CV_Score", color = "CV_Score", orientation = "h")


# ## 7.3 Feature Importances

# In this section, I characterize the important features for each of the classification models (minus SVM). I look at this on a per-model and average basis.

# In[ ]:


# Capturing feature importances in Plotly figures and in a dictionary
feature_imps = defaultdict(list)
figs = []
for i, model in enumerate(models):
    if model.__name__ == "SVC":
        continue
        
    trace = go.Scatter(
        y = model.feature_importances(X_train,Y_train),
        x = features,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
            color = model.feature_importances(X_train,Y_train),
            colorscale='Portland',
            showscale=True
            )
    )
    layout = go.Layout(
            autosize= True,
            title= model.__name__,
            hovermode= 'closest',
            yaxis=dict(
                title= 'Feature Importance',
                ticklen= 5,
                gridwidth= 2
            ),
            showlegend= False
            )
    figs.append(dict(data = [trace], layout = layout))
    
    feature_imps[i].append(model.__name__)
    feature_imps[i].extend(model.feature_importances(X_train,Y_train))
feature_imps = pd.DataFrame.from_dict(feature_imps, orient = "index", columns = ['Model_Name'] + features)


# In[ ]:


# Plotting feature importances by classification model
for fig in figs:
    iplot(fig)


# In[ ]:


# Extract mean feature importance by model and plot
mean_imp = pd.concat([feature_imps.mean(axis = 0),feature_imps.std(axis = 0)], axis = 1).reset_index()
mean_imp.columns = ["Feature","Mean_Importance","Std"]
fig = px.bar(mean_imp.sort_values(by="Mean_Importance"), x = "Feature", y = "Mean_Importance", color = "Mean_Importance", error_y = "Std")
fig.update_traces(error_y_color = "black")


# ### Observations:
# - Title and Gender are the most important features for our models. This suggests that the most important reason for survival was down to being a female (because title and gender are very correlated)
# - PClass and Cabin deck (they too, are correlated) have similar importances and suggest that your social class was the second most important factor in survival; this was probably because position will play a larger role during emergency evacuations
# - Family, Age, Fare and Embarkation point are much less important

# ## 7.5 Parameter Tuning Using GridSearch

# In this section, we will tune each of the 3 aforementioned models (XGB, ADAB and SVC) using GridSearch.

# In[ ]:


# Setting the parameter grid for each model
# Extra Trees
et_grid = {
    'n_estimators': [250, 500, 1000],
    'max_depth' : [2,4,8],
    'min_samples_leaf': [2,4,8]
}

# RF
rf_grid = {
    'n_estimators' : [250, 500, 1000],
    'max_depth' : [2,4,6],
    'min_samples_leaf': [2,4,8]
}


# XGB
xgb_param_grid = {
    "learning_rate": [0.01, 0.1, 1],
    "n_estimators": [1000, 2000, 4000],
    "max_depth": [2,3, 4]
}


# In[ ]:


# Setting up Grid Search Models to tune
et_gs = GridSearchCV(
    estimator = ExtraTreesClassifier(),
    param_grid = et_grid,
    cv = 5,
    scoring = 'accuracy'
)

rf_gs = GridSearchCV(
    estimator = RandomForestClassifier(),
    param_grid = rf_grid,
    cv = 5,
    scoring = 'accuracy'
)

xgb_gs = GridSearchCV(
    estimator = XGBClassifier(    
        min_child_weight = 2,
        gamma = 1,                        
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary:logistic',
        nthread = -1,
        scale_pos_weight = 1),
    param_grid = xgb_param_grid,
    cv = 5,
    scoring = 'accuracy'
)


# In[ ]:


# Determining best parameters for each model
best_params = {}
best_params['Extra Trees'] = et_gs.fit(X_train,Y_train).best_params_


# In[ ]:


best_params['RF'] = rf_gs.fit(X_train,Y_train).best_params_


# In[ ]:


best_params['XGB'] = xgb_gs.fit(X_train,Y_train).best_params_


# In[ ]:


best_params


# ## 7.6 Ensemble Model

# In this section, we combine the 3 aforementioned classifiers into one ensemble model

# In[ ]:


# Creating the ensemble model with hard voting
ensemble = VotingClassifier(
    estimators = [('ET',ExtraTreesClassifier(max_depth = 8, min_samples_leaf = 4, n_estimators = 1000)),
                  ('RF',RandomForestClassifier(max_depth = 4, min_samples_leaf = 2, n_estimators = 500)), 
                  ('XGB',XGBClassifier(min_child_weight = 2,gamma = 1,  subsample = 0.8, colsample_bytree = 0.8, objective = 'binary:logistic', nthread = -1,\
                                       scale_pos_weight = 1, learning_rate = 0.01, max_depth = 4, n_estimators = 4000))],
    voting = 'hard'
)


# In[ ]:


# Fitting the ensemble to the training data
ensemble_fit = ensemble.fit(X_train,Y_train)


# # 8. Submission File

# In[ ]:


# Preparing submission
submission = pd.concat([test['PassengerId'],pd.Series(ensemble_fit.predict(X_test))], axis = 1,)
submission.columns = ['PassengerId','Survived']
submission.head()


# In[ ]:


submission.to_csv("submission.csv",index=False)

