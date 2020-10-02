# Hi. Got a ship to sink? You're in luck, cause I have some data for you.

###############################################################################
# Section 1: In which I do some horrendous things to the data
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from itertools import compress
from math import isnan
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree

# Let's read the csv files

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# Okay, so, now I'm gonna do things a little bit backwards, and extract and 
# transform the columns I find relevant from the csv files.
# I'll explain why I chose particular columns over others in a little bit.

# Here we have two helper functions that transform certain data into flags: 
# assumed_age and family_size:

def assumed_age(age):
    "Takes the passenger's age and returns a flag that represents it"
    
    if isnan(age):
        age_label = 1
    else:
        if age >= 18:
            age_label = 1
        else:
            age_label = 0
    
    return age_label
	
def family_size(size):
    "Takes the size of the passenger's family size \
    and returns a flag that represents it"
    
    if size == 0:
        flag = 0
    else:
        if size <= 3:
            flag = 1
        else:
            if size <= 5:
                flag = 2
            else:
                flag = 3
                
    return flag
	
# Here's a function that uses both assumed_age and family_size and 
# transforms all of the relevant data in the csv files: transform_data:

def transform_data(data):
    "Extracts and transforms the relevant Titanic data from the dataframe"
     
    # 0 - Female
    # 1 - Male
    passenger_sex = np.array([int(val) for val in data["Sex"] == "male"])
    
    # 0 - Child (Age != NaN && Age < 18)
    # 1 - Adult (Age == NaN || Age >= 18)
    passenger_age = np.array([assumed_age(val) for val in data["Age"]])
    
    # 0 - First class
    # 1 - Second class
    # 2 - Third class
    passenger_class = np.array(data["Pclass"] - 1)
    
    # 0 - Alone (SibSp + Parch == 0)
    # 1 - Small famlily (SibSp + Parch <= 3)
    # 2 - Medium famlily (SibSp + Parch <= 5)
    # 3 - Large famlily (SibSp + Parch > 5)
    passenger_family = np.array([family_size(val) for val in 
                                 (data["SibSp"] + data["Parch"])])
    
    X = np.array([passenger_sex, passenger_age, 
                  passenger_class, passenger_family])
    X = X.transpose()
    
    return X
	
# Now we use transform_data:
    
# First, we transform our train data:

X_train = transform_data(train_data)
y_train = np.array(train_data["Survived"])

# Just an arbitrary dataframe used to display the transformed data in a table
x_train_df = pd.DataFrame(columns=["SexFlag", "AgeFlag", 
                                   "ClassFlag", "FamilySizeFlag"])

x_train_df["SexFlag"] = X_train[:, 0]
x_train_df["AgeFlag"] = X_train[:, 1]
x_train_df["ClassFlag"] = X_train[:, 2]
x_train_df["FamilySizeFlag"] = X_train[:, 3]

print("Raw train data:")
display(train_data.head())

print("Transformed train data:")
display(x_train_df.head())

print("Raw train data:")
display(train_data.tail())

print("Transformed train data:")
display(x_train_df.tail())

# Second, we transform our test data:

X_test = transform_data(test_data)

# Again, just an arbitrary dataframe
x_test_df = pd.DataFrame(columns=["SexFlag", "AgeFlag", 
                                  "ClassFlag", "FamilySizeFlag"])

x_test_df["SexFlag"] = X_test[:, 0]
x_test_df["AgeFlag"] = X_test[:, 1]
x_test_df["ClassFlag"] = X_test[:, 2]
x_test_df["FamilySizeFlag"] = X_test[:, 3]

print("Raw test data:")
display(test_data.head())

print("Transformed test data:")
display(x_test_df.head())

print("Raw test data:")
display(test_data.tail())

print("Transformed test data:")
display(x_test_df.tail())

# A not so small sidenote when it comes to the SibSp and Parch columns:

# Regarding a single passenger, I found it to be a guessing game whether the 
# numbers in SibSp and Parch relate to Siblings/Spouses or Parents/Children 
# respectively, and how those would affect one's chances of survival. 
# Moreover, I realized that I couldn't find all the people a person was 
# traveling with in a given file. That led me to conclude that looking at the 
# sum of those two columns and taking into account the number of people a 
# person was travelling with was a more reliable tactic than trying to assume 
# whether someone would be more likely to survive if they had their sister 
# onboard or something. In short, I labeled everyone as traveling Alone or 
# with a Small, Medium or Large family.

###############################################################################
# Section 2: In which I make some bold assumptions
###############################################################################

# Time to draw some plots. Yeah!

# First, we're gonna need some labels:

survived_label = ["No", "Yes"]
sex_label = ["Female", "Male"]
age_label = ["Child", "Adult"]
class_label = ["First", "Second", "Third"]
family_label = ["Alone", "Small", "Medium", "Large"]
embarked_label = ["Cherbourg", "Queenstown", "Southampton"]

# Second, let's make a neat little dataframe:

demographics = pd.DataFrame(columns=["Survived", "Sex", "Age", 
                                     "Class", "Family", "Embarked"])

demographics["Survived"] = np.take(survived_label, train_data["Survived"])
demographics["Sex"] = np.take(sex_label,  X_train[:, 0])
demographics["Age"] = np.take(age_label, X_train[:, 1])
demographics["Class"] = np.take(class_label, X_train[:, 2])
demographics["Family"] = np.take(family_label, X_train[:, 3])
demographics["Embarked"] = np.array(train_data["Embarked"])

for i, val in enumerate(demographics["Embarked"]):
    if val == "C":
        demographics.Embarked[i]="Cherbourg"
    if val == "Q":
        demographics.Embarked[i]="Queenstown"
    if val == "S":
        demographics.Embarked[i]="Southampton"
		
# Third, we're gonna be drawing a lot of count plots, 
# so a function we'll come in handy:

def draw_passenger_countplot(title, x, hue=None, order=None, hue_order=None):
    "Draws a count plot based on passenger information"
    axes = plt.axes()
    axes.set_title(title)
    
    cnt = sns.countplot(x=x, hue=hue, 
                        order=order, hue_order=hue_order,
                        palette="Set3")

    cnt.set(ylabel="Passenger Number")

    plt.show()
    
    return
	
# All set!

# Let's address the Age column.
# The number of NaN values in it is kind of a problem. 
# I thought about it for some time and concluded it would 
# be best to label all of them as adults. 
# Look at the graph produced by the code bellow:

nan_family_size = list(compress(X_train[:, 3], train_data["Age"].isnull()))
nan_family_size = np.take(family_label, nan_family_size)

draw_passenger_countplot(title="Plot #1: NaN Family Size", x=nan_family_size)

# The majority of NaNs traveled alone, which, to me, is enough to immediately 
# put them in the adults category. As for the rest, the number of adults 
# onboard was larger than the number of children, so I decided not to try and 
# guess which one's which and label all NaNs as adults.

# As for the Embarked column, I disregarded it, because I don't see how the 
# place you embarked on could affect your chances of survival.
# Take a look at the graph bellow:

draw_passenger_countplot(title="Plot #2: Embarked Surviral Ratio", 
                         x=demographics["Embarked"], 
                         hue=demographics["Survived"])

# The largest amount of passengers (from this csv file) embarked in 
# Southampton. There isn't a big difference between the number of people who 
# survived and who didn't among those who embarked in Cherbourg or Queenstown.
# About 50% of the people who embarked in Southampton did die, which is a much 
# larger percentage than the other two.

# To figure out what happened, let's look at the Southampton demographics:

southhampton_sex = list(compress(demographics["Sex"], 
                                 demographics["Embarked"] == "Southampton"))

southhampton_class = list(compress(demographics["Class"], 
                                   demographics["Embarked"] == "Southampton"))

draw_passenger_countplot(title="Plot #3: Southampton Demographics", 
                         x=southhampton_class, 
                         hue=southhampton_sex,
                         order=class_label, 
                         hue_order=sex_label)
						 
# Here we see that there is an overwhelming number of third class male 
# passengers from Southhampton. We'll later prove that group had the worst 
# chances of survival. I think this distribution justifies the 50% survival 
# rate of Southhampton's passengers. Just to be sure, let's look at the 
# Queenstown and Cherbourg demographics:

queenstown_sex = list(compress(demographics["Sex"], 
                               demographics["Embarked"] == "Queenstown"))

queenstown_class = list(compress(demographics["Class"], 
                                 demographics["Embarked"] == "Queenstown"))

draw_passenger_countplot(title="Plot #4: Queenstown Demographics", 
                         x=queenstown_class, 
                         hue=queenstown_sex,
                         order=class_label, 
                         hue_order=sex_label)
						  
# Here we see that the number of third class men and women is about equal. 
# The percentage of first and second class passengers is smaller than that in 
# Southhampton, yet the survival rate of people who embarked in Queenstown 
# seems to be a bit larger. Do keep in mind that the sample of passengers who 
# embarked in Queenstown and Cherbourg (from this csv file) is much, much 
# smaller than the sample of passengers who embarked in Southhampton.
# We don't have control over how the data was distributed between csv files 
# and have no idea of knowing what the entirety of Titanic's data would look 
# like if plotted. We can make do with small inconsistencies. As for Cherbourg:

cherbourg_sex = list(compress(demographics["Sex"], 
                              demographics["Embarked"] == "Cherbourg"))

cherbourg_class = list(compress(demographics["Class"], 
                                demographics["Embarked"] == "Cherbourg"))

draw_passenger_countplot(title="Plot #5: Cherbourg Demographic", 
                         x=cherbourg_class, 
                         hue=cherbourg_sex,
                         order=class_label, 
                         hue_order=sex_label)
						  
# The passengers who embarked in Cherbourg have shown to have the largest 
# survival rate, which proves to make sense because the majority of them are 
# first class passengers. The supposed survival chances associated with 
# passengers on account of the place they embarked on have shown to be closely 
# tied to the passenger's class. We already have a column that reflects that 
# information. With that said, I officialy render the Embarked column useless.

# As for the remaining columns: I disregarded the Ticket column, because I 
# don't see how the number/label of your ticket could affect your chances of 
# survival. I disregarded the Fare column, because the class of the passenger 
# is already reflected in the Pclass Column. I disregarded the Cabin column, 
# because the data is too sparse. You could argue that these columns could 
# indicate the cabin the passenger was situated in, which we could use to 
# determine how far the passenger could have been from the boat deck when the 
# ship crashed. I would in turn argue that the cabins for wealthier passengers 
# were on higher decks, i.e. closer to the boat deck, which is again reflected 
# in the Pclass column and stands to justify my reasoning.

# If you think there are flaws in my logic and reasons why I should use these 
# columns, please let me know.

###############################################################################
# Section 3: In which I tell you how much I love count plots
###############################################################################

# We are also gonna be using a lot of facet grids, 
# so let's make a function for that too:
    
def draw_passenger_facetgrid(title, data, row=None, col=None, hue=None, 
                             order=None, col_order=None, hue_order=None):
    "Draws a facet grid based on passenger information"
    
    fg = sns.FacetGrid(data, row=row, col=col, size=4,
                       row_order=order,
                       col_order=col_order)
    
    fg.map(sns.countplot, hue,
           order=hue_order,
           palette="Set3")
    
    fg.fig.suptitle(title)
    
    fg.fig.subplots_adjust(top=0.9)

    plt.show()
    
    return
	
# Now that we got all of that out the way, 
# what do our overall passenger demographics look like?

draw_passenger_facetgrid("Plot #6: Sex/Age/Class Passenger Demographics", 
                         demographics,
                         row="Sex", col="Age", hue="Class",
                         order=sex_label, 
                         col_order=age_label, 
                         hue_order=class_label)
						 
# As we can see, there is a large amount of third class men on board. We 
# already figured that one out. The majority of children are also from the 
# Third class. The number of First class and Second class passengers 
# are not that far apart.

# We haven't included the passenger's family size in the previous plot.
# Let's look at that now.

draw_passenger_facetgrid("Plot #7: Sex/Family/Class Passenger Demographics", 
                         demographics,
                         row="Sex", col="Family", hue="Class",
                         order=sex_label, 
                         col_order=family_label, 
                         hue_order=class_label)
						 
# Most of the third class men traveled alone. Generally, most of the 
# passengers seemed to be traveling alone or with a small family. The ones 
# with a bigger family are largely members of the Third class.

# The basic assumption is that more women survived than men. 
# The following graph shows that:

draw_passenger_countplot("Plot #8: Sex Survival Ratio", 
                         x=demographics["Sex"], 
                         hue=demographics["Survived"], 
                         hue_order=survived_label)
						 
# That was to be expected. A similar graph can be drawn to show the survival 
# ratio between different classes.

draw_passenger_countplot("Plot #9: Class Survival Ratio", 
                         x=demographics["Class"], 
                         hue=demographics["Survived"], 
                         order=class_label, 
                         hue_order=survived_label)
						 
# How about we combine these two things into one graph?

draw_passenger_facetgrid("Plot #10: Sex/Class Survival Ratio", 
                         demographics,
                         row="Sex", col="Class", hue="Survived",
                         order=sex_label, 
                         col_order=class_label, 
                         hue_order=survived_label)
						 
# First and second class women survived almost exclusively. Third class women 
# come up as a 50/50. Men weren't so lucky. The majority of third and second 
# class men died, and the first class men didn't do much better. We can now 
# say with certainty that women had bigger chances of survival than men, 
# as well as that first class passengers had an advantage against second class 
# passengers, and the second class passengers against third class passengers.

# Those were easy assumptions to make. What about age and family size?

draw_passenger_facetgrid("Plot #11: Age/Sex Survival Ratio", 
                         demographics,
                         row="Age", col="Sex", hue="Survived",
                         order=age_label, 
                         col_order=sex_label, 
                         hue_order=survived_label)

# Not much to say about this one. The adult male/female survival ratio stayed 
# the same. The number of children onboard is not high enough to change those 
# statistics and they mostly correspond to their adult counterparts, albeit 
# the ratios aren't so drastic. It almost makes Age a thorwaway column. Almost.

draw_passenger_facetgrid("Plot #12: Family/Class Survival Ratio", 
                         demographics,
                         row="Class", col="Family", hue="Survived",
                         order=class_label, 
                         col_order=family_label, 
                         hue_order=survived_label)

# This one's far more interesting. You can see that almost none of the first 
# and second class passengers traveled with medium or large families, and 
# those from the third class who did, mostly died. First and second class 
# passengers who travelled alone or with a small family have roughly the same 
# survival ratio, as well as the third class passengers with a small family. 
# Third class passengers who traveled alone mostly died. Remember that they 
# account for the huge number of third class men (who seemed to be the target 
# audience for drowning).

###############################################################################
# Section 4: In which we pick our weapon of choice
###############################################################################

# Onto the classifiers!

# I tested Decision Trees and Random Forests (because Kaggle said so), and 
# then I followed the flowchart on scikit-learn 
# (this one: http://scikit-learn.org/stable/tutorial/machine_learning_map/) 
# and decided to throw in Extra Trees, KNeighbors, LinearSVC, SVC, Gaussian 
# classifier, Ada Boost and Gradient Boosting just for the fun of it.

# As always, let's break this thing down into functions:

def decision_tree_evaluation(X, y):
    "Evaluates the Decision Tree classifier"
    
    dec_tree = tree.DecisionTreeClassifier()
    
    min_samples_leaf_range = list(range(1, 10))
    max_depth_range = list(range(1, 15))
    
    param_grid = dict(min_samples_leaf=min_samples_leaf_range, 
                      max_depth=max_depth_range)
    
    grid = GridSearchCV(dec_tree, param_grid, cv=10, scoring="roc_auc")
    grid.fit(X, y)
    
    return grid.best_score_, grid.best_params_, grid.best_estimator_

def random_forest_evaluation(X, y):
    "Evaluates the Random Forest classifier"
    
    rand_tree = RandomForestClassifier()
    
    n_estimators_range = list(range(1, 10))
    max_features_range = list(range(1, 4))
    
    param_grid = dict(n_estimators=n_estimators_range, 
                      max_features=max_features_range)
    
    grid = GridSearchCV(rand_tree, param_grid, cv=10, scoring="roc_auc")
    grid.fit(X, y)
    
    return grid.best_score_, grid.best_params_, grid.best_estimator_
	
def extra_trees_evaluation(X, y):
    "Evaluates the Extra Trees classifier"
    
    extra_tree = ExtraTreesClassifier()
    
    n_estimators_range = list(range(1, 10))
    max_features_range = list(range(1, 4))
    
    param_grid = dict(n_estimators=n_estimators_range, 
                      max_features=max_features_range)
    
    grid = GridSearchCV(extra_tree, param_grid, cv=10, scoring="roc_auc")
    grid.fit(X, y)
    
    return grid.best_score_, grid.best_params_, grid.best_estimator_

def knn_evaluation(X, y):
    "Evaluates the KNN classifier"
    
    knn = KNeighborsClassifier()
    
    k_range = list(range(1, 31))
    
    param_grid = dict(n_neighbors=k_range)
    
    grid = GridSearchCV(knn, param_grid, cv=10, scoring="roc_auc")
    grid.fit(X, y)
    
    return grid.best_score_, grid.best_params_, grid.best_estimator_

def linear_svc_evaluation(X, y):
    "Evaluates the LinearSVC classifier"
    
    linear_svc = svm.LinearSVC()
    
    scores = cross_val_score(linear_svc, X, y, cv=10, scoring="roc_auc")
    
    return scores.mean(), "None", "None"

def svc_evaluation(X, y):
    "Evaluates the SVC classifier"
    
    svc = svm.SVC()
    
    scores = cross_val_score(svc, X, y, cv=10, scoring="roc_auc")
    
    return scores.mean(), "None", "None"

def gaussian_naive_bayes_evaluation(X, y):
    "Evaluates the Gaussian Naive Bayes classifier"
    
    gnb = GaussianNB()
    
    scores = cross_val_score(gnb, X, y, cv=10, scoring="roc_auc")
    
    return scores.mean(), "None", "None"

def bernoulli_naive_bayes_evaluation(X, y):
    "Evaluates the Bernoulli Naive Bayes classifier"
    
    bnb = BernoulliNB()
    
    scores = cross_val_score(bnb, X, y, cv=10, scoring="roc_auc")
    
    return scores.mean(), "None", "None"

def ada_boost_evaluation(X, y):
    "Evaluates the Ada Boost classifier"
    
    ab = AdaBoostClassifier()
    
    n_estimators_range = list(range(40, 60))
    
    param_grid = dict(n_estimators=n_estimators_range)
    
    grid = GridSearchCV(ab, param_grid, cv=10, scoring="roc_auc")
    grid.fit(X, y)
    
    return grid.best_score_, grid.best_params_, grid.best_estimator_

def gradient_boosting_evaluation(X, y):
    "Evaluates the Gradient Boosting classifier"
    
    gb = GradientBoostingClassifier()
    
    n_estimators_range = list(range(100, 120))
    
    param_grid = dict(n_estimators=n_estimators_range)
    
    grid = GridSearchCV(gb, param_grid, cv=10, scoring="roc_auc")
    grid.fit(X, y)
    
    return grid.best_score_, grid.best_params_, grid.best_estimator_

# Once more, let's make some labels:

models = ["DecisionTreeClassifier", "RandomForestClassifier", 
          "ExtraTreesClassifier", "KNeighborsClassifier", "LinearSVC", "SVC", 
          "GaussianNB", "BernoulliNB", "AdaBoost", "GradientBoosting"]
		  
# Then a dataframe:

options = {models[0] : decision_tree_evaluation,
           models[1] : random_forest_evaluation,
           models[2] : extra_trees_evaluation,
           models[3] : knn_evaluation,
           models[4] : linear_svc_evaluation,
           models[5] : svc_evaluation,
           models[6] : gaussian_naive_bayes_evaluation,
           models[7] : bernoulli_naive_bayes_evaluation,
           models[8] : ada_boost_evaluation,
           models[9] : gradient_boosting_evaluation
           }
		   
# And lastly, a function to encapsulate all of the madness:

def model_evaluation(X, y, model):
    "Evaluates the specified model"
    
    best_score, best_params, best_estimator = options[model](X, y)
    
    return [best_score, best_params, best_estimator]
	
# Done! The next step would be to evaluate all the models.
# We want to save the evaluation scores so we can plot them later. 
# Fair warning, this part takes awhile to execute.

scores = []

for i, model in enumerate(models):
    scores.append(model_evaluation(X_train, y_train, model))

scores = np.array(scores)

# In this section we'll be using bar plots. So (you've guessed it), 
# let's make a function for it:
def draw_model_barplot(title, x, y, xlabel=None, ylabel=None):
    "Draws a bar plot based on model evaluation scores"
    
    axes = plt.axes()
    
    axes.set_title(title)
    
    bar = sns.barplot(x=x, y=y, palette="Set3")
    
    bar.set(xlabel=xlabel)
    bar.set(ylabel=ylabel)
    
    # Gets the index of the best evaluation score
    max_index = np.argmax(x)
    
    for i, patch in enumerate(axes.patches):
        x = 0.9
        y = patch.get_y()
        
        if y < 0:
            y = 0.1
        else:
            y = y + 0.5
             
        if i == max_index:
            color = "red"
        else:
            color = "black"
        
        # Writes the evaluation score next to each bar
        axes.annotate(str(round(scores[i, 0], 4)), (x, y), color=color)

    plt.show()
    
    return
	
# Now that it's finally done, it's time to plot this monster.

draw_model_barplot("Plot #13: Model Evaluation", scores[:, 0], 
                   y=models, xlabel="Score", ylabel="Model")

# One of two things can happen when you run this script: Random Forests or 
# Extra Trees can turn out to be the best classifier. You could run the 
# evaluation for them a couple more times and figure out which one does best 
# more often. But, both of these classifiers work by taking random subsets of 
# the features. That means that they never give the same result each time they 
# run and if you look closely, their scores are never that far appart.
# Kaggle recommends Random Forests, so let's use them.

###############################################################################
# Section 5: In which we all proceed to the lifeboats
###############################################################################

# You excited yet? Only a couple more steps left.

# Given the random nature of the classifier, the best values for its parameters 
# are also going to be inconsistent. Let's run the evaluation for Random 
# Forests a couple more times and find out the mean values for its parameters.

rand_forests_scores = []

for _ in range(10):
    rand_forests_scores.append(model_evaluation(X_train, y_train, 
                                                "RandomForestClassifier"))
    
rand_forests_scores = np.array(rand_forests_scores)

n_estimators = []
max_features = []

for i in range(10):
    param = rand_forests_scores[i, 1]
    n_estimators.append(param["n_estimators"])
    max_features.append(param["max_features"])
    
n_estimators_best = int(round(np.mean(n_estimators)))
max_features_best = int(round(np.mean(max_features)))

print("Number of estimators: ", n_estimators_best)
print("Maximum number of features: ", max_features_best)

# It's finally time to make ourselves some 
# Random Forests and predict this thing.

rand_forests = RandomForestClassifier(n_estimators=n_estimators_best, 
                                      max_features=max_features_best)

rand_forests.fit(X_train, y_train)

y_test = rand_forests.predict(X_test)

# Simple enough. One more thing to do.

results = pd.DataFrame(columns=["PassengerId", "Survived"])
    
results["PassengerId"] = test_data["PassengerId"]
results["Survived"] = y_test
    
results.to_csv("results.csv", index=False)

# And that's it! Ladies and gentlemen, the show's over.

# Thanks for sticking around this long. This is my first take on data science, 
# so you have my sincerest apologies for any obvious beginner mistakes I made. 
# If you have any suggestions or anything else to say about my solution, 
# I'd be more than happy to hear it.
# Either way, farewell and have a pleasent day!