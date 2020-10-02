#!/usr/bin/env python
# coding: utf-8

# In[99]:


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 


# In[100]:


holdout = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# In[101]:


def remove_border(plt, legend=None, frameon=False):
    if plt is None:
        return
    
    # Remove plot border
    for sphine in plt.gca().spines.values():
        sphine.set_visible(False)
        
    # Create legend and remove legend box
    if legend is not None:
        plt.legend(legend, frameon=False)
    
    # Remove ticks 
    plt.tick_params(left=False, bottom=False) 


# ## Exploring and converting age column
# ![conversion](img/cut.svg)
# 
# Conversion of age column
# - Uses the pandas.fillna() method to fill all of the missing values with -0.5
# - Cuts the Age column into following segments using pandas.cut():
#     - Age categories `(labels, start, end not including)` are as follows:
#         - ("Missing", -1, 0)
#         - ("Infant", 0, 5)
#         - ("Child", 5, 12), 
#         - ("Teenager", 12, 18)
#         - ("Young Adult", 18, 35)
#         - ("Adult", 35, 60),
#         - ("Senior", 60, 100)

# In[102]:


def process_age(df, cut_point=None, label_names=None):
    # Fill missing age with negative values to denote missing category
    df["Age"] = df["Age"].fillna(-0.5)
    
    # If cut point and label names are not provided, default ranges used.
    if cut_point is None and label_names is None:
        age_ranges = [("Missing", -1, 0), ("Infant", 0, 5), ("Child", 5, 12), 
                      ("Teenager", 12, 18), ("Young Adult", 18, 35), ("Adult", 35, 60),
                      ("Senior", 60, 100)]
    
        cut_points = [x for _, x, _ in age_ranges]
        cut_points.append(age_ranges[-1][-1])
        label_names = [labels for labels, *_ in age_ranges]
    
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df


# In[103]:


train = process_age(train)
holdout = process_age(holdout)
train_original_columns = train.columns


# ## Create dummies 

# In[104]:


def create_dummies(df, column):
    return pd.concat([df, pd.get_dummies(df[column], prefix = column)], axis=1)


# In[105]:


for col in ["Age_categories", 'Pclass', "Sex"]:
    train = create_dummies(train, col)
    holdout = create_dummies(holdout, col)


# ## Preparing more features

# In[106]:


from sklearn.preprocessing import minmax_scale

# The holdout set has a missing value in the Fare column which
# we'll fill with the mean.
holdout["Fare"] = holdout["Fare"].fillna(train["Fare"].mean())

# Fill missing Embarked values with S, where Titanic first started 
for df in [train, holdout]:
    col = "Embarked"
    df[col] = df[col].fillna("S")

# Create dummy columns for Embarked    
train = create_dummies(train, col)
holdout = create_dummies(holdout, col)
    
#This estimator scales and translates each feature individually 
# such that it is in the given range on the training set, i.e. between zero and one.
for col in ["SibSp", "Parch", "Fare"]:
    train[col+"_scaled"] = minmax_scale(train[col].astype("float"))
    holdout[col+"_scaled"] = minmax_scale(holdout[col].astype("float"))


# In[107]:


print(train.columns)


# In[108]:


target = "Survived"
features = train.drop(columns=train_original_columns).columns
features


# ## Determining the most revelant features

# In[109]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')
lr.fit(train[features], train[target])
coefficients = lr.coef_
feature_importance = pd.Series(coefficients[0], index=train[features].columns)
print(feature_importance)
feature_importance.plot.barh()
remove_border(plt)
plt.show()


# In[110]:


ordered_feature_importance = feature_importance.abs().sort_values(ascending=True)
ordered_feature_importance.plot.barh()
remove_border(plt)
plt.show()

# Since index 0 is smallest value, we want index 0 to be biggest value
ordered_feature_importance = feature_importance.abs().sort_values(ascending=False)


# We will train a new model with the top 8 scores and check our accuracy using cross validation.

# In[111]:


# Using top 8 features
features = ordered_feature_importance[:8].index.tolist()
features


# ## Train a model using relevant features

# In[112]:


from sklearn.model_selection import cross_val_score
from numpy import mean 

lr = LogisticRegression(solver='liblinear')
scores = cross_val_score(lr, train[features], train[target], cv=10)
accuracy = mean(scores)
print(scores)
print(accuracy)


# The cross validation score of 81.48% is marginally higher than the cross validation score for the model we created in the previous mission, which had a score of 80.2%.
# 
# Hopefully, this improvement will translate to previously unseen data. Let's train a model using the columns from the previous step, make some predictions on the holdout data and submit it to Kaggle for scoring.

# In[113]:


lr = LogisticRegression()
lr.fit(train[features], train[target])
holdout_predictions = lr.predict(holdout[features])
holdout_predictions


# ## Submit improved model to kaggle

# In[114]:


def submission(holdout, holdout_predictions):
    submission = pd.DataFrame({"PassengerId":holdout["PassengerId"], "Survived":holdout_predictions})
    submission.to_csv("gender_submission.csv", index=False)
submission(holdout, holdout_predictions)


# You advanced 2,651 places on the leaderboard!
# Your submission scored 0.7703, which is an improvement of your previous score of 0.75598. Great job!

# It's only a small improvement, but we're moving in the right direction.
# 
# ## Engineering a new feature using binning
# A lot of the gains in accuracy in machine learning come from **Feature Engineering**. Feature engineering is the practice of creating new features from your existing data.
# 
# One common way to engineer a feature is using a technique called **binning**. Binning is when you take a continuous feature, like the fare a passenger paid for their ticket, and separate it out into several ranges (or 'bins'), turning it into a categorical variable.
# 
# This can be useful when there are patterns in the data that are non-linear and you're using a linear model (like logistic regression). We actually used binning in the previous mission when we dealt with the Age column, although we didn't use the term.
# 
# Let's look at histograms of the Fare column for passengers who died and survived, and see if there are patterns that we can use when creating our bins.

# In[115]:


survived = train[train["Survived"] == 1]
died = train.drop(survived.index)

col_hist = "Fare"
# Survived and died histogram in a single plot
for df, color in [(survived, "red"), (died,"blue")]:
    ax = df[col_hist].plot.hist(alpha=.5, color=color, bins=50)
    ax.set_xlabel(col_hist)
    ax.set_xlim([0, 250])

remove_border(plt, legend=["Survived", "Died"])

plt.title("Survived and Died Histogram")
plt.show()

print(train[col_hist].describe())


# Looking at the values, it looks like we can separate the feature into four bins to capture some patterns from the data:
# - 0-12
# - 12-50
# - 50-100
# - 100+
# 
# 

# In[116]:


def process_fare(df, cut_point=None, label_names=None):
    
    # If cut point and label names are not provided, default ranges used.
    if cut_point is None and label_names is None:
        fare_ranges = [("0-12", 0, 12), ("12-50", 12, 50), ("50-100", 50, 100), 
                      ("100+", 100, 1000)]
    
        cut_points = [x for _, x, _ in fare_ranges]
        cut_points.append(fare_ranges[-1][-1])
        label_names = [labels for labels, *_ in fare_ranges]
    
    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels=label_names)
    return df


# In[117]:


train = process_fare(train)
holdout = process_fare(holdout)
# Create dummy columns for newly Fare bins    
train = create_dummies(train, "Fare_categories")
holdout = create_dummies(holdout, "Fare_categories")


# In[118]:


train.columns


# In[119]:


# To construct list of all possible feature columns
train_original_columns = train_original_columns.tolist()
train_original_columns.append("Fare_categories")

features = train.drop(columns=train_original_columns).columns
features


# ## Engineering features from text columns
# Looking at the Name column, There is a title like 'Mr' or 'Mrs' within each, as well as some less common titles, like the 'Countess' from the final row of our table above. By spending some time researching the different titles, we can categorize these into six types:
# 
# - Mr
# - Mrs
# - Master
# - Miss
# - Officer
# - Royalty
# 
# We can use the `Series.str.extract` [method](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.extract.html) and a [regular expression](https://en.wikipedia.org/wiki/Regular_expression) to extract the title from each name and then use the `Series.map()` method and a predefined dictionary to simplify the titles.
# 

# In[120]:


def create_titles_n_cabins(df):
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }

    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)

    # Form a new feature from cabin by using the first letter 
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    
    df = create_dummies(df, "Title")
    df = create_dummies(df, "Cabin_type")
    return df 


# In[121]:


train = create_titles_n_cabins(train)
holdout = create_titles_n_cabins(holdout)
train_original_columns.append("Cabin_type")
train_original_columns.append("Title")


# In[122]:


train["Title"].value_counts()


# In[123]:


pd.pivot_table(train, values="Survived", index="Title")


# ## Finding correlated features

# In[124]:


features = train.drop(columns=train_original_columns).columns
features


# We now have 34 possible feature columns we can use to train our model. One thing to be aware of as you start to add more features is a concept called collinearity. Collinearity occurs where more than one feature contains data that are similar.
# 
# The effect of collinearity is that your model will overfit - you may get great results on your test data set, but then the model performs worse on unseen data (like the holdout set).
# 
# One easy way to understand collinearity is with a simple binary variable like the Sex column in our dataset. Every passenger in our data is categorized as either male or female, so 'not male' is exactly the same as 'female'.
# 
# As a result, when we created our two dummy columns from the categorical Sex column, we've actually created two columns with identical data in them. This will happen whenever we create dummy columns, and is called the [dummy variable trap](https://www.algosome.com/articles/dummy-variable-trap-regression.html). The easy solution is to choose one column to drop any time you make dummy columns.

# In[125]:


import numpy as np
import seaborn as sns

def plot_correlation_heatmap(df):
    corr = df.corr()
    
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)


    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


# Collinearity can happen in other places, too. A common way to spot collinearity is to plot correlations between each pair of variables in a heatmap. An example of this style of plot is below:

# In[126]:


plot_correlation_heatmap(train[features])


# The darker squares, whether the darker red or darker blue, indicate pairs of columns that have higher correlation and may lead to collinearity. The easiest way to produce this plot is using the DataFrame.corr() method to produce a correlation matrix, and then use the Seaborn library's seaborn.heatmap() function to plot the values.
# 
# 
# ## Final selection using RFECV
# We can see that there is a high correlation between `Sex_female`/`Sex_male` and `Title_Miss`/`Title_Mr`/`Title_Mrs`. We will remove the columns `Sex_female` and `Sex_male` since the title data may be more nuanced.
# 
# Apart from that, we should remove one of each of our dummy variables to reduce the collinearity in each. We'll remove:
# - Pclass_2
# - Age_categories_Teenager
# - Fare_categories_12-50
# - Title_Master
# - Cabin_type_A
# 
# n an earlier step, we manually used the logit coefficients to select the most relevant features. An alternate method is to use one of scikit-learn's inbuilt feature selection classes. We will be using the [feature_selection.RFECV class](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) which performs recursive feature elimination with cross-validation.
# 
# The RFECV class starts by training a model using all of your features and scores it using cross validation. It then uses the logit coefficients to eliminate the least important feature, and trains and scores a new model. At the end, the class looks at all the scores, and selects the set of features which scored highest.
# 
# Like the LogisticRegression class, RFECV must first be instantiated and then fit. The first parameter when creating the RFECV object must be an estimator, and we need to use the cv parameter to specific the number of folds for cross-validation.
# 
# ```python
# from sklearn.feature_selection import RFECV
# lr = LogisticRegression()
# selector = RFECV(lr,cv=10)
# selector.fit(all_X,all_y)
# ```
# 
# Once the RFECV object has been fit, we can use the RFECV.support_ attribute to access a boolean mask of True and False values which we can use to generate a list of optimized columns:
# ```python
# optimized_columns = all_X.columns[selector.support_]
# ```

# In[127]:


to_drop = ["Pclass_2", "Age_categories_Teenager", "Fare_categories_12-50", "Title_Master", "Cabin_type_A"]
features = features.drop(to_drop)
features


# In[128]:


from sklearn.feature_selection import RFECV
original_features = features
lr = LogisticRegression(solver='liblinear')
selector = RFECV(estimator=lr, cv=10)
selector.fit(train[features], train[target])
optimized_features = train[features].columns[selector.support_]
print("{} features were removed.".format(len(original_features)-len(optimized_features)))
print("Features removed are:", [col for col in original_features if col not in optimized_features])
print("Optimized {} features are: {}".format(len(optimized_features), optimized_features))


# In[129]:


lr = LogisticRegression(solver='liblinear')
scores = cross_val_score(lr, train[optimized_features], train[target], cv=10)
accuracy = mean(scores)
accuracy


# This 18-feature model scores 82.6%, a modest improvement compared to the 81.5% from our earlier model. Let's train these columns on the holdout set, save a submission file and see what score we get from Kaggle.

# In[130]:


lr = LogisticRegression(solver='liblinear')
lr.fit(train[optimized_features], train[target])
holdout_predictions = lr.predict(holdout[optimized_features])
submission(holdout, holdout_predictions)


# In[ ]:




