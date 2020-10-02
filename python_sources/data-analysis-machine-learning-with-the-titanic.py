#!/usr/bin/env python
# coding: utf-8

# # Exploring/analysing the Titanic dataset with Python and Logistic Regression for survival estimation

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#read the datasets
train = pd.read_csv("../input/train.csv")
holdout = pd.read_csv("../input/test.csv")


# In[ ]:


#lets describe our train df...

columns = ['SibSp','Parch','Fare','Cabin','Embarked']
train[columns].describe(include='all',percentiles=[])


# In[ ]:


#and holdout df

holdout[columns].describe()


# ### Do some exploratory visualisation

# In[ ]:


chance_survive = len(train[train["Survived"] == 1]) / len(train["Survived"])


# #### The probability of surviving the Titanic was only 0.38. We can plot the general probability against each category

# In[ ]:


def plot_survival(df, index, color = "blue", use_index = True, num_xticks = 0, xticks = "", position = 0.5, legend = 
                  ["General probabilty of survival"]):
    df_pivot = df.pivot_table(index=index,values="Survived")
    df_pivot.plot.bar(ylim = [0,1], color = color, use_index = use_index, position = position)
    plt.axhline(chance_survive, color = "red", linewidth = 1)
    if num_xticks>0:
        plt.xticks(range(num_xticks), xticks)
    plt.legend(legend)
    plt.title("Plotting the survival probability by "+index+"\n")


# In[ ]:


plot_survival(train, "Sex", color = ["pink", "blue"], use_index = False, num_xticks = 2, xticks = ['Female', 'Male'])


# In[ ]:


plot_survival(train,"Pclass", color = "blue", use_index = True)


# In[ ]:


plot_survival(train,'Parch', "red", True , position = 0.3)
plot_survival(train,'SibSp', "blue", True)


# #### The "age" column contains too many values ; we can group by age class

# In[ ]:


cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df


# In[ ]:


train = process_age(train,cut_points,label_names)
holdout = process_age(holdout,cut_points,label_names)


# ### Visualising our new age features

# In[ ]:


plot_survival(train,"Age_categories", use_index = False, num_xticks = len(train["Age_categories"].unique()), 
              xticks = train["Age_categories"].unique().sort_values())


# ## Process the fare feature column. Adapt the process_age column seen earlier to the fare column

# In[ ]:


def process_fare(df, cut_points, label_names):
    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels = label_names)
    return df


# In[ ]:


train = process_fare(train, [0,12,50,100,1000], ["0-12$","12-50$","50-100$","100+$"])


# In[ ]:


holdout = process_fare(holdout, [0,12,50,100,1000], ["0-12$","12-50$","50-100$","100+$"])


# ### Visualising our new fare features

# In[ ]:


plot_survival(train,"Fare_categories", use_index = False, num_xticks = len(train["Fare_categories"].unique())-1, 
              xticks = train["Fare_categories"].unique().sort_values())


# ### Create a dummy coding function

# In[ ]:


def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# In[ ]:


#create dummies for the Age_cat, Class, Sex, Embarked features & Fare

for col in ["Age_categories", "Pclass", "Sex", "Embarked", "Fare_categories"]:
    train = create_dummies(train, col)
    holdout = create_dummies(holdout, col)


# ### Data preprocessing : Fare and Embarked columns missing data

# In[ ]:


# the holdout DF has one missing value for the Fare feature. We can easliy replace it by the mean of that column.
# Same for the embarked column. We can easily replace it by the most common value, "S"

holdout["Fare"] = holdout["Fare"].fillna(train["Fare"].mean())
train["Embarked"] = train["Embarked"].fillna("S")
holdout["Embarked"] = holdout["Embarked"].fillna("S")


# ### Data preprocessing : Scaling down our values
# #### Scale our numerical columns so that they have the same range and does not falsely influence our prediction.

# In[ ]:


from sklearn.preprocessing import minmax_scale

cols = ["SibSp", "Parch", "Fare"]
new_cols = ["SibSp_scaled", "Parch_scaled", "Fare_scaled"]

for col, new_col in zip(cols, new_cols):
    train[new_col] = minmax_scale(train[col])
    holdout[new_col] = minmax_scale(holdout[col])


# In[ ]:


# our SibSp and Parch got converted to floats when scaled, we should keep that in mind 

dtypes = train[cols + new_cols].dtypes


# ### Exploratory analysis : plotting correlations coefficients for each feature

# #### Let's explore the relative importance of each feature thanks to regression coefficients. First, we define which columns we will explore

# In[ ]:


columns = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled']

columns_not_scaled = ['Age_categories_Missing', 'Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'SibSp', 'Parch', 'Fare']


# #### Then fit our model to the data and get the coefs for each feature

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(train[columns], train["Survived"])
coefficients = lr.coef_
feature_importance = pd.Series(coefficients[0], index = train[columns].columns)

lr.fit(train[columns_not_scaled], train["Survived"])
coefficients2 = lr.coef_
feature_importance_2 = pd.Series(coefficients2[0], index = train[columns_not_scaled].columns)


# #### And plot it by descending order

# In[ ]:


ordered_feature_importance = feature_importance.abs().sort_values()
#ordered_feature_importance_2 = feature_importance_2.abs().sort_values()
ordered_feature_importance.plot.barh(color = "blue")
#ordered_feature_importance_2.plot.barh(color = "red")
#plt.legend(['Scaling', 'No scaling'], loc = 4)


# ###### The 5th first features are the most important in the regression process

# ## Create a feature that gets the title of the person

# In[ ]:


def get_titles(train, test):
    titles_train = train["Name"].str.split(".", expand = True)
    titles_test = test["Name"].str.split(".", expand = True)
    titles_train = titles_train[0].str.split(",", expand = True)
    titles_test = titles_test[0].str.split(",", expand = True)
    titles_train = titles_train[1]
    titles_test = titles_test[1]
    train["titles"] = titles_train.astype("category")
    test["titles"] = titles_test.astype("category")
    #train = train.drop("Name", axis = 1)
    #test = test.drop("Name", axis = 1)
    return train, holdout


# In[ ]:


train, holdout = get_titles(train,holdout)


# ### Plot the chance of survival by title

# In[ ]:


import numpy as np


# In[ ]:


plot_survival(train,"titles", use_index = False, num_xticks = len(train["titles"].unique())-1, 
              xticks = train["titles"].unique().sort_values())


# In[ ]:


train = create_dummies(train,"titles")
holdout = create_dummies(holdout,"titles")


# #### We can see that the feminine titles (Mlle, mme) have a one-on-one chance of survival based on the training set. On the other hand. Creating this feature has allowed us to have a more in depth understanding of the logic behind the lifeboat management.

# ## Now engineer the Cabin feature

# In[ ]:


print("number of null values :",train["Cabin"].isnull().sum())
print("number of non_null values :",train["Cabin"].notnull().sum())


# ### We have a lot of missing columns.

# #### We can get the type of Cabin based on the first letter of the variables

# In[ ]:


train["Cabin"] = train["Cabin"].fillna("unknown")
holdout["Cabin"] = holdout["Cabin"].fillna("unknown")


# In[ ]:


cabins = train["Cabin"].tolist()
cabins_h = holdout["Cabin"].tolist()


# #### Extract the first letter of each variable with the string method i[0:1]

# In[ ]:


cabins_type = []
for i in cabins:
    cabins_type.append(i[0:1])
    
cabins_type_holdout = []
for i in cabins_h:
    cabins_type_holdout.append(i[0:1])


# In[ ]:


train["Cabin"] = cabins_type
holdout["Cabin"] = cabins_type_holdout


# In[ ]:


train = create_dummies(train, "Cabin")
holdout = create_dummies(holdout, "Cabin")


# In[ ]:


plot_survival(train,"Cabin", use_index = True)


# #### The Cabin Type feature does not appear to give any particular insight on survival probability.

# In[ ]:


print("We have now ",len(train.columns), "columns as predictors to fit")


# ### Now let's fit a logistic regression model. 
# #### First define the columns we use as predictors

# In[ ]:


columns_cabins_titles = ['Age_categories_Missing', 'Age_categories_Infant',
'Age_categories_Child', 'Age_categories_Teenager',
'Age_categories_Young Adult', 'Age_categories_Adult',
'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
'Sex_female', 'SibSp_scaled', 'Parch_scaled',
'Fare_categories_0-12$', 'Fare_categories_12-50$',
'Fare_categories_50-100$', 'Fare_categories_100+$', 'titles_ Capt', 'titles_ Col', 'titles_ Don',
'titles_ Dr', 'titles_ Jonkheer', 'titles_ Lady', 'titles_ Major',
'titles_ Master', 'titles_ Miss', 'titles_ Mlle', 'titles_ Mme',
'titles_ Mr', 'titles_ Mrs', 'titles_ Ms', 'titles_ Rev', 'titles_ Sir',
'titles_ the Countess', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D',
'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_u']

other_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3']


# #### Then define the actual model

# In[ ]:


from sklearn import model_selection


# In[ ]:


from sklearn.linear_model import LogisticRegression


# ### Let's plot the regression coefficients : regression coefficients are the importance of features in a model

# In[ ]:


logreg = LogisticRegression()


logreg.fit(train[columns_cabins_titles], train["Survived"])
feature_importance_2 = logreg.coef_


# In[ ]:


feature_importance_2 = pd.Series(feature_importance_2[0], index = train[columns_cabins_titles].columns)
ordered_feature_importance = feature_importance_2.abs().sort_values()


# In[ ]:


ordered_feature_importance.plot.barh(color = "blue", figsize = (10,10))


# ## Now let's look at possible colinearity between our features

# Colinearity is when two or more feature provide the same information. It happens when you have : 
#     - dummy variables
#     - similar information (for example, Sex & Titles can provide the same information)
#     
# Colinearity leads to overfitting.
# 
# We can plot a Correlation heatmap for a dataset to see if there is collinearity.

# In[ ]:


columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior', "titles_ Col", "titles_ Don", "titles_ Dr", "titles_ Master", "titles_ Miss", 
        "titles_ Mr", "titles_ Mrs", "titles_ Ms", "titles_ Rev"]


# In[ ]:


columns_cabins_titles = ['Age_categories_Missing', 'Age_categories_Infant',
'Age_categories_Child', 'Age_categories_Teenager',
'Age_categories_Young Adult', 'Age_categories_Adult',
'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',
'Sex_female', 'SibSp_scaled', 'Parch_scaled',
'Fare_categories_0-12$', 'Fare_categories_12-50$',
'Fare_categories_50-100$', 'Fare_categories_100+$', 'titles_ Capt', 'titles_ Col', 'titles_ Don',
'titles_ Dr', 'titles_ Jonkheer', 'titles_ Lady', 'titles_ Major',
'titles_ Master', 'titles_ Miss', 'titles_ Mlle', 'titles_ Mme',
'titles_ Mr', 'titles_ Mrs', 'titles_ Ms', 'titles_ Rev', 'titles_ Sir',
'titles_ the Countess', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D',
'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T']


# In[ ]:


import seaborn as sns


# In[ ]:


def plot_correlation_heatmap(df):
    corr = df.corr()
    
    sns.set(style="white")
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(520, 10, as_cmap=True)


    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


# In[ ]:


corrs = train[columns_cabins_titles].corr()


# In[ ]:


plot_correlation_heatmap(corrs.corr())


# ### Now that we have all our coefficients, we can manually select columns that have a too high collinearity with one another. We can also use scikit learn to do that job for us

# In[ ]:


from sklearn.feature_selection import RFECV
lr = LogisticRegression()
selector = RFECV(lr,cv=10)
selector.fit(train[columns_cabins_titles],train["Survived"])


# In[ ]:


optimized_columns = train[columns_cabins_titles].columns[selector.support_]


# In[ ]:


print("We only have", len(optimized_columns), "columns left for our model. Let's test it")


# In[ ]:


lr = LogisticRegression()
scores = model_selection.cross_val_score(lr, train[optimized_columns], train["Survived"], cv=10)
accuracy_optimized = scores.mean()


# In[ ]:


lr = LogisticRegression()
scores = model_selection.cross_val_score(lr, train[columns_cabins_titles], train["Survived"], cv=10)
accuracy_n_optimized = scores.mean()


# In[ ]:


print("We have an accuracy of ", accuracy_optimized, "with optimized predictors", "vs ", accuracy_n_optimized, "before")


# In[ ]:


#Before fitting our model to our real holdout dataset we should remove the "Captain" column since it does not exist in the 
#holdout.

optimized_columns = optimized_columns.drop("titles_ Capt")


# In[ ]:


lr = LogisticRegression()
lr.fit(train[optimized_columns],train["Survived"])
holdout_predictions = lr.predict(holdout[optimized_columns])


# In[ ]:


holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv("submission_2.csv",index=False)


# ### Our score on the test set is about 78% which is quite good considering the fact that we used a fairly simple logistic regression model and that we did not use the totality of the features at disposition.

# In[ ]:




