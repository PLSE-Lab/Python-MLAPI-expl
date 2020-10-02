#!/usr/bin/env python
# coding: utf-8

# # Random Forest and Logistic regression on Titanic dataset
# Before we perform the model evaluation we shall analyse the dataset. Also, clean the data and change to the right formats for the modeling 

# In[ ]:


#importing required packages and data
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")


# ## Glimpse the data
# Let's analyse the features available for analysis.

# In[ ]:


print("Training data has {} rows and {} columns".format(data_train.shape[0], data_train.shape[1]))
data_train.head()


# - **PassengerId**: The unique id assigned for each person
# - **Survived**: The target feature, which specifies whether the person survived or not
# - **Pclass**: The ticket class (which is synonymous to socio-economic class)
# - **Name**: Name of the person
# - **Sex**: Gender
# - **Age**: The age of the individual
# - **SibSp**: number of siblings / spouses aboard the Titanic
# - **Parch**: number of parents / children aboard the Titanic
# - **Ticket**: ticket number
# - **Fare**: ticket cost
# - **Cabin**: Cabin number
# - **Embarked**: The port at which the individual embarked<br>
# <br>
# From the above features **Ticket** and **Name** columns doesn't give much sense to the target variable. However, we can derive the salutaion of the person (Mr, Mrs etc.), which may give insight like social status of the person, married/ unmarried (in females) etc.

# In[ ]:


print("Test data has {} rows and {} columns".format(data_test.shape[0], data_test.shape[1]))
data_test.head()


# ## Check for missing values and cleaning data

# In[ ]:


data_train.isnull().sum()


# In[ ]:


data_test.isnull().sum()


# From the dataset we shall drop the following features:
# - PassengerId : As it is a unique ID for each person it doesnt play any role for prediction
# - Ticket: Again due to unique ticket number being assigned for each ticket ticket number would not present any insight
# - Cabin: This feature is being dropeed due to the very high missing values, especially in the test dataset
# - Name: The name of the person doesn't give any value.However the edsignation might provide some insight. So Name would be dropped after deriving the salutation info

# In[ ]:


drop_cols = ["PassengerId", "Ticket", "Cabin"]
data_train.drop(drop_cols, inplace=True, axis=1, errors="ignore")
data_test.drop(drop_cols, inplace=True, axis=1, errors="ignore")
print(data_train.info())
print("==================================")
print(data_test.info())


# There are still missing values in **Age** and **Embarked** in training set and **Fare** in test set. Since **embarked** is a categorical feature we shall impute the values with the mode of the data. For **Age** we shall use the Title category, which would be derived from the name to impute the median age for each category. For **Fare** the value can be filled with the median value.

# In[ ]:


#imputing embarked missing values in training data
data_train.loc[data_train["Embarked"].isnull(),"Embarked"] = data_train["Embarked"].mode()[0]
data_test.loc[data_test["Fare"].isnull(),"Fare"] = data_test["Fare"].median()
print("The 2 missing values in Embarked feature has been filled with {}".format(data_train["Embarked"].mode()[0]))
print("The 1 missing values in Fare feature has been filled with {}".format(data_test["Fare"].median()))


# In[ ]:


#Creating derived feature "Title"
combined = [data_train, data_test]
for dataset in combined:
    dataset["Title"] = dataset["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
pd.concat([data_train["Title"], data_test["Title"]]).value_counts()


# We shall consider the titles that appear frequently (Mr, Miss, Mrs and Master) and categorize all others into category called *Other*.

# In[ ]:


valid_title = ["Mr","Miss","Mrs","Master"]
data_train["Title"] = data_train.Title.apply(lambda x: x if x in valid_title else "Other")
data_test["Title"] = data_test.Title.apply(lambda x: x if x in valid_title else "Other")
pd.concat([data_train["Title"], data_test["Title"]]).value_counts()


# In[ ]:


#Drop the name column
data_train.drop("Name", inplace=True, axis=1, errors="ignore")
data_test.drop("Name", inplace=True, axis=1, errors="ignore")


# Lets analyse the age distribution for each of the Title.

# In[ ]:


f,axes= plt.subplots(1,2, figsize=(10,5))
p1 = sns.boxplot(x="Title", y="Age", data=data_train, ax=axes[0])
p2 = sns.boxplot(x="Title", y="Age", data=data_test, ax=axes[1])
p1.set(title="Training data")
p2.set(title="Test data")


# We see a similar pattern for each category in both train and test dataset. So we shall impute them with the median ages in both the data sets.

# In[ ]:


for dataset in combined:
    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Mr"),"Age"] = dataset.loc[dataset["Title"]=="Mr", "Age"].median()
    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Mrs"), "Age"] = dataset.loc[dataset["Title"]=="Mrs", "Age"].median()
    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Miss"), "Age"] = dataset.loc[dataset["Title"]=="Miss", "Age"].median()
    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Master"),"Age"] = dataset.loc[dataset["Title"]=="Master", "Age"].median()
    dataset.loc[(dataset["Age"].isnull()) & (dataset["Title"]=="Other"), "Age"] = dataset.loc[dataset["Title"]=="Other", "Age"].median()

print(data_train.isnull().sum())
print("================================")
print(data_test.isnull().sum())


# In[ ]:


data_train.head()


# For numerical features (**Age** and **Fare**) we shall bin them, rather than using the absolute values.

# In[ ]:


f,axes = plt.subplots(1,2, figsize=(20,5))
p1=sns.boxplot(data=data_train, x="Age", ax=axes[0])
p2=sns.boxplot(data=data_train, x="Fare", ax=axes[1])
p2.set_xlim(right=100)


# We have age ranging from 0 to 80. Thus we can group them as 0-5,5-15,15-24,24-35,35-45,45-55,55-65,65-90. <br>
# For **Fare** we see outliers in the data. Since we have less data we shall not remove the rows from the data, instead we shall cap it while we bin the values.

# In[ ]:


def bin_age(data):
    labels = ("0-5","5-15","15-24","24-35","35-45","45-55","55-65","65-90")
    bins = (0,5,15,24,35,45,55,65,90)
    data["Age"] = pd.cut(data.Age, bins, labels=labels)
    
def bin_fare(data):
    labels = ("very_low", "low","moderate","high","very_high")
    bins=(-1,10,15,30,50,700)
    data["Fare"] = pd.cut(data.Fare, bins, labels=labels)


# In[ ]:


datasets = [data_train, data_test]
for dataset in datasets:
    bin_age(dataset)
    bin_fare(dataset)


# In[ ]:


data_train.head()


# ## Data Exploration
# Now that data has been cleaned and features has been derived, we shall anayse how each of the variables relate to the target variable *Survived* and how the variables relate to each other.

# In[ ]:


def plot_graph(data, x, y, hue=None):
    if hue==None:
        f,axes = plt.subplots(1,2, figsize=(15,5))
        sns.barplot(data=data, x=x, y=y, ax=axes[0])
        sns.countplot(data=data, x=x, ax=axes[1])
    else:
        f,axes = plt.subplots(1,2,figsize=(15,5))
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=axes[0])
        sns.countplot(data=data, x=x, hue=hue, ax=axes[1])


# ### Gender
# As we know from the movies children and women were the first one to be evacuated. Thus we can expect a strong relation between the gender and survival rate.

# In[ ]:


plot_graph(data_train, "Sex","Survived")


# ### Age

# In[ ]:


plot_graph(data_train,"Age","Survived", "Sex")


# As expected Males survival rate is significantly less than that of Female. Also age has an important role to play in the survival.

# ### Pclass

# In[ ]:


plot_graph(data_train, x="Pclass", y="Survived", hue="Sex")


# The **1st** class people has higher survival rate, followed by **2nd** and **3rd** class people. So it can be infered that the higher class people had access to the evaculation facilities than the lower class.

# ### Title
# From the graph below married women had slightly higher chances of survival compared to unmarried women.

# In[ ]:


plot_graph(data=data_train, x="Title",y="Survived")


# ### Embarked

# In[ ]:


plot_graph(data_train, "Embarked", "Survived")


# People embarked from port **C** seems to have higher chances of survival compared to other ports (15-20% higher)

# ### Fare

# In[ ]:


plot_graph(data_train,"Fare","Survived")


# In[ ]:


plot_graph(data_train,"Fare","Survived","Pclass")


# Even though the Fare may seem to have a relation to the survival rate in the first graph, the pattern is not evident when we group based on Pclass. Even some of the lower class people have paid hugher fares for the trip. However the survival rate is pretty much same for a particulat class irrespective of the fare they paid. So survival would be closely related to Pclass rather than Fare.

# ## Model
# Let's move to building a model for the prediction of titanic disaster survival
# - Random Forest model
# - Logistic Regresion model

# ## Random Forest model
# 
# The slected features would be used to train the model. Before we proceed we shall shuffle the data randomly.

# In[ ]:


#shuffling the data
data_train = data_train.reindex(np.random.permutation(data_train.index))

features = ['Pclass','Sex','Age','SibSp','Parch','Embarked','Title']
data_X = data_train[features]
test_X = data_test[features]
data_Y = data_train["Survived"]
data_X.head()


# Since we are using a Random Forest model (tree based) we shall label encode the categrical featues. On hot encoding my reduce the performance of the tree based models (ref: https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769)

# ### Encoding

# In[ ]:


from sklearn import preprocessing

def encode_features(data_X,test_X):
    features_to_label = ["Sex","Age","Embarked","Title"]
    combined_X = pd.concat([data_X[features_to_label],test_X[features_to_label]])
    for feature in features_to_label:
        encoder = preprocessing.LabelEncoder()
        encoder = encoder.fit(combined_X[feature])
        data_X[feature] = encoder.transform(data_train[feature])
        test_X[feature] = encoder.transform(test_X[feature])
    return (data_X, test_X)

data_X,test_X = encode_features(data_X,test_X)
data_X.head()


# ### Create vlaidation set
# To validate the model as it runs we shall split the data into train and validation set. 15% of data would be used for validation.

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size=0.15, shuffle=False)
print("Training set shape : {}".format(x_train.shape))
print("Validation set shape : {}".format(x_val.shape))


# ### Model and tuning

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

model = RandomForestClassifier()

#choosing a set of parameters to try on
params = {
    'n_estimators':[4,5,9,64,100,250],
    'criterion':['gini','entropy'],
    'max_features':['sqrt','log2',None],
    'max_depth':[4,8,16,32,None],
    'max_depth':[3,5,7,9,11]
}

accuracy_scorer = make_scorer(accuracy_score)

search_params = GridSearchCV(model, params, scoring=accuracy_scorer)
search_params = search_params.fit(x_train, y_train)

model = search_params.best_estimator_
print("The model has an accuracy of {}".format(search_params.best_score_))
model.fit(x_train, y_train)


# In[ ]:


predict = model.predict(x_val)
print("The validation set accuracy is {}".format(accuracy_score(y_val, predict)))


# ## Logistic Regression
# For logistic regression we shall use one-hot encoding of the categorical features. Label encoding them might case bias.

# ### One-hot encoding

# In[ ]:


features = ['Pclass','Sex','Age','SibSp','Parch','Embarked','Title']
lr_data_X = data_train[features]
lr_test_X = data_test[features]
lr_data_Y = data_train["Survived"]
lr_data_X.head()


# In[ ]:


pd.set_option('display.max_columns',100)
def lr_encoding(data_X,test_X):
    len_train = len(data_X)
    features_to_label = ["Pclass","Sex","Age","Embarked","Title"]
    combined_X = pd.concat([data_X[features_to_label],test_X[features_to_label]])
    combined_X = pd.get_dummies(combined_X, columns=features_to_label)
    data_X = combined_X.iloc[0:len_train,:]
    test_X = combined_X.iloc[len_train:,:]
    return(data_X, test_X)

lr_data_X,lr_test_X = lr_encoding(data_X,test_X)
lr_data_X.head()


# ### splitting data

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(lr_data_X, lr_data_Y, test_size=0.15, shuffle=False)
print("Training set shape : {}".format(x_train.shape))
print("Validation set shape : {}".format(x_val.shape))


# ### Model

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='liblinear')
params = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty':["l1","l2"],
    'max_iter':[100,150,200,300,400,500]
}

accuracy_scorer = make_scorer(accuracy_score)

search_params = GridSearchCV(lr_model, params, scoring=accuracy_scorer)
search_params = search_params.fit(x_train, y_train)

lr_model = search_params.best_estimator_
print("The training set accuracy is {}".format(search_params.best_score_))
lr_model.fit(x_train, y_train)


# In[ ]:


predict = lr_model.predict(x_val)
print("The validation set accuracy is {}".format(accuracy_score(y_val, predict)))


# #### Predicting test data
# Since the Random Forest model has slightly higher accuracy than the Logistic Regression model we shall use the random forest model to make the predictions for the test data.

# In[ ]:


p_id = pd.read_csv("../input/test.csv")['PassengerId']
predict = model.predict(test_X)

out = pd.DataFrame({'PassengerId' : p_id, 'Survived': predict})

out.head()


# # References
# - Scikit-Learn ML from Start to Finish (https://www.kaggle.com/jeffd23/scikit-learn-ml-from-start-to-finish)
# - EASY ML WALKTHROUGH (Regression/SVM/KNN)(https://www.kaggle.com/jojothebufferlo/easy-ml-walkthrough-regression-svm-knn)
