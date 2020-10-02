#!/usr/bin/env python
# coding: utf-8

# ### Titanic EDA & ML

#    In this notebook, I will be study on Titanic Dataset on Kaggle. Since this is my first time on EDA & ML notebook on kaggle, I chose the most worked on dataset on Kaggle. So that, I am very glad to share this notebook with everyone and be thankful anyone who were shared on Kaggle which was really helpful for me.
# 
#    So firstly, I wil be focusing on the data in general. Then, I will work on the features separately. While I am working on features, I make my own dataset for ML techniques which I will focusing on ML part. So that, sorry for that disorganized structure which you maybe need to jump from branch(feature) to branch(feature) :).
#    So lets import our libraries first.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# preprocessing
import sklearn
from sklearn.preprocessing import LabelEncoder

# ML
from sklearn. model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# Then, import our dataset as well.

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
df_set = [train, test]


# Lets check them briefly

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe().T


# For the first impressions on the dataset, has consists of 891 rows and 12 features with some of the features has missing values. We will look at those in **Missing Values** part. 
# 
# We can also make some analysis before we look at the dataset thoroughly.
#     * PassengerId looks like a unique for each individual when we look at that values.
#     * Age derivated between 15 to 40 so that most of the passengers are young.
#     * Fare has very wide range and it can be useful for visualization and analyzing.
#     * Pclass looks like a categorical variable which can be take 1, 2 or 3 values and 3 are more than other two values( %50 and %75 are the
#     same).So that it can use also to analyzing the dataset. 
#     

# ## Correlations

# In order to understand the relationship between features, we can look at the correlations by using heatmap.

# In[ ]:


# Correlation matrix
heatmap = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# In[ ]:


pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))


# So that, when we looked at the correlations, we can say that Pclass and Fare are most important features for Survived target feature.

# In[ ]:


train.info()


# When we are looking at the info of our data, we can see that there are some features which need to be transformed into other data types but for now, lets look at the missing values first. 

# ## Missing Values

# When we are looking at the dataset, we can say that "age" feature has some missing values. However, lets see that in a visual way to see it more clearly.

# In[ ]:


msno.matrix(train, figsize = (30,10));


# As we expect, Age feature has many missing values by looking at the graph which is above. In addition, our Cabin feature has lots of missing values as well. So lets see is there any correlation between Age and Cabin features

# In[ ]:


msno.heatmap(train);


# When we looked at the nullity correlation which is the graph at above, we can see there is 0.1 correlation between Cabin and Age features which can be ignored. Thus, We do not consider both of them are together while we are filling the missing values at each part.

# In[ ]:


train.isnull().sum()


# For a beginning, it is okay to leave it there and try to analyze features first. While we are analysing the data, we can make our own dataset for the prediction after our EDA.

# ## EDA

# ### Survived

# **Survived** feature is consist of two integer which are 0 is referring to the people who are not survived and 1 is referring to the people who are survived at the disaster. Thus, there is no need to change in this feature. Lets just visualize and see the number of survivals.

# In[ ]:


# Survived
fig = plt.figure( figsize = (17,1))
sns.countplot( y = "Survived", data = train);
print(train.Survived.value_counts())


# Unfortunately, there are many people who are not survived at the disaster. Best thing we can do for them is to make sure this type of accidents will never happen again. This is also why we are doing this kind of analysis. 

# ### Pclass

# Pclass is reffering to socieconomic class of passengers. Lets see how many class on dataset.

# In[ ]:


# Pclass 
train.Pclass.unique()


# In[ ]:


# Barplot for categorical feature
print(train["Survived"][train['Pclass'] == 1].value_counts(normalize = True)[1])
print(train["Survived"][train['Pclass'] == 2].value_counts(normalize = True)[1])
print(train["Survived"][train['Pclass'] == 3].value_counts(normalize = True)[1])
sns.barplot(x = "Pclass", y = "Survived", data = train);


# Sadly, we can say that socioeconomic condition of a passenger has a large impact of him/her chance of survival.

# ### Sex

# Sex feature is referring to the gender of the passengers. 

# In[ ]:





# In[ ]:


a = train["Sex"].value_counts()
colors = ['#ff9999', '#ffcc99']
genders = ["female","male"]
plt.pie(a, labels= genders, colors = colors, startangle=90, autopct='%.1f%%', shadow = True )
plt.tight_layout()
plt.show()


# In[ ]:


sns.barplot(x = "Sex", y = "Survived", data = train);
print("Probability of Survived Male Passenger: ", train["Survived"][train['Sex'] == "male"].value_counts(normalize = True)[1])
print("Probability of Survived Female Passenger: ", train["Survived"][train['Sex'] == "female"].value_counts(normalize = True)[1])


# Even though, there are more number of female passenger than male passenger, we can see that females have more chance to survive than males have. Also, its understandable when we are thinking gender difference when a disaster occurs. Women and children has precision at evacuation.

# In[ ]:


for dataset in df_set:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# At the above, we mapped our feature as integer.

# ### Age

# Age is referring to ages of the passengers as you can predict. However, we know Age feature has many missing values in dataset. So that, we need to fill these missing values with respect to correlated features

# In[ ]:


train['Age'].hist(bins=30, color='red',alpha=0.8, edgecolor='black');


# In[ ]:


train.Age.value_counts()


# In[ ]:


train["Age"].isnull().sum()


# Age can be useful for making some prediction like in disasters "Children first" sentences are used during the evacuations.

# In[ ]:


heatmap = sns.heatmap(train[["Survived","Sex","Pclass","Age","Fare","SibSp","Parch"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# When we are looking at the heatmap, we can see the most correlated features with Age are Pclass and SibSp. So that, we can use Pclass or SibSp for filling the missing values in Age feature.

# In[ ]:


print(train[["Pclass","Age"]].groupby(["Pclass"], as_index = False).median())
sns.boxplot(x = "Pclass", y= "Age" , data = train);


# In[ ]:


print(train[["SibSp","Age"]].groupby(["SibSp"], as_index = False).median())
sns.boxplot(x = "SibSp", y= "Age" , data = train);


# In order to fill the missing values, we will use median of **Age** and **SibSp** features which are get from boxplot at above.

# In[ ]:


for dataset in df_set:
    miss_age_index = list(dataset["Age"][dataset["Age"].isnull()].index)
    for i in miss_age_index :
        median_age= dataset["Age"].median()
        fill_age = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(fill_age) :
            dataset['Age'].iloc[i] = fill_age
        else :
            dataset['Age'].iloc[i] = median_age
    dataset['Age'] = dataset['Age'].astype(int)


# In[ ]:


train["Age"].isnull().sum()
train['Age'].hist(bins=30, color='red',alpha=0.8, edgecolor='black');


# In[ ]:


sns.factorplot(x="Survived", y = "Age", data = train, kind="violin");


# As we can see it more clearly in violin plot, the more younger people are survived in disaster.

# In[ ]:


# After we filled the missing values, we can convert it into integer which was float data type and make binning.
train['Age_bin'] = pd.cut(train['Age'], 5)
train[['Age_bin', 'Survived']].groupby(['Age_bin'], as_index=False).mean().sort_values(by='Age_bin', ascending=True)


# In[ ]:


for dataset in df_set:
   dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
   dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), "Age"] = 1
   dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), "Age"] = 2
   dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), "Age"] = 3
   dataset.loc[ dataset['Age'] > 64, "Age"] = 4

train = train.drop(["Age_bin"], axis = 1)
df_set = [train, test]


# In[ ]:


print(train["Survived"][train["Age"] == 0 ].value_counts(normalize = True)[1])
print(train["Survived"][train["Age"] == 1].value_counts(normalize = True)[1])
print(train["Survived"][train["Age"] == 2 ].value_counts(normalize = True)[1])
print(train["Survived"][train["Age"] == 3 ].value_counts(normalize = True)[1])
print(train["Survived"][train["Age"] == 4 ].value_counts(normalize = True)[1])
sns.barplot(x = train["Age"], y = train["Survived"], data = train );


# In[ ]:


sns.barplot(x = train["Age"], y = train["Survived"], hue = train["Sex"], data = train);


# When we looked at the barplot, we can see the effect of **Sex** feature clearly. Ratio between gender are increased rapidly passenger are older. However, when the age is low, gender has less precision.

# ### Embarked

# Embarked feature is showing the port where the passenger boarded the Titanic.

# In[ ]:


print(train[["Embarked", "Survived"]].groupby(["Embarked"], as_index = False).mean())
sns.barplot(x='Embarked', y='Survived', data=train ,order=['S','C','Q']);


# In[ ]:


pd.crosstab([train.Embarked, train.Pclass], [train.Sex, train.Survived], margins = True).style.background_gradient(cmap = "Reds")


# In[ ]:


train["Embarked"].isnull().sum()


# As we can see, there are only two missing values in "Embarked" feature and it can be useful for our prediction even if it is nothing to do with the disaster. So that, we can fill them with most occured place in our dataset. 

# In[ ]:


train["Embarked"].value_counts()


# Lets fill that two missing values with "S" value which is referring to Southampton and save it our df.

# In[ ]:


for dataset in df_set:
    dataset["Embarked"] = dataset["Embarked"].fillna("S")


# In[ ]:


for dataset in df_set:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# ### Parch

# Parch is the number of parents/children the passenger has aboard the Titanic. Lets look at the distribution of parent/children.

# In[ ]:


train["Parch"].value_counts()


# In[ ]:


sns.barplot(x='Parch', y='Survived', data=train);


# Before we make a comment, lets look at the **SibSp** feature too because of the similarity between them.

# ### SibSp

# **SibSp** is the number of siblings/spouses the passenger has aboard the Titanic. So it is very relatable feature with **Parch**.

# In[ ]:


train["SibSp"].value_counts()


# In[ ]:


sns.barplot(x= "SibSp", y="Survived", data = train);


# 
# Even I look both SibSp and Parch features, I can not say so much about them. However, if we merge them, we can say that having a relative can change the chance of survival or not? 

# ### Family

# Since, we did not satisfy with **Parch** and **SibSp** features, we can make more general feature since they were related. 

# In[ ]:


for dataset in df_set:
    dataset['Has_Family'] =  dataset["Parch"] + dataset["SibSp"]
    dataset['Has_Family'].loc[dataset['Has_Family'] > 0] = 1
    dataset['Has_Family'].loc[dataset['Has_Family'] == 0] = 0

train = train.drop(["SibSp","Parch"], axis = 1 )
test = test.drop(['Parch', 'SibSp'], axis=1)
df_set = [train, test]


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,6))
sns.countplot(x='Has_Family', data=train, order=[0,1], ax = axis1);
mean_family = train[["Has_Family", "Survived"]].groupby(['Has_Family'],as_index=False).mean()
sns.barplot(x='Has_Family', y='Survived', data=mean_family, order=[0,1], ax = axis2);
axis1.set_xticklabels(["Alone","With Family"], rotation=0);


# As we can see, even the people those who are with their family with are less than who are alone, people who are with their family are more likely to survive. So that we keep **Has_Family** feature as a combination of **Parch** and **SibSp**. 

# ### Name 

# Name feature is referring to the name of passengers.

# In[ ]:


train["Name"].nunique()


# Since we know, there are 891 rows in our dataset, that means all the values on Name feature are unique values. So that, it can be bothersome for grouping them into categories. It can be done with many ways. If you are interested in a way to do it, I would recommend https://www.kaggle.com/sinakhorami/titanic-best-working-classifier/. Since, this is my first EDA, I dont work on these feature due to my lack of knowledge.

# In[ ]:


train = train.drop(["Name", "PassengerId"], axis = 1)
test = test.drop(["Name"], axis = 1)
df_set = [train, test]


# ### Fare

# **Fare** is referring to cost of the ticket.Since we know the socio-economic status is the most important feature for our analysis,  we can think of the cost can be important too. Then, we can say that the people who bought more expensive or more cheaper tickets are more likely to survive or not.

# In[ ]:


test.Fare.isnull().sum()


# We can fill the missing values with the median for **Fare**

# In[ ]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)


# Now , lets look at the how many unique value do we have.

# In[ ]:


train["Fare"].nunique()


# Since there are so many unique values in Fare feature, we can split them into group. It makes the Fare feature more easy to understand and useful for our operations.

# In[ ]:


train['Fare_bin'] = pd.qcut(train['Fare'], 4)
train[['Fare_bin', 'Survived']].groupby(['Fare_bin'], as_index=False).mean().sort_values(by='Fare_bin', ascending=True)


# After binning our Fare feature, we can mapping our **Fare** values as respect to bins that we divide.

# In[ ]:


for dataset in df_set:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train = train.drop(["Fare_bin"], axis = 1)
df_set = [train, test]


# Now, lets look at the bin size and survived ratio of them.

# In[ ]:


print(train["Survived"][train["Fare"] == 0 ].value_counts(normalize = True)[1])
print(train["Survived"][train["Fare"] == 1].value_counts(normalize = True)[1])
print(train["Survived"][train["Fare"] == 2 ].value_counts(normalize = True)[1])
print(train["Survived"][train["Fare"] == 3 ].value_counts(normalize = True)[1])

sns.barplot(x = train["Fare"], y = train["Survived"], data = train );


# Now, we can say that people who bought more expensive tickets are more likely to survive. However, we can also think that "Can we be more spesific about that estimation?" and add another feature.

# In[ ]:


sns.barplot(x = train["Fare"], y = train["Survived"], hue = train["Sex"]);


# As we can see, there are more women then men who are survived even though they bought same price of ticket.

# ### Cabin

# Cabin feature has the cabin number where passenger stayed in Titanic. It can be divided into groups, but I think, more information about Titanic architecture is needed to make some assumption. Maybe I am so picky about these features since it is my first time. It can also be used for making new features. However, I rather think to drop from datasets.

# In[ ]:


train.Cabin.value_counts()


# In[ ]:


train = train.drop(["Cabin"], axis = 1)
test = test.drop(["Cabin"], axis = 1)
df_set = [train, test]


# ### Ticket

# Ticket feature has the ticket number of the boarding passenger on the dataset. It can be grouped within themselves, but I dont think it can be useful due to the lack of information. If we know which ticket refers to where in the Titanic, we can make it more useful. Maybe I misunderstood this feature.

# In[ ]:


train.Ticket.value_counts()


# In[ ]:


train = train.drop(["Ticket"], axis=1)
test = test.drop(["Ticket"], axis=1)
df_set = [train, test]


# That concludes the EDA part. Now, we need to check our dataset before the ML part.

# In[ ]:


train.head()


# In[ ]:


test.head()


# ## ML

# Now, we can use our **df_set** to make prediction with our ML techniques. However, lets select our dependent and indepent variables for classification first.

# In[ ]:


X_train = train.drop(["Survived"], axis=1)
y_train = train["Survived"]
X_test = test.drop("PassengerId", axis = 1).copy()


# I selected the **X_train** , **y_train** and **X_test** from the dataset. I will use 9 techniques on our model and which are :
#     * KNN
#     * Logistic Regression
#     * Gaussian Naive Bayes
#     * Support Vector Classifier
#     * Classification and Regression Trees
#     * Random Forest
#     * Gradient Boosting Machine
#     * Extreme Gradient Boost (XGBoost)
#     * Light GBM
# I will also use Grid Search Cross-Validation for finding best parameters for the our model. Then, I will get the score of them and sort with barplot. Finally, I will decide which model is the best for our data.

# ### KNN

# In[ ]:


knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
accuracy = round(knn.score(X_train, y_train)*100, 2)
print("Accuracy before the model tuning is :" + str(accuracy))
#model tuning
knn_params = {'n_neighbors' : np.arange(1, 50),
              'weights': ['uniform', 'distance']}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv= 10)
knn_cv.fit(X_train, y_train)
print(knn_cv.best_params_)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 48, weights = "distance")
knn_tuned = knn.fit(X_train, y_train)
y_pred_knn_tuned = knn.predict(X_test)
tuned_accuracy = round(knn_tuned.score(X_train, y_train)*100, 2)
print("Accuracy after the model tuning : " + str(tuned_accuracy))
knn_cv_score = cross_val_score(knn_tuned, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(knn_cv_score))


# ### Logistic Regression

# In[ ]:


loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train, y_train)
y_pred_lr = loj_model.predict(X_test)
accuracy = round(loj.score(X_train, y_train)*100, 2)
print("Accuracy is :" + str(accuracy))
loj_cv_score = cross_val_score(loj, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(loj_cv_score))


# ### Gaussian Naive Bayes

# In[ ]:


nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
accuracy = round(nb.score(X_train, y_train)*100, 2)
print("Accuracy is :" + str(accuracy))
nb_cv_score = cross_val_score(nb, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(nb_cv_score))


# ### Support Vector Classifier

# In[ ]:


svm= SVC()
svm_model = svm.fit(X_train, y_train)
y_pred_svc = svm_model.predict(X_test)
accuracy = round(svm.score(X_train, y_train)*100, 2)
print("Accuracy before the model tuning : " + str(accuracy))

#model tuning
svc_params = {"C": np.arange(1,5),
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'gamma': [.1, .25, .5, .75, 1.0]}

svc = SVC()

svc_cv_model = GridSearchCV(svc,svc_params, 
                            cv = 10, 
                            n_jobs = -1,)

svc_cv_model.fit(X_train, y_train)
print(svc_cv_model.best_params_)


# In[ ]:


svc_tuned = SVC(kernel = "rbf", C =3 , gamma = 0.1).fit(X_train, y_train)
y_pred_svc_tuned = svc_tuned.predict(X_test)
tuned_accuracy = round(svc_tuned.score(X_train, y_train)*100, 2)
print("Accuracy after the model tuning : " + str(tuned_accuracy))
svc_cv_score = cross_val_score(svc_tuned, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(svc_cv_score))


# ### Classification And Regression Trees (CART)

# In[ ]:


cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)
accuracy = round(cart.score(X_train, y_train)*100, 2)
print("Accuracy before the model tuning : " + str(accuracy))
#model tuning
cart_grid = {"max_depth" : range(1,10),
              "min_samples_split" : list(range(2, 50))}
cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1)
cart_cv_model = cart_cv.fit(X_train, y_train)
print(cart_cv_model.best_params_)


# In[ ]:


cart = tree.DecisionTreeClassifier(max_depth = 4, min_samples_split = 2)
cart_tuned = cart.fit(X_train, y_train)
y_pred_cart_tuned = cart_tuned.predict(X_test)
tuned_accuracy = round(cart_tuned.score(X_train, y_train)*100, 2)
print("Accuracy after the model tuning : " + str(tuned_accuracy))
cart_cv_score = cross_val_score(cart_tuned, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(cart_cv_score))


# ### Random Forest

# In[ ]:


rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy = round(rf_model.score(X_train, y_train)*100, 2)
print("Accuracy before the model tuning :"+ str(accuracy))

#model tuning
rf_params = {"max_depth": [2,5,8,10],
             'criterion' : ['gini', 'entropy'],
            "max_features" : [1,2,3,4,5,6],
            "n_estimators" : [10,500,1000],
            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1) 
rf_cv_model.fit(X_train, y_train)
print("Best parameters are: " + str(rf_cv_model.best_params_))


# In[ ]:


rf_tuned = RandomForestClassifier(max_depth = 5, 
                                  max_features = 4, 
                                  min_samples_split = 5,
                                  n_estimators = 500,
                                 criterion = 'entropy')

rf_tuned.fit(X_train, y_train)
y_pred_rf_tuned = rf_tuned.predict(X_test)
tuned_accuracy = round(rf_tuned.score(X_train, y_train)*100, 2)
print("Accuracy after model tuning", str(tuned_accuracy))
rf_cv_score = cross_val_score(rf_tuned, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(rf_cv_score))
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "b")

plt.xlabel("Importances of Parameters")


# ### Gradient Boosting Machine (GBM)

# In[ ]:


gbm_model = GradientBoostingClassifier().fit(X_train, y_train)
y_pred_gbm = gbm_model.predict(X_test)
accuracy = round(gbm_model.score(X_train, y_train)*100, 2)
print("Accuracy before model tuning : " + str(accuracy))
gbm_params = {"learning_rate" : [ 0.01, 0.1, 0.05],
             "n_estimators": [100,500,1000],
             "max_depth": [3,5,10],
             "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()

gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1)
gbm_cv.fit(X_train, y_train)
print("Best Parameters are: " + str(gbm_cv.best_params_))


# In[ ]:


gbm = GradientBoostingClassifier(learning_rate = 0.01, 
                                 max_depth = 3,
                                min_samples_split = 2,
                                n_estimators = 1000)
gbm_tuned =  gbm.fit(X_train,y_train)
y_pred_gbm_tuned = gbm_tuned.predict(X_test)
tuned_accuracy = round(gbm_tuned.score(X_train, y_train)*100, 2)
print("Accuracy after model tuning :" + str(tuned_accuracy))
gbm_cv_score = cross_val_score(gbm_tuned, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(gbm_cv_score))


# ### Extreme Gradient Boosting(XGBoost)
# 

# In[ ]:


xgb_model = XGBClassifier().fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy = round(xgb_model.score(X_train, y_train)*100, 2)
print("Accuracy before the model tuning : " + str(accuracy))

#model tuning
xgb_params = {
        'n_estimators': [100, 500, 1000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.05],
        "min_samples_split": [2,5,10]}
xgb = XGBClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1)
xgb_cv_model.fit(X_train, y_train)
print("Best parameters are : " + str(xgb_cv_model.best_params_))


# In[ ]:


xgb = XGBClassifier(learning_rate = 0.1, 
                    max_depth = 3,
                    min_samples_split = 2,
                    n_estimators = 500,
                    subsample = 0.8)
xgb_tuned =  xgb.fit(X_train,y_train)
y_pred_xgb_tuned = xgb_tuned.predict(X_test)
tuned_accuracy = round(xgb_tuned.score(X_train, y_train)*100, 2)
print("Accuracy after the model tuning : " + str(tuned_accuracy))
xgb_cv_score = cross_val_score(xgb_tuned, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(xgb_cv_score))


# ### Light GBM

# In[ ]:


lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)
accuracy = round(knn.score(X_train, y_train)*100, 2)
print("Accuracy before the model tuning : " + str(accuracy))
lgbm_params = {
        'n_estimators': [100, 500, 1000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.05],
        "min_child_samples": [5,10,20]}
lgbm = LGBMClassifier()

lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 
                             cv = 10, 
                             n_jobs = -1)
lgbm_cv_model.fit(X_train, y_train)
print(" Best parameters are : " + str(lgbm_cv_model.best_params_))


# In[ ]:



lgbm = LGBMClassifier(learning_rate = 0.05, 
                       max_depth = 3,
                       num_leaves =10,
                       subsample = 0.6,
                       n_estimators = 500,
                       min_child_samples = 10)
lgbm_tuned = lgbm.fit(X_train,y_train)
y_pred_lgbm_tuned = lgbm_tuned.predict(X_test)
tuned_accuracy = round(lgbm_tuned.score(X_train, y_train)*100, 2)
print("Accuracy after the model tuning : " + str(tuned_accuracy))
lgbm_cv_score = cross_val_score(lgbm_tuned, X_train, y_train, cv = 10).mean()*100
print("Cross validation score is " + str(lgbm_cv_score))


# In[ ]:


models = [
    knn_tuned,
    loj_model,
    svc_tuned,
    nb_model,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    lgbm_tuned,
    xgb_tuned
    
]


for model in models:
    names = model.__class__.__name__
    accuracy = round(model.score(X_train, y_train),2)
    print("-"*28)
    print(names + ":" )
    print("Accuracy: {:.4%}".format(accuracy))


# In[ ]:


result = []

results = pd.DataFrame(columns= ["Models","Accuracy"])

for model in models:
    names = model.__class__.__name__
    accuracy = round(model.score(X_train, y_train),2)    
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])
    results = results.append(result)
    
sns.set_color_codes("muted")
sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="b")
plt.xlabel('Accuracy %')
plt.title('Accuracy Ratio Of Models');   


# In[ ]:


cv_models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Gaussian Naive Bayes', 
              'XGBoost', 'Linear SVC', 
              'Random Forest', 'Gradient Boosting Trees', 'CART', 'LGBM'],
    'Score': [
        knn_cv_score, 
        loj_cv_score,      
        nb_cv_score, 
        xgb_cv_score, 
        svc_cv_score, 
        rf_cv_score,
        gbm_cv_score,
        cart_cv_score,
        lgbm_cv_score
    ]})
print('---Cross-validation Accuracy Scores---')
print(cv_models.sort_values(by='Score', ascending=False))

sns.set_color_codes("muted")
sns.barplot(x= 'Score', y = 'Model', data=cv_models, color="b")
plt.xlabel('Accuracy %')
plt.title('Cross-validation Accuracy Scores');   


# For model selection, KNN is very good before K-fold CV operation but, after the CV, XGBoost looks more consistent and successful. The reason why we care about K-fold CV so much, it makes the model more robust.For me, the robustness is more important than high accuracy. Thus, I pick XGBoost over KNN and others.

# Finally, lets make our submission.

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_xgb_tuned
    })
submission.to_csv('submission.csv', index=False)

