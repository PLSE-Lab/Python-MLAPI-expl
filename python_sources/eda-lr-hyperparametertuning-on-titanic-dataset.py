#!/usr/bin/env python
# coding: utf-8

# # <div align="center">TITANIC</div>
# <div align="center">![Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/300px-RMS_Titanic_3.jpg)
# 
# RMS Titanic was a British passenger liner operated by the White Star Line that sank in the North Atlantic Ocean in the early morning hours of April 15, 1912, after striking an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making the sinking one of modern history's deadliest peacetime commercial marine disasters. RMS Titanic was the largest ship afloat at the time she entered service and was the second of three Olympic-class ocean liners operated by the White Star Line. She was built by the Harland and Wolff shipyard in Belfast. Thomas Andrews, chief naval architect of the shipyard at the time, died in the disaster. For more information [Click Here](https://en.wikipedia.org/wiki/RMS_Titanic). </div>

# <font color="purple">
# Table of content:
# 1. [Load And Check The Data](#1)
# 1. [Description Of The Variables](#2)
# 1. [Univariate variable Analysis](#3)
#     * [Categorical Variable Analysis](#4)
#     * [Numerical Variable Analysis](#5)
# 1. [Basic Data Analysis](#6)
#     * [Pclass  --  Survived](#pclass-survived)
#     * [Sex  --  Survived](#sex-survived)
#     * [SibSp  --  Survived](#sibsp-survived)
#     * [Parch  --  Survived](#parch-survived)
# 1. [Outlier Detection](#7)
# 1. [Visualization](#8)
#     * [Correlation between Survived --- Pclass --- Age --- SibSp --- Parch --- Fare](#correlation)
#     * [Survived --- SibSp](#SSibSp)
#     * [Survived --- Parch](#SParch)
#     * [Survived --- Pclass](#SPclass)
#     * [Survived --- Age](#SAge)
#     * [Survived --- Age --- Gender](#SAgeGender)
#     * [Survived --- Pclass --- Gender](#SPclassGender)
# 1. [Missing Values](#9)
#     * [Finding Missing Values](#10)
#     * [Filling Missing Values](#11)
#         * [Missing Values Of Embarked](#fillEmbarked)
#         * [Missing Values Of Fare](#fillFare)
#         * [Missing Values Of Age](#fillAge)
# 1. [Feature Engineering](#12)
#     * [Name To Title](#13)
#     * [Pclass](#14)
#     * [Embarked](#15)
#     * [Sex](#16)
#     * [Dropping PassengerId, Ticket and Cabin](#17)
# 1. [Modeling](#18)
#     * [Train-Test Split](#19)
#     * [Simple Logistic Regression](#20)
#     * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#21)
#     * [Ensemble Modeling](#22)
# 1. [Prediction And Submission](#23)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 999;')


# <a id="1"></a><br>
# # <div align="center"> Loading And Checking The Data </div>
# 

# In[ ]:


dfTrain = pd.read_csv("/kaggle/input/titanic/train.csv")
dfTest = pd.read_csv("/kaggle/input/titanic/test.csv")

print("\nTrain dataframe info\n")
dfTrain.info()
print("\nTest dataframe info\n")
dfTest.info()


# In[ ]:


dfTrain.head(10)


# In[ ]:


dfTrain.describe()


# <a id="2"></a><br>
# # <div align="center"> Description Of The Variables </div>

# |Variable   | Data Type | Definition                                     | Key                                            |
# |-----------|-----------|------------------------------------------------|------------------------------------------------|
# |PassengerId| int64     | Unique ID of the passanger                     | 0 = No, 1 = Yes                                |
# |Survived   | int64     | Survival status                                | 0 = No, 1 = Yes                                |
# |Pclass     | int64     | Ticket class                                   | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
# |Name       | object    | Passengers name                                | 	                                              |
# |Sex        | object    | Gender of the passenger                        | 	                                              |
# |Age        | float64   | Age in years                                   | 	                                              |
# |Sibsp      | int64     | Number of siblings / spouses aboard the Titanic| 	                                              |
# |Parch      | int64     | Number of parents / children aboard the Titanic| 	                                              |
# |Ticket     | object    | Ticket number                                  | 	                                              |
# |Fare       | float64   | Passenger fare                                 | 	                                              |
# |Cabin      | object    | Cabin number                                   | 	                                              |
# |Embarked   | object    | Port of Embarkation                            | C = Cherbourg, Q = Queenstown, S = Southampton |

# <a id="3"></a><br>
# # <div align="center"> Univariate Variable Analysis </div>

# * [Categorical variable analysis](#4)
# >     Labels: Survived, Pclass, Sex, SibSp, Parch, Embarked
# * [Numerical variable analysis](#5)
# >     Labels: Age, Fare 
# 

# <a id="4"></a>
# ### Categorical Variable Analysis
# 

# In[ ]:


import plotly
from plotly.offline import iplot
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

def PlotPieChart(df,label):
    trace = go.Pie(labels=df[label])
    layout = dict(title = str(label))
    fig = dict(data=[trace], layout=layout)
    iplot(fig)


# In[ ]:


categoricalLabels = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked"]
for label in categoricalLabels:
    PlotPieChart(dfTrain,label)


# <a id="5"></a><br>
# ### Numerical Variable Analysis

# In[ ]:


import plotly.express as px
def PlotHistogram(df,label):
    fig = px.histogram(df, x=label)
    fig.show()


# In[ ]:


numericalLabels = ["Age", "Fare"]
for label in numericalLabels:
    PlotHistogram(dfTrain,label)


# 
# <a id="6"></a><br>
# # <div align="center"> Basic Data Analysis </div>

# In this section we will examine the relationships between the two labels based on the [Description of the variables](#2)
# 
# These peer tags:
# * [Pclass  --  Survived](#pclass-survived)
# * [Sex  --  Survived](#sex-survived)
# * [SibSp  --  Survived](#sibsp-survived)
# * [Parch  --  Survived](#parch-survived)

# In[ ]:


import plotly.express as px
def relationPieChart(df,value,name):
    fig = px.pie(df, values=value, names=name, title=str(value+" -- "+name))
    fig.show()


# <a id="pclass-survived"></a><br>
# #### Pclass  --  Survived

# In[ ]:


print("Surviving probability of Pclasses")
print(dfTrain[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending=False))
relationPieChart(dfTrain,"Survived","Pclass")


# <a id="sex-survived"></a><br>
# #### Sex  --  Survived

# In[ ]:


print("Surviving probability of genders")
print(dfTrain[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending=False))
relationPieChart(dfTrain,"Survived","Sex")


# <a id="sibsp-survived"></a><br>
# #### SibSp  --  Survived

# In[ ]:


print("Surviving probability of Sibsps")
print(dfTrain[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending=False))
relationPieChart(dfTrain,"Survived","SibSp")


# <a id="parch-survived"></a><br>
# #### Parch  --  Survived

# In[ ]:


print("Surviving probability of Parchs")
print(dfTrain[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending=False))
relationPieChart(dfTrain,"Survived","Parch")


# 
# <a id="7"></a><br>
# # <div align="center"> Outlier Detection </div>

# In[ ]:


def detectOutlier(df,features, minOutlierCount):
    outlierList = []
    
    for feature in features:
        #1st quartile
        Q1 = np.percentile(df[feature],25)
        #3rd quartile
        Q3 = np.percentile(df[feature],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier Step
        outlierStep = IQR * 1.5
        #detect outlier and their indices
        outlierListCol = df[(df[feature] < Q1 - outlierStep) | (df[feature] > Q3 + outlierStep)].index
        #store indices
        outlierList.extend(outlierListCol)
    
    outlierIndices = Counter(outlierList)
    multipleOutliers = list(i for i,v in outlierIndices.items() if v > minOutlierCount)
    
    return multipleOutliers


# Checking and dropping the rows which has outliers in 2 or more columns(in Age, SibSp, Parch, Fare)

# In[ ]:


dfTrain.loc[detectOutlier(dfTrain,["Age","SibSp","Parch","Fare"],2)]


# In[ ]:


dfTrain.drop(detectOutlier(dfTrain,["Age","SibSp","Parch","Fare"],2),axis=0,inplace=True)
dfTrain.reset_index(inplace=True,drop=True)


# <a id="8"></a><br>
# # <div align="center"> Visualization </div>

# * [Correlation between Survived --- Pclass --- Age --- SibSp --- Parch --- Fare](#correlation)
# * [Survived --- SibSp](#SSibSp)
# * [Survived --- Parch](#SParch)
# * [Survived --- Pclass](#SPclass)
# * [Survived --- Age](#SAge)
# * [Survived --- Age --- Gender](#SAgeGender)
# * [Survived --- Pclass --- Gender](#SPclassGender)

# <a id="correlation"></a><br>
# #### Correlation between Survived --- Pclass --- Age --- SibSp --- Parch ---[](http://) Fare

# In[ ]:


correlationList = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
sns.heatmap(dfTrain[correlationList].corr(), annot=True, fmt=".2f")


# <a id="SSibSp"></a><br>
# #### Survived --- SibSp

# In[ ]:


g = sns.factorplot(x = "SibSp", y = "Survived", kind="bar", data=dfTrain,size = 5)
g.set_ylabels("Survival Probablity")
plt.show()


# Having a lot of siblings or spouses means less chance to survive according to this dataset. Whereas passengers have 1 or 2 siblings or spouses, have more chance to survive

# <a id="SParch"></a><br>
# #### Survived --- Parch

# In[ ]:


g = sns.factorplot(x = "Parch", y = "Survived", kind="bar", data=dfTrain,size = 5)
g.set_ylabels("Survival Probablity")
plt.show()


# * Number of parents/children(Parch) and siblings/spouses(SibSp) aboard the Titanic can be used together to extract a new feature until th=3. 
# * small families have more chance to survive.
# * there is a high standard deviation in survival of passenger with parch=3.

# <a id="SPclass"></a><br>
# #### Survived --- Pclass

# In[ ]:


g = sns.factorplot(x = "Pclass", y = "Survived", kind="bar", data=dfTrain,size = 5)
g.set_ylabels("Survival Probablity")
plt.show()


# * Higher Pclass passengers has higher survival probablity

# <a id="SAge"></a><br>
# #### Survived --- Age

# In[ ]:


g = sns.FacetGrid(dfTrain,col="Survived",size = 5)
g.map(sns.distplot,"Age",bins=30)
plt.show()


# * child ages has higher survival rate.
# * large number of 20s ages couldnt survive
# * most passengers are in 15-35 age range

# <a id="SAgeGender"></a><br>
# #### Survived --- Age --- Gender

# In[ ]:


import plotly.express as px
fig = px.histogram(dfTrain[dfTrain.Survived == 1], x="Age", color="Sex", marginal="violin", title ="Survived passengers by their ages", color_discrete_map={"male": "#187196","female": "#fab7cc"})
fig.show()
fig2 = px.histogram(dfTrain[dfTrain.Survived == 0], x="Age", color="Sex", marginal="violin", title ="Couldnt survived passengers by their ages", color_discrete_map={"male": "#187196","female": "#fab7cc"})
fig2.show()


# <a id="SPclassGender"></a><br>
# #### Survived --- Pclass --- Gender

# In[ ]:


fig = px.histogram(dfTrain[dfTrain.Survived == 1], x="Pclass", color="Sex", marginal="violin", title ="Survived passengers by their ticket class", color_discrete_map={"male": "#187196","female": "#fab7cc"})
fig.show()
fig2 = px.histogram(dfTrain[dfTrain.Survived == 0], x="Pclass", color="Sex", marginal="violin", title ="Couldnt survived passengers by their ticket class", color_discrete_map={"male": "#187196","female": "#fab7cc"})
fig2.show()


# <a id="9"></a><br>
# # <div align="center"> Missing Values </div>

# * [Finding missing values](#9)
# * [Filling missing values](#10)

# <a id="10"></a>
# ## Finding Missing Values
# 

# In[ ]:


print("\nTrain dataframe columns that include null values:\n")
print(dfTrain.columns[dfTrain.isna().any()])
print("\nTrain dataframe null rows count:\n")
print(dfTrain.isna().sum())
print("\n================================================")
print("\nTest dataframe columns that include null values:\n")
print(dfTest.columns[dfTest.isna().any()])
print("\nTest dataframe null rows count:\n")
print(dfTest.isna().sum())


# In[ ]:


dfTrain[dfTrain.Embarked.isna()]


# In[ ]:


dfTest[dfTest.Fare.isna()]


# <a id="11"></a>
# ## Filling Missing Values

# Labels which will be firstly filled in dataframes
# * Train:
#     * [Embarked](#fillEmbarked) > 2 missing value
#     * [Age](#fillAge) > 170 missing value
# * Test:
#     * [Fare](#fillFare) > 1 missing value
#     * [Age](#fillAge) > 86 missing value

# We will temporarily combine the train and test data to fill with more consistent data

# In[ ]:


dfCombined = pd.concat([dfTrain, dfTest], axis=0)
dfCombined.info()


# <a id="fillEmbarked"></a>
# ##### Filling Missing Values Of Embarked Column
# Maybe we can analyze train the data by saying that there can be a connection between the ***ticket fare*** and the ***port of embarkation*** information.

# In[ ]:


fig = px.box(dfCombined, x="Embarked", y="Fare", points="all")
fig.show()


# In[ ]:


dfTrain[dfTrain.Embarked.isna()][["Fare","Embarked"]]


# As we can see from the above box plot and nan values, we can fill the embarked column as **"C"**

# In[ ]:


dfTrain["Embarked"].fillna("C",inplace=True)
dfTrain.iloc[[60,821]]


# <a id="fillFare"></a>
# ##### Filling Missing Values Of Fare Column
# Now we can fill the nan value of **test dataframes fare column**
# 
# Here, also, we examine the data by saying that there might be a relationship between ticket ***fares***, ***port of embarkation*** and ***ticket class*** information.

# In[ ]:


dfTest[dfTest.Fare.isna()]


# We can fill it by the average fare of the passengers who has same port of embarkation and ticket class 

# In[ ]:


np.mean(dfCombined[(dfCombined["Pclass"] == 3) & (dfCombined["Embarked"] == "S")]["Fare"])


# In[ ]:


dfTest["Fare"].fillna(np.mean(dfCombined[(dfCombined["Pclass"] == 3) & (dfCombined["Embarked"] == "S")]["Fare"]) , inplace=True)
dfTest.iloc[[152]]


# <a id="fillAge"></a>
# ##### Filling Missing Values Of Age Columns

# In[ ]:


dfTrain[dfTrain["Age"].isna()]


# In[ ]:


dfTest[dfTest["Age"].isna()]


# In[ ]:


dfCombined["Gender"] = [1 if i == "male" else 0 for i in dfCombined["Sex"]] # make sex variable numerical and store them in gender column to show in heatmap.
correlationList = ["Age", "Gender", "Pclass", "SibSp", "Parch", "Fare"]
sns.heatmap(dfCombined[correlationList].corr(), annot=True, fmt=".2f")
plt.show()
dfCombined.drop(["Gender"],axis=1,inplace=True) # drop the Gender column, because it was necessary for only heatmap.


# In[ ]:


fig = px.box(dfCombined, x = "Sex", y = "Age", color="Pclass", points="all", title="Correlation between Sex --- Age --- Pclass")
fig.show()


# Regardless of gender, 1stclass passengers are older than 2nd class passengers, which are older than 3rd class passengers.

# In[ ]:


fig = px.box(dfCombined, x = "SibSp", y = "Age", points="all", title="Correlation between Age and SibSp")
fig.show()


# In[ ]:


fig = px.histogram(dfCombined, x = "Fare", y = "Age", histfunc='avg', title="Correlation between Average Age and Fare")
fig.show()


# There is no significant effect between ticket fares and age. The average age of each ticket is very close to each other

# In[ ]:


fig = px.box(dfCombined, x = "Parch", y = "Age", points="all", title="Correlation between Age and Parch")
fig.show()


# Certain number ranges can cover different age ranges, in parent/child(Parch) values

# In[ ]:


trainIndexNanAge = list(dfTrain[dfTrain["Age"].isna()].index)
print("number of nan age train indexes : {}".format(len(trainIndexNanAge)))
testIndexNanAge = list(dfTest[dfTest["Age"].isna()].index)
print("number of nan age test indexes : {}".format(len(testIndexNanAge)))
combinedIndexNanAge = list(dfCombined[dfCombined["Age"].isna()].index)
print("number of total nan age indexes : {}".format(len(combinedIndexNanAge)))


# We will fill the nan values with the ages median of the same passengers which has same siblings/spouses, parent/childs and pclass

# In[ ]:


for index in trainIndexNanAge:
    age_pred = dfCombined["Age"][((dfCombined["SibSp"] == dfTrain.iloc[index]["SibSp"]) & (dfCombined["Parch"] == dfTrain.iloc[index]["Parch"]) & (dfCombined["Pclass"] == dfTrain.iloc[index]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dfTrain["Age"].iloc[index] = age_pred


# In[ ]:


for index in testIndexNanAge:
    age_pred = dfCombined["Age"][((dfCombined["SibSp"] == dfTest.iloc[index]["SibSp"]) & (dfCombined["Parch"] == dfTest.iloc[index]["Parch"]) & (dfCombined["Pclass"] == dfTest.iloc[index]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dfTest["Age"].iloc[index] = age_pred


# After this filling we have still two age values empty in the test dataframe 

# In[ ]:


trainIndexNanAge = list(dfTrain[dfTrain["Age"].isna()].index)
print("number of nan age train indexes : {}".format(len(trainIndexNanAge)))
testIndexNanAge = list(dfTest[dfTest["Age"].isna()].index)
print("number of nan age test indexes : {}".format(len(testIndexNanAge)))
dfCombined = pd.concat([dfTrain, dfTest], axis=0)
combinedIndexNanAge = list(dfCombined[dfCombined["Age"].isna()].index)
print("number of total nan age indexes : {}".format(len(combinedIndexNanAge)))


# Since this passenger data does not match other passengers, we fill them using the median of all passengers age value

# In[ ]:


age_med = dfCombined["Age"].median()
for index in trainIndexNanAge:
    dfTrain["Age"].iloc[index] = age_med
for index in testIndexNanAge:
    dfTest["Age"].iloc[index] = age_med


# In[ ]:


trainIndexNanAge = list(dfTrain[dfTrain["Age"].isna()].index)
print("number of nan age train indexes : {}".format(len(trainIndexNanAge)))
testIndexNanAge = list(dfTest[dfTest["Age"].isna()].index)
print("number of nan age test indexes : {}".format(len(testIndexNanAge)))
dfCombined = pd.concat([dfTrain, dfTest], axis=0)
combinedIndexNanAge = list(dfCombined[dfCombined["Age"].isna()].index)
print("number of total nan age indexes : {}".format(len(combinedIndexNanAge)))
del dfCombined, combinedIndexNanAge, testIndexNanAge, trainIndexNanAge


# Now we have no empty row on **age** column in both dataframes.

# <a id="12"></a><br>
# # <div align="center"> Feature Engineering </div>

# * [Name To Title](#13)
# * [Pclass](#14)
# * [Embarked](#15)
# * [Sex](#16)
# * [Dropping Passenger Id, Ticket and Cabin](#17)

# <a id="13"></a>
# ## Name To Title

# There can not be a relationship between the name and the possibility of survive, but there may be a relationship between the title information inside the names.

# In[ ]:


def find_title(name):
    return name.split(",")[1].split(".")[0].strip()


# In[ ]:


dfTrain["Title"] = dfTrain["Name"].apply(find_title)
dfTest["Title"] = dfTest["Name"].apply(find_title)
print("Used Different Titles Are:")
print(pd.concat([dfTrain,dfTest],axis=0)["Title"].value_counts())


# Considering that we can not learn very few passing data, we can name them as "Other".

# In[ ]:


other_list = ["Rev", "Dr", "Col", "Major", "Ms", "Mlle", "Jonkheer", "Lady", "Mme", "Dona", "Capt", "the Countess", "Sir", "Don"]
dfTrain["Title"] = dfTrain["Title"].replace(other_list, "Other")
dfTest["Title"] = dfTest["Title"].replace(other_list, "Other")
print("Used Different Titles Are:")
print(pd.concat([dfTrain,dfTest],axis=0)["Title"].value_counts())


# In[ ]:


g = sns.factorplot(x = "Title", y = "Survived", kind="bar", data=dfTrain, size = 5)
g.set_ylabels("Survival Probablity")
plt.show()


# Now we can convert this 5 title variables into dummy/indicator variables.

# In[ ]:


dfTrain.drop(["Name"], axis=1, inplace=True)
dfTest.drop(["Name"], axis=1, inplace=True)

dfTrain["Title"] = dfTrain["Title"].astype("category")
dfTrain = pd.get_dummies(dfTrain,columns=["Title"])
dfTest["Title"] = dfTest["Title"].astype("category")
dfTest = pd.get_dummies(dfTest,columns=["Title"])

print("Train dataframe columns: {}".format(dfTrain.columns.values))
print("Test dataframe columns: {}".format(dfTest.columns.values))


# <a id="14"></a>
# ## Pclass

# the 3 Pclass variables.

# In[ ]:


dfTrain["Pclass"] = dfTrain["Pclass"].astype("category")
dfTrain = pd.get_dummies(dfTrain,columns=["Pclass"])

dfTest["Pclass"] = dfTest["Pclass"].astype("category")
dfTest = pd.get_dummies(dfTest,columns=["Pclass"])

print("Train dataframe columns: {}".format(dfTrain.columns.values))
print("Test dataframe columns: {}".format(dfTest.columns.values))


# <a id="15"></a>
# ## Embarked

# the 3 Embarked variables.

# In[ ]:


dfTrain["Embarked"] = dfTrain["Embarked"].astype("category")
dfTrain = pd.get_dummies(dfTrain,columns=["Embarked"])

dfTest["Embarked"] = dfTest["Embarked"].astype("category")
dfTest = pd.get_dummies(dfTest,columns=["Embarked"])

print("Train dataframe columns: {}".format(dfTrain.columns.values))
print("Test dataframe columns: {}".format(dfTest.columns.values))


# <a id="16"></a>
# ## Sex

# And finally the 2 gender variables.

# In[ ]:


dfTrain["Sex"] = dfTrain["Sex"].astype("category")
dfTrain = pd.get_dummies(dfTrain,columns=["Sex"], prefix="S")

dfTest["Sex"] = dfTest["Sex"].astype("category")
dfTest = pd.get_dummies(dfTest,columns=["Sex"], prefix="S")

print("Train dataframe columns: {}".format(dfTrain.columns.values))
print("Test dataframe columns: {}".format(dfTest.columns.values))


# <a id="17"></a>
# ## Dropping Passenger Id, Ticket and Cabin

# We drop the unnecessary **Ticket**, **Cabin** and **PassengerId** columns. But we do not drop **PassengerID** in the **test** database. Because we will use it in the *submission phase*.

# In[ ]:


dfTrain.drop(["Ticket","Cabin","PassengerId"], axis=1, inplace=True)
dfTest.drop(["Ticket","Cabin"], axis=1, inplace=True)

print("Train dataframe columns: {}".format(dfTrain.columns.values))
print("Test dataframe columns: {}".format(dfTest.columns.values))


# In[ ]:


dfTrain.head()


# In[ ]:


dfTest.head()


# <a id="18"></a><br>
# # <div align="center"> Modeling </div>

# * [Train-Test Split](#19)
# * [Simple Logistic Regression](#20)
# * [Hyperparameter Tuning -- Grid Search -- Cross Validation](#21)
# * [Ensemble Modeling](#22)

# In[ ]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


dfTrain = (dfTrain - np.min(dfTrain)) / (np.max(dfTrain) - np.min(dfTrain)).values
dfTestTemp = dfTest["PassengerId"]
dfTest = (dfTest - np.min(dfTest)) / (np.max(dfTest) - np.min(dfTest)).values
dfTest["PassengerId"] = dfTestTemp
del dfTestTemp
dfTest.describe()


# <a id="19"></a>
# ## Train-Test Split

# In[ ]:


xTrain = dfTrain.drop(["Survived"], axis = 1)
yTrain = dfTrain["Survived"]
xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size = 0.33, random_state = 50)
print("sizes: ")
print("xTrain: {}, xValidation: {}, yTrain: {}, yValidation: {}".format(len(xTrain),len(xVal),len(yTrain),len(yVal)))
print("\t\t\t test: {}".format(len(dfTest)))


# <a id="20"></a>
# ## Simple Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(xTrain,yTrain)
acc_logreg_train = logreg.score(xTrain,yTrain)*100
acc_logreg_validation = logreg.score(xVal,yVal)*100
print("Train data accuracy: {}".format(acc_logreg_train))
print("Validation data accuracy: {}".format(acc_logreg_validation))


# <a id="21"></a>
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation

# **In this section the 5 popular ML classifier will be compared by their mean accuracy using stratified cross validation**
# 
# > Decision Tree
# 
# > SVM
# 
# > Random Forest
# 
# > KNN
# 
# > Logistic Regression

# In[ ]:


classifiers = [DecisionTreeClassifier(random_state = 50),
               SVC(random_state = 50),
               RandomForestClassifier(random_state = 50),
               KNeighborsClassifier(),
               LogisticRegression(random_state = 50)]

decisionTree_params_grid = {"min_samples_split":range(10,500,20),
                            "max_depth":range(1,20,2)}
svc_params_grid = {"kernel":["rbf"],
                   "gamma":[0.001,0.01,0.1,1],
                   "C":[1,10,50,100,200],
                   "probability":[True]}
randomForest_params_grid = {"max_features":[1,3,10],
                            "min_samples_split":[2,3,10],
                            "min_samples_leaf":[1,3,10],
                            "bootstrap":[False],
                            "n_estimators":[100,300],
                            "criterion":["gini"]}
knn_params_grid = {"n_neighbors":np.linspace(1,19,10,dtype=int).tolist(),
                   "weights":["uniform","distance"],
                   "metric":["euclidean","manhattan"]}
logisticRegression_params_grid = {"C":np.logspace(-3,3,7),
                                  "penalty":["l1","l2"]}


classifier_params = [decisionTree_params_grid,
                     svc_params_grid,
                     randomForest_params_grid,
                     knn_params_grid,
                     logisticRegression_params_grid]


# In[ ]:


cvResult = []
bestEstimators = []
for classifierIndex in range(len(classifiers)):
    classifier = GridSearchCV(classifiers[classifierIndex], param_grid=classifier_params[classifierIndex], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs=-1 , verbose=1)
    classifier.fit(xTrain,yTrain)
    cvResult.append(classifier.best_score_)
    bestEstimators.append(classifier.best_estimator_)
    print("current best score = {}".format(cvResult[classifierIndex]))


# In[ ]:


cvResult = pd.DataFrame({"Cross Validation Means": cvResult, "ML Models":[
    "Decision Tree Classifier",
    "SVM",
    "Random Forest Classifier",
    "K Neighbors Classifier",
    "Logistic Regression"]})


# In[ ]:


fig = px.bar(cvResult, x='ML Models', y='Cross Validation Means', title="Cross Validation Scores")
fig.show()


# <a id="22"></a>
# ## Ensemble Modeling

# In[ ]:


votingClassifier = VotingClassifier(estimators = [("decissionTree",bestEstimators[0]),
                                                  ("randomForest",bestEstimators[2]),
                                                  ("logisticRegression",bestEstimators[4])],
                                    voting = "soft",
                                    n_jobs = -1)


# In[ ]:


votingClassifier = votingClassifier.fit(xTrain,yTrain)
print("Train data accuracy: {}".format(accuracy_score(votingClassifier.predict(xTrain),yTrain)))
print("Validation data accuracy: {}".format(accuracy_score(votingClassifier.predict(xVal),yVal)))


# <a id="23"></a><br>
# # <div align="center"> Prediction And Submission </div>

# In[ ]:


survivedTest = pd.Series(votingClassifier.predict(dfTest.drop(["PassengerId"],axis=1)),name="Survived").astype(int)
results = pd.concat([dfTest["PassengerId"], survivedTest],axis = 1)
results.head(10)


# In[ ]:


results.info()


# In[ ]:


results.describe()


# In[ ]:


results.to_csv("titanic.csv", index = False)

