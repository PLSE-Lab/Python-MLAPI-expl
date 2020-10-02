#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
plt.style.use('ggplot') # to plot graphs with gggplot2 style
# Any results you write to the current directory are saved as output.


# In[ ]:


#Reading the dataset on pandas
strains = pd.read_csv("../input/cannabis.csv")


# In[ ]:


strains.shape


# In[ ]:


strains.info()


# In[ ]:


# check the null value
strains.isnull().sum()


# Flavor and Description have nan value

# In[ ]:


strains.head()


# In[ ]:


strains['Type']= strains['Type'].astype(str)


# In[ ]:


print(strains.nunique())


# ### EDA

# In[ ]:


print("Numerical describe of distribuition Type")
print(strains.groupby("Type")["Strain"].count())
print("Percentage of distribuition Type ")
print((strains.groupby("Type")["Strain"].count() / len(strains.Type) * 100).round(decimals=2))


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x="Type", data=strains, palette='hls')
plt.xlabel('Species', fontsize=15)
plt.ylabel('Count', fontsize=20)
plt.title("Cannabis Species Count ", fontsize=20)
plt.show()


# ### Looking the distribuition of Rating and type by Rating

# In[ ]:


print("Top 10 Rating by consumers")
print(strains["Rating"].value_counts().head(10))

plt.figure(figsize=(8,6))

#Total rating distribuition
g = sns.distplot(strains["Rating"], bins=50)
g.set_title("Rating distribuition", size = 20)
g.set_xlabel('Rating', fontsize=15)


#  Almost all species have rating higher than 4
# 
# Now I will Look the distribuition separted by Type
# 

# In[ ]:


print("Rating Distribuition by Species Type")
print(pd.crosstab(strains[strains.Rating > 4.0]['Rating'], strains.Type))

plt.figure(figsize=(10,14))

#Let's look the Rating distribuition by Type.
g = plt.subplot(311)
g = sns.distplot(strains[(strains.Type == 'hybrid') & 
                               (strains.Rating > 0)]["Rating"], color='y')
g.set_xlabel("Rating", fontsize=15)
g.set_ylabel("Distribuition", fontsize=15)
g.set_title("Rating Distribuition Hybrids", fontsize=20)

g1 = plt.subplot(312)
g1 = sns.distplot(strains[(strains.Type == 'sativa') & 
                               (strains.Rating > 0)]["Rating"], color='g')
g1.set_xlabel("Rating", fontsize=15)
g1.set_ylabel("Distribuition", fontsize=15)
g1.set_title("Rating Distribuition Sativas", fontsize=20)

g2 = plt.subplot(313)
g2 = sns.distplot(strains[(strains.Type == 'indica') & 
                               (strains.Rating > 0)]["Rating"], color='r')
g2.set_xlabel("Rating", fontsize=15)
g2.set_ylabel("Distribuition", fontsize=15)
g2.set_title("Rating Distribuition Indicas", fontsize=20)

plt.subplots_adjust(wspace = 0.1, hspace = 0.6,top = 0.9)

plt.show()


# Sativa and Indica have a similar rating distribuition, and we can see that almost of all species in dataset have rating higher than 4

# ### Lets try a better look of the Rating Distribuition by species considering just the values higher than 2

# In[ ]:


plt.figure(figsize=(10,6))
#I will now explore the Rating distribuition by Type
g = sns.boxplot(x="Type",y="Rating",data=strains[strains["Rating"] > 2],palette="hls")
g.set_title("Rating distribuition by Species Type", fontsize=20)
g.set_xlabel("Species", fontsize=15)
g.set_ylabel("Rating", fontsize=15)
plt.show()


# WE can see that the Sativa have a less median than Hybrids and indicas

# In[ ]:


#Looking the Rating distribuition description 
print("Rating less than 4: ")
print(strains[strains.Rating <= 4].groupby("Type")["Strain"].count())
print("Rating between 4 and 4.5: ")
print(strains[(strains.Rating > 4) & (strains.Rating <= 4.5)].groupby("Type")["Strain"].count())
print("Top Strains - Rating > 4.5: ")
print(strains[strains["Rating"] > 4.5].groupby("Type")["Strain"].count())
print("Distribuition by type of Ratings equal 5: ")
print(strains[strains["Rating"] == 5].groupby("Type")["Strain"].count())
print("Total of: 2350 different Strains")


# In[ ]:


#I will extract the values in Effects and Flavor and pass to a new column
df_effect = pd.DataFrame(strains.Effects.str.split(',',4).tolist(),
             columns = ['Effect_1','Effect_2','Effect_3','Effect_4','Effect_5'])

df_flavors = pd.DataFrame(strains.Flavor.str.split(',',n=2,expand=True).values.tolist(),
                          columns = ['Flavor_1','Flavor_2','Flavor_3'])


# In[ ]:


#Concatenating the new variables with strains
strains = pd.concat([strains, df_effect], axis=1)
strains = pd.concat([strains, df_flavors], axis=1)

#Looking the result
strains.head()


# In[ ]:


strains.columns


# We can se the Effects and Flavors are in separated columns... Now I will explore the main related effects

# In[ ]:


print("The top 5 First Effects related")
print(strains['Effect_1'].value_counts()[:5])

plt.figure(figsize=(13,6))

g = sns.boxplot(x = 'Effect_1', y="Rating",
                hue="Type",
                data=strains[strains["Rating"] > 3],
                palette="hls")
g.set_xlabel("Related Effect", fontsize=15)
g.set_ylabel("Rating Distribuition", fontsize=15)
g.set_title("First Effect Related x Rating by Species Type",fontsize=20)

plt.show()


# ### The second most related effects and respective Rating

# In[ ]:


print("The top 5 Second related Effects")
print(strains['Effect_2'].value_counts()[:5])

plt.figure(figsize=(13,6))

g = sns.boxplot(x = 'Effect_2', y="Rating",
                hue="Type",
                data=strains[strains["Rating"] > 3],
                palette="hls")
g.set_xlabel("Related Effect", fontsize=15)
g.set_ylabel("Rating Distribuition", fontsize=15)
g.set_title("Second Effect Related x Rating by Species Type",fontsize=20)

plt.show()


# ### Now let's see the first Flavor related
# 
# We have 33 flavors in total

# In[ ]:


strains.head()


# In[ ]:


strains.shape


# ### Exploring Flavors

# In[ ]:


print("TOP 10 Flavors related")
print(strains.Flavor_1.value_counts()[:10])

plt.figure(figsize=(14,6))
sns.countplot('Flavor_1', data=strains)
plt.xticks(rotation=90)
plt.xlabel('Flavors', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title("First flavors described ", fontsize=20)
plt.show()


# ### Let's explore the Strains with Rating equal 5

# In[ ]:


#Whats the type with most strains with rating 5?
print("Percentual of Species with Rating equal 5")
five_rating = (strains[strains["Rating"] == 5].groupby("Type")["Strain"].count()                / len(strains[strains["Rating"] == 5]) *100).round(decimals=2)
print(five_rating)
plt.figure(figsize=(10,6))
g = sns.countplot(x="Type",data=strains[strains["Rating"] == 5])
g.set_xlabel('Species', fontsize=15)
g.set_ylabel('Frequency', fontsize=15)
g.set_title("Distribuition of Types by Rating 5.0  ", fontsize=20)

plt.show()


# ### Exploring the principal effects and Flavors Related in Rating five strains

# In[ ]:


strains_top = strains[strains["Rating"] == 5]

fig, ax = plt.subplots(2,1, figsize=(12,10))

sns.countplot(x ='Effect_1',data = strains_top,hue="Type",ax=ax[0], palette='hls')

sns.countplot(x ='Flavor_1',data = strains_top,hue="Type",ax=ax[1], palette='hls')

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=45)


# ### 
# 
# Curious!
# We can see that in all types, the most related flavors are Sweet and Earthly.
# Is important to remember that we have alot of another flavors that are related with this Sweet and Earthly tastes.
# 
# We can also remember that the first cannabis strain was Skunk #1, that have a high earthly and pungent taste.
# 
# The distribuition total of data set is almost nearly of this values:
# 
#     hybrid 51.55
#     indica 29.73
#     sativa 18.72
# 
# Now I will Explore the total Effects and Flavors related to each strain
# 

# In[ ]:


#Let's create subsets by each type and explore their Flavors and Effects
hibridas = strains[strains.Type == 'hybrid']
indicas = strains[strains.Type == 'indica']
sativas = strains[strains.Type == 'sativa']


# In[ ]:


#Now we can delete some columns that will not be useful
del strains["Effects"]
del strains["Flavor"]


# In[ ]:


#Creating the spliter -- copied by LiamLarsen -- 
def get_effects(dataframe):
    ret_dict = {}
    for list_ef in dataframe.Effects:
        effects_list = list_ef.split(',')
        for effect in effects_list:
            if not effect in ret_dict:
                ret_dict[effect] = 1
            else:
                ret_dict[effect] += 1
    return ret_dict


# ### Sativas effects

# In[ ]:


#Creating the counting of effects
sativa_effects = get_effects(sativas)

#Let see the distribuition of effects by types
plt.figure(figsize=(10,8))
sns.barplot(list(sativa_effects.values()), list(sativa_effects.keys()), orient='h')
plt.xlabel("Count", fontsize=12)
plt.ylabel("Related effects", fontsize=12)
plt.title("Sativas strain effects distribution", fontsize=16)
plt.show()


# ### Indicas effects

# In[ ]:


# Couting effects of indicas 
indica_effects = get_effects(indicas)

# Ploting Indica Effects
plt.figure(figsize=(10,8))
sns.barplot(list(indica_effects.values()),list(indica_effects.keys()), orient='h')
plt.xlabel("Count", fontsize=15)
plt.ylabel("Related effects", fontsize=15)
plt.title("Indica strain effects distribution", fontsize=20)
plt.show()


# ### Hibrids flavors

# In[ ]:


hibridas_effects = get_effects(hibridas)

# Ploting Hybrid effects
plt.figure(figsize=(10,8))
sns.barplot(list(hibridas_effects.values()),list(hibridas_effects.keys()), orient='h')
plt.xlabel("Count", fontsize=15)
plt.ylabel("Related effects", fontsize=15)
plt.title("Hibrids strain effects distribution", fontsize=20)
plt.show()


# 5. Some observations:
# 
# Some observations:
# 
# We can clearly see that Happy, Uplified, Relaxed, Euphoric have a high ranking at all 3 types
# 
# Its interesting that almost 350 people of 440 in Sativas related Happy and Uplifted Effects
# 
#     'Happy': 342
#     'Uplifted': 328
#     'Euphoric': 276
#     'Energetic': 268
# 
# 78% has described Happy to Sativas strains
# 
# Indicas we have 699 votes and Relaxed with most frequency at distribuition:
# 
#     'Relaxed': 628
#     'Happy': 562
#     'Euphoric': 516
# 
# 90% has described Relaxed to Indica strains
# 
# Hybrids We have 1212 votes and distribuition of effects is
# 
#     'Happy': 967
#     'Relaxed': 896
#     'Uplifted': 848
#     'Euphoric': 843
# 
# 80% has described Happy and 74% related Relaxed to Hybrids strains
# Very Interesting!
# 
# We also need to remember that's possible to vote in more than 1 effect or flavor in each vote.
# 

# ### 6. Exploring general Flavors and Effects:
# 
# Now, let's check the flavors
# I will use the same loop to known the most related flavors

# In[ ]:


#Creating flavors to cut each flavor by row -- inspired in LiamLarsen --
def flavors(df):
    ret_dict = {}
    for list_ef in df.Flavor.dropna():
        flavors_list = list_ef.split(',')
        for flavor in flavors_list:
            if not flavor in ret_dict:
                ret_dict[flavor] = 1
            else:
                ret_dict[flavor] += 1
    return ret_dict


# ### Sativas Flavors

# In[ ]:


#Runing flavors counts to sativas
sativa_flavors = flavors(sativas)

plt.figure(figsize=(10,12))
sns.barplot(list(sativa_flavors.values()),list(sativa_flavors.keys()), orient='h')
plt.xlabel("Count", fontsize=12)
plt.ylabel("Most related flavors", fontsize=12)
plt.title("Sativa flavors distribution", fontsize=16)
plt.show()


# ### Most frequent flavors in Sativas:
# Sweet: 207 <br>
# Earthy: 178<br>
# Citrus: 143 <br>

# ### Indicas Flavors
# 

# In[ ]:


indica_flavors = flavors(indicas)

plt.figure(figsize=(10,12))
sns.barplot(list(indica_flavors.values()),list(indica_flavors.keys()), orient='h')
plt.xlabel("Count", fontsize=12)
plt.ylabel("Most related flavors",fontsize=12)
plt.title("Indica flavors distribution", fontsize=16)
plt.show()


# Most frequent values in indicas
# Earthy: 378 <br>
# Sweet: 312<br>
# Pungent: 157<br>
# Berry: 145 <br>

# ### Hibrids flavors

# In[ ]:


#Getting hibridas flavors
hibridas_flavors = flavors(hibridas)

plt.figure(figsize=(10,12))
sns.barplot(list(hibridas_flavors.values()),list(hibridas_flavors.keys()), alpha=0.8,orient='h')
plt.xlabel("Count", fontsize=15)
plt.ylabel("Most related flavors", fontsize=15)
plt.title("Hibrids flavors distribution", fontsize=20)
plt.show()


# The most frequent values in Hybrid type is: <br>
# Earthy: 549 <br>
# Sweet: 534<br>
# Citrus: 301 <br>

# ### Librarys to WordCloud
# 

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import nltk.tokenize as word_tokenize
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction import stop_words


# ### Word Clouds

# In[ ]:


stopwords = set(STOPWORDS)
newStopWords = ['strain','effect', 'genetic', 'effects','flavor',
                'dominant','known','cross']
stopwords.update(newStopWords)

wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=600,
                          random_state=42,
                         ).generate(" ".join(strains['Description'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - DESCRIPTION", fontsize=25)
plt.axis('off')
plt.show()


# ### Word Cloud Sativas

# In[ ]:


stopwords = set(STOPWORDS)
newStopWords = ['strain','effect', 'genetic', 'sativa', 'effects',
                'aroma','flavor','dominant','known','cross','genetics']
stopwords.update(newStopWords)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=200, 
                          width=1000, height=600,
                          random_state=42,
                         ).generate(" ".join(strains[strains.Type == 'sativa']['Description'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - SATIVAS", fontsize=25)
plt.axis('off')
plt.show()


# ### Word Cloud Indicas

# In[ ]:


stopwords = set(STOPWORDS)
newStopWords = ['strain','effect', 'genetic', 'indica', 'effects','aroma', 
                'genetics','flavor','dominant','known','cross']
stopwords.update(newStopWords)

wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=150, 
                          width=1000, height=600,
                          random_state=42,
                         ).generate(" ".join(strains[strains.Type == 'indica']['Description'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - INDICAS", fontsize=25)
plt.axis('off')
plt.show()


# ### Word Cloud Hybrids

# In[ ]:


stopwords = set(STOPWORDS)
newStopWords = ['strain','effect', 'genetic', 'hybrid', 'effects', 'aroma',
                'genetics', 'flavor', 'genetics','cross','dominant','known']
stopwords.update(newStopWords)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=150, 
                          width=1000, height=600,
                          random_state=42,
                         ).generate(" ".join(strains[strains.Type == 'hybrid']['Description'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - HYBRIDS", fontsize=25)
plt.axis('off')
plt.show()


# ### Word Cloud Rating 5 Strains

# In[ ]:


stopwords = set(STOPWORDS)
newStopWords = ['strain','effect', 'genetic','effects','cross','genetics',
                'aroma','consumer','known','dominant']
stopwords.update(newStopWords)

wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=1500,
                          max_font_size=150, 
                          width=1000, height=600,
                          random_state=42,
                         ).generate(" ".join(strains[strains.Rating == 5]['Description'].astype(str)))

fig = plt.figure(figsize = (12,12))
plt.imshow(wordcloud)
plt.title("WORD CLOUD - RATING 5", fontsize=25)
plt.axis('off')
plt.show()


# ### 7. Preprocessing dataset:
# 
# Knowing all this...Let's get high!
# 
# and try to predict the type of strain using flavors, effects and rating?

# In[ ]:


# Lets do some transformation in data

print(strains.head())


# In[ ]:


#Transformin the Type in numerical 
strains["Type"] = pd.factorize(strains["Type"])[0]
del strains["Description"]
# Now we have 3 numerical Types
# 0 - Hybrid
# 1 - Sativa
# 2 - Indica


# In[ ]:


# Creating the dummies variable of Effects and Flavors
#effect_dummy = strains['Effects'].str.get_dummies(sep=',',)
#flavor_dummy = strains['Flavor'].str.get_dummies(sep=',')

dummy = pd.get_dummies(strains[['Effect_1','Effect_2','Effect_3','Effect_4','Effect_5','Flavor_1','Flavor_2','Flavor_3']])


# In[ ]:


#Concatenating the result and droping the used variables 
strains = pd.concat([strains, dummy], axis=1)

strains = strains.drop(['Strain','Effect_1','Effect_2','Effect_3','Effect_4','Effect_5','Flavor_1','Flavor_2','Flavor_3'], axis=1)

strains.shape


# ### 8. Importing Sklearn and Modeling:

# In[ ]:


#Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score

#Models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, RandomTreesEmbedding


# In[ ]:


strains.head(2)


# In[ ]:


# setting X and y
X = strains.drop("Type",1)
y = strains["Type"]
feature_name = X.columns.tolist()
X = X.astype(np.float64, copy=False)
y = y.astype(np.float64, copy=False)


# In[ ]:


#Spliting the variables in train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)


# ### Feature Selection

# In[ ]:


thresh = 5 * 10**(-3)
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
#select features using threshold
selection = SelectFromModel(model, threshold=thresh, prefit=True)

X_important_train = selection.transform(X_train)
X_important_test = selection.transform(X_test)


# In[ ]:


print("X_important_train Shape: ", X_important_train.shape)
print("X_important_test Shape: ", X_important_test.shape)


# ### Let's take a look at some models and compare their score

# In[ ]:


clfs = []
seed = 3

clfs.append(("LogReg", 
             Pipeline([("Scaler", StandardScaler()),
                       ("LogReg", LogisticRegression())])))

clfs.append(("XGBClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("XGB", XGBClassifier())]))) 
clfs.append(("KNN", 
             Pipeline([("Scaler", StandardScaler()),
                       ("KNN", KNeighborsClassifier())]))) 

clfs.append(("DecisionTreeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeClassifier())]))) 

clfs.append(("RandomForestClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestClassifier())]))) 

clfs.append(("GradientBoostingClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, n_estimators=150))]))) 

clfs.append(("RidgeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RidgeClassifier", RidgeClassifier())])))

clfs.append(("BaggingRidgeClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("BaggingClassifier", BaggingClassifier())])))

clfs.append(("ExtraTreesClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("ExtraTrees", ExtraTreesClassifier())])))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'accuracy'
n_folds = 7

results, names  = [], [] 

for name, model  in clfs:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, X_important_train, y_train, cv= 5, scoring=scoring, n_jobs=-1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()


# ### I will select the top 3 models and set some hyperParameters to try increase their prediction power.
# - The top 3 will be:
#     - GradientBoostingClassifier
#     - XGBClassifier
#     - RidgeClassifier
# 
# 

# In[ ]:


from sklearn.grid_search import GridSearchCV

params_ridge = {'alpha':[0.001, 0.1, 1.0],
                'tol':[0.1, 0.01, 0.001], 
                'solver':['auto', 'svd', 'cholesky','lsqr', 'sparse_cg', 'sag', 'saga']}

ridge = RidgeClassifier()
    
Ridge_model = GridSearchCV(estimator = ridge, param_grid=params_ridge, verbose=2, n_jobs = -1)

# Fit the random search model
Ridge_model.fit(X_important_train, y_train)


# In[ ]:


# Printing the Training Score
print("Training score data: ")
print(Ridge_model.score(X_important_train, y_train) )
print("Ridge Best Parameters: ")
print(Ridge_model.best_params_ )


# - We got a nice improvement in our model compared with the first model without HyperParameters.

# Now, let's Predict with this model

# In[ ]:


# Predicting with X_test
Ridge_model = RidgeClassifier(solver='sparse_cg', tol=0.001, alpha=1.0)
Ridge_model.fit(X_important_train, y_train)
y_pred = Ridge_model.predict(X_important_test)

# Print the results
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ### Now, I will evaluate the best params to XGBoost model

# In[ ]:


param_xgb = {
 'n_estimators':[100,150,200],
 'max_depth':[3,4,5,6],
 'min_child_weight':[2,3,4,5],
 'colsample_bytree':[.1, 0.2, 0.3,0.6,0.7,0.8],
 'colsample_bylevel':[0.2,0.6,0.8]
}


# In[ ]:


xgb = XGBClassifier()

xgb_model = GridSearchCV(estimator = xgb, 
                        param_grid = param_xgb, 
                        scoring='accuracy',
                        cv=2,
                        verbose = 1)

xgb_model.fit(X_important_train, y_train)


# In[ ]:


print("Results of the GridSearchCV of XGB: ")
print(xgb_model.best_params_)
print(xgb_model.score(X_important_train, y_train))


# In[ ]:


# let's set the best parameters to our model and fit again
xgb = XGBClassifier(colsample_bylevel=0.6, colsample_bytree=0.1, objective='multi', max_depth= 4, min_child_weight= 2, n_estimators= 200)
xgb.fit(X_important_train, y_train)

# Predicting with X_test
y_pred = xgb.predict(X_important_test)

# Print the results
print("METRICS \nAccuracy Score: ", accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ### Now let's fit and predict with Gradient Boosting Classifier model

# In[ ]:


param_gb = {
    'n_estimators':[50, 125, 150],
    'max_depth':[2,3,4],
    'max_features':[3,4,5,6],
    'learning_rate':[0.0001, 0.001, 0.01,0.1,1]

}

gb = GradientBoostingClassifier()

gb_model = GridSearchCV(estimator = gb, 
                        param_grid = param_gb, 
                        scoring='accuracy',
                        cv=5,
                        verbose = 1)

gb_model.fit(X_important_train, y_train)


# In[ ]:


print("Results of the GridSearchCV of Gradient Boosting Classifier: ")
print(gb_model.best_params_)
print(gb_model.score(X_important_train, y_train))


# In[ ]:


gb = GradientBoostingClassifier(learning_rate=.1, max_depth= 3, max_features=6, n_estimators= 150)
gb.fit(X_important_train, y_train)

# Predicting with X_test
y_pred = gb.predict(X_important_test)

# Print the results
print("METRICS \nAccuracy Score: ", accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




