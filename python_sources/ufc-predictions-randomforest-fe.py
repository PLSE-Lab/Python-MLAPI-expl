#!/usr/bin/env python
# coding: utf-8

# # UFC prediction
# 
# Hello everyone, the purpose of this notebooks is to predict the result of UFC matches (UFC 245 event is the latest one as I'm curently writting this introduction). 
# 
# But first, let's speak a little bit about the context of this dataset.  
# The UFC is nowadays the biggest mixed martial arts competition organization in the world in terms of views and prestige. The fighters fighting in this one are usually considered as the best in the world (they often came from different organisations and got promoted there after a long road). People have always wanted to predict the winner of those kind of fights and even more today as bettings on such games are now becoming so huge that it is becoming closer to what a boxing match could make for sponsors and sport bet organisations concerning earnings. To be able to predict (showing probabilities) the final result of a UFC match could be a good betting decision helper and therefore making fights even more exciting to watch. 
# 
# The special thing about this dataset to keep in mind is that each row is a compilation of both fighter statistics UP UNTIL THIS FIGHT (and this is very important to understand !). 

# In[ ]:


from IPython.display import Image
Image("/kaggle/input/ufc245/UFC 245.jpg")


# In[ ]:


import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[ ]:


df = pd.read_csv('/kaggle/input/ufcdata/data.csv')
b_age = df['B_age']  #we replace B_age to put it among B features
df.drop(['B_age'], axis = 1, inplace = True)
df.insert(76, "B_age", b_age)
df1 = df.copy() #We make a copy of the dataframe for the feature engineering part later
df.head(4)


# In[ ]:


print(df.shape)
len(df[df['Winner'] == 'Draw'])


# In[ ]:


for col in df: print(col)


# The last fight (and ufc event) recorded on this dataset was on the 8th November of 2019.

# In[ ]:


last_fight = df.loc[0, ['date']]
print(last_fight)


# ## Data Cleaning

# Before April 2001, there were almost no rules in UFC (no judges, no time limits, no rounds, etc.). It's up to this precise date that UFC started to implement a set of rules known as "Unified Rules of Mixed Martial Arts" in accordance with the New Jersey State Athletic Control Board in United States. Therefore we will delete all fights before this major update in UFC's rules history. 

# In[ ]:


limit_date = '2001-04-01'
df = df[(df['date'] > limit_date)]
print(df.shape)


# In[ ]:


print("Total NaN in dataframe :" , df.isna().sum().sum())
print("Total NaN in each column of the dataframe")
na = []
for index, col in enumerate(df):
    na.append((index, df[col].isna().sum())) 
na_sorted = na.copy()
na_sorted.sort(key = lambda x: x[1], reverse = True) 

for i in range(len(df.columns)):
    print(df.columns[na_sorted[i][0]],":", na_sorted[i][1], "NaN")


# Most NaN values can be explained because of empty statistics of new fighters joining UFC and fighting for their first time (they get NaN values until their first fight, so according to how the dataset is built, those statistics are filled in their second fight). 

# In[ ]:


print('Number of features with NaN values :', len([x[1] for x in na if x[1] > 0]))


# We delete all NaN rows from na_features columns.  
# We also delete B_draw and R_draw columns as all those feature's values are fixed to 0 and hence it won't have any impact in our predictive model.  
# We delete every match where the result is a draw (equality), indeed we don't want to add an additional class to our target variable. Now we only have a winner and a looser. 

# In[ ]:


na_features = ['B_Reach_cms', 'B_avg_BODY_att', 'R_avg_BODY_att', 'R_Stance', 'B_Stance', 'R_Reach_cms', 'Referee', 'B_age']
df.dropna(subset = na_features, inplace = True)

df.loc[:,'B_draw'].value_counts() 
df.drop(['B_draw', 'R_draw'], axis = 1, inplace = True)
df = df[df['Winner'] != 'Draw']


# In[ ]:


print(df.shape)
print("Total NaN in dataframe :" , df.isna().sum().sum())


# ## Feature Engineering

# In[ ]:


df.info()


# There are 133 quantitative features and 10 categorical features (by looking at the dataset). 
# Let's see which features are categorical features.

# In[ ]:


list(df.select_dtypes(include=['object', 'bool']))


# Let's drop Referee and Location columns as they are kinda useless for our model. We still need to keep date to filter our dataframe and also the rest of the categorical features. 

# In[ ]:


df.drop(['Referee', 'location'], axis = 1, inplace = True)


# In[ ]:


df.corr(method = 'pearson').abs()


# In[ ]:


corr_matrix = df.corr(method = 'pearson').abs()
sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1)
                 .astype(np.bool))
                 .stack()
                 .sort_values(ascending=False)
      )
print(sol[0:10])


# We can notice that variables which are higly correlated between each other are the "attempted" and "landed" ones. Intuitively, the more strikes a fighter attempts, the more strikes are actually landed to his opponent.  
# For instance, the higher the average significant head strikes "landed of attempted" for one fighter, the higher the average significant strikes "landed of attempted" he will get.  
# 
# Though, we can't only keep "landed" variables because we need to know the accuracy of the fighter (whether on leg kicks, head or body strikes, submission , clinches, etc.). Maybe he attempts a lot of shots but has a hard time touching his opponent.  
# We will try to transform some features and hence reduce the number of variables to build a more comprehensive and lighter dataframe later. The goal is to keep main informations while avoiding dependence between features. But first, we are going to use all our dataframe features on the model. 
# 
# We can also notice that the number of rounds is highly correlated to the title bout. The rule is that for a title bout, the fights must last 5 rounds maximum and only 3 for a non title bout. But UFC changed rules to allow non title bout fights to last 5 rounds (those on the main cards acctually). 

# ## Data preparation

# As we only need the last fights of UFC fighters to get their last updated statistics and hence feed it into our model, we don't have to keep the previous fights from the active fighters as it will not make any difference in the model's performance.  
# In another words, we want to train our model on every fighter's fight at a moment T - 1 where T is the last fight of the fighter. We will then test our model on every fighter's fight at the moment T. We drop fights at moments T-2, T-3, ...

# In[ ]:


#i = index of the fighter's fight, 0 means the last fight, -1 means first fight
def select_fight_row(df, name, i): 
    df1 = df[(df['R_fighter'] == name) | (df['B_fighter'] == name)]
    df1.reset_index(drop=True, inplace=True) #as we created a new temporary dataframe, we have to reset indexes
    idx = max(df1.index)  #get the index of the oldest fight
    if i > idx:  #if we are looking for a fight that didn't exist, we return an empty array
        return []
    arr = df1.iloc[i,:].values
    return arr

select_fight_row(df, 'Amanda Nunes', 0) #we get the last fight of Amanda Nunes


# In[ ]:


#function to make sure that we don't add the same figther twice in the list
def is_in(liste, name):
    if len(liste) == 0:
        liste = []
        liste.append(name)
    for i in range(len(liste)):
        if name == liste[i]:
            return liste 
    liste.append(name)
    return liste

#get all active UFC fighters (according to the limit_date parameter)
def list_fighters(df, limit_date):
    names = []
    for i, row in df.iterrows():
        if (row['date'] > limit_date):  
            names = is_in(names, row['R_fighter']) 
            names = is_in(names, row['B_fighter']) 
    return names


# In[ ]:


fighters = list_fighters(df, '2017-01-01')
print(len(fighters))


# We build a new DataFrame by adding the last fight of every active UFC fighter. 

# In[ ]:


arr = [select_fight_row(df, fighters[i], 0) for i in range(len(fighters)) if len(select_fight_row(df, fighters[i], 0)) > 0]
cols = [col for col in df] 

last_fights0 = pd.DataFrame(data = arr, columns = cols)
last_fights0.drop_duplicates(inplace = True)
last_fights0.head(4)


# We build another Dataframe by adding the second last fight of every active UFC fighter.

# In[ ]:


arr = [select_fight_row(df, fighters[i], 1) for i in range(len(fighters)) if len(select_fight_row(df, fighters[i], 1)) > 0]
last_fights1 = pd.DataFrame(data = arr, columns = cols)
last_fights1.drop_duplicates(inplace = True)
last_fights1.head(4)


# In[ ]:


print(last_fights0.shape)
print(last_fights1.shape)


# ## Data preprocessing
#   
# Let's create a pipeline to make transformation on each categorical feature.  
# We will rather use ordinal/label encoder than one-hot-encoder as it leads to a poorer accuracy on Random Forest model when dummie features get high. Here is a good explanation why : https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769.  
# Furthermore, we will be able to interpret feature importances easily. 

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

#If the winner is from the Red corner, Winner value will be encoded as 1, otherwise it will be 0 (Blue corner)
#preprocess = make_column_transformer(
#    (OrdinalEncoder(), ['Winner', 'title_bout', 'weight_class', 'B_Stance', 'R_Stance']),  
#    remainder='passthrough'
#)
preprocess = make_column_transformer(
    (OrdinalEncoder(), [3, 4, 5, 69, 136]),  
    remainder='passthrough'
)

last_fights0_preprocess = preprocess.fit_transform(last_fights0)
last_fights1_preprocess = preprocess.transform(last_fights1) 


# In[ ]:


last_fights0_preprocess[0,:]


# In[ ]:


last_fights0_preprocess = np.delete(last_fights0_preprocess, [5,6,7], axis = 1) #We remove names and date columns
last_fights1_preprocess = np.delete(last_fights1_preprocess, [5,6,7], axis = 1)


# In[ ]:


last_fights0_preprocess[0,:]


# In[ ]:


X_train, y_train = last_fights1_preprocess[:,1:], last_fights1_preprocess[:,0]
X_test, y_test = last_fights0_preprocess[:,1:], last_fights0_preprocess[:,0]
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ## Random Forest Model
# 
# Random Forest is a tree-based model and hence does not require feature scaling. Those algorithm computations aren't based on distance (euclidian distance or whatever), therefore, normalizing data is useless.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

# Random Forest composed of 500 decision trees. We optimized parameters using cross-validation and GridSearch tool paired together
classifier = RandomForestClassifier(n_estimators = 500, 
                                    criterion = 'entropy', 
                                    max_depth = 20, 
                                    min_samples_split = 2,
                                    min_samples_leaf = 3, 
                                    random_state = 0
                                   )
classifier.fit(X_train, y_train)

#We use cross-validation with 10-folds to have a more precise accuracy (reduce variation)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('Accuracy mean : ', accuracies.mean())
print('Accuracy standard deviation : ', accuracies.std())

#print('Training accuracy : ', classifier.score(X_train, y_train))
y_pred = classifier.predict(X_test)
print('Testing accuracy : ', accuracy_score(y_test, y_pred), '\n')

target_names = ["Blue","Red"]
print(classification_report(y_test, y_pred, labels = [0, 1], target_names = target_names))


# We get an accuracy of 0.66 so as average f1-score which is pretty good for such an uncertain sport as MMA.

# In[ ]:


#from sklearn.model_selection import GridSearchCV
#parameters = [{'n_estimators': [10, 50, 100, 500, 1000],
#               'criterion': ['gini', 'entropy'],
#               'max_depth': [5, 10, 50],
#               'min_samples_split': [2, 3, 4],
#               'min_samples_leaf': [1, 2, 3],
#              }]

#grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_

#best_params = grid_search.best_params_
#print('Best accuracy : ', best_accuracy)
#print('Best parameters : ', best_params)


# Remind that Blue => 0 and Red => 1 for the target value (Winner).

# In[ ]:


from sklearn.metrics import confusion_matrix

#The confusion matrix is like that:
#[TN FP
# FN TP]
cm = confusion_matrix(y_test, y_pred) 
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax, fmt = "d")
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(['Blue', 'Red'])
ax.yaxis.set_ticklabels(['Blue', 'Red'])


# In[ ]:


feature_importances = classifier.feature_importances_
indices = np.argsort(feature_importances)[::-1]
n = 20 #maximum feature importances displayed
feature_names = [name for name in cols if name not in ["R_fighter", "B_fighter", "date", "Winner","R_Stance", "B_Stance"]]
feature_names.insert(2, "B_Stance") 
feature_names.insert(3, "R_Stance")
idx = indices[0:n] 
std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)

#for f in range(n):
#    print("%d. feature %s (%f)" % (f + 1, feature_names[idx[f]], feature_importances[idx[f]])) 

plt.figure(figsize=(20, 6))
plt.title("Feature importances")
plt.bar(range(n), feature_importances[idx], color="r", yerr=std[idx], align="center")
plt.xticks(range(n), [feature_names[id] for id in idx], rotation = 45) 
plt.xlim([-1, n]) 
plt.show()


# There are no big significant features that could explain by themselves the model results. Almost all variables have an impact on the model even if it's on a very small proportion.  
# But we can notice that takedowns attempts, significant strikes that opponents are doing on fighters and age play a slightly bigger role which makes sense.  
# 
# 
# Let's visualize now the construction of a single Decision Tree among the Forest. 

# In[ ]:


from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

tree_estimator = classifier.estimators_[10]
export_graphviz(tree_estimator, 
                out_file='tree.dot', 
                filled=True, 
                rounded=True)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
Image(filename = 'tree.png')


# No tree has a higher max_depth than 20. The main features used to split the tree at each step are often the ones represented in the feature importances plot earlier. 

# ## Predictions
# 
# Let's make predictions on the next UFC event introducing one waited fight on the main card between Kamaru Usman and Colby Covington (event occuring the 15/12/2019, UFC 245).

# In[ ]:


def predict(df, preprocess, classifier, blue_fighter, red_fighter, weightclass, rounds, title_bout = False): 
    #We build two dataframes, one for each figther 
    f1 = df[(df['R_fighter'] == blue_fighter) | (df['B_fighter'] == blue_fighter)]
    f1.reset_index(drop=True, inplace=True)
    f1 = f1[:1]
    f2 = df[(df['R_fighter'] == red_fighter) | (df['B_fighter'] == red_fighter)] 
    f2.reset_index(drop=True, inplace=True)
    f2 = f2[:1]
    #if the fighter was red/blue corner on his last fight, we filter columns to only keep his statistics (and not the other fighter)
    #then we rename columns according to the color of  the corner in the parameters using re.sub()
    if (f1.loc[0, ['R_fighter']].values[0]) == blue_fighter:
        result1 = f1.filter(regex='^R', axis=1) #here we keep the red corner stats
        result1.rename(columns = lambda x: re.sub('[R]','B', x), inplace=True)  #we rename it with "B_" prefix because he's in the blue_corner
    else: 
        result1 = f1.filter(regex='^B', axis=1)
    if (f2.loc[0, ['R_fighter']].values[0]) == red_fighter:
        result2 = f2.filter(regex='^R', axis=1)
    else:
        result2 = f2.filter(regex='^B', axis=1)
        result2.rename(columns = lambda x: re.sub('[B]','R', x), inplace=True)
    result = pd.concat([result1, result2], axis = 1) #we concatenate the red and blue fighter dataframes (in columns)
    #We must prepare the data as we did when we trained our model because we use the same pipeline object (preprocess)
    result.drop(['R_fighter','B_fighter'], axis = 1, inplace = True) 
    result.insert(0, 'R_fighter', red_fighter) 
    result.insert(1, 'B_fighter', blue_fighter) 
    result.insert(2, 'date', '2019') #this value is random, the column will be removed anyway but we need it for the preprocess step 
    result.insert(3, 'Winner', 'Red') #same here
    result.insert(4, 'title_bout', title_bout) 
    result.insert(5, 'weight_class', weightclass)
    result.insert(6, 'no_of_rounds', rounds)
    preprocess_result = preprocess.transform(result)
    fights = np.delete(preprocess_result, [5,6,7], axis = 1)[:,1:] #we remove fighter names, date and winner
    pred = classifier.predict(fights)
    proba = classifier.predict_proba(fights)
    if (pred == 1.0): 
        print("The predicted winner is", red_fighter, 'with a probability of', round(proba[0][1] * 100, 2), "%")
    else:
        print("The predicted winner is", blue_fighter, 'with a probability of ', round(proba[0][0] * 100, 2), "%")
    return proba


# In[ ]:


predict(df, preprocess, classifier, 'Kamaru Usman', 'Colby Covington', 'Welterweight', 5, True) 


# In[ ]:


predict(df, preprocess, classifier, 'Max Holloway', 'Alexander Volkanovski', 'Featherweight', 5, True) 


# In[ ]:


predict(df, preprocess, classifier, 'Amanda Nunes', 'Germaine de Randamie', "Women's Bantamweight", 5, True)


# In[ ]:


predict(df, preprocess, classifier, 'Jose Aldo', 'Marlon Moraes', 'Bantamweight', 3, False)


# In[ ]:


predict(df, preprocess, classifier, 'Urijah Faber', 'Petr Yan', 'Bantamweight', 3, False)


# In[ ]:





# In[ ]:





# After testing our Random Forest model on almost all features on our first dataframe, we are going to apply some logical transformations on features that make sense like only keeping the ratio (accuracy) between attempts and landed shots for instance. We'll also drop some extra features like the winning's types which don't bring a lot of information to our model. The purpose is to see how well those changes are improving our model's performance (if it's statisticaly significant or not). 

# In[ ]:


features = ['avg_BODY', 'avg_CLINCH', 'avg_DISTANCE', 'avg_GROUND', 'avg_HEAD', 'avg_LEG', 'avg_SIG_STR', 'avg_TD', 'avg_TOTAL_STR', 'avg_opp_BODY', 'avg_opp_CLINCH', 'avg_opp_DISTANCE', 'avg_opp_GROUND', 'avg_opp_HEAD', 'avg_opp_LEG', 'avg_opp_SIG_STR', 'avg_opp_TD', 'avg_opp_TOTAL_STR']

for i in range(len(features)):
    feature_att = 'B_' + features[i] + '_att'
    feature_landed = 'B_' + features[i] + '_landed'
    feature_acc_name = 'B_' + features[i] + '_acc'
    feature_acc = df1[feature_landed] / df1[feature_att]
    df1.drop([feature_landed, feature_att], axis = 1, inplace = True)
    df1.insert(12 + i, feature_acc_name, feature_acc)
    
for i in range(len(features)):
    feature_att = 'R_' + features[i] + '_att'
    feature_landed = 'R_' + features[i] + '_landed'
    feature_acc_name = 'R_' + features[i] + '_acc'
    feature_acc = df1[feature_landed] / df1[feature_att]
    df1.drop([feature_landed, feature_att], axis = 1, inplace = True)
    df1.insert(62 + i, feature_acc_name, feature_acc)

#Let's drop the percentage variables that are no longer needed and the type of wins (decision, submision, TKO, etc.)
for feat in df1:
    if ("pct") in feat:
        df1.drop([feat], axis = 1, inplace = True)
    if ("by") in feat: 
        df1.drop([feat], axis = 1, inplace = True)

print(df1.shape)


# We reduced features number from 145 to 89. 

# In[ ]:


df1 = df1[(df1['date'] > limit_date)]
cols = df1.columns
df1.dropna(subset = cols, inplace = True)
df1.loc[:,'B_draw'].value_counts() 
df1.drop(['B_draw', 'R_draw'], axis = 1, inplace = True)
df1 = df1[df1['Winner'] != 'Draw']
print(df1.shape)
print("Total NaN in dataframe :" , df1.isna().sum().sum())


# In[ ]:


df1.drop(['Referee', 'location'], axis = 1, inplace = True)
fighters = list_fighters(df1, '2017-01-01')
print(len(fighters))


# In[ ]:


arr = [select_fight_row(df1, fighters[i], 0) for i in range(len(fighters)) if len(select_fight_row(df1, fighters[i], 0)) > 0]
cols = [col for col in df1] 
last_fights_0 = pd.DataFrame(data = arr, columns = cols)
last_fights_0.drop_duplicates(inplace = True)
last_fights_0.head(4)
arr = [select_fight_row(df1, fighters[i], 1) for i in range(len(fighters)) if len(select_fight_row(df1, fighters[i], 1)) > 0]
last_fights_1 = pd.DataFrame(data = arr, columns = cols)
last_fights_1.drop_duplicates(inplace = True)
last_fights_1.head(4)
preprocess1 = make_column_transformer(
    (OrdinalEncoder(), ['Winner', 'title_bout', 'weight_class', 'B_Stance', 'R_Stance']),  
    remainder='passthrough'
)
last_fights_0_preprocess = preprocess1.fit_transform(last_fights_0)
last_fights_1_preprocess = preprocess1.transform(last_fights_1) 
last_fights_0_preprocess = np.delete(last_fights_0_preprocess, [5,6,7], axis = 1) 
last_fights_1_preprocess = np.delete(last_fights_1_preprocess, [5,6,7], axis = 1)
X_train1, y_train1 = last_fights_1_preprocess[:,1:], last_fights_1_preprocess[:,0]
X_test1, y_test1 = last_fights_0_preprocess[:,1:], last_fights_0_preprocess[:,0]
y_train1 = y_train1.astype('int')
y_test1 = y_test1.astype('int')
print(X_train1.shape)
print(y_train1.shape)
print(X_test1.shape)
print(y_test1.shape)


# In[ ]:


classifier1 = RandomForestClassifier(n_estimators = 500, 
                                    criterion = 'entropy', 
                                    max_depth = 10, 
                                    min_samples_split = 2,
                                    min_samples_leaf = 2, 
                                    random_state = 0
                                   )
classifier1.fit(X_train1, y_train1)

accuracies = cross_val_score(estimator = classifier1, X = X_train1, y = y_train1, cv = 5)
print('Accuracy mean : ', accuracies.mean())
print('Accuracy standard deviation : ', accuracies.std())

y_pred1 = classifier1.predict(X_test1)
print('Testing accuracy : ', accuracy_score(y_test1, y_pred1), '\n')

target_names = ["Blue","Red"]
print(classification_report(y_test1, y_pred1, labels = [0, 1], target_names = target_names))


# In[ ]:


cm = confusion_matrix(y_test1, y_pred1) 
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax, fmt = "d")
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(['Blue', 'Red'])
ax.yaxis.set_ticklabels(['Blue', 'Red'])


# We get a final accuracy of around 0.66 on the test set which is quite similar to the model containing all variables even if general training accuracy is smaller (0.58). Maybe the reason is because the model trained on a dataset too small as the number of NaN values were higher on this one due to feature transformation . Anyway, it still performs well on test set.  
# 
# I will try to update my work when I got some time. 
