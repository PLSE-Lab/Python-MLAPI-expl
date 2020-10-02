#!/usr/bin/env python
# coding: utf-8

# # **World Cup 2018 Prediction by Ang Peng Seng**
# 
# The purpose of this is to try and predict the top 3 teams for World Cup 2018 using classification models coupled with poisson distribution to predict the exact results of the semi-finals, third place playoff and final. 
# 
# ## **Final Predictions based on this notebook:**
# 
# **Winner**: Germany
# 
# **2nd Place**: Spain
# 
# **3rd Place**: France
# 
# **Final Score**: Germany VS Spain: 2-1
# 
# **Third place playoff score**: France vs England: 1-1 (France win Penalty Shootout)
# 
# 
# 
# ## **Contents:**
# 
# **1. Import Necessary Packages/Datasets** 
# 
# **2. Data Cleaning**
# 
# **3. Classification Models to predict match results (Win/Draw/Lose)**
# - Variables used:
#     - Which stadium is it played at (0 -neutral, 1-away team's stadium, 2- home team's stadium)
#     - Whether the match is an important match or a friendly match (0 - Friendly, 1- Important)
#     - How much the Home team's rank changes compared to the past period 
#     - How much the Away team's rank changes compared to the past period 
#     - Difference in the 2 team's ranking
#     - Difference in the 2 team's mean weighted ratings over the past 3 years
# 
# **4. Classification Models to predict exact goals scored by Home and Away Sides**
# 
# - Variables used same as in (3)
# 
# **5. Visualizing ability/potential of players of the 32 countries**
# 
# **6. Adding variables to build a poisson model**
#  
# - Variables used:
#     - Soccer Power Index
#     - Average Age
#     - Average Height
#     - Total World Cup Appearances
#     - Average goals scored per game
#     - Average goals conceded per game
#     - Potential
# 
# **7. Predicting World Cup 2018**
# - Detailed Methods on how to simulate and predict the World Cup 2018 Matches will be explained here

# # **1. Import Necessary Packages/Datasets**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import random
import numpy as np 
import scipy as sp 
from scipy.stats import poisson
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import pandas as pd 
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import xgboost as xgb
import scikitplot as skplt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


countries = pd.read_csv("../input/world-cup-dataset/01 2018 World Cup Team Statistics.csv")
historical = pd.read_csv("../input/world-cup-dataset/02 Historical Data on World Cup Matches.csv")
player_stats_18 = pd.read_csv("../input/world-cup-dataset/03 2018 FIFA Players Database.csv")
results = pd.read_csv("../input/cleaned-matches/cleaned_matches.csv")
squads = pd.read_csv("../input/2018-fifa-world-cup-squads/2018 FIFA World Cup Squads.csv")
fifa18 = pd.read_csv("../input/fifa-18-more-complete-player-dataset/complete.csv")
results_so_far = pd.read_csv("../input/updatedfixtures/05 2018 World Cup Match Fixtures.csv")
stats = pd.read_csv("../input/world-cup-dataset/01 2018 World Cup Team Statistics.csv")
world_cup = pd.read_csv("../input/fifa-worldcup-2018-dataset/World Cup 2018 Dataset.csv")


# # **2. Data Cleaning**

# In[ ]:


squads.Player = squads.Player.apply(lambda x: x[:-10] if "captain" in x else x)


# In[ ]:


fifa18 = fifa18.replace({"Korea Republic":"South Korea"})
wc_player_stats_18 = fifa18[fifa18["name"].isin(squads.Player) | fifa18["full_name"].isin(squads.Player)]
wc_fifa18_stats = fifa18[fifa18.nationality.isin(squads.Team)]


# Updated Results as at 23/06

# In[ ]:


results_so_far = results_so_far.replace({"Korea Republic":"South Korea"})
results_so_far["Result"] = np.where(results_so_far["Home Team Goals"] < results_so_far["Away Team Goals"], 0, np.where(results_so_far["Home Team Goals"]==results_so_far["Away Team Goals"],1,2))
results_so_far["Matches"] = results_so_far["Home Team Name"] + "," + results_so_far["Away Team Name"]
results_so_far = results_so_far.dropna(how="any")
results_so_far = results_so_far.drop(["Year","Match date","Stage","Stadium","Group"],axis=1)
results_so_far["Home Team Goals"] = results_so_far["Home Team Goals"].apply(lambda x: int(x))
results_so_far["Away Team Goals"] = results_so_far["Away Team Goals"].apply(lambda x: int(x))
results_so_far.tail(2)


# In[ ]:


results = results.drop(["Unnamed: 0"],axis=1)
results.reset_index(inplace=True,drop=True)
results.tail(2)


# In[ ]:


world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
world_cup = world_cup.dropna(how='all')
world_cup = world_cup.replace({"IRAN": "Iran", 
                               "Costarica": "Costa Rica", 
                               "Porugal": "Portugal", 
                               "Columbia": "Colombia", 
                               "Korea" : "South Korea"})
world_cup = world_cup.set_index('Team')
world_cup.head(4)


# In[ ]:


wc_countries = countries[countries.team.isin(squads.Team)]
wc_countries.head(2)


# In[ ]:


squads = pd.read_csv("../input/2018-fifa-world-cup-squads/2018 FIFA World Cup Squads.csv")
rankings = pd.read_csv("../input/fifa-international-soccer-mens-ranking-1993now/fifa_ranking.csv")

rankings_prev = rankings[rankings["rank_date"]=="2018-05-17"]
rankings_prev = rankings_prev.replace({"IR Iran":"Iran","Korea Republic":"South Korea"})
rankings_prev = rankings_prev.loc[rankings_prev["country_full"].isin(list(squads["Team"]))]
rankings_prev=rankings_prev.drop(["country_abrv","confederation"],axis=1)
rankings_prev.reset_index(inplace=True,drop=True)
rankings_prev = rankings_prev.set_index("country_full",drop=True)

rankings_18 = rankings[rankings["rank_date"]=="2018-06-07"]
rankings_18 = rankings_18.replace({"IR Iran":"Iran","Korea Republic":"South Korea"})
rankings_18 = rankings_18.loc[rankings_18["country_full"].isin(list(squads["Team"]))]
rankings_18=rankings_18.drop(["country_abrv","confederation"],axis=1)
rankings_18.reset_index(inplace=True,drop=True)
rankings_18 = rankings_18.set_index("country_full",drop=False)

rankings_18["mean_weighted_over_years"] = (rankings_18["cur_year_avg_weighted"]+rankings_18["last_year_avg_weighted"]+
                                            rankings_18["two_year_ago_weighted"]+rankings_18["three_year_ago_weighted"])/4
rankings_18.head()


# # **3. Classification Models to predict match results (Win/Draw/Lose)**

# **The 6 variables used to predict the results of a match are: **
# - Which stadium is it played at (0 -neutral, 1-away team's stadium, 2- home team's stadium)
# - Whether the match is an important match or a friendly match (0 - Friendly, 1- Important)
# - How much the Home team's rank changes compared to the past period 
# - How much the Away team's rank changes compared to the past period 
# - Difference in the 2 team's ranking
# - Difference in the 2 team's mean weighted ratings over the past 3 years

# In[ ]:


x = results.loc[:,["country","impt","home_rank_change","away_rank_change","diff_in_ranking","diff_in_mean_weighted_over_years"]]
y = results.loc[:,"Result"]


# ## **3.1 Splitting into training and test set**

# We shall use 80% of our dataset as our training set and 20% as our test set. We will also apply 5-fold Cross Validation

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=0)


# Let's define some function to evaluate our models

# In[ ]:


def train_acc_score(model):
    return round(np.mean(cross_val_score(model,x_train,y_train,cv=k_fold,scoring="accuracy")),2)

def test_acc_score(model):
    return round(accuracy_score(y_test, model.predict(x_test)),2)

def train_prec_score(model):
    return round(precision_score(y_train,model.predict(x_train),average='macro'),2)

def test_prec_score(model):
    return round(precision_score(y_test,model.predict(x_test),average='macro'),2)

def train_f1(model):
    return round(f1_score(y_train,model.predict(x_train),average='macro'),2)

def test_f1(model):
    return round(f1_score(y_test,model.predict(x_test),average='macro'),2)

def confusion_matrix_model(model_used):
    cm=confusion_matrix(y_test,model_used.predict(x_test))
    col=["Predicted Away Win","Predicted Draw","Predicted Home Win"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Away Win","Predicted Draw","Predicted Home Win"]
    cm.index=["Actual Away Win","Actual Draw","Actual Home Win"]
    return cm.T

def confusion_matrix_model_train(model_used):
    cm=confusion_matrix(y_train,model_used.predict(x_train))
    col=["Predicted Away Win","Predicted Draw","Predicted Home Win"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Away Win","Predicted Draw","Predicted Home Win"]
    cm.index=["Actual Away Win","Actual Draw","Actual Home Win"]
    return cm.T

def importance_of_features(model):
    features = pd.DataFrame()
    features['feature'] = x_train.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    return features.plot(kind='barh', figsize=(6,6))


# ## **3.2 Building classification models to predict results**

# The models will be optimised using GridSearchCV based on F1 score. F1 score gives a weighted average between precision and accuracy/recall. It tells you how precise your classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances).
# 
# I have typed in some of the optimised parameters based on the GridSearchCV code output, then commented out the GridSearchCV codes to make the notebook run faster as it won't be re-optimised.
# 
# Confusion matrix table and details will only be shown for the final selected models in order to save space. There would be a summary of each models in the evaluation section below

# **3.2.1. Logistic Regression (Lasso)**

# In[ ]:


param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
log_reg1 = GridSearchCV(LogisticRegression(penalty="l1"),param_grid=param_grid,scoring="f1_macro")
#log_reg1=LogisticRegression(penalty="l1")
log_reg1.fit(x_train,y_train)
print(log_reg1.best_params_)
print("In-sample accuracy: " + str(train_acc_score(log_reg1)))
print("Test accuracy: " + str(test_acc_score(log_reg1)))
print ("In-sample Precision Score: " + str(train_prec_score(log_reg1)))
print ("Test Precision Score: " + str(test_prec_score(log_reg1)))
print ("In-sample F1 Score: " + str(train_f1(log_reg1)))
print ("Test F1 Score: " + str(test_f1(log_reg1)))
#confusion_matrix_model_train(log_reg1)


# **3.2.2. Logistic Regression (Ridge)**

# In[ ]:


param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
log_reg2 = GridSearchCV(LogisticRegression(penalty="l2"),param_grid=param_grid,scoring="f1_macro")
#log_reg2=LogisticRegression(penalty="l2",C=0.01)
log_reg2.fit(x_train,y_train)
print(log_reg2.best_params_)
print("In-sample accuracy: " + str(train_acc_score(log_reg2)))
print("Test accuracy: " + str(test_acc_score(log_reg2)))
print ("In-sample Precision Score: " + str(train_prec_score(log_reg2)))
print ("Test Precision Score: " + str(test_prec_score(log_reg2)))
print ("In-sample F1 Score: " + str(train_f1(log_reg2)))
print ("Test F1 Score: " + str(test_f1(log_reg2)))
#confusion_matrix_model_train(log_reg2)


# **3.2.3. SVM (RBF Kernel)**

# In[ ]:


#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#svc_rbf = GridSearchCV(SVC(kernel="rbf",random_state=0),param_grid=param_grid,scoring="f1_macro")
svc_rbf = SVC(kernel='rbf', gamma=0.001, C=0.5,random_state=0)
svc_rbf.fit(x_train, y_train)
#print(svc_rbf.best_params_)
print("In-sample accuracy: " + str(train_acc_score(svc_rbf)))
print("Test accuracy: " + str(test_acc_score(svc_rbf)))
print ("In-sample Precision Score: " + str(train_prec_score(svc_rbf)))
print ("Test Precision Score: " + str(test_prec_score(svc_rbf)))
print ("In-sample F1 Score: " + str(train_f1(svc_rbf)))
print ("Test F1 Score: " + str(test_f1(svc_rbf)))
#confusion_matrix_model_train(svc_rbf)


# **3.2.4. SVM (Linear Kernel)**

# In[ ]:


#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#svc_lin= GridSearchCV(SVC(kernel="linear",random_state=0),param_grid=param_grid,scoring="f1_macro")
svc_lin = SVC(kernel='linear', gamma=0.001, C=0.1,random_state=0)
svc_lin.fit(x_train, y_train)
#print(svc_lin.best_params_)
print("In-sample accuracy: " + str(train_acc_score(svc_lin)))
print("Test accuracy: " + str(test_acc_score(svc_lin)))
print ("In-sample Precision Score: " + str(train_prec_score(svc_lin)))
print ("Test Precision Score: " + str(test_prec_score(svc_lin)))
print ("In-sample F1 Score: " + str(train_f1(svc_lin)))
print ("Test F1 Score: " + str(test_f1(svc_lin)))
#confusion_matrix_model_train(svc_lin)


# **3.2.5. K-Nearest Neighbour**

# In[ ]:


#param_grid = dict(n_neighbors=np.arange(10,70),weights=("uniform","distance"),p=(1,2))
#KNN = GridSearchCV(KNeighborsClassifier(),param_grid=param_grid,scoring="f1_macro")
KNN=KNeighborsClassifier(n_neighbors=16,p=1,weights='uniform')
KNN.fit(x_train,y_train)
#print(KNN.best_params_)
print("In-sample accuracy: " + str(train_acc_score(KNN)))
print("Test accuracy: " + str(test_acc_score(KNN)))
print ("In-sample Precision Score: " + str(train_prec_score(KNN)))
print ("Test Precision Score: " + str(test_prec_score(KNN)))
print ("In-sample F1 Score: " + str(train_f1(KNN)))
print ("Test F1 Score: " + str(test_f1(KNN)))
#confusion_matrix_model_train(KNN)


# **3.2.6. Decision Tree**

# In[ ]:


#param_grid = dict(max_depth=np.arange(4,10),min_samples_leaf=np.arange(1,8),min_samples_split=np.arange(2,8),max_leaf_nodes=np.arange(30,100,10))
#Dec_tree = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,scoring="f1_macro")
Dec_tree=DecisionTreeClassifier(max_depth= 8, max_leaf_nodes= 40, min_samples_leaf= 1, min_samples_split= 7)
Dec_tree.fit(x_train,y_train)
#print(Dec_tree.best_params_)
print("In-sample accuracy: " + str(train_acc_score(Dec_tree)))
print("Test accuracy: " + str(test_acc_score(Dec_tree)))
print ("In-sample Precision Score: " + str(train_prec_score(Dec_tree)))
print ("Test Precision Score: " + str(test_prec_score(Dec_tree)))
print ("In-sample F1 Score: " + str(train_f1(Dec_tree)))
print ("Test F1 Score: " + str(test_f1(Dec_tree)))
#confusion_matrix_model_train(Dec_tree)


# **3.2.7. Random Forest**

# In[ ]:


#param_grid = dict(max_depth=np.arange(3,10),min_samples_leaf=np.arange(1,10),min_samples_split=np.arange(2,6),max_leaf_nodes=np.arange(50,120,10))
#param_grid = dict(n_estimators = np.arange(50,500,50))
#ranfor = GridSearchCV(RandomForestClassifier(max_depth= 7, max_leaf_nodes=50, min_samples_leaf= 7, min_samples_split= 4,random_state=0),param_grid=param_grid,scoring="f1_macro")
ranfor = RandomForestClassifier(n_estimators=50,max_depth= 7, max_leaf_nodes=50, min_samples_leaf= 7, min_samples_split= 4,random_state=0)
ranfor.fit(x_train,y_train)
#print(ranfor.best_params_)
print("In-sample accuracy: " + str(train_acc_score(ranfor)))
print("Test accuracy: " + str(test_acc_score(ranfor)))
print ("In-sample Precision Score: " + str(train_prec_score(ranfor)))
print ("Test Precision Score: " + str(test_prec_score(ranfor)))
print ("In-sample F1 Score: " + str(train_f1(ranfor)))
print ("Test F1 Score: " + str(test_f1(ranfor)))
#confusion_matrix_model_train(ranfor)


# **3.2.8. XGBoosting**

# In[ ]:


#param_grid = dict(n_estimators=np.arange(50,500,50),max_depth=np.arange(6,12),learning_rate=(0.0001,0.001,0.01,0.1))
#xgclass = GridSearchCV(xgb.XGBClassifier(random_state=0),param_grid=param_grid,scoring="f1_macro")
xgclass = xgb.XGBClassifier(max_depth=9, n_estimators=450, learning_rate=0.01)
xgclass.fit(x_train,y_train)
#print(xgclass.best_params_)
print("In-sample accuracy: " + str(train_acc_score(xgclass)))
print("Test accuracy: " + str(test_acc_score(xgclass)))
print ("In-sample Precision Score: " + str(train_prec_score(xgclass)))
print ("Test Precision Score: " + str(test_prec_score(xgclass)))
print ("In-sample F1 Score: " + str(train_f1(xgclass)))
print ("Test F1 Score: " + str(test_f1(xgclass)))
confusion_matrix_model_train(xgclass)


# In[ ]:


importance_of_features(xgclass)


# ## **3.3 Evaluation for models predicting results**

# In[ ]:


Classifiers=["Logistic Regression (Lasso)","Logistic Regression (Ridge)","Support Vector Machine (RBF)","Support Vector Machine(Linear)","K-Nearest Neighbours","Decision Tree","Random Forest","XGBoost"]
in_sample_acc=[round(train_acc_score(x),2) for x in [log_reg1,log_reg2,svc_rbf,svc_lin,KNN,Dec_tree,ranfor,xgclass]]
test_acc=[round(test_acc_score(x),2) for x in [log_reg1,log_reg2,svc_rbf,svc_lin,KNN,Dec_tree,ranfor,xgclass]]
train_prec = [round(train_prec_score(x),2) for x in [log_reg1,log_reg2,svc_rbf,svc_lin,KNN,Dec_tree,ranfor,xgclass]]
test_prec = [round(test_prec_score(x),2) for x in [log_reg1,log_reg2,svc_rbf,svc_lin,KNN,Dec_tree,ranfor,xgclass]]
trainf1 = [train_f1(x) for x in [log_reg1,log_reg2,svc_rbf,svc_lin,KNN,Dec_tree,ranfor,xgclass]]
testf1 = [test_f1(x) for x in [log_reg1,log_reg2,svc_rbf,svc_lin,KNN,Dec_tree,ranfor,xgclass]]
cols=["Classifier","Training Accuracy","Test Accuracy","Training Precision","Test Precision","Training F1 Score","Test F1 Score"]
pred_results = pd.DataFrame(columns=cols)
pred_results["Classifier"]=Classifiers
pred_results["Training Accuracy"]=in_sample_acc
pred_results["Test Accuracy"]=test_acc
pred_results["Training Precision"]=train_prec
pred_results["Test Precision"]=test_prec
pred_results["Training F1 Score"]=trainf1
pred_results["Test F1 Score"]=testf1
pred_results


# **Selected model to predict W/D/L result: **
# 
# **XGBoost** --> Highest Test set F1 score, along with highest test set accuracy and precision 

# # **4. Classification Models to predict exact goals scored by Home and Away Sides**

# ## **4.1 Splitting into training and test set**

# Again, we will use 80-20 to split the training and test set with 5-fold cross validation. 

# In[ ]:


x = results.loc[:,["country","impt","home_rank_change","away_rank_change","diff_in_ranking","diff_in_mean_weighted_over_years"]]
y_home=results.loc[:,"home_score"]
y_away=results.loc[:,"away_score"]


# In[ ]:


x_home_train,x_home_test,y_home_train,y_home_test=train_test_split(x,y_home,test_size=0.2,random_state=0)
x_away_train,x_away_test,y_away_train,y_away_test=train_test_split(x,y_away,test_size=0.2,random_state=0)
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)


# Functions to evaluate the models for goals scored for home and away:

# In[ ]:


#home goals
def home_train_acc_score(model):
    return round(np.mean(cross_val_score(model,x_home_train,y_home_train,cv=k_fold,scoring="accuracy")),2)
def home_test_acc_score(model):
    return round(accuracy_score(y_home_test, model.predict(x_home_test)),2)
def home_train_prec_score(model):
    return round(precision_score(y_home_train,model.predict(x_home_train),average='macro'),2)
def home_test_prec_score(model):
    return round(precision_score(y_home_test,model.predict(x_home_test),average='macro'),2)
def home_train_f1(model):
    return round(f1_score(y_home_train,model.predict(x_home_train),average='macro'),2)
def home_test_f1(model):
    return round(f1_score(y_home_test,model.predict(x_home_test),average='macro'),2)
def home_confusion_matrix_model_train(model_used):
    cm=confusion_matrix(y_home_train,model_used.predict(x_home_train))
    col=["Predicted Home Goals: 0","Predicted Home Goals: 1","Predicted Home Goals: 2","Predicted Home Goals: 3","Predicted Home Goals: 4","Predicted Home Goals: 5","Predicted Home Goals: 6"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Home Goals: 0","Predicted Home Goals: 1","Predicted Home Goals: 2","Predicted Home Goals: 3","Predicted Home Goals: 4","Predicted Home Goals: 5","Predicted Home Goals: 6"]
    cm.index=["Actual Home Goals: 0","Actual Home Goals: 1","Actual Home Goals: 2","Actual Home Goals: 3","Actual Home Goals: 4","Actual Home Goals: 5","Actual Home Goals: 6"]
    #cm[col]=np(cm[col])
    return cm.T
def home_confusion_matrix_model_test(model_used):
    cm=confusion_matrix(y_home_test,model_used.predict(x_home_test))
    col=["Predicted Home Goals: 0","Predicted Home Goals: 1","Predicted Home Goals: 2","Predicted Home Goals: 3","Predicted Home Goals: 4","Predicted Home Goals: 5","Predicted Home Goals: 6"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Home Goals: 0","Predicted Home Goals: 1","Predicted Home Goals: 2","Predicted Home Goals: 3","Predicted Home Goals: 4","Predicted Home Goals: 5","Predicted Home Goals: 6"]
    cm.index=["Actual Home Goals: 0","Actual Home Goals: 1","Actual Home Goals: 2","Actual Home Goals: 3","Actual Home Goals: 4","Actual Home Goals: 5","Actual Home Goals: 6"]
    #cm[col]=np(cm[col])
    return cm.T
def home_importance_of_features(model):
    features = pd.DataFrame()
    features['feature'] = x_home_train.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    return features.plot(kind='barh', figsize=(10,10))

#away goals
def away_train_acc_score(model):
    return round(np.mean(cross_val_score(model,x_away_train,y_away_train,cv=k_fold,scoring="accuracy")),2)
def away_test_acc_score(model):
    return round(accuracy_score(y_away_test, model.predict(x_away_test)),2)
def away_train_prec_score(model):
    return round(precision_score(y_away_train,model.predict(x_away_train),average='macro'),2)
def away_test_prec_score(model):
    return round(precision_score(y_away_test,model.predict(x_away_test),average='macro'),2)
def away_train_f1(model):
    return round(f1_score(y_away_train,model.predict(x_away_train),average='macro'),2)
def away_test_f1(model):
    return round(f1_score(y_away_test,model.predict(x_away_test),average='macro'),2)
def away_confusion_matrix_model_train(model_used):
    cm=confusion_matrix(y_away_train,model_used.predict(x_away_train))
    col=["Predicted Away Goals: 0","Predicted Away Goals: 1","Predicted Away Goals: 2","Predicted Away Goals: 3","Predicted Away Goals: 4","Predicted Away Goals: 5","Predicted Away Goals: 6"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Away Goals: 0","Predicted Away Goals: 1","Predicted Away Goals: 2","Predicted Away Goals: 3","Predicted Away Goals: 4","Predicted Away Goals: 5","Predicted Away Goals: 6"]
    cm.index=["Actual Away Goals: 0","Actual Away Goals: 1","Actual Away Goals: 2","Actual Away Goals: 3","Actual Away Goals: 4","Actual Away Goals: 5","Actual Away Goals: 6"]
    #cm[col]=np(cm[col])
    return cm.T
def away_confusion_matrix_model_test(model_used):
    cm=confusion_matrix(y_away_test,model_used.predict(x_away_test))
    col=["Predicted Away Goals: 0","Predicted Away Goals: 1","Predicted Away Goals: 2","Predicted Away Goals: 3","Predicted Away Goals: 4","Predicted Away Goals: 5","Predicted Away Goals: 6"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Away Goals: 0","Predicted Away Goals: 1","Predicted Away Goals: 2","Predicted Away Goals: 3","Predicted Away Goals: 4","Predicted Away Goals: 5","Predicted Away Goals: 6"]
    cm.index=["Actual Away Goals: 0","Actual Away Goals: 1","Actual Away Goals: 2","Actual Away Goals: 3","Actual Away Goals: 4","Actual Away Goals: 5","Actual Away Goals: 6"]
    #cm[col]=np(cm[col])
    return cm.T
def away_importance_of_features(model):
    features = pd.DataFrame()
    features['feature'] = x_away_train.columns
    features['importance'] = model.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    return features.plot(kind='barh', figsize=(10,10))


# ## **4.2 Classification models for goals scored by Home Side**

# Just like in (3.2), the models will be optimised using GridSearchCV based on F1 score. 
# 
# I have typed in some of the optimised parameters based on the GridSearchCV code output, then commented out the GridSearchCV codes to make the notebook run faster as it won't be re-optimised.
# 
# Confusion matrix table and details will only be shown for the final selected models in order to save space. There would be a summary of each models in the evaluation section below

# **4.2.1. Logistic Regression (Lasso)**

# In[ ]:


param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
homelog_reg1 = GridSearchCV(LogisticRegression(penalty="l1"),param_grid=param_grid,scoring="f1_macro")
#homelog_reg1=LogisticRegression(penalty="l1")
homelog_reg1.fit(x_home_train,y_home_train)
#predicted=log_reg1.predict(x_test)
print(homelog_reg1.best_params_)
print("In-sample accuracy: " + str(home_train_acc_score(homelog_reg1)))
print("Test accuracy: " + str(home_test_acc_score(homelog_reg1)))
print ("In-sample Precision Score: " + str(home_train_prec_score(homelog_reg1)))
print ("Test Precision Score: " + str(home_test_prec_score(homelog_reg1)))
print ("In-sample F1 Score: " + str(home_train_f1(homelog_reg1)))
print ("Test F1 Score: " + str(home_test_f1(homelog_reg1)))
home_confusion_matrix_model_train(homelog_reg1)


# **4.2.2. Logistic Regression (Ridge)**

# In[ ]:


param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
homelog_reg2 = GridSearchCV(LogisticRegression(penalty="l2"),param_grid=param_grid,scoring="f1_macro")
#homelog_reg1=LogisticRegression(penalty="l1")
homelog_reg2.fit(x_home_train,y_home_train)
#predicted=log_reg1.predict(x_test)
print(homelog_reg2.best_params_)
print("In-sample accuracy: " + str(home_train_acc_score(homelog_reg2)))
print("Test accuracy: " + str(home_test_acc_score(homelog_reg2)))
print ("In-sample Precision Score: " + str(home_train_prec_score(homelog_reg2)))
print ("Test Precision Score: " + str(home_test_prec_score(homelog_reg2)))
print ("In-sample F1 Score: " + str(home_train_f1(homelog_reg2)))
print ("Test F1 Score: " + str(home_test_f1(homelog_reg2)))
#home_confusion_matrix_model_train(homelog_reg2)


# **4.2.3. SVM (RBF Kernel)**

# In[ ]:


#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#homesvc_rbf = GridSearchCV(SVC(kernel="rbf",random_state=0),param_grid=param_grid,scoring="f1_macro")
homesvc_rbf = SVC(kernel='rbf', gamma=0.001, C=1,random_state=0)
homesvc_rbf.fit(x_home_train, y_home_train)
#print(homesvc_rbf.best_params_)
print("In-sample accuracy: " + str(home_train_acc_score(homesvc_rbf)))
print("Test accuracy: " + str(home_test_acc_score(homesvc_rbf)))
print ("In-sample Precision Score: " + str(home_train_prec_score(homesvc_rbf)))
print ("Test Precision Score: " + str(home_test_prec_score(homesvc_rbf)))
print ("In-sample F1 Score: " + str(home_train_f1(homesvc_rbf)))
print ("Test F1 Score: " + str(home_test_f1(homesvc_rbf)))
#home_confusion_matrix_model_train(homesvc_rbf)


# **4.2.4. KNN** 

# In[ ]:


#param_grid = dict(n_neighbors=np.arange(10,70),weights=("uniform","distance"),p=(1,2))
#homeKNN = GridSearchCV(KNeighborsClassifier(),param_grid=param_grid,scoring="f1_macro")
homeKNN=KNeighborsClassifier(n_neighbors=10,p=1,weights='uniform')
homeKNN.fit(x_home_train,y_home_train)
predicted=homeKNN.predict(x_home_test)
#print(homeKNN.best_params_)
print("In-sample accuracy: " + str(home_train_acc_score(homeKNN)))
print("Test accuracy: " + str(home_test_acc_score(homeKNN)))
print ("In-sample Precision Score: " + str(home_train_prec_score(homeKNN)))
print ("Test Precision Score: " + str(home_test_prec_score(homeKNN)))
print ("In-sample F1 Score: " + str(home_train_f1(homeKNN)))
print ("Test F1 Score: " + str(home_test_f1(homeKNN)))
#home_confusion_matrix_model_train(homeKNN)


# **4.2.5. Decision Tree**

# In[ ]:


#param_grid = dict(max_depth=np.arange(4,10),min_samples_leaf=np.arange(1,8),min_samples_split=np.arange(2,8),max_leaf_nodes=np.arange(30,100,10))
#homeDec_tree = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,scoring="f1_macro")
homeDec_tree=DecisionTreeClassifier(max_depth= 8, max_leaf_nodes= 50, min_samples_leaf= 1, min_samples_split= 6,random_state=0)
homeDec_tree.fit(x_home_train,y_home_train)
predicted=homeDec_tree.predict(x_home_test)
#print(homeDec_tree.best_params_)
print("In-sample accuracy: " + str(home_train_acc_score(homeDec_tree)))
print("Test accuracy: " + str(home_test_acc_score(homeDec_tree)))
print ("In-sample Precision Score: " + str(home_train_prec_score(homeDec_tree)))
print ("Test Precision Score: " + str(home_test_prec_score(homeDec_tree)))
print ("In-sample F1 Score: " + str(home_train_f1(homeDec_tree)))
print ("Test F1 Score: " + str(home_test_f1(homeDec_tree)))
#home_confusion_matrix_model_train(homeDec_tree)


# **4.2.6. Random Forest**

# In[ ]:


#param_grid = dict(max_depth=np.arange(3,10),min_samples_leaf=np.arange(1,10),min_samples_split=np.arange(2,6),max_leaf_nodes=np.arange(50,120,10))
#param_grid = dict(n_estimators = np.arange(50,500,50))
#homeranfor = GridSearchCV(RandomForestClassifier(max_depth= 4, max_leaf_nodes=50, min_samples_leaf= 1, min_samples_split= 3,random_state=0),param_grid=param_grid,scoring="f1_macro")
homeranfor = RandomForestClassifier(n_estimators=250,max_depth= 4, max_leaf_nodes=50, min_samples_leaf= 1, min_samples_split= 3,random_state=0)
homeranfor.fit(x_home_train,y_home_train)
predicted=homeranfor.predict(x_home_test)
#print(ranfor.best_params_)
print("In-sample accuracy: " + str(home_train_acc_score(homeranfor)))
print("Test accuracy: " + str(home_test_acc_score(homeranfor)))
print ("In-sample Precision Score: " + str(home_train_prec_score(homeranfor)))
print ("Test Precision Score: " + str(home_test_prec_score(homeranfor)))
print ("In-sample F1 Score: " + str(home_train_f1(homeranfor)))
print ("Test F1 Score: " + str(home_test_f1(homeranfor)))
#home_confusion_matrix_model_train(homeranfor)


# **4.2.7. XGBooost**

# In[ ]:


#param_grid = dict(n_estimators=np.arange(50,500,50),max_depth=np.arange(6,12),learning_rate=(0.0001,0.001,0.01,0.1))
#homexgclass = GridSearchCV(xgb.XGBClassifier(random_state=0),param_grid=param_grid,scoring="f1_macro")
homexgclass = xgb.XGBClassifier(max_depth=11, n_estimators=350, learning_rate=0.01)
homexgclass.fit(x_home_train,y_home_train)
predicted=homexgclass.predict(x_home_test)
#print(homexgclass.best_params_)
print("In-sample accuracy: " + str(home_train_acc_score(homexgclass)))
print("Test accuracy: " + str(home_test_acc_score(homexgclass)))
print ("In-sample Precision Score: " + str(home_train_prec_score(homexgclass)))
print ("Test Precision Score: " + str(home_test_prec_score(homexgclass)))
print ("In-sample F1 Score: " + str(home_train_f1(homexgclass)))
print ("Test F1 Score: " + str(home_test_f1(homexgclass)))
#home_confusion_matrix_model_train(homexgclass)


# ## **4.3 Evaluation for Models predicting Home Goals**

# In[ ]:


Classifiers=["Logistic Regression (Lasso)","Logistic Regression (Ridge)","Support Vector Machine (RBF)","K-Nearest Neighbours","Decision Tree","Random Forest","XGBoost"]
in_sample_acc=[round(home_train_acc_score(x),2) for x in [homelog_reg1,homelog_reg2,homesvc_rbf,homeKNN,homeDec_tree,homeranfor,homexgclass]]
test_acc=[round(home_test_acc_score(x),2) for x in [homelog_reg1,homelog_reg2,homesvc_rbf,homeKNN,homeDec_tree,homeranfor,homexgclass]]
train_prec = [round(home_train_prec_score(x),2) for x in [homelog_reg1,homelog_reg2,homesvc_rbf,homeKNN,homeDec_tree,homeranfor,homexgclass]]
test_prec = [round(home_test_prec_score(x),2) for x in [homelog_reg1,homelog_reg2,homesvc_rbf,homeKNN,homeDec_tree,homeranfor,homexgclass]]
trainf1 = [home_train_f1(x) for x in [homelog_reg1,homelog_reg2,homesvc_rbf,homeKNN,homeDec_tree,homeranfor,homexgclass]]
testf1 = [home_test_f1(x) for x in [homelog_reg1,homelog_reg2,homesvc_rbf,homeKNN,homeDec_tree,homeranfor,homexgclass]]
cols=["Classifier","Training Accuracy","Test Accuracy","Training Precision","Test Precision","Training F1 Score","Test F1 Score"]
Home_goals_pred = pd.DataFrame(columns=cols)
Home_goals_pred["Classifier"]=Classifiers
Home_goals_pred["Training Accuracy"]=in_sample_acc
Home_goals_pred["Test Accuracy"]=test_acc
Home_goals_pred["Training Precision"]=train_prec
Home_goals_pred["Test Precision"]=test_prec
Home_goals_pred["Training F1 Score"]=trainf1
Home_goals_pred["Test F1 Score"]=testf1
Home_goals_pred


# We will use **Logistic Regression (Lasso)** to predict goals scored by home side. Although the F1 score for XGBoost is highest, logistic regression accuracy and precision are higher for the test set, while the precision metric for training and test differs alot, which makes it overfitted. Logistic Regression is also a simpler model to use.

#  

# ## **4.4 Classification models for goals scored by Away Side**

# Just like in (4.2), the models will be optimised using GridSearchCV based on F1 score. 
# 
# I have typed in some of the optimised parameters based on the GridSearchCV code output, then commented out the GridSearchCV codes to make the notebook run faster as it won't be re-optimised.
# 
# Confusion matrix table and details will only be shown for the final selected models in order to save space. There would be a summary of each models in the evaluation section below

# **4.4.1. Logistic Regression (Lasso)**

# In[ ]:


param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
awaylog_reg1 = GridSearchCV(LogisticRegression(penalty="l1"),param_grid=param_grid,scoring="f1_macro")
#awaylog_reg1=LogisticRegression(penalty="l1")
awaylog_reg1.fit(x_away_train,y_away_train)
#predicted=awaylog_reg1.predict(x_test)
print(awaylog_reg1.best_params_)
print("In-sample accuracy: " + str(away_train_acc_score(awaylog_reg1)))
print("Test accuracy: " + str(away_test_acc_score(awaylog_reg1)))
print ("In-sample Precision Score: " + str(away_train_prec_score(awaylog_reg1)))
print ("Test Precision Score: " + str(away_test_prec_score(awaylog_reg1)))
print ("In-sample F1 Score: " + str(away_train_f1(awaylog_reg1)))
print ("Test F1 Score: " + str(away_test_f1(awaylog_reg1)))
away_confusion_matrix_model_train(awaylog_reg1)


# **4.4.2. Logistic Regression (Ridge)**

# In[ ]:


param_grid = dict(C=(0.0001,0.001,0.005,0.01,0.1,0.5,1))
awaylog_reg2 = GridSearchCV(LogisticRegression(penalty="l2"),param_grid=param_grid,scoring="f1_macro")
#awaylog_reg1=LogisticRegression(penalty="l1")
awaylog_reg2.fit(x_away_train,y_away_train)
#predicted=awaylog_reg1.predict(x_test)
print(awaylog_reg2.best_params_)
print("In-sample accuracy: " + str(away_train_acc_score(awaylog_reg2)))
print("Test accuracy: " + str(away_test_acc_score(awaylog_reg2)))
print ("In-sample Precision Score: " + str(away_train_prec_score(awaylog_reg2)))
print ("Test Precision Score: " + str(away_test_prec_score(awaylog_reg2)))
print ("In-sample F1 Score: " + str(away_train_f1(awaylog_reg2)))
print ("Test F1 Score: " + str(away_test_f1(awaylog_reg2)))
#away_confusion_matrix_model_train(awaylog_reg2)


# **4.4.3. SVM (RBF Kernel)**

# In[ ]:


#param_grid = dict(C=(0.001,0.01,0.1,0.5,1,2),gamma=(0.001,0.01,0.1,0.5,1,2))
#awaysvc_rbf = GridSearchCV(SVC(kernel="rbf",random_state=0),param_grid=param_grid,scoring="f1_macro")
awaysvc_rbf = SVC(kernel='rbf', gamma=0.01, C=2,random_state=0)
awaysvc_rbf.fit(x_away_train, y_away_train)
#print(awaysvc_rbf.best_params_)
print("In-sample accuracy: " + str(away_train_acc_score(awaysvc_rbf)))
print("Test accuracy: " + str(away_test_acc_score(awaysvc_rbf)))
print ("In-sample Precision Score: " + str(away_train_prec_score(awaysvc_rbf)))
print ("Test Precision Score: " + str(away_test_prec_score(awaysvc_rbf)))
print ("In-sample F1 Score: " + str(away_train_f1(awaysvc_rbf)))
print ("Test F1 Score: " + str(away_test_f1(awaysvc_rbf)))
#away_confusion_matrix_model_train(awaysvc_rbf)


# **4.4.4. KNN** 

# In[ ]:


#param_grid = dict(n_neighbors=np.arange(10,70),weights=("uniform","distance"),p=(1,2))
#awayKNN = GridSearchCV(KNeighborsClassifier(),param_grid=param_grid,scoring="f1_macro")
awayKNN=KNeighborsClassifier(n_neighbors=18,p=1,weights='distance')
awayKNN.fit(x_away_train,y_away_train)
predicted=awayKNN.predict(x_away_test)
#print(awayKNN.best_params_)
print("In-sample accuracy: " + str(away_train_acc_score(awayKNN)))
print("Test accuracy: " + str(away_test_acc_score(awayKNN)))
print ("In-sample Precision Score: " + str(away_train_prec_score(awayKNN)))
print ("Test Precision Score: " + str(away_test_prec_score(awayKNN)))
print ("In-sample F1 Score: " + str(away_train_f1(awayKNN)))
print ("Test F1 Score: " + str(away_test_f1(awayKNN)))
#away_confusion_matrix_model_train(awayKNN)


# **4.4.5. Decision Tree**

# In[ ]:


#param_grid = dict(max_depth=np.arange(4,10),min_samples_leaf=np.arange(1,8),min_samples_split=np.arange(2,8),max_leaf_nodes=np.arange(30,100,10))
#awayDec_tree = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid,scoring="f1_macro")
awayDec_tree=DecisionTreeClassifier(max_depth= 8, max_leaf_nodes= 40, min_samples_leaf= 2, min_samples_split= 5,random_state=0)
awayDec_tree.fit(x_away_train,y_away_train)
predicted=awayDec_tree.predict(x_away_test)
#print(awayDec_tree.best_params_)
print("In-sample accuracy: " + str(away_train_acc_score(awayDec_tree)))
print("Test accuracy: " + str(away_test_acc_score(awayDec_tree)))
print ("In-sample Precision Score: " + str(away_train_prec_score(awayDec_tree)))
print ("Test Precision Score: " + str(away_test_prec_score(awayDec_tree)))
print ("In-sample F1 Score: " + str(away_train_f1(awayDec_tree)))
print ("Test F1 Score: " + str(away_test_f1(awayDec_tree)))
#away_confusion_matrix_model_train(awayDec_tree)


# **4.4.6. Random Forest**

# In[ ]:


#param_grid = dict(max_depth=np.arange(3,10),min_samples_leaf=np.arange(1,10),min_samples_split=np.arange(2,6),max_leaf_nodes=np.arange(50,120,10))
#param_grid = dict(n_estimators = np.arange(50,500,50))
#awayranfor = GridSearchCV(RandomForestClassifier(max_depth= 7, max_leaf_nodes=50, min_samples_leaf= 4, min_samples_split= 2,random_state=0),param_grid=param_grid,scoring="f1_macro")
awayranfor = RandomForestClassifier(n_estimators=150,max_depth= 7, max_leaf_nodes=50, min_samples_leaf= 4, min_samples_split= 2,random_state=0)
awayranfor.fit(x_away_train,y_away_train)
predicted=awayranfor.predict(x_away_test)
#print(awayranfor.best_params_)
print("In-sample accuracy: " + str(away_train_acc_score(awayranfor)))
print("Test accuracy: " + str(away_test_acc_score(awayranfor)))
print ("In-sample Precision Score: " + str(away_train_prec_score(awayranfor)))
print ("Test Precision Score: " + str(away_test_prec_score(awayranfor)))
print ("In-sample F1 Score: " + str(away_train_f1(awayranfor)))
print ("Test F1 Score: " + str(away_test_f1(awayranfor)))
#away_confusion_matrix_model_train(awayranfor)


# **4.4.7. XGBooost**

# In[ ]:


#param_grid = dict(n_estimators=np.arange(50,500,50),max_depth=np.arange(6,12),learning_rate=(0.0001,0.001,0.01,0.1))
#awayxgclass = GridSearchCV(xgb.XGBClassifier(random_state=0),param_grid=param_grid,scoring="f1_macro")
awayxgclass = xgb.XGBClassifier(max_depth=7, n_estimators=100, learning_rate=0.01)
awayxgclass.fit(x_away_train,y_away_train)
predicted=awayxgclass.predict(x_away_test)
#print(awayxgclass.best_params_)
print("In-sample accuracy: " + str(away_train_acc_score(awayxgclass)))
print("Test accuracy: " + str(away_test_acc_score(awayxgclass)))
print ("In-sample Precision Score: " + str(away_train_prec_score(awayxgclass)))
print ("Test Precision Score: " + str(away_test_prec_score(awayxgclass)))
print ("In-sample F1 Score: " + str(away_train_f1(awayxgclass)))
print ("Test F1 Score: " + str(away_test_f1(awayxgclass)))
#away_confusion_matrix_model_train(awayxgclass)


# ## **4.5 Evaluation for Models predicting Away Goals**

# In[ ]:


Classifiers=["Logistic Regression (Lasso)","Logistic Regression (Ridge)","Support Vector Machine (RBF)","K-Nearest Neighbours","Decision Tree","Random Forest","XGBoost"]
in_sample_acc=[round(away_train_acc_score(x),2) for x in [awaylog_reg1,awaylog_reg2,awaysvc_rbf,awayKNN,awayDec_tree,awayranfor,awayxgclass]]
test_acc=[round(away_test_acc_score(x),2) for x in [awaylog_reg1,awaylog_reg2,awaysvc_rbf,awayKNN,awayDec_tree,awayranfor,awayxgclass]]
train_prec = [round(away_train_prec_score(x),2) for x in [awaylog_reg1,awaylog_reg2,awaysvc_rbf,awayKNN,awayDec_tree,awayranfor,awayxgclass]]
test_prec = [round(away_test_prec_score(x),2) for x in [awaylog_reg1,awaylog_reg2,awaysvc_rbf,awayKNN,awayDec_tree,awayranfor,awayxgclass]]
trainf1 = [away_train_f1(x) for x in [awaylog_reg1,awaylog_reg2,awaysvc_rbf,awayKNN,awayDec_tree,awayranfor,awayxgclass]]
testf1 = [away_test_f1(x) for x in [awaylog_reg1,awaylog_reg2,awaysvc_rbf,awayKNN,awayDec_tree,awayranfor,awayxgclass]]
cols=["Classifier","Training Accuracy","Test Accuracy","Training Precision","Test Precision","Training F1 Score","Test F1 Score"]
away_goals_pred = pd.DataFrame(columns=cols)
away_goals_pred["Classifier"]=Classifiers
away_goals_pred["Training Accuracy"]=in_sample_acc
away_goals_pred["Test Accuracy"]=test_acc
away_goals_pred["Training Precision"]=train_prec
away_goals_pred["Test Precision"]=test_prec
away_goals_pred["Training F1 Score"]=trainf1
away_goals_pred["Test F1 Score"]=testf1
away_goals_pred


# We shall use **logistic regression (Lasso)** to predict the number of goals scored by away side due to the highest Test set F1 score. 

# # **5. Visualizing ability/potential of players of the 32 countries**

# **Let's get the top 30 players from each WC countries as a representation for the country.**

# In[ ]:


wc_fifa18_stats.replace([False,"False",True,"True"],[0,0,1,1])
unused_var=[]
for i in fifa18.columns:
    if "prefers" in i:
        unused_var.append(i)
wc_fifa18_stats = wc_fifa18_stats.drop(["club_logo","ID","real_face","birth_date","flag","photo"],axis=1)
wc_fifa18_stats = wc_fifa18_stats.drop(unused_var,axis=1)
grouped_wc_fifa18_stats = wc_fifa18_stats.groupby(['nationality']).apply(lambda x: (x.sort_values('overall',ascending=False)).head(30).mean()).sort_values('potential',ascending=True)
grouped_wc_fifa18_stats.head()


# In[ ]:


grouped_wc_fifa18_stats["Points_to_max_potential"] = grouped_wc_fifa18_stats["potential"] - grouped_wc_fifa18_stats["overall"]


# In[ ]:


curr = grouped_wc_fifa18_stats["overall"]
pot = grouped_wc_fifa18_stats["Points_to_max_potential"]
ind = np.arange(32)
width = 0.7
plt.figure(figsize=(10,10))
p1 = plt.bar(ind,curr, width)
p2 = plt.bar(ind,pot, width, bottom=curr)

plt.ylabel('Ability')
plt.title('Current and Potential Ability for each country')
plt.xticks(ind,(grouped_wc_fifa18_stats.index),rotation=90)
plt.legend((p1[0], p2[0]), ('Current', 'Potential'))

plt.show()


# We can see that France has the highest potential while Spain has the highest current ability.

# # **6. Adding variables to build a poisson model**

# We shall use the following variables of each country:
# 
# - Soccer Power Index
# - Average Age
# - Average Height
# - Total World Cup Appearances
# - Average goals scored per game
# - Average goals conceded per game
# - Potential
# 
# and compare it with the opponent to build a poisson distribution model to predict the number of goals scored. Detailed method will be explained below in **(7)**.

# In[ ]:


stats = stats.set_index("team",drop=True)
stats = stats.fillna(0)
stats["Ave_goals_scored_per_game"] = stats['total_worldcup_match_goals_scored'] / stats['total_world_cup_matches_played']
stats["Ave_goals_conceded_per_game"] = stats['total_worldcup_match_goals_against'] / stats['total_world_cup_matches_played']
stats["Potential"] = 0
for i in stats.index:
    stats.loc[i,"Potential"] = grouped_wc_fifa18_stats.loc[i,"potential"]
    
#Normalise the variables
for i in ["Soccer_power_index","Average_age","Average_height","total_worldcup_appearances","Potential"]:
    stats[i] = stats[i].apply(lambda x: (x - stats[i].mean())/stats[i].std())

stats = stats.loc[:,['Soccer_power_index','Average_age','Average_height','total_worldcup_appearances',"Ave_goals_scored_per_game","Ave_goals_conceded_per_game","Potential"]]
stats = stats.fillna(0)  #for panama and iceland (First time WC)


# In[ ]:


stats.head(5)


# # **7. Predicting World Cup 2018**
# 
# ## **Overall Method used:**
# 
# **1)** XGBoost would be used first to predict just the result of the matches (Win,Lose,Draw) for the **group stages** as I am only interested in the match result and who progresses. Real results up to 23/06 was used. 
# 
# **2)** For the **elimination rounds onwards**, Logistic Regression (Lasso) model along with the poisson distribution would be used to predict the exact score.
#     * Weight of 0.8 would be given to the poisson distribution and weight of 0.2 would be given to 
#       the Logistic Regression (Lasso)
#       
# ### **Explanation on how I combine poisson distribution with Logistic Regression to predict the goals**
# 
# We shall use the method and formula:
# - **Base goals** = Average goals scored from X against Y = max(Ave_goals_scored_per_game (X), Ave_goals_conceded_per_game(Y))
# 
# - **difference in countries** = 0.4 * (diff. in Soccer_Power_index) + 0.25 * (diff. in Potential) + 0.25 * (diff. in total_worldcup_appearances) + 0.05 * (diff. in Average_height) - 0.05 * (diff. in Average_age)
#                               
#     E.g diff in Soccer_Power_index --> Soccer_Power_index (X) - Soccer_Power_index (Y)
# 
#     Most weight (0.4) is given to difference in soccer power index because I feel it is the main determining factor of a soccer match. 
#     
#     The potential of the players and the experience (total_worldcup_experience) would also play a important role in a match, especially in a semi-final/final. 
#     
#     Least weights were given to the height (Headers advantage) and age (might be correlated with their stamina) as they might play a small part in some parts of the match. Age difference was "subtracted" because the younger the player is, there is a higher change of having more stamina, hence lower age is "better". 
# 
# - **Mean goals scored from X against Y** = Max(0, Base goals + difference in countries)
# 
# A poisson distribution would then be used to calculate the probability of them scoring 0,1,2,3,4,5,6 goals and combined with the classification models used to predict the number of goals for the match. 
# 
# ## **Combining poisson distribution with logistic regression together**
# 
# 1) Using the mean goals scored from Home side against Away side (Calculated as above), we can generate a list of probabilities of Home Side scoring 0 - 6 goals against the Away side. 
#     
#     e.g [ Prob(0 goals), Prob(1 Goal), Prob(2Goals)....Prob(6 goals)]
#     
# 2) We then use the Logistic Regression (Lasso) Model to generate a list of probabilities of Home Side scoring 0-6 goals against the Away Side. 
# 
#     e.g [ Prob(0 goals), Prob(1 Goal), Prob(2Goals)....Prob(6 goals)]
#     
# 3) We then combine the probabilities by multiplying the poisson probabilities by 0.8 and logistic regression probabilities by 0.2 
# 
#     e.g Prob (0 goals) = 0.8 * Poisson Probability (0 goals) + 0.2 * Logisti Regression Probability(0 goals)
#         etc
#         etc
#         
#     (More weight is given to poisson probabilities as I feel they are more accurate and stronger compared to the logistic regression models).
# 
# 4) Our prediction for the number of goals scored by the home side against the away side would be the one with the highest probability.
# 
# 5) Repeat step 1-4 for the away side. 
#     
#     

# ### **Group Stage Matches**

# In[ ]:


from itertools import combinations
#variables: ["country","impt","home_rank_change","away_rank_change","diff_in_ranking","diff_in_mean_weighted_over_years"]
opponents = ['First match \nagainst', 'Second match\n against', 'Third match\n against']

world_cup['points'] = 0
world_cup['total_prob'] = 0

for group in set(world_cup['Group']):
    print('___Starting group {}:___'.format(group))
    for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
        print("{} vs. {}: ".format(home, away), end='')
        
        if (home + "," + away) in list(results_so_far["Matches"]):
            real_result = results_so_far[results_so_far["Matches"] == (home + "," + away)]
            real_result.reset_index(inplace=True,drop=True)
            real_result = real_result.loc[0,"Result"]
            world_cup.loc[home, 'total_prob'] += 1
            world_cup.loc[away, 'total_prob'] += 1
            if real_result == 0:
                print("{} wins ".format(away))
                world_cup.loc[away, 'points'] += 3
                continue
            elif real_result == 1:
                points = 1
                print("Draw")
                world_cup.loc[home, 'points'] += 1
                world_cup.loc[away, 'points'] += 1
                continue
            elif real_result == 2:
                points = 3
                world_cup.loc[home, 'points'] += 3
                print("{} wins ".format(home))
                continue
        elif (away + "," + home) in list(results_so_far["Matches"]):
            real_result = results_so_far[results_so_far["Matches"] == (away + "," + home)]
            real_result.reset_index(inplace=True,drop=True)
            real_result = real_result.loc[0,"Result"]
            world_cup.loc[home, 'total_prob'] += 1
            world_cup.loc[away, 'total_prob'] += 1
            if real_result == 0:
                print("{} wins ".format(home))
                world_cup.loc[home, 'points'] += 3
                continue
            elif real_result == 1:
                points = 1
                print("Draw")
                world_cup.loc[home, 'points'] += 1
                world_cup.loc[away, 'points'] += 1
                continue
            elif real_result == 2:
                points = 3
                world_cup.loc[away, 'points'] += 3
                print("{} wins ".format(away))
                continue
        
        
        row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan,np.nan,np.nan]]), columns=x_test.columns)
        home_rank = rankings_18.loc[home,"rank"]
        away_rank =  rankings_18.loc[away,"rank"]
        

        row["country"] = 2 if home == "Russia" else 0
        row["impt"] = 1
        row["home_rank_change"] = rankings_18.loc[home,"rank"] - rankings_prev.loc[home,"rank"]
        row["away_rank_change"] = rankings_18.loc[away,"rank"] - rankings_prev.loc[away,"rank"]
        row['diff_in_ranking'] = home_rank - away_rank
        row["diff_in_mean_weighted_over_years"] = rankings_18.loc[home,"mean_weighted_over_years"] - rankings_18.loc[away,"mean_weighted_over_years"]
        
        home_win_prob = xgclass.predict_proba(row)[:,2][0]
        away_win_prob = xgclass.predict_proba(row)[:,0][0]
        draw_prob = xgclass.predict_proba(row)[:,1][0]

        
        points = 0
        if max(home_win_prob,away_win_prob,draw_prob) == away_win_prob:
            print("{} wins with a probability of {:.2f}% ".format(away, away_win_prob))
            world_cup.loc[away, 'points'] += 3
            world_cup.loc[home, 'total_prob'] += home_win_prob
            world_cup.loc[away, 'total_prob'] += away_win_prob
        if max(home_win_prob,away_win_prob,draw_prob) == draw_prob:
            points = 1
            print("Draw with probability of {:.2f}%".format(draw_prob))
            world_cup.loc[home, 'points'] += 1
            world_cup.loc[away, 'points'] += 1
            world_cup.loc[home, 'total_prob'] += draw_prob
            world_cup.loc[away, 'total_prob'] += draw_prob
        if max(home_win_prob,away_win_prob,draw_prob) == home_win_prob:
            points = 3
            world_cup.loc[home, 'points'] += 3
            world_cup.loc[home, 'total_prob'] += home_win_prob
            world_cup.loc[away, 'total_prob'] += away_win_prob
            print("{} wins with a probability of {:.2f}%".format(home, home_win_prob))


# Note that some probabilities are not more than 50% because the maximum of (Win Probability, Draw Probability and Loss Probability) is taken to make the decision, and hence the maximum probability selected might not be more than 50%.

# ### **Elimination Round**

# In[ ]:


pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]  # pair up for round-of-16

labels = []
odds = []

world_cup = world_cup.sort_values(by=['Group', 'points', 'total_prob'], ascending=False).reset_index()
next_round_wc = world_cup.groupby('Group').nth([0, 1]) # select the top 2
next_round_wc = next_round_wc.reset_index()
next_round_wc = next_round_wc.loc[pairing]
next_round_wc = next_round_wc.set_index('Team')

finals = ['round_of_16', 'quarterfinal']

for f in finals:
    print("___Starting of the {}___".format(f))
    iterations = int(len(next_round_wc) / 2)
    winners = []

    for i in range(iterations):
        home = next_round_wc.index[i*2]
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home,
                                   away), 
                                   end='')
        row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan,np.nan,np.nan]]), columns=x_test.columns)
        home_rank = rankings_18.loc[home,"rank"]
        away_rank =  rankings_18.loc[away,"rank"]


        row["country"] = 2 if home == "Russia" else 0
        row["impt"] = 1
        row["home_rank_change"] = rankings_18.loc[home,"rank"] - rankings_prev.loc[home,"rank"]
        row["away_rank_change"] = rankings_18.loc[away,"rank"] - rankings_prev.loc[away,"rank"]
        row['diff_in_ranking'] = home_rank - away_rank
        row["diff_in_mean_weighted_over_years"] = rankings_18.loc[home,"mean_weighted_over_years"] - rankings_18.loc[away,"mean_weighted_over_years"]
        base_home_goals = max(stats.loc[home,"Ave_goals_scored_per_game"],stats.loc[away,"Ave_goals_conceded_per_game"])
        base_away_goals = max(stats.loc[away,"Ave_goals_scored_per_game"],stats.loc[home,"Ave_goals_conceded_per_game"])
        home_diff_in_countries = (0.4 * (stats.loc[home,"Soccer_power_index"] - stats.loc[away,"Soccer_power_index"]) +
                                  0.25 * (stats.loc[home,"Potential"] - stats.loc[away,"Potential"]) +
                                  0.25 * (stats.loc[home,"total_worldcup_appearances"] - stats.loc[away,"total_worldcup_appearances"]) +
                                  0.05 * (stats.loc[home,"Average_height"] - stats.loc[away,"Average_height"]) -
                                  0.05 * (stats.loc[home,"Average_age"] - stats.loc[away,"Average_age"]))
        away_diff_in_countries = (0.4 * (stats.loc[away,"Soccer_power_index"] - stats.loc[home,"Soccer_power_index"]) +
                                  0.25 * (stats.loc[away,"Potential"] - stats.loc[home,"Potential"]) +
                                  0.25 * (stats.loc[away,"total_worldcup_appearances"] - stats.loc[home,"total_worldcup_appearances"]) +
                                  0.05 * (stats.loc[away,"Average_height"] - stats.loc[home,"Average_height"]) -
                                  0.05 * (stats.loc[away,"Average_age"] - stats.loc[home,"Average_age"]))

        mean_home_goals = max(0,base_home_goals + home_diff_in_countries)   #max (0,goals) because cant be negative goals
        mean_away_goals = max(0,base_away_goals + away_diff_in_countries) 

        home_prob_of_goals = []
        home_prob_of_goals_logmodel = list(homelog_reg1.predict_proba(row)[0])
        away_prob_of_goals = []
        away_prob_of_goals_logmodel = list(awaylog_reg1.predict_proba(row)[0])
        for i in range(7):
            home_prob_of_goals.append(0.8 * poisson.pmf(i, mean_home_goals) + 0.2 * home_prob_of_goals_logmodel[i])
            away_prob_of_goals.append(0.8 * poisson.pmf(i, mean_away_goals) + 0.2 * away_prob_of_goals_logmodel[i])


        home_goals = np.argmax(home_prob_of_goals)
        away_goals = np.argmax(away_prob_of_goals)
        if home_goals > away_goals:
            print("{} wins {} with score of {}:{}".format(home,away,str(home_goals),str(away_goals)),end='')
            winners.append(home)

        elif home_goals < away_goals:
            print("{} wins {} with score of {}:{}".format(away,home,str(away_goals),str(home_goals)),end='')
            winners.append(away)

        else:
            team=[home,away]
            win = random.choice(team)
            print("{} draws with {} with a score of {}:{} after extra time and {} wins the penalty shootout".format(home,away,str(home_goals),str(away_goals),win))
            winners.append(win)
        print("\n")
    next_round_wc = next_round_wc.loc[winners]
    
print("Top 4: " + winners[0]+ "," + winners[1] + "," + winners[2] + "," +winners[3])


# ### **Predict exact scores for the final and 3rd place play-off**

# In[ ]:


home_sides = winners[0:3:2]
away_sides = winners[1::2]
winner = []
loser = []
print("__Starting of the Semifinal__")
for i in range(2):
    home = home_sides[i]
    away = away_sides[i]
    print("{} vs. {}: ".format(home,
                                away), 
                                end='')
    row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan,np.nan,np.nan]]), columns=x_home_test.columns)
    home_rank = rankings_18.loc[home,"rank"]
    away_rank =  rankings_18.loc[away,"rank"]
    row["country"] = 0
    row["impt"] = 1
    row["home_rank_change"] = rankings_18.loc[home,"rank"] - rankings_prev.loc[home,"rank"]
    row["away_rank_change"] = rankings_18.loc[away,"rank"] - rankings_prev.loc[away,"rank"]
    row['diff_in_ranking'] = home_rank - away_rank
    row["diff_in_mean_weighted_over_years"] = rankings_18.loc[home,"mean_weighted_over_years"] - rankings_18.loc[away,"mean_weighted_over_years"]
    base_home_goals = max(stats.loc[home,"Ave_goals_scored_per_game"],stats.loc[away,"Ave_goals_conceded_per_game"])
    base_away_goals = max(stats.loc[away,"Ave_goals_scored_per_game"],stats.loc[home,"Ave_goals_conceded_per_game"])
    home_diff_in_countries = (0.4 * (stats.loc[home,"Soccer_power_index"] - stats.loc[away,"Soccer_power_index"]) +
                              0.25 * (stats.loc[home,"Potential"] - stats.loc[away,"Potential"]) +
                              0.25 * (stats.loc[home,"total_worldcup_appearances"] - stats.loc[away,"total_worldcup_appearances"]) +
                              0.05 * (stats.loc[home,"Average_height"] - stats.loc[away,"Average_height"]) -
                              0.05 * (stats.loc[home,"Average_age"] - stats.loc[away,"Average_age"]))
    away_diff_in_countries = (0.4 * (stats.loc[away,"Soccer_power_index"] - stats.loc[home,"Soccer_power_index"]) +
                              0.25 * (stats.loc[away,"Potential"] - stats.loc[home,"Potential"]) +
                              0.25 * (stats.loc[away,"total_worldcup_appearances"] - stats.loc[home,"total_worldcup_appearances"]) +
                              0.05 * (stats.loc[away,"Average_height"] - stats.loc[home,"Average_height"]) -
                              0.05 * (stats.loc[away,"Average_age"] - stats.loc[home,"Average_age"]))
    
    mean_home_goals = max(0,base_home_goals + home_diff_in_countries)   #max (0,goals) because cant be negative goals
    mean_away_goals = max(0,base_away_goals + away_diff_in_countries) 
    
    home_prob_of_goals = []
    home_prob_of_goals_logmodel = list(homelog_reg1.predict_proba(row)[0])
    away_prob_of_goals = []
    away_prob_of_goals_logmodel = list(awaylog_reg1.predict_proba(row)[0])
    for i in range(7):
        home_prob_of_goals.append(0.8 * poisson.pmf(i, mean_home_goals) + 0.2 * home_prob_of_goals_logmodel[i])
        away_prob_of_goals.append(0.8 * poisson.pmf(i, mean_away_goals) + 0.2 * away_prob_of_goals_logmodel[i])
        

    home_goals = np.argmax(home_prob_of_goals)
    away_goals = np.argmax(away_prob_of_goals)
    if home_goals > away_goals:
        print("{} wins {} with score of {}:{}".format(home,away,str(home_goals),str(away_goals)),end='')
        winner.append(home)
        loser.append(away)
    elif home_goals < away_goals:
        print("{} wins {} with score of {}:{}".format(away,home,str(away_goals),str(home_goals)),end='')
        winner.append(away)
        loser.append(home)
    else:
        team=[home,away]
        win = random.choice(team)
        team.remove(win)
        lose = team[0]
        print("{} draws with {} with a score of {}:{} after extra time and {} wins the penalty shootout".format(home,away,str(home_goals),str(away_goals),win))
        winner.append(win)
        loser.append(lose)
    print("\n")
print("__Third place playoff__")
home = loser[0]
away = loser[1]
print("{} vs. {}: ".format(home,
                            away), 
                            end='')
row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan,np.nan,np.nan]]), columns=x_home_test.columns)
home_rank = rankings_18.loc[home,"rank"]
away_rank =  rankings_18.loc[away,"rank"]
row["country"] = 0
row["impt"] = 1
row["home_rank_change"] = rankings_18.loc[home,"rank"] - rankings_prev.loc[home,"rank"]
row["away_rank_change"] = rankings_18.loc[away,"rank"] - rankings_prev.loc[away,"rank"]
row['diff_in_ranking'] = home_rank - away_rank
row["diff_in_mean_weighted_over_years"] = rankings_18.loc[home,"mean_weighted_over_years"] - rankings_18.loc[away,"mean_weighted_over_years"]
base_home_goals = max(stats.loc[home,"Ave_goals_scored_per_game"],stats.loc[away,"Ave_goals_conceded_per_game"])
base_away_goals = max(stats.loc[away,"Ave_goals_scored_per_game"],stats.loc[home,"Ave_goals_conceded_per_game"])
home_diff_in_countries = (0.4 * (stats.loc[home,"Soccer_power_index"] - stats.loc[away,"Soccer_power_index"]) +
                          0.25 * (stats.loc[home,"Potential"] - stats.loc[away,"Potential"]) +
                          0.25 * (stats.loc[home,"total_worldcup_appearances"] - stats.loc[away,"total_worldcup_appearances"]) +
                          0.05 * (stats.loc[home,"Average_height"] - stats.loc[away,"Average_height"]) -
                          0.05 * (stats.loc[home,"Average_age"] - stats.loc[away,"Average_age"]))
away_diff_in_countries = (0.4 * (stats.loc[away,"Soccer_power_index"] - stats.loc[home,"Soccer_power_index"]) +
                          0.25 * (stats.loc[away,"Potential"] - stats.loc[home,"Potential"]) +
                          0.25 * (stats.loc[away,"total_worldcup_appearances"] - stats.loc[home,"total_worldcup_appearances"]) +
                          0.05 * (stats.loc[away,"Average_height"] - stats.loc[home,"Average_height"]) -
                          0.05 * (stats.loc[away,"Average_age"] - stats.loc[home,"Average_age"]))
    
mean_home_goals = max(0,base_home_goals + home_diff_in_countries)   #max (0,goals) because cant be negative goals
mean_away_goals = max(0,base_away_goals + away_diff_in_countries) 
    
home_prob_of_goals = []
home_prob_of_goals_logmodel = list(homelog_reg1.predict_proba(row)[0])
away_prob_of_goals = []
away_prob_of_goals_logmodel = list(awaylog_reg1.predict_proba(row)[0])
for i in range(7):
    home_prob_of_goals.append(0.8 * poisson.pmf(i, mean_home_goals) + 0.2 * home_prob_of_goals_logmodel[i])
    away_prob_of_goals.append(0.8 * poisson.pmf(i, mean_away_goals) + 0.2 * away_prob_of_goals_logmodel[i])
        

home_goals = np.argmax(home_prob_of_goals)
away_goals = np.argmax(away_prob_of_goals)
if home_goals > away_goals:
    print("{} wins {} with score of {}:{}".format(home,away,str(home_goals),str(away_goals)),end='')

elif home_goals < away_goals:
    print("{} wins {} with score of {}:{}".format(away,home,str(away_goals),str(home_goals)),end='')

else:
    team=[home,away]
    win = random.choice(team)
    team.remove(win)
    lose = team[0]
    print("{} draws with {} with a score of {}:{} after extra time and {} wins the penalty shootout".format(home,away,str(home_goals),str(away_goals),win))

print("\n")
print("__World Cup Final__")
home = winner[0]
away = winner[1]
print("{} vs. {}: ".format(home,
                            away), 
                            end='')
row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, np.nan,np.nan,np.nan]]), columns=x_home_test.columns)
home_rank = rankings_18.loc[home,"rank"]
away_rank =  rankings_18.loc[away,"rank"]
row["country"] = 0
row["impt"] = 1
row["home_rank_change"] = rankings_18.loc[home,"rank"] - rankings_prev.loc[home,"rank"]
row["away_rank_change"] = rankings_18.loc[away,"rank"] - rankings_prev.loc[away,"rank"]
row['diff_in_ranking'] = home_rank - away_rank
row["diff_in_mean_weighted_over_years"] = rankings_18.loc[home,"mean_weighted_over_years"] - rankings_18.loc[away,"mean_weighted_over_years"]
base_home_goals = max(stats.loc[home,"Ave_goals_scored_per_game"],stats.loc[away,"Ave_goals_conceded_per_game"])
base_away_goals = max(stats.loc[away,"Ave_goals_scored_per_game"],stats.loc[home,"Ave_goals_conceded_per_game"])
home_diff_in_countries = (0.4 * (stats.loc[home,"Soccer_power_index"] - stats.loc[away,"Soccer_power_index"]) +
                          0.25 * (stats.loc[home,"Potential"] - stats.loc[away,"Potential"]) +
                          0.25 * (stats.loc[home,"total_worldcup_appearances"] - stats.loc[away,"total_worldcup_appearances"]) +
                          0.05 * (stats.loc[home,"Average_height"] - stats.loc[away,"Average_height"]) -
                          0.05 * (stats.loc[home,"Average_age"] - stats.loc[away,"Average_age"]))
away_diff_in_countries = (0.4 * (stats.loc[away,"Soccer_power_index"] - stats.loc[home,"Soccer_power_index"]) +
                          0.25 * (stats.loc[away,"Potential"] - stats.loc[home,"Potential"]) +
                          0.25 * (stats.loc[away,"total_worldcup_appearances"] - stats.loc[home,"total_worldcup_appearances"]) +
                          0.05 * (stats.loc[away,"Average_height"] - stats.loc[home,"Average_height"]) -
                          0.05 * (stats.loc[away,"Average_age"] - stats.loc[home,"Average_age"]))
    
mean_home_goals = max(0,base_home_goals + home_diff_in_countries)   #max (0,goals) because cant be negative goals
mean_away_goals = max(0,base_away_goals + away_diff_in_countries) 
    
home_prob_of_goals = []
home_prob_of_goals_logmodel = list(homelog_reg1.predict_proba(row)[0])
away_prob_of_goals = []
away_prob_of_goals_logmodel = list(awaylog_reg1.predict_proba(row)[0])
for i in range(7):
    home_prob_of_goals.append(0.8 * poisson.pmf(i, mean_home_goals) + 0.2 * home_prob_of_goals_logmodel[i])
    away_prob_of_goals.append(0.8 * poisson.pmf(i, mean_away_goals) + 0.2 * away_prob_of_goals_logmodel[i])
        

home_goals = np.argmax(home_prob_of_goals)
away_goals = np.argmax(away_prob_of_goals)
if home_goals > away_goals:
    print("{} wins {} with score of {}:{}".format(home,away,str(home_goals),str(away_goals)),end='')

elif home_goals < away_goals:
    print("{} wins {} with score of {}:{}".format(away,home,str(away_goals),str(home_goals)),end='')

else:
    team=[home,away]
    win = random.choice(team)
    team.remove(win)
    lose = team[0]
    print("{} draws with {} with a score of {}:{} after extra time and {} wins the penalty shootout".format(home,away,str(home_goals),str(away_goals),win))


#  

# ## **Final Predictions:**
# 
# **Winner**: Germany
# 
# **2nd Place**: Spain
# 
# **3rd Place**: France
# 
# **Final Score**: Germany VS Spain: 2-1
# 
# **Third place playoff score**: France vs England: 1-1 (France win Penalty Shootout)

# In[ ]:




