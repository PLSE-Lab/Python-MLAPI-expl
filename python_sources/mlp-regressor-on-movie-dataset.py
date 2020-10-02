#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


#the usual
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#colored printing output
from termcolor import colored

#I/O
import io
import os
import requests

#pickle
import pickle

#math
import math

#scipy
from scipy import stats

#sk learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from itertools import combinations
from mlxtend.feature_selection import ColumnSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


#sns style
import seaborn as sns
sns.set_style("whitegrid")
sns.despine()
sns.set_context("talk") #larger display of plots axis labels etc..



# In[ ]:


#use TPUs for faster processing (not sure if enabled by default)
if 'COLAB_TPU_ADDR' not in os.environ:
  print('ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!')
else:
  tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  print ('TPU address is', tpu_address)


# # Functions

# In[ ]:


#scatter plots for all numerical values of a df
def ScatterPlots(df, n_plots=3):
  fig, axes = plt.subplots(1, n_plots)
  i=0
  for key, value in df.iloc[:, :-1].iteritems(): 
    print(key) 
    df.plot(kind="scatter",x=key, y="revenue",color="green",ax=axes[i],figsize=(20,10))
    i+=1


# In[ ]:


#check for missing value, zeros etc..
#prints with colour output depending on set limit value
def print_color(text, values, limit=math.inf):
  to_compare=False
  if isinstance(values, float):
    to_compare=(values>limit)
  else:
    to_compare=(values>limit).sum()    
  if to_compare==True:
    print(colored(text,color="magenta",attrs=['reverse', 'blink'])+colored(values,color="magenta"))
  else:
    print(colored(text,color="green")+colored(values,color="green"))
        
def CheckValues(X, Y, detail=False):
  print("---- in X-----")
  print ("shape:", train_x.shape)
  if detail==True:  
    print_color("percentage of Nan:\n", values=X.isna().mean().round(4)*100,limit=5.)
    print_color("percentage of zeros:\n", values=X.eq(0).mean().round(4)*100,limit=5.)
    print_color("percentage of negative:\n", values=X.lt(0).mean().round(4)*100,limit=5.)
  
  print_color("percentage of Nan for entire df:",values=round(X.isna().mean().mean()*100,2),limit=5.)
  print_color("percentage of zeros for entire df:",values=round(X.eq(0).mean().mean()*100,2),limit=5.)
  print_color("percentage of negative values for entire df:",values=round(X.lt(0).mean().mean()*100,2),limit=5.)
  
  print("---- in Y-----")
  print("shape:", Y.shape)
  if detail==True:
    print_color("percentage of Nan:\n", values=Y.isna().mean().round(4)*100,limit=5.)
    print_color("percentage of zeros:\n", values=Y.eq(0).mean().round(4)*100,limit=5.)
    print_color("percentage of negative:\n", values=Y.lt(0).mean().round(4)*100,limit=5.)
  
  print_color("percentage of Nan for entire df:",values=round(Y.isna().mean().mean()*100,2),limit=5.)
  print_color("percentage of zeros for entire df:",values=round(Y.eq(0).mean().mean()*100,2),limit=5.)
  print_color("percentage of negative values for entire df:",values=round(Y.lt(0).mean().mean()*100,2),limit=5.)
  


#remove missing values replace by mean
def RemoveMissVal(X,Y, verbose=False):
  if(verbose):
    print("percentage of Nan in X before removal:\n",X.isna().mean().round(4)*100)
    print("percentage of Nan in Y before removal:\n",Y.isna().mean().round(4)*100)
  X_clean=X.fillna(X.mean())
  Y_clean=Y.fillna(Y.mean())
  
  if(verbose):
    print("-----DONE------------")
    print("percentage of Nan in X after removal:\n",X_clean.isna().mean().round(4)*100)
    print("percentage of Nan in Y after removal:\n",Y_clean.isna().mean().round(4)*100)
  
  return X_clean, Y_clean

#remove outliers (values larger than std_dev will be removed)
def RemoveSigma(X,Y,std_dev,verbose=False):
    if(verbose):
      print("---- before----")
      print(X.shape)
      print(Y.shape)
    #print(np.abs(stats.zscore(X)))
    X_cut = X[(np.abs(stats.zscore(X)) < float(std_dev)).all(axis=1)]
    Y_cut = Y[(np.abs(stats.zscore(X)) < float(std_dev)).all(axis=1)]
    if(verbose):
      print("---- after----")
      print(X_cut.shape)
      print(Y_cut.shape)
    return X_cut, Y_cut


# In[ ]:





# In[ ]:





# In[ ]:


#plot histograms from two df on same plot
def DoubleDfHist(df,df2,n_hist):
  fig, ax = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(20, 6))
  for j in range(n_hist):
    df.hist(column=df.columns[j], bins=100, ax=ax[j], alpha=1, color='red',log=True)
    df2.hist(column=df2.columns[j], bins=100, ax=ax[j], alpha=0.5, color='blue',log=True)


# In[ ]:


def rmsle(y_pred, y_test) : 
    #clip zero values
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(list(np.asarray(y_pred).clip(min=0) + 1)) - np.log(list(np.asarray(y_test).clip(min=0) + 1)))**2))


# In[ ]:


def cross_val_predict(train_X,train_y, model, k_fold=5, use_scaling=True, Verbose=False, score_rmsle=True):
    cv = KFold(n_splits = k_fold)
    test_y_overall = []
    predict_y_overall = []
    train_X=train_X.values
    train_y=train_y.values    
    for train_index, test_index in cv.split(train_X):
      train_X_fi, train_y_fi = train_X[train_index], train_y[train_index]
      test_X_fi, test_y_fi = train_X[test_index], train_y[test_index]
      
      #if train_X, Y are not np arrays use thise:
      #train_X_fi, train_y_fi = train_X.iloc[train_index], train_y.iloc[train_index]
      #test_X_fi, test_y_fi = train_X.iloc[test_index], train_y.iloc[test_index]

      #scale, train the model and evaluate it
      scaler = StandardScaler()
      train_scaled = scaler.fit_transform(train_X_fi)
      test_scaled  = scaler.fit_transform(test_X_fi)
      
      if use_scaling:
        model.fit(train_scaled, train_y_fi)
        prediction = model.predict(test_scaled)
      else:
        model.fit(train_X_fi, train_y_fi)
        prediction = model.predict(test_X_fi)
      
            
      #store the target var and the prediction for later analysis
      test_y_overall.extend(test_y_fi)
      predict_y_overall.extend(prediction)     
    
      cross_val_error_rmsle = rmsle(predict_y_overall, test_y_overall)
      cross_val_error_r2 = r2_score(predict_y_overall, test_y_overall)    
      
     #calculate and pring both rmsle and r2 scores, return only one of them 
    if(Verbose==True):
      print("cross_val_error_rmsle is:",cross_val_error_rmsle)
      print("cross_val_error_r2 is:",cross_val_error_r2)
      
    if score_rmsle:
      cross_val_error=cross_val_error_rmsle
    else:
      cross_val_error=cross_val_error_r2
      
    return cross_val_error


# In[ ]:


#plot validation curve
def PlotValidationCurve(train_scores,valid_scores,param_range,param_name,logx=False,verbose=False):
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  if verbose==True:
    print("train_scores_mean:",train_scores_mean)
    print("valid_scores_mean:",valid_scores_mean)
  plt.figure(figsize=(10, 5), dpi=80)
  plt.title("Validation curve")
  plt.plot(param_range, train_scores_mean, label="Training score",
             color="orange", lw=2,marker=".")  
  plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="orange", lw=0)
    
  plt.plot(param_range, valid_scores_mean, label="Cross-validation score",
             color="black", lw=2)
  plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.2,
                 color="black", lw=0)
  if(logx==True):
    plt.xscale('log')
  plt.ylim(-.2, 1.1)
  plt.xlabel(str(param_name))
  plt.ylabel("score")
  plt.ylabel("Score")
  
  plt.legend(loc=0)


# In[ ]:


def PlotLearningCurve(train_sizes,train_scores,valid_scores,param_range,logx=False,verbose=False,ymin=0,ymax=1.):
#plot validation curve
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  valid_scores_mean = np.mean(valid_scores, axis=1)
  valid_scores_std = np.std(valid_scores, axis=1)
  if verbose==True:
    print("train_scores_mean:",train_scores_mean)
    print("valid_scores_mean:",valid_scores_mean)
  plt.figure(figsize=(10, 5), dpi=80)
  plt.title("Learning curve")
  plt.grid()
  plt.plot(train_sizes, train_scores_mean, label="Training score",
             color="red", lw=2,marker=".")  
  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="red", lw=0)
    
  plt.plot(train_sizes, valid_scores_mean, label="Cross-validation score",
             color="navy", lw=2)
  plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.2,
                 color="navy", lw=0)
  if(logx==True):
    plt.xscale('log')
  plt.ylim(ymin, ymax)
  plt.xlabel("training set size")
  plt.ylabel("score")
  plt.ylabel("Score")
  
  plt.legend(loc=0)


# In[ ]:


#Calculates the best model with all combinations fo features, of up to max_size features of X.
#TODO: need to check implementation with rmsle
def best_subset(estimator, X, y, max_size=8, cv=5, use_rmsle=False, verbose=False):
  n_features = X.shape[1]
  subsets = (combinations(range(n_features), k + 1) 
               for k in range(min(n_features, max_size)))
  best_size_subset = []
  for subsets_k in subsets:  # for each list of subsets of the same size      
      best_score = -np.inf
      best_subset = None
      for subset in subsets_k: # for each subset
          estimator.fit(X.iloc[:, list(subset)], y)
           # get the subset with the best score among subsets of the same size
          score = estimator.score(X.iloc[:, list(subset)], y)         
          #score=rmsle(X.iloc[:, list(subset)].values, y.values) #TODO: this needs to be fixed
          if score > best_score:
                best_score, best_subset = score, subset      
        # first store the best subset of each size
      best_size_subset.append(best_subset)

    # compare best subsets of each size
  best_score = -np.inf
  best_subset = None
  list_scores = []
  for subset in best_size_subset:
      if(use_rmsle):
        score=cross_val_predict(X.iloc[:, list(subset)].astype(float), y, estimator, Verbose=verbose, score_rmsle=True) #home made scorer with rmsle
      else:
        score = cross_val_score(estimator, X.iloc[:, list(subset)], y, cv=cv).mean()
      list_scores.append(score)
      if score > best_score:
        best_score, best_subset = score, subset
  return best_subset, best_score, best_size_subset, list_scores


# # Load dataframe

# In[ ]:


import os
from google.colab import drive
drive.mount('/gdrive')
get_ipython().system('pwd')


# In[5]:


#load test and train data
train_data = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")
test_data = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")


# In[ ]:


pd.set_option('display.max_columns', 500)
train_data.head()


# In[ ]:


train_data.describe()


# In[ ]:


train_x = train_data.select_dtypes(include=['float',"int"])
train_x = train_x.drop(columns=["id"])#irelevant feature
train_x = train_x.drop(columns=["revenue"])#drop the label
train_y=train_data["revenue"]
train_x.head()


# In[ ]:


test_x = test_data.select_dtypes(include=['float',"int"])
test_x = test_x.drop(columns=["id"])
test_x.head()


# In[ ]:


#print shapes
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)


# In[ ]:


#make a df with only the selected x and y columns
df_train=pd.concat([train_x, train_y], axis=1)
df_train.head()


# # EDA on dataframe

# In[ ]:


#plt.subplot()
df_train.hist(figsize=(20,15),bins=100,color="red",log=True,layout=(4,1))


# In[ ]:


ScatterPlots(df_train)


# not many Nans only very small percentage in runtime feature.  Many zeros in budget

# In[ ]:


CheckValues(train_x,train_y,detail=True)


# In[ ]:


train_x_clean,train_y_clean= RemoveMissVal(train_x,train_y)
train_x_cut,train_y_cut= RemoveSigma(train_x_clean,train_y_clean,3.)
test_x_clean,test_x_clean = RemoveMissVal(test_x,test_x)


# In[ ]:


CheckValues(train_x_clean,train_y_clean,detail=True)


# In[ ]:


df_train_cut=pd.concat([train_x_cut, train_y_cut], axis=1)
ScatterPlots(df_train_cut)


# maybe films with english original language have more revenue? (on the other hand the pie chart shows that most films are in english). Let's try anyhow to add this categorical variable to a new df as a one hot encoded value (en=1, all others=0)

# In[ ]:


one_hot = pd.get_dummies(train_data['original_language'])
train_hot_lang = train_data.drop('original_language',axis = 1)
one_hot["en"]
train_x_lang=train_x.join(one_hot["en"])
train_x_lang.head()


# In[ ]:


train_data["original_language"].value_counts().plot(kind='pie',figsize= (10,10));


# In[ ]:


df_train_lang=pd.concat([train_x_lang, train_y], axis=1)
ScatterPlots(df_train_lang,n_plots=4)


#  from the scatter plots budget seems to be the most correlated to revenue, create a df with only budget as feature

# In[ ]:


train_x_only_budget = train_x.drop(columns=["popularity","runtime"])
train_x_only_budget.head()


# In[ ]:


train_x_budget_clean,train_y_budget_clean= RemoveMissVal(train_x_only_budget,train_y)
train_x_lang_clean,train_y_lang_clean= RemoveMissVal(train_x_lang,train_y)


# # Linear regression

# In[ ]:



lin_reg = linear_model.LinearRegression(normalize=True)
lin_reg.fit(train_x_clean, train_y_clean)

print("---coeffs-----")
print("coef:",lin_reg.coef_)
print("intercept:",lin_reg.intercept_)

y_pred = lin_reg.predict(train_x_clean)
y_pred_test= lin_reg.predict(test_x_clean)

print("---scoring----")
print("r2 score on y_train is:", r2_score(train_y_clean, y_pred))
print("lin reg score:",lin_reg.score(train_x_clean, train_y_clean))

plt.scatter(train_y,y_pred,color="teal")
plt.xlabel("y train")
plt.ylabel("y pred")


# In[ ]:


coeff_df = pd.DataFrame(lin_reg.coef_, train_x_clean.columns, columns=['Coefficient'])  
coeff_df


# In[ ]:


df_pred = pd.DataFrame({'Actual': train_y, 'Predicted': y_pred, "difference":train_y-y_pred, "ratio":train_y/y_pred})
df_pred.head(10)


# show the true and predicet on a hist for first 100 rows

# In[ ]:


df_pred = pd.DataFrame({'Actual': train_y[:100], 'Predicted': y_pred[:100]})
df_pred.plot(kind='bar',figsize=(50,8))
plt.show()


# cross validation on the training set, compare output of rmsle, r2 with homemade function and r2 with built in sk learn (not sure why we get a difference between the homemade r1 and sk learn r2 here..)

# In[ ]:


cross_val_predict(train_x_clean,train_y_clean, lin_reg, Verbose=True)
print("cross_val_r2 with standard sk learn lib is: ",(cross_val_score(lin_reg, train_x_clean, train_y_clean, cv=5,scoring="r2")).mean())


# get the regressor parameters that give the best fit with GridSearchCV. Results show the defaults were indeed the best with the addition that normalize must be  set to True (which I was doing)

# In[ ]:


lin_reg.get_params().keys()


# In[ ]:


#try something new here with gridSearchCV
params = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(estimator=lin_reg,param_grid=params, cv=5, n_jobs=-1,scoring="r2")

grid.fit(train_x_clean, train_y_clean)
print('Best parameters:', grid.best_params_)
print('Best performance:', grid.best_score_)


# look for the best subset with the built in function. Based on the r2 score the subset (0,1,2) (i.e containing columns budget, popularity and runtime) is the best. RMSLE suggests to take only the oth colum (i.e only budget)

# In[ ]:


best_subset(lin_reg, train_x_clean, train_y_clean, max_size=8, cv=5)


# In[ ]:


best_subset(lin_reg, train_x_clean, train_y_clean, max_size=8, cv=5, use_rmsle=True,verbose=False)


# In[ ]:


best_subset(lin_reg, train_x_cut, train_y_cut, max_size=8, cv=5, use_rmsle=True,verbose=False)


# # MLP regressor

# In[ ]:


mlp=MLPRegressor(hidden_layer_sizes=(100,),alpha=1e-4,learning_rate_init=0.001,verbose=10,tol=0.00001,solver='adam',activation="relu")
mlp.fit(train_x_clean, train_y_clean)
print("score:",mlp.score(train_x_clean,train_y_clean))
print('current loss computed with the loss function: ',mlp.loss_)
#print('coefs: ', mlp.coefs_)
#print('intercepts: ',mlp.intercepts_)
print(' number of iterations for the solver: ', mlp.n_iter_)
print('num of layers: ', mlp.n_layers_)
#print('Num of o/p: ', mlp.n_outputs_)


# In[ ]:





# Score is comparable to the one from linear regression. Don't understand why the loss is soo high though, tried tweeking various input parameters in the MLP but doesn't change..

# In[ ]:


#plot the output and compare with lin reg
y_pred_mlp=mlp.predict(train_x_clean)
y_pred_test_mlp=mlp.predict(test_x_clean)
plt.figure(figsize=(6, 4), dpi=100)
plt.scatter(train_y_clean,y_pred_mlp, c="orange",label="MLP-regressor",marker=".")
plt.scatter(train_y_clean,y_pred, color="teal",alpha=0.5,label="lin reg.",marker=".")
plt.xlabel("y train")
plt.ylabel("y pred")
plt.legend(loc=0)

print(mlp.coefs_[0].shape)
print(mlp.coefs_[1].shape)


# In[ ]:


plots to compare linear and MLP


# In[ ]:


print("---coeffs-----")
print("coef:",lin_reg.coef_)
print("intercept:",lin_reg.intercept_)

print(train_x_clean["budget"].values)
f = plt.figure(figsize=(20,5))
f.add_subplot(131)
plt.gca().set_title('budget')
plt.xlabel("budget")
plt.ylabel("revenue")
plt.scatter(train_x_clean["budget"],train_y_clean,label="train")
plt.scatter(train_x_clean["budget"],y_pred,label="predicted lin. reg")
plt.scatter(train_x_clean["budget"],y_pred_mlp,label="predicted MLP")
x=np.linspace(0,4e8,1000)
plt.plot(x,lin_reg.coef_[0]*x+lin_reg.intercept_,color="r")
plt.legend(loc=0)

f.add_subplot(132)
plt.gca().set_title('popularity')
plt.xlabel("popularity")
plt.ylabel("revenue")
plt.xlim(-10,100)
plt.scatter(train_x_clean["popularity"],train_y_clean,label="train")
plt.scatter(train_x_clean["popularity"],y_pred,label="predicted lin. reg")
plt.scatter(train_x_clean["popularity"],y_pred_mlp,label="predicted MLP")
x=np.linspace(0,1000,100)
plt.plot(x,lin_reg.coef_[1]*x+lin_reg.intercept_,color="r")
#plt.scatter(test_x_clean["popularity"],y_pred_test_mlp,label="predicted MLP test")
plt.legend(loc=0)

f.add_subplot(133)
plt.gca().set_title('runtime')
plt.xlabel("runtime")
plt.ylabel("revenue")
#plt.xlim(0,100)
plt.scatter(train_x_clean["runtime"],train_y_clean,label="train")
plt.scatter(train_x_clean["runtime"],y_pred,label="predicted lin. reg")
plt.scatter(train_x_clean["runtime"],y_pred_mlp,label="predicted MLP")
x=np.linspace(0,1000,100)
plt.plot(x,lin_reg.coef_[2]*x+lin_reg.intercept_,color="r")
plt.legend(loc=0)


# find the best subset, based on r2 score the best subset is 0,1 with score of 0.549

# In[ ]:


mlp=MLPRegressor(hidden_layer_sizes=(100,),alpha=1e-4,learning_rate_init=0.001,tol=0.00001,solver='adam',activation="relu")
best_subset(mlp, train_x_clean, train_y_clean, max_size=8, cv=5, use_rmsle=False,verbose=False)


# In[ ]:





# perform grid search to find best set of  parameters (only some scans shown there, to be tunes in params)

# In[ ]:


params = {'hidden_layer_sizes': [i for i in range(95,105)],
              'activation': ['relu'],
              'solver': ['adam'],
              'learning_rate': ['constant'],
              'learning_rate_init': [0.001],
              'power_t': [0.5],
              'alpha': [0.0001],
              'max_iter': [1000],
              'early_stopping': [False,True],
              'warm_start': [False]}

#grid = GridSearchCV(estimator=lin_reg,param_grid=params, cv=5, n_jobs=-1,scoring="r2")

grid = GridSearchCV(mlp, param_grid=params, scoring="r2",
                   cv=5, pre_dispatch='2*n_jobs')
grid.fit(train_x_clean, train_y_clean)
print('Best parameters:', grid.best_params_)
print('Best performance:', grid.best_score_)


# validation and learning curves. The cross validation score follows closely the training score, something strange there. To be checked

# In[ ]:


#hidden_layer_sizes
scan_range=np.arange(1, 100, 20)
print(scan_range)
train_scores, valid_scores = validation_curve(MLPRegressor(tol=1e-4), train_x_clean, train_y_clean, "hidden_layer_sizes",scan_range,cv=5)
PlotValidationCurve(train_scores,valid_scores,scan_range,"hidden_layer_size",verbose=True)


# In[ ]:


model = MLPRegressor(activation="relu",random_state=9,hidden_layer_sizes=(150,100,20,),learning_rate_init=0.01)
model.fit(train_x_clean, train_y_clean)


# In[ ]:


# when does the parameter alpha kick in ?
sumit = 0
alpha = 1e+20
loss_all = 4101055500334117.5 # I took the overall loss previously computed
for i in range(len(model.coefs_[0][0])): # let's take the last layer of MLP
  #print(sumit,model.coefs_[0][0][i])
  sumit+= alpha*model.coefs_[0][0][i]**2
print(sumit/loss_all) # the effect


# In[ ]:


model.loss_curve_


# In[ ]:


scan_range=np.logspace(15, 19, 5)
train_scores, valid_scores = validation_curve(MLPRegressor(activation="relu",random_state=9,hidden_layer_sizes=(150,100,20,),learning_rate_init=0.01), train_x_clean, train_y_clean, "alpha",scan_range,cv=2)
PlotValidationCurve(train_scores,valid_scores,scan_range,"alpha",logx=True)


# No improvement with the regularisation term. Even with alpha 1e6 seems we don't overfit.. This is to be checked

# In[ ]:





# In[ ]:


train_sizes, train_scores, valid_scores = learning_curve(MLPRegressor(activation="relu",random_state=9), train_x_clean, train_y_clean, train_sizes=[0.1, 0.15,0.5, 1], cv=5)
PlotLearningCurve(train_sizes,train_scores,valid_scores,scan_range,"train size",ymin=0.2,ymax=0.8)


# In[ ]:


train_sizes, train_scores, valid_scores = learning_curve(linear_model.LinearRegression(), train_x_clean, train_y_clean, train_sizes=[0.1, 0.15,0.5, 1], cv=5)
PlotLearningCurve(train_sizes,train_scores,valid_scores,scan_range,"train size",ymin=0.2,ymax=0.8)


# In[ ]:


get_ipython().run_cell_magic('html', '', "<marquee style='width: 30%; color: blue;'><b>Try to see if we get improvements with the only buget and with added language feature</b></marquee>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<!--<marquee style='width: 30%; color: blue;'><b>Well  not quite, try with df adding language and also df with only budget </b></marquee>-->")


# In[ ]:


train_x_lang_clean.head()


# In[ ]:


df_train_lang=pd.concat([train_x_lang_clean, train_y_clean], axis=1)
ScatterPlots(df_train_lang,n_plots=4)


# In[ ]:


mlp=MLPRegressor(hidden_layer_sizes=(20,20,20,),alpha=0.0001,random_state=9,max_iter=200)
mlp.fit(train_x_lang_clean, train_y_lang_clean)
print("score:",mlp.score(train_x_lang_clean,train_y_lang_clean))
print('current loss computed with the loss function: ',mlp.loss_)
#print('coefs: ', mlp.coefs_)
#print('intercepts: ',mlp.intercepts_)
print(' number of iterations for the solver: ', mlp.n_iter_)
print('num of layers: ', mlp.n_layers_)
print('Num of o/p: ', mlp.n_outputs_)


# In[ ]:


best_subset(mlp, train_x_lang_clean, train_y_lang_clean, max_size=8, cv=5, use_rmsle=True,verbose=False)


# In[ ]:


best_subset(mlp, train_x_budget_clean, train_y_budget_clean, max_size=8, cv=5, use_rmsle=False,verbose=False)


# In[ ]:


grid.fit(train_x_lang_clean, train_y_lang_clean)
print('Best parameters:', grid.best_params_)
print('Best performance:', grid.best_score_)


# In[ ]:


grid.fit(train_x_budget_clean, train_y_budget_clean)
print('Best parameters:', grid.best_params_)
print('Best performance:', grid.best_score_)


# In[ ]:


scan_range=np.arange(1, 100, 20)
print(scan_range)
train_scores, valid_scores = validation_curve(MLPRegressor(tol=1e-4), train_x_budget_clean, train_y_budget_clean, "hidden_layer_sizes",scan_range,cv=5)
PlotValidationCurve(train_scores,valid_scores,scan_range,"hidden_layer_size",verbose=True)


# * validation curve and gridsearchCV: Tried many parameters tuning on the MLP, the most sensitive seems to be the hidden_layer_size (should be above 20 no point in going above), and the activation function (relu is the best)
# * learning curve: points to a dataset of more than 1000 from the training score (although the cv score is flat independently of the data set size, so something strange there)
# * seems I cannot overfit even with a very large reg. term
# * best_subset: adding the language feature doesn't improve the score. The best score of 0.55  is acheived from fitting only the budget with the MLP.
# 

# In[ ]:


get_ipython().run_cell_magic('html', '', "<marquee style='width: 30%; color: blue;'><b>Whee finished!</b></marquee>")

