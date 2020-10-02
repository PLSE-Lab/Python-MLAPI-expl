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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#sns style
import seaborn as sns
#sns.set_style("whitegrid")
sns.despine()
sns.set_context("talk") #larger display of plots axis labels etc..
sns.set(style='darkgrid')


# # Functions

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
  
def ScatterPlots(df, n_plots=3):
  fig, axes = plt.subplots(1, n_plots)
  i=0
  for key, value in df.iloc[:, :-1].iteritems(): 
    print(key) 
    df.plot(kind="scatter",x=key, y="Survived",color="orange",ax=axes[i],figsize=(20,10))
    i+=1

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


def rmsle(y_pred, y_test) : 
    #clip zero values
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(list(np.asarray(y_pred).clip(min=0) + 1)) - np.log(list(np.asarray(y_test).clip(min=0) + 1)))**2))
  
  
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


# In[ ]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[ ]:


def conf_mat(train_y,train_x):
  # Creating the confusion matrix:
  lr_cm = confusion_matrix(train_y, mlp.predict(train_x))

  #Visualization:
  f, ax = plt.subplots(figsize=(5,5))
  sns.heatmap(lr_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax, cmap='inferno')
  plt.title('Logistic Regression Classification Confusion Matrix')
  plt.xlabel('y_pred')
  plt.ylabel('y_truth')
  plt.show()


# In[ ]:


def undersample(train_x, train_y, print_val=False):
  exit_indices= np.array(train_y[train_y.Exited==1].index)
  stayed_indices = np.array(train_y[train_y.Exited==0].index)
  if(print_val):
    print(exit_indices.shape)
    print(stayed_indices.shape)
  stayed_indices_undersampled = np.random.choice(stayed_indices, len(exit_indices), replace=False)
  if(print_val):
    print("shape of stayed undersampled",stayed_indices_undersampled.shape)
    print("shape of exited",exit_indices.shape)
  undersampled_indices = np.concatenate([stayed_indices_undersampled, exit_indices]) 
  #print(undersampled_indices.shape)

  train_x_us = train_x.loc[undersampled_indices]
  train_y_us = train_y.loc[undersampled_indices]
  return train_x_us, train_y_us


# # load dataset

# In[ ]:





# In[ ]:


train_data= pd.read_csv('../input/Churn_Modelling.csv')


# #EDA 

# In[ ]:


train_data.head()


# In[ ]:


train_data.hist(bins="auto",figsize=(12,7),grid=False);
#train_data.hist(figsize=(20,15),bins=100,color="orange",log=True,layout=(11,1))


# In[ ]:


train_data.info()


# In[ ]:


sns.pairplot(train_data,hue="Exited",height=2)


# In[ ]:


fig = plt.figure(figsize = (18,18)); ax = fig.gca()
#one_hot_churn = pd.get_dummies(churn, columns = ['Gender', 'Geography', 'HasCrCard', 'IsActiveMember'])
sns.heatmap(train_data.corr(), annot = True, vmin= -0.5, vmax = 0.5, ax=ax)


# In[ ]:


train_data.head()


# In[ ]:


train_data["Gender"].value_counts().plot(kind='pie',figsize= (8,8));


# In[ ]:


def bar_chart(feature,input_df):
    Exited = input_df[input_df['Exited']==1][feature].value_counts()
    Stayed = input_df[input_df['Exited']==0][feature].value_counts()
    df = pd.DataFrame([Exited,Stayed])
    df.index = ['Exited','Stayed']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


bar_chart("Gender",train_data)
#train_data["Gender"].value_counts().plot(kind='pie',figsize= (8,8));


# In[ ]:


bar_chart("IsActiveMember",train_data)


# Males tend to stay a bit more also active members, no strong bias for geography

# In[ ]:





# In[ ]:


train_data["Geography"].value_counts().plot(kind='pie',figsize= (8,8));


# In[ ]:


bar_chart("Geography",train_data)


# In[ ]:


train_data.head()


# * Males tend to stay a bit more, so do active members, no strong bias for geography
# * quite strong correlation with age and balance
# * hot encode Gender and Geography
# * drop  Rownumber , customer Id surname,

# In[ ]:


#train_data_red=train_data.drop(["RowNumber","CreditScore", "CustomerId", "Surname", "Tenure","Salary"], axis=1)
train_data.head()


# In[ ]:


train_data_red=train_data.drop(["RowNumber","CreditScore", "CustomerId", "Surname", "Tenure","EstimatedSalary"], axis=1)
train_data_red.head()
#train_data=train_data_red


# In[ ]:


train_data_red['Gender'] = train_data['Gender'].map({'Female': 1, 'Male': 0})
train_data_red.head()
#CheckValues(train_x_clean,train_y_clean,detail=True)


# In[ ]:


one_hot = pd.get_dummies(train_data_red['Geography'])
train_hot_geo = train_data_red.drop('Geography',axis = 1)
#one_hot["Spain"]
#train_hot_geo=train_data_red.join(one_hot["Spain"],one_hot["France"],one_hot["Germany"])
train_hot_geo=train_hot_geo.join(one_hot["Spain"])
train_hot_geo=train_hot_geo.join(one_hot["France"])
train_hot_geo=train_hot_geo.join(one_hot["Germany"])
train_hot_geo.head()


# keep this DF

# In[ ]:


train_x = train_hot_geo.drop(columns=["Exited"])#drop the label
train_x.head()


# In[ ]:


train_y=train_hot_geo[["Exited"]]


# In[ ]:


print(train_x.shape)
print(train_y.shape)
print(type(train_y))
print(type(train_x))


# In[ ]:


df_train=pd.concat([train_x, train_y], axis=1)
df_train.head()


# In[ ]:


CheckValues(train_x,train_y,detail=True)


# In[ ]:


fig = plt.figure(figsize = (18,18)); ax = fig.gca()
#one_hot_churn = pd.get_dummies(churn, columns = ['Gender', 'Geography', 'HasCrCard', 'IsActiveMember'])
sns.heatmap(df_train.corr(), annot = True, vmin= -0.5, vmax = 0.5, ax=ax)


# # First dirty linear regression
# 1. Select a small number of features which need small or no preparation
# 2. Train a first MLPClassifier with Scikit learn
# 3. Look at the predicted values (to see if they make sense)

# In[ ]:



train_x.head()


# In[ ]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=0, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(train_x, train_y)

print("Training set score: %f" % mlp.score(train_x, train_y))
print ("cross_val F1 score:",cross_val_score(mlp, train_x, train_y, scoring="f1_macro",cv=3)) 

cm = confusion_matrix(train_y, mlp.predict(train_x))
print(cm)

conf_mat(train_y,train_x)
#plt.hist(mlp.predict_proba(train_x)[:,1])
#plt.hist(mlp.predict_proba(train_x)[:,0])

print(classification_report(train_y,mlp.predict(train_x)))


# # with undersampling

# In[ ]:


train_x.head()


# In[ ]:





# try to improve the F1 score 

# In[ ]:


train_x_us, train_y_us= undersample(train_x,train_y,print_val=True)

mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000,
                    solver='adam', verbose=0, tol=1e-4, random_state=1,
                    learning_rate_init=.001,early_stopping=True)

mlp.fit(train_x_us, train_y_us)
print("Training set score: %f" % mlp.score(train_x_us, train_y_us))
cv_score=cross_val_score(mlp, train_x_us, train_y_us, scoring="f1_macro",cv=3).mean()
#print ("cross_val F1 score:",cross_val_score(mlp, train_x_us, train_y_us, scoring="f1_macro",cv=3)) 
print ("cross_val F1 score:",cv_score) 
cm = confusion_matrix(train_y_us, mlp.predict(train_x_us))

print(cm)

conf_mat(train_y_us,train_x_us)
#plt.hist(mlp.predict_proba(train_x_us)[:,1])
#plt.hist(mlp.predict_proba(train_x_us)[:,0])
print("the mean F1 score from 3-fold cross-validation is:",cv_score)
print(classification_report(train_y_us,mlp.predict(train_x_us)))


# In[ ]:


#f1_macro

#params = {'hidden_layer_sizes': [i for i in range(95,105)],
#              'activation': ['relu'],
#              'solver': ['adam',"sgd"],
#              'learning_rate': ['constant'],
#              'learning_rate_init': [0.001],
#              'power_t': [0.5],
#              'alpha': [0.0001],
#              'max_iter': [1000],
#              'early_stopping': [False,True],
#              'warm_start': [False]}

#grid = GridSearchCV(estimator=lin_reg,param_grid=params, cv=5, n_jobs=-1,scoring="r2")

#grid = GridSearchCV(mlp, param_grid=params, scoring="f1_macro",
#                   cv=5, pre_dispatch='2*n_jobs')
#grid.fit(train_x, train_y)
#print('Best parameters:', grid.best_params_)
#print('Best performance:', grid.best_score_)


# #feature engineering

# In[ ]:


a = sns.FacetGrid(df_train, hue = 'Exited', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , df_train['Age'].max()))
a.add_legend()


# In[ ]:


df_train.head()


# In[ ]:


df_train.hist("Age")


# In[ ]:


df_ones=df_train.loc[(df_train.Exited == 1)]
df_zeros=df_train.loc[(df_train.Exited == 0)]


# In[ ]:


df_ones.hist("Age")
df_zeros.hist("Age")


# In[ ]:


print(df_ones["Age"].values.mean())
print(df_ones["Age"].values.std())


# In[ ]:


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#gaussian(50,44,9.7)


# In[ ]:


print(df_zeros["Age"].values.mean())
print(df_zeros["Age"].values.std())


# In[ ]:


#df_train=df_train.drop(columns=["gauss_age","sqr_age","tran_age"])
df_train.head()


# In[ ]:


df_train["Age_stay"]=df_train["Age"].apply(lambda x: gaussian(x,44,9.7))


# In[ ]:


df_train.head()


# In[ ]:


df_train["Age_exit"]=df_train["Age"].apply(lambda x: gaussian(x,37,10))


# In[ ]:


df_train.head(50)


# In[ ]:


#from sklearn.mixture import GaussianMixture 
#gmm = GaussianMixture(n_components = 2) 
#gmm.fit(df_train[['Age']].values)
#labels = gmm.predict(df_train[['Age']].values) 
#print(labels)


# In[ ]:


#a = sns.FacetGrid(df_train, hue = 'Exited', aspect=3 )
#a.map(sns.kdeplot, 'gauss_age', shade= True)
#a.set(xlim=(df_train['sqr_age'].min(), df_train['sqr_age'].max()))
#a.add_legend()


# In[ ]:


train_x=df_train.drop(columns=["Exited"])
train_y=df_train[["Exited"]]
print(train_x.shape)
print(train_y.shape)


# In[ ]:


train_x.head()


# With feature scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features=scaler.fit_transform(train_x)
train_x_scaled = pd.DataFrame(scaled_features, index=train_x.index, columns=train_x.columns)
train_x_scaled.head()


# In[ ]:





# In[ ]:



#!should normalize after the splitting!
x_train, x_test, y_train, y_test = train_test_split(train_x_scaled, train_y, test_size=0.2, random_state=42)

train_x_us, train_y_us= undersample(x_train,y_train,print_val=True)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                    solver='adam', verbose=0, tol=1e-4, random_state=1,
                    learning_rate_init=.01,early_stopping=True)

mlp.fit(train_x_us, train_y_us)
print(mlp.predict(train_x_us))
print("Training set score: %f" % mlp.score(train_x_us, train_y_us))
print ("cross_val F1 score:",cross_val_score(mlp, train_x_us, train_y_us, scoring="f1_macro",cv=3)) 

cm = confusion_matrix(train_y_us, mlp.predict(train_x_us))

print(cm)

conf_mat(train_y_us,train_x_us)

print("---- report for train ----- ")
print(classification_report(train_y_us,mlp.predict(train_x_us)))

print("---- report for test ----- ")
print(classification_report(y_test,mlp.predict(x_test)))

#plt.hist(mlp.predict_proba(train_x_us)[:,1])
#plt.hist(mlp.predict_proba(train_x_us)[:,0])


# In[ ]:


#train_y_us,mlp.predict(train_x_us))
train_x.hist("Age")


# In[ ]:


x_test.hist("Age")


# I I had more time:
# * fine tune the MLP (e.g grid search while scoring on the F1)
# * Correlation with balance seems quite high, maybe something to do there
# * training curve

# In[ ]:


get_ipython().run_cell_magic('html', '', "<marquee style='width: 30%; color: blue;'><b>Whee finished!</b></marquee>")

