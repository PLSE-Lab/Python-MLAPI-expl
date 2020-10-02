#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. (Hey, I'm just a simple kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split # Import train_test_split from sklearn library


from sklearn import model_selection #models selection importing
from sklearn import metrics #selection metrics
from sklearn.metrics import confusion_matrix #confusion matrix
from sklearn.linear_model import LogisticRegression #Logistic regression
from sklearn.model_selection import cross_val_score #cross validarion import
from sklearn.metrics import classification_report #classification report import
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score #import the ROC curve 
from sklearn.metrics import roc_curve


# There is 1 csv file in the current version of the dataset:
# 

# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: ../input/Churn_Modelling.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# Churn_Modelling.csv has 10000 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/Churn_Modelling.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Churn_Modelling.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


print(os.listdir('../input'))


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1, 8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df1, 20, 10)


# ## Conclusion 1`
# This concludes your starter analysis! To go forward from here, click the blue "Fork Notebook" button at the top of this kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!

# # #My work start from here!

# In[ ]:


df1.head()


# In[ ]:


#As we are predicting the exit status of the customer we need to see how some important features are collerated with the predictive feature
#plt.bar(df1.Balance,df1.Exited)


# In[ ]:


#checking the correlation between age and balance!
np.corrcoef(df1.Balance,df1.Age)


# Oh! in real life I was expecting these two variables to be more correlated but they are not!!
# This gives me an idea on how the data might hold some new different amazing relationship! 

# In[ ]:


#copying the original dataset
df2=df1.copy()
df2['Zero_Balance']=np.where(df2['Balance']==0, 1, 0)
gender1=np.array(df2.groupby(['Gender'])['HasCrCard','IsActiveMember','Exited','Zero_Balance'].mean().reset_index())
Geography1=np.array(df2.groupby(['Geography'])['HasCrCard','IsActiveMember','Exited','Zero_Balance'].mean().reset_index())


# In[ ]:


#This method plot the the statistics of given entry
def plotData(data,row,label,subplt,fig):
    ax = fig.add_subplot(subplt)
    N = 4
    gen = data[row,1:]
    ind = np.array(['HasCard','IsActive','Exited','Zero_Balance'])    # the x locations for the groups
    p2 = ax.bar(ind, gen, 0.6, color=(0.2588,0.4433,1.0))
    p1 = ax.bar(ind, 1-gen, 0.6,color=(1.0,0.5,0) ,bottom=gen)
    plt.ylabel("Level")
    plt.title("%s statistics" %label)


# In[ ]:


#plotting the gender statistics
fig = plt.figure(figsize=(10,6))
plotData(gender1,0,"Female",121,fig)
plotData(gender1,1,"Male",122,fig)
#plotting the Geography statistics
fig2 = plt.figure(figsize=(15,6))
plotData(Geography1,0,"France",131,fig2)
plotData(Geography1,1,"Germany",132,fig2)
plotData(Geography1,2,"Spain",133,fig2)


# From the Above analysis we can see that the trend is likely to be the same for men and women in a way of being active or having a card or not but as we can see the Female are more likely to exit comparing to the men.
# 

# From the above figures we can easily get the insight on what is in the data regarding some potential identifiers on account balances and how it goes with being active or Exited accross the countries given. This will help me understand the feature importances in selection. For example for Germany.. No customer has zero balance and yet It has a higher percentage of exited customers..  Thus is more important feature to consider when building model. 

# # Feature extractions

# From the dataset we have we can easily throw out the unimportant columns for our prediction..
# Following are column that are obvious that has no help on prediction:
# - RowNumber
# - CustomerId
# - Surname
# 
# We will simply ignore them and they might come back when we want to identify what customer we've predicted

# In[ ]:


df=df1.copy()
df.columns


# #Because we will be using Gender and Geography columns which have string values we need to assign some dummie  values to each of the entry of those column.
# There are two ways of doing this:
# 1. By defaults I'm assignning like:
# - Female= 1
# - Male=0
# Or use the dummie function to get the expanded columns of each of the entries 
# 
# Again for the Geography case we can use the dummie function as by using the mormal assigning for the case  we have many countries it can bias our prediction.
# Thus I will use the dummie function too.
# 

# In[ ]:


gend=pd.get_dummies(df.Gender)
geo=pd.get_dummies(df.Geography)


# In[ ]:


#combining the object so we can concatenate with the bigger dataframe
obj=[df,gend,geo]
df=pd.concat(obj, axis=1)


# In[ ]:


#creating the dataset we will be using 
#this contain only helpfull feature by ignoring the Gender and geography columns
df3 = df[['CreditScore','Female','Male','France','Germany','Spain','Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember','Balance','EstimatedSalary','Exited']]


# **Now on as we are going to build model we need to identify which is a dependent(Predictive) variable and Independent. For our case dependent is "Exited " column and  other are predictors**

# In[ ]:


X=df3[['CreditScore','Female','Male','France','Germany','Spain','Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Balance']]
Y=df3[['Exited']]


# # #Splitting data

# As now we have the idea on what is in the data basically, We know that in machine learning, before running any algorithm in our dataset we need to divide our dataset into two sets one called training_set and another test_set. This splitting helps us to prevent Overfiiting or Underfitting of our machine learning model.

# In[ ]:


# random_state below is a metric that is used by the function to shuffle datas while splitting. This is chosen randomly.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 2) # 0.2 test_size means 20% for testing

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# 1. # #MODEL SELECTION
# As we have the pre-defined label then we will be doing a **Supervised learning**

# Supervised learning is the task of inferring a function from labeled training data. By fitting to the labeled training set, we want to find the most optimal model parameters to predict unknown labels on other objects.

# There are several techniques to use for this problem byut we keep in mind that we are predicting a binary classification problem. There are some appropriate model to use for this problem:
# - Logistic regression
# - Support vector  machine
# - Decision tree classification
# - Random forest classification
# - KNN classification.
# 
# Some of them are more complex but they don't really improves the performance! I will be starting using Logistic regression

# # Cross-validation

# Cross-validation is a technique for evaluating ML models by training several ML models on subsets of the available input data and evaluating them on the complementary subset of the data. Use cross-validation to detect overfitting, ie, failing to generalize a pattern.
# 
# You can use the k-fold cross-validation method to perform cross-validation. In k-fold cross-validation, you split the input data into k subsets of data (also known as folds). You train an ML model on all but one (k-1) of the subsets, and then evaluate the model on the subset that was not used for training. This process is repeated k times, with a different subset reserved for evaluation (and excluded from training) each time.  //source: Amazon.com
# 
# #This techniques prevent overfitting of the training dataset! //Following is for Logistic Regression

# In[ ]:


kfold = model_selection.KFold(n_splits=5, random_state=2)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train.values.ravel(), cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# ***Score of 78.4% is good for our dataset to build our model.*****

# # # 1. Logistic regression MODEL 

# In[ ]:


#getting the Logistic Regression model
logreg = LogisticRegression()
#fitting the data
logreg.fit(X_train, y_train.values.ravel())
#predicting 
y_pred = logreg.predict(X_test)
#printing the accuracy
print('Accuracy of logistic regression classifier on test set:'+str(format(logreg.score(X_test, y_test))))
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix
print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
#printing the confusion matrix graphic
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# # #Precision & recall**

# The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.The recall is intuitively the ability of the classifier to find all the positive samples.
# The F-beta score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall and precision are equally important. The support is the number of occurrences of each class in y_test.

# In[ ]:


print(classification_report(y_test, y_pred))


# # Feature prunning

# As we can see some of the features are not so more important like others. I choose to use extra tree classifier. This technique implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

# In[ ]:


#USING THE TREE CLASSIFIER
clf = ExtraTreesClassifier()
#fitting the features 
clf = clf.fit(X_train, y_train.values.ravel())
#
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
#extracting the selected features
feature_idx=model.get_support()
feature_name=X.columns[feature_idx]
#Getting the new features extracted 
X_test_new=X_test.loc[:,feature_idx]
X_train_new=X_train.loc[:,feature_idx]


# In[ ]:


#Those are more important features then I will reuse them again in the model
feature_name


# # Fitting the Logistic Regression again!!

# In[ ]:


logre = LogisticRegression()
logre.fit(X_train_new ,y_train.values.ravel())

y_pred = logre.predict(X_test_new)
print('Accuracy of logistic regression classifier on test set:'+str(format(logre.score(X_test_new, y_test))))
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix
print(conf_matrix)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title('Confusion matrix of the classifier IMPROVED')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
y= (conf_matrix[0,0]+conf_matrix[1,1])/(conf_matrix[0,0]+conf_matrix[0,1]+conf_matrix[1,0]+conf_matrix[1,1])
print('Accurracy= '+str(y))
print(classification_report(y_test, y_pred))


# The above changes didn't really improves our model performance comparing to when using all the 13 features!
# 
# Due to the above results let me try to use The random forest classification methods to improve the performnce.

# # #Support vector machine

# In[ ]:


from sklearn.svm import SVC 
model = SVC(probability=True)  
# Fitting the model
model = model.fit(X_train_new, y_train)  
# Predictions/probs on the test dataset
predicted = pd.DataFrame(model.predict(X_test_new))  
# Store metrics
accuracy = metrics.accuracy_score(y_test, predicted)
print("Support vector machine acuuracy is ",accuracy)


# Wow! O.82 for this SVM is an improvment. It improved my the model by 0.4%. Which is good somehow. Let me try to use Random forest.
# **The random forest is the powerful one for binary classification predictions.**

# In[ ]:





# # # 2. USING RANDOM FOREST

# In[ ]:


# The  number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
#fitting the classifier
rfecv = rfecv.fit(X_train_new, y_train.values.ravel())

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train_new.columns[rfecv.support_])

# Plot number of features VS. cross-validation scores

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(X_train_new,y_train.values.ravel())
#printing the score 
ypred_new=clf_rf.predict(X_test_new)
ac = accuracy_score(y_test,ypred_new)
print('Accuracy is: ',ac)


# # !!! Accuracy is:  0.86 WHICH IS AN IMPROVEMENT!!!!

# Considering the size of data this kind of occuracy is good 

# Conclusion:
# Considering the size of dataset we got and the data structure we had I can confidently say that 86% ACCURACY is good. I am very sure that if I would be having more sufficent data and time to preprocess and to train well my model I would reach up to 98% accuracy!.
