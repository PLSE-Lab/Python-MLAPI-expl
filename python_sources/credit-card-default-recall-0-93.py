#!/usr/bin/env python
# coding: utf-8

# # Predicting Credit Card Default 

# ## Importing Libraries

# In[ ]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# ## Reading the Dataset

# In[ ]:


# Reading the csv file and putting it into 'df' object.
df = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
df.head()


# ## Importing Train Test Split

# In[ ]:


# Importing test_train_split from sklearn library
from sklearn.model_selection import train_test_split


# ## Creating Dependent and Independent Variables

# In[ ]:


# Putting feature variable to X
X = df.drop('default.payment.next.month',axis=1)

# Putting response variable to y
y = df['default.payment.next.month']

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ## Building A Random Forest Model

# In[ ]:


# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()


# In[ ]:


# fit
rfc.fit(X_train,y_train)


# ## Making Predictions

# In[ ]:


# Making predictions
predictions = rfc.predict(X_test)


# ## Importing Confusion Matrix

# In[ ]:


# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


# In[ ]:


# Let's check the report of our default model
print(classification_report(y_test,predictions))


# As we can infer we get a recall of 0.36 , goal is to maximize the recall score in order to identify maximum defaulters. 

# In[ ]:


# Printing confusion matrix
print(confusion_matrix(y_test,predictions))


# ## Printing Accuracy

# In[ ]:


print(accuracy_score(y_test,predictions))


# ## Renaming the Columns of the Dataset

# In[ ]:


# Renaming Columns into more understandable/user-friendly terms

# SEX changed to GENDER
# PAY_0 changed to PAY_1
# default.payment.next.month is too long and changed to something simplier, DEFAULT
df.rename(columns={'SEX':'GENDER',
                   'PAY_0':'PAY_1',
                   'default.payment.next.month':'DEFAULT',} , inplace=True)

df.drop('ID', axis=1, inplace=True) # Drop column ID

df.info()  # we see that we have 30,000 observations and no null values


# In[ ]:


# We inspect the data as a whole|
df.describe().T


# # Feature Engineering

# ## Creating New Features

# In[ ]:


sum_column = df["PAY_1"] + df["PAY_2"]+ df["PAY_3"]+ df["PAY_4"]+ df["PAY_5"]+ df["PAY_6"]
df['pay_sum'] = sum_column
bill_sum = df["BILL_AMT1"]+df["BILL_AMT2"]+df["BILL_AMT3"]+df["BILL_AMT4"]+df["BILL_AMT5"]+df["BILL_AMT6"]
df["bill_sum"]=bill_sum
pay_amt = df['PAY_1']+df['PAY_2']+df['PAY_3']+df['PAY_4']+df['PAY_5']+df['PAY_6']
df["pay_amt_sum"]=pay_amt


# In[ ]:


print('Education Column Values: ', df['EDUCATION'].unique())


# In[ ]:


fig, ax = plt.subplots()
sns.countplot(data=df,x='EDUCATION', order = df['EDUCATION'].value_counts().index, color='salmon')


# In[ ]:


df['EDUCATION'].value_counts()


# In[ ]:


# There exists values 0, 5 and 6 in this column.
# Since these are unknown (undefined), they can be grouped into the category 4: "Others"

df['EDUCATION'] = df['EDUCATION'].apply(lambda edu_value: edu_value 
                                        if ((edu_value > 0 and edu_value < 4)) 
                                        else 4) # Changes every value of x not within (and inclusive of) 1 ~ 3 to 4  

# Corrected changes
df['EDUCATION'].unique()


# In[ ]:


# Countplt
fig, ax = plt.subplots()
sns.countplot(data=df,x='EDUCATION', order = df['EDUCATION'].value_counts().index, color='salmon');


# In[ ]:


print("Marriage Column Values: ", df['MARRIAGE'].unique())


# In[ ]:


df['MARRIAGE'] = df['MARRIAGE'].apply(lambda marriage_value: marriage_value
                                     if (marriage_value > 0 and marriage_value < 3)
                                     else 3) # changes every value of x not within (and inclusive of) 1 and 2 to 3

# Corrected changes
df['MARRIAGE'].unique()


# In[ ]:


df['AGE'].unique()


# In[ ]:


## Creating a Function to Distribute the Age
def func(x):
    if(x >=20 and x<30 ):
        return 1
    elif(x>=30 and x<40):
        return 2
    elif(x>=40 and x<50):
        return 3
    elif(x>=50 and x<60):
        return 4
    elif(x>=60 and x<=80):
        return 5


# In[ ]:


## Applying the function
df['AGE'] = df['AGE'].apply(func)


# In[ ]:



fig, ax = plt.subplots()
sns.countplot(data=df,x='AGE', order = df['AGE'].value_counts().index, color='salmon');


# In[ ]:


# Creating a new dataframe with just the categorical explanatory variables
df_categorical = df[['GENDER', 'EDUCATION', 'MARRIAGE','AGE','PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
                     ,'DEFAULT']]


# In[ ]:


f, axes = plt.subplots(3, 3, figsize=(19,14), facecolor='white')
f.suptitle("FREQUENCY OF CATEGORICAL VARIABLES (BY TARGET)",size=20)

# Creating plots of each categorical variable to target 
ax1 = sns.countplot(x='GENDER', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[0,0])
ax2 = sns.countplot(x='EDUCATION', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[0,1])
ax3 = sns.countplot(x='MARRIAGE', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[0,2])
ax4 = sns.countplot(x='PAY_1', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[1,0])
ax5 = sns.countplot(x='PAY_2', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[1,1])
ax6 = sns.countplot(x='PAY_3', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[1,2])
ax7 = sns.countplot(x='PAY_4', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[2,0])
ax8 = sns.countplot(x='PAY_5', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[2,1])
ax9 = sns.countplot(x='PAY_6', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[2,2])
ax10 = sns.countplot(x='AGE', hue = 'DEFAULT', data=df_categorical, palette='Reds', ax=axes[2,2])
# Setting legends to upper right
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
ax3.legend(loc="upper right")
ax4.legend(loc="upper right")
ax5.legend(loc="upper right")
ax6.legend(loc="upper right")
ax7.legend(loc="upper right")
ax8.legend(loc="upper right")
ax9.legend(loc="upper right")
ax10.legend(loc="upper right")
# Changing ylabels to horizontal and changing their positions
ax1.set_ylabel('COUNTS', rotation=0, labelpad=40)  # Labelpad adjusts distance of the title from the graph
ax1.yaxis.set_label_coords(-0.1,1.02)              # (x, y)
ax2.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax2.yaxis.set_label_coords(-0.1,1.02)
ax3.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax3.yaxis.set_label_coords(-0.1,1.02)
ax4.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax4.yaxis.set_label_coords(-0.1,1.02)
ax5.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax5.yaxis.set_label_coords(-0.1,1.02)
ax6.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax6.yaxis.set_label_coords(-0.1,1.02)
ax7.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax7.yaxis.set_label_coords(-0.1,1.02)
ax8.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax8.yaxis.set_label_coords(-0.1,1.02)
ax9.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax9.yaxis.set_label_coords(-0.1,1.02)
ax10.set_ylabel('COUNTS', rotation=0, labelpad=40)
ax10.yaxis.set_label_coords(-0.1,1.02)

# Shifting the Super Title higher
f.tight_layout()  # Prevents graphs from overlapping with each other
f.subplots_adjust(top=0.9);


# In[ ]:


# generate binary values using get_dummies
age = pd.get_dummies(df['AGE'], prefix='AGE' )
mr = pd.get_dummies(df['MARRIAGE'], prefix='MARRIAGE' )
ed = pd.get_dummies(df['EDUCATION'],prefix='EDUCATION')
# merge with main df bridge_df on key values
df = df.join(age)
df = df.join(mr)
df= df.join(ed)


# In[ ]:


df = df.drop(['AGE','MARRIAGE','EDUCATION'],axis=1)


# In[ ]:


print(df['DEFAULT'].value_counts(),'\n')
print(len(df['DEFAULT']))


# #### We can infer it as Highly Imbalanced Dataset

# In[ ]:


# Frequency of the defaults
default = df['DEFAULT'].sum() # adds up all the default cases in the df
no_default = len(df['DEFAULT']) - default  # entire dataset - default cases

# Percentage of the defaults
default_perc = round(default/len(df['DEFAULT']) * 100, 1)
no_default_perc = round(no_default/len(df['DEFAULT']) * 100, 1)

# Plotting Target
fig, ax = plt.subplots(figsize=(10,7))  # Sets size of graph
sns.set_context('notebook', font_scale=1.2)  # Affects things like size of label, lines and other elements of the plot.

sns.countplot('DEFAULT',data=df, palette="Reds")   
plt.annotate('Non-default: {}'.format(no_default), 
             xy=(-0.25, 3000), # xy = (x dist from 0, y dist from 0)
            size=15.5)

plt.annotate('Default: {}'.format(default), 
             xy=(0.8, 3000), # xy = (x dist from 0, y dist from 0)
            size=15)
plt.annotate('{}%'.format(no_default_perc), xy=(-0.1, 8000),size=15)
plt.annotate('{}%'.format(default_perc), xy=(0.9, 8000),size=15)
plt.title('CREDIT CARD COUNT', size=18)
plt.xlabel("Default",size=15)
plt.ylabel('Count', rotation=0, 
           labelpad=40, # Adjusts distance of the title from the graph
           size=15)
ax.yaxis.set_label_coords(-0.1,.9)

plt.box(False)        # Removes the bounding area
plt.savefig('target_skew.png', transparent = True)


# #### 22 % of Defaulters 

# In[ ]:


# Freq distribution of all data
fig, ax = plt.subplots(figsize=(15,15))
pd.DataFrame.hist(df,ax=ax)
plt.tight_layout();


# In[ ]:


# Can we infer more? what about the columns for lIMIT_BALANCE?
x1 = list(df[df['DEFAULT'] == 1]['LIMIT_BAL'])
x2 = list(df[df['DEFAULT'] == 0]['LIMIT_BAL'])

fig2, ax_lim_bal = plt.subplots(figsize=(12,4))
sns.set_context('notebook', font_scale=1.2)
#sns.set_color_codes("pastel")
plt.hist([x1, x2], bins = 40, density=False, color=['firebrick', 'salmon'])
plt.xlim([0,600000])
plt.legend(['Yes', 'No'], title = 'Default', loc='upper right', facecolor='white')
plt.xlabel('Limit Balance (NT dollar)')
plt.ylabel('Frequency', rotation=0,labelpad=40)
plt.title('LIMIT BALANCE HISTOGRAM BY TYPE OF CREDIT CARD', SIZE=15)
plt.box(False)
plt.savefig('ImageName', format='png', dpi=200, transparent=True);


# In[ ]:


# Now that we have our features, let's plot them on a correlation matrix to remove anything that might 
# cause multi-colinearity within our model

sns.set(style="white")
# Creating the data
data = df.corr()


# Generate a mask for the upper triangle
mask = np.zeros_like(data, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# Set up the matplotlib figure to control size of heatmap
fig, ax = plt.subplots(figsize=(60,50))


# Create a custom color palette
cmap = sns.diverging_palette(133, 10,
                      as_cmap=True)  
# as_cmap returns a matplotlib colormap object rather than a list of colors
# Green = Good (low correlation), Red = Bad (high correlation) between the independent variables

# Plot the heatmap
g = sns.heatmap(data=data, annot=True, cmap=cmap, ax=ax, 
                mask=mask, # Splits heatmap into a triangle
                annot_kws={"size":20},  #Annotation size
               cbar_kws={"shrink": 0.8} # Color bar size
               );


# Prevent Heatmap Cut-Off Issue
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

# Changes size of the values on the label
ax.tick_params(labelsize=25) 

ax.set_yticklabels(g.get_yticklabels(), rotation=0);
ax.set_xticklabels(g.get_xticklabels(), rotation=80);

plt.savefig('correlation_heatmap.png', transparent = True)


# In[ ]:


df_default_corrs = data.filter(like='DEFAULT')


# In[ ]:


df_default_corrs


# In[ ]:


df_default_corrs.plot(kind='bar',figsize=(15,10))


# PAY_1,PAY_SUM,PAY_AMT_SUM and PAY_2 highly correlate with the Defaulters
# 

# In[ ]:


# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)
y = df['DEFAULT']

# Data splitting for 80% Train/Val and 20% Test 
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2, random_state=69) # 20% holdout 
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.25, random_state=69) # Train/Val

# Initializing the scaler  (Just scale every single time lol)
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
std.fit(X_train_val.values)

## Scale the Predictors on the train/val dataset
X_train_val_scaled = std.transform(X_train_val.values) 

## This line instantiates the model. 
rf = RandomForestClassifier() 

## Fit the model on your training data.
rf.fit(X_train_val_scaled, y_train_val) 

# Obtain the feature importance
feature_importance = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                   columns=['Variable_Importance']).sort_values('Variable_Importance',ascending=True)

# Set seaborn contexts 
sns.set(style="whitegrid")

feature_importance.plot.barh(figsize=(15,10))


# ## Importing Required Libraries

# In[ ]:


# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

# Classifier Metrics 
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import fbeta_score, cohen_kappa_score

# Pre-processing packages
from sklearn.preprocessing import StandardScaler


# CV, Gridsearch, train_test_split, model selection packages
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split


# ## Building Base Line Model (No Scaling)

# In[ ]:


## Baseline model performance evaluation

# to give model baseline report with cross-validation in dataframe 
def baseline_report_cv_(model, X, y, n_splits, name):
    """
    Accepts a model object, X (independent variables), y (target), n_splits and name of the model
    and returns a model with various scoring metrics of each classifier model on a cross-validation split
    ----
    Input: model object, X, y, n_splits (integer), name (str)
    Output: Various metric scores of a model.
    """
    # Splitting the data into 80% training/validation data and 20% testing data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
#     # Splitting the training data into 60% training data and 20% validation data.
#     X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=69)
     
    # Creating a shuffled kfold of 5
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=1000) 
    
    accuracy     = np.mean(cross_val_score(model, X_train_val, y_train_val,cv=cv, scoring='accuracy'))
    precision    = np.mean(cross_val_score(model, X_train_val, y_train_val,cv=cv, scoring='precision'))
    recall       = np.mean(cross_val_score(model, X_train_val, y_train_val,cv=cv, scoring='recall'))
    f1score      = np.mean(cross_val_score(model, X_train_val, y_train_val,cv=cv, scoring='f1'))
    rocauc       = np.mean(cross_val_score(model, X_train_val, y_train_val,cv=cv, scoring='roc_auc'))
    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'rocauc'       : [rocauc],
                             'timetaken'    : [0]       })   # timetaken for comparison later
    return df_model


# to evaluate baseline models
gnb = GaussianNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
linearsvc = LinearSVC()

# Scaling the inputs into model
# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)
y = df['DEFAULT']

# to concat all models
df_models = pd.concat([baseline_report_cv_(gnb, X, y, 5, 'GaussianNB'),
                       baseline_report_cv_(logit, X, y, 5, 'LogisticRegression'),
                       baseline_report_cv_(knn, X, y, 5, 'KNN'),
                       baseline_report_cv_(decisiontree, X, y, 5, 'DecisionTree'),
                       baseline_report_cv_(randomforest, X, y, 5, 'RandomForest'),
                       baseline_report_cv_(linearsvc, X, y, 5, 'LinearSVC')
                       ], axis=0).reset_index()

df_models_no_scale = df_models.drop('index', axis=1)
df_models_no_scale


# #### GuassianNB with Recall of 0.86

# ## Building Base Line Model(Scaled)

# In[ ]:


## Scaled Dataset Model performance evaluation

# to evaluate baseline models
gnb = GaussianNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
linearsvc = LinearSVC()

# Scaling the inputs into model
# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)
y = df['DEFAULT']

## Scale data (just scale everything lol)
std = StandardScaler()
std.fit(X.values)

## Scale the Predictors
X = std.transform(X.values)


# to concat all models
df_models = pd.concat([baseline_report_cv_(gnb, X, y, 5, 'GaussianNB'),
                       baseline_report_cv_(logit, X, y, 5, 'LogisticRegression'),
                       baseline_report_cv_(knn, X, y, 5, 'KNN'),
                       baseline_report_cv_(decisiontree, X, y, 5, 'DecisionTree'),
                       baseline_report_cv_(randomforest, X, y, 5, 'RandomForest'),
                       baseline_report_cv_(linearsvc, X, y, 5, 'LinearSVC')
                       ], axis=0).reset_index()

df_models_scale = df_models.drop('index', axis=1)
df_models_scale


# ## Poor Perfomance as compared to scale

# ## SMOTE 

# In[ ]:


## SMOTE Dataset model performance evaluation
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE

# Smote with no scaling with cross-validation in dataframe 
def baseline_report_cv_smote(model, X, y, n_splits, name):
    """
    Accepts a model object, X (independent variables), y (target), n_splits and name of the model, SMOTE's the data
    and returns a model with various scoring metrics of each classifier model on a cross-validation split
    ----
    Input: model object, X, y, n_splits (integer), name (str)
    Output: Various metric scores of a model.
    """
    from imblearn.over_sampling import SMOTE # Allows for smoting if you forget to initialize it before running func
    
    # Splitting the data into 80% training/validation data and 20% testing data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    # Splitting the training data into 60% training data and 20% validation data.
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=69)
    
    
    #this helps with the way kf will generate indices below
    X_train_val, y_train_val = np.array(X_train_val), np.array(y_train_val)
    
    
    # Creating a shuffled kfold of 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1000) 
    
    clf_model_acc_scores_cv = []
    clf_model_precision_scores_cv = []
    clf_model_recall_scores_cv = []
    clf_model_f1_scores_cv = []
    clf_model_rocauc_scores_cv = []
    
    # Manual Cross-Validation
    for train_ind, val_ind in kf.split(X_train_val, y_train_val):

        # Assigning train and validation values for an individual fold
        X_train, y_train = X_train_val[train_ind], y_train_val[train_ind]
        X_val, y_val = X_train_val[val_ind], y_train_val[val_ind] 

        # Creating the SMOTE data
        X_smoted, y_smoted = SMOTE(random_state=69).fit_sample(X_train, y_train)
        
        # Initializing model
        clf_model = model.fit(X_smoted, y_smoted) # Train model on SMOTE'd data
        y_pred = clf_model.predict(X_val)  # Y pred after testing on validation data split
        
        # Save scores of model
        clf_model_acc_score = accuracy_score(y_val, y_pred)
        clf_model_precision_score = precision_score(y_val, y_pred)
        clf_model_recall_score = recall_score(y_val, y_pred)
        clf_model_f1_score = f1_score(y_val, y_pred)   
        clf_model_rocauc_score = roc_auc_score(y_val, y_pred)
        
        # Append scores of model their scoring lists
        clf_model_acc_scores_cv.append(clf_model_acc_score)
        clf_model_precision_scores_cv.append(clf_model_precision_score)
        clf_model_recall_scores_cv.append(clf_model_recall_score)
        clf_model_f1_scores_cv.append(clf_model_f1_score)
        clf_model_rocauc_scores_cv.append(clf_model_rocauc_score)
        

    
    accuracy     = np.mean(clf_model_acc_scores_cv)
    precision    = np.mean(clf_model_precision_scores_cv)
    recall       = np.mean(clf_model_recall_scores_cv)
    f1score      = np.mean(clf_model_f1_scores_cv)
    rocauc       = np.mean(clf_model_rocauc_scores_cv)
    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'rocauc'       : [rocauc],
                             'timetaken'    : [0]       })   # timetaken for comparison later
    return df_model


# to evaluate baseline models
gnb = GaussianNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
linearsvc = LinearSVC()


# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)
y = df['DEFAULT']

# to concat all models
df_models = pd.concat([baseline_report_cv_smote(gnb, X, y, 5, 'GaussianNB'),
                       baseline_report_cv_smote(logit, X, y, 5, 'LogisticRegression'),
                       baseline_report_cv_smote(knn, X, y, 5, 'KNN'),
                       baseline_report_cv_smote(decisiontree, X, y, 5, 'DecisionTree'),
                       baseline_report_cv_smote(randomforest, X, y, 5, 'RandomForest'),
                       baseline_report_cv_smote(linearsvc, X, y, 5, 'LinearSVC')
                       ], axis=0).reset_index()

df_models_no_scale_cv_smote = df_models.drop('index', axis=1)
df_models_no_scale_cv_smote


# #### GaussianNB with recall of 0.93 !!!

# ## Oversampling the Data

# In[ ]:


## Oversample Dataset model performance evaluation

def baseline_report_cv_oversampling(model, X, y, n_splits, name):
    """
    Accepts a model object, X (independent variables), y (target), n_splits and name of the model, oversamples the data
    and returns a model with various scoring metrics of each classifier model on a cross-validation split
    ----
    Input: model object, X, y, n_splits (integer), name (str)
    Output: Various metric scores of a model.
    """
    # Allows for oversampling if you forget to initialize it before running func
    from imblearn.over_sampling import RandomOverSampler
    
    # Splitting the data into 80% training/validation data and 20% testing data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    # Splitting the training data into 60% training data and 20% validation data.
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=69)
    
    
    #this helps with the way kf will generate indices below
    X_train_val, y_train_val = np.array(X_train_val), np.array(y_train_val)
    
    
    # Creating a shuffled kfold of 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1000) 
    
    clf_model_acc_scores_cv = []
    clf_model_precision_scores_cv = []
    clf_model_recall_scores_cv = []
    clf_model_f1_scores_cv = []
    clf_model_rocauc_scores_cv = []
    
    # Manual Cross-Validation
    for train_ind, val_ind in kf.split(X_train_val, y_train_val):

        # Assigning train and validation values for an individual fold
        X_train, y_train = X_train_val[train_ind], y_train_val[train_ind]
        X_val, y_val = X_train_val[val_ind], y_train_val[val_ind] 

        # Creating the OverSampled data
        X_resampled, y_resampled = RandomOverSampler(random_state=69).fit_sample(X_train, y_train)
        
        # Initializing model
        clf_model = model.fit(X_resampled, y_resampled) # Train model on SMOTE'd data
        y_pred = clf_model.predict(X_val)  # Y pred after testing on validation data split
        
        # Save scores of model
        clf_model_acc_score = accuracy_score(y_val, y_pred)
        clf_model_precision_score = precision_score(y_val, y_pred)
        clf_model_recall_score = recall_score(y_val, y_pred)
        clf_model_f1_score = f1_score(y_val, y_pred)   
        clf_model_rocauc_score = roc_auc_score(y_val, y_pred)
        
        # Append scores of model their scoring lists
        clf_model_acc_scores_cv.append(clf_model_acc_score)
        clf_model_precision_scores_cv.append(clf_model_precision_score)
        clf_model_recall_scores_cv.append(clf_model_recall_score)
        clf_model_f1_scores_cv.append(clf_model_f1_score)
        clf_model_rocauc_scores_cv.append(clf_model_rocauc_score)
        
   
    accuracy     = np.mean(clf_model_acc_scores_cv)
    precision    = np.mean(clf_model_precision_scores_cv)
    recall       = np.mean(clf_model_recall_scores_cv)
    f1score      = np.mean(clf_model_f1_scores_cv)
    rocauc       = np.mean(clf_model_rocauc_scores_cv)
    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'rocauc'       : [rocauc],
                             'timetaken'    : [0]       })   # timetaken for comparison later
    return df_model


# to evaluate baseline models
gnb = GaussianNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
linearsvc = LinearSVC()

# Scaling the inputs into model
# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)  
y = df['DEFAULT']

# to concat all models
df_models = pd.concat([baseline_report_cv_oversampling(gnb, X, y, 5, 'GaussianNB'),
                       baseline_report_cv_oversampling(logit, X, y, 5, 'LogisticRegression'),
                       baseline_report_cv_oversampling(knn, X, y, 5, 'KNN'),
                       baseline_report_cv_oversampling(decisiontree, X, y, 5, 'DecisionTree'),
                       baseline_report_cv_oversampling(randomforest, X, y, 5, 'RandomForest'),
                       baseline_report_cv_oversampling(linearsvc, X, y, 5, 'LinearSVC')
                       ], axis=0).reset_index()

df_models_no_scale_oversampled = df_models.drop('index', axis=1)
df_models_no_scale_oversampled


# #### Recall of 0.93 with f1 0.38

# ## Undersampling the data

# In[ ]:


## Undersample Dataset model performance evaluation


def baseline_report_cv_undersampling(model, X, y, n_splits, name):
    """
    Accepts a model object, X (independent variables), y (target), n_splits and name of the model, undersamples the data
    and returns a model with various scoring metrics of each classifier model on a cross-validation split
    ----
    Input: model object, X, y, n_splits (integer), name (str)
    Output: Various metric scores of a model.
    """
    # Allows for undersampling if you forget to initialize it before running func
    from imblearn.under_sampling import RandomUnderSampler
    
    # Splitting the data into 80% training/validation data and 20% testing data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    # Splitting the training data into 60% training data and 20% validation data.
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=69)
    
    
    #this helps with the way kf will generate indices below
    X_train_val, y_train_val = np.array(X_train_val), np.array(y_train_val)
    
    
    # Creating a shuffled kfold of 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1000) 
    
    clf_model_acc_scores_cv = []
    clf_model_precision_scores_cv = []
    clf_model_recall_scores_cv = []
    clf_model_f1_scores_cv = []
    clf_model_rocauc_scores_cv = []
    
    # Manual Cross-Validation
    for train_ind, val_ind in kf.split(X_train_val, y_train_val):

        # Assigning train and validation values for an individual fold
        X_train, y_train = X_train_val[train_ind], y_train_val[train_ind]
        X_val, y_val = X_train_val[val_ind], y_train_val[val_ind] 

        # Creating the UnderSampled data
        X_resampled, y_resampled = RandomUnderSampler(random_state=69).fit_sample(X_train, y_train)
        
        # Initializing model
        clf_model = model.fit(X_resampled, y_resampled) # Train model on SMOTE'd data
        y_pred = clf_model.predict(X_val)  # Y pred after testing on validation data split
        
        # Save scores of model
        clf_model_acc_score = accuracy_score(y_val, y_pred)
        clf_model_precision_score = precision_score(y_val, y_pred)
        clf_model_recall_score = recall_score(y_val, y_pred)
        clf_model_f1_score = f1_score(y_val, y_pred)   
        clf_model_rocauc_score = roc_auc_score(y_val, y_pred)
        
        # Append scores of model their scoring lists
        clf_model_acc_scores_cv.append(clf_model_acc_score)
        clf_model_precision_scores_cv.append(clf_model_precision_score)
        clf_model_recall_scores_cv.append(clf_model_recall_score)
        clf_model_f1_scores_cv.append(clf_model_f1_score)
        clf_model_rocauc_scores_cv.append(clf_model_rocauc_score)
        

    
    accuracy     = np.mean(clf_model_acc_scores_cv)
    precision    = np.mean(clf_model_precision_scores_cv)
    recall       = np.mean(clf_model_recall_scores_cv)
    f1score      = np.mean(clf_model_f1_scores_cv)
    rocauc       = np.mean(clf_model_rocauc_scores_cv)
    df_model = pd.DataFrame({'model'        : [name],
                             'accuracy'     : [accuracy],
                             'precision'    : [precision],
                             'recall'       : [recall],
                             'f1score'      : [f1score],
                             'rocauc'       : [rocauc],
                             'timetaken'    : [0]       })   # timetaken for comparison later
    return df_model


# to evaluate baseline models
gnb = GaussianNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
linearsvc = LinearSVC()

# Scaling the inputs into model
# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)  
y = df['DEFAULT']

# to concat all models
df_models = pd.concat([baseline_report_cv_undersampling(gnb, X, y, 5, 'GaussianNB'),
                       baseline_report_cv_undersampling(logit, X, y, 5, 'LogisticRegression'),
                       baseline_report_cv_undersampling(knn, X, y, 5, 'KNN'),
                       baseline_report_cv_undersampling(decisiontree, X, y, 5, 'DecisionTree'),
                       baseline_report_cv_undersampling(randomforest, X, y, 5, 'RandomForest'),
                       baseline_report_cv_undersampling(linearsvc, X, y, 5, 'LinearSVC')
                       ], axis=0).reset_index()

df_models_no_scale_undersample = df_models.drop('index', axis=1)
df_models_no_scale_undersample


# #### Recall of 0.934 f1 of 0.381

# ## SMOTE(SCALED)

# In[ ]:


## SMOTE Datset with Scaling for Model Performance Evaluation


# to evaluate baseline models
gnb = GaussianNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
linearsvc = LinearSVC()

# Scaling the inputs into model
# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)  
y = df['DEFAULT']

# Creating n_splits for the function since it already has kfold creation in them
n_splits = 5

## Scale data (just scale everything lol)
std = StandardScaler()
std.fit(X.values)

## Scale the Predictors
X = std.transform(X.values)

# to concat all models
df_models = pd.concat([baseline_report_cv_smote(gnb, X, y, 5, 'GaussianNB'),
                       baseline_report_cv_smote(logit, X, y, 5, 'LogisticRegression'),
                       baseline_report_cv_smote(knn, X, y, 5, 'KNN'),
                       baseline_report_cv_smote(decisiontree, X, y, 5, 'DecisionTree'),
                       baseline_report_cv_smote(randomforest, X, y, 5, 'RandomForest'),
                       baseline_report_cv_smote(linearsvc, X, y, 5, 'LinearSVC')
                       ], axis=0).reset_index()

df_models_scale_cv_smote = df_models.drop('index', axis=1)
df_models_scale_cv_smote


# #### Recall of 0.82 with improved f1 of 0.41

# ## Oversampled(Scaled)

# In[ ]:


## Oversampling & Scaled Dataset on Model Performance Evaluation

# to evaluate baseline models
gnb = GaussianNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
linearsvc = LinearSVC()

# Scaling the inputs into model
# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)  
y = df['DEFAULT']

# Creating n_splits for the function since it already has kfold creation in them
n_splits = 5

## Scale data (just scale everything lol)
std = StandardScaler()
std.fit(X.values)

## Scale the Predictors
X = std.transform(X.values)

# to concat all models
df_models = pd.concat([baseline_report_cv_oversampling(gnb, X, y, 5, 'GaussianNB'),
                       baseline_report_cv_oversampling(logit, X, y, 5, 'LogisticRegression'),
                       baseline_report_cv_oversampling(knn, X, y, 5, 'KNN'),
                       baseline_report_cv_oversampling(decisiontree, X, y, 5, 'DecisionTree'),
                       baseline_report_cv_oversampling(randomforest, X, y, 5, 'RandomForest'),
                       baseline_report_cv_oversampling(linearsvc, X, y, 5, 'LinearSVC')
                       ], axis=0).reset_index()

df_models_scale_oversampled = df_models.drop('index', axis=1)
df_models_scale_oversampled


# #### Recall of 0.78 with improved f1score of0.43!!
# 

# In[ ]:


## Undersampling & Scaled Dataset for model performance evaluation

# to evaluate baseline models
gnb = GaussianNB()
logit = LogisticRegression()
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
linearsvc = LinearSVC()

# Scaling the inputs into model
# Separate data into X and Y components
X = df.drop('DEFAULT',axis=1)  
y = df['DEFAULT']

# Creating n_splits for the function since it already has kfold creation in them
n_splits = 5

## Scale data (just scale everything lol)
std = StandardScaler()
std.fit(X.values)

## Scale the Predictors
X = std.transform(X.values)

# to concat all models
df_models = pd.concat([baseline_report_cv_undersampling(gnb, X, y, 5, 'GaussianNB'),
                       baseline_report_cv_undersampling(logit, X, y, 5, 'LogisticRegression'),
                       baseline_report_cv_undersampling(knn, X, y, 5, 'KNN'),
                       baseline_report_cv_undersampling(decisiontree, X, y, 5, 'DecisionTree'),
                       baseline_report_cv_undersampling(randomforest, X, y, 5, 'RandomForest'),
                       baseline_report_cv_undersampling(linearsvc, X, y, 5, 'LinearSVC')
                       ], axis=0).reset_index()

df_models_scale_undersample = df_models.drop('index', axis=1)
df_models_scale_undersample


# ## Recall of 0.74 with F1 score of 0.44 !!! Much better model

# ## Let's Check all the Models and Select the Best!

# In[ ]:


# All the scores of the models
df_models_no_scale


# ### Recall is good !!

# In[ ]:


df_models_scale


# ### Pretty bad model!

# In[ ]:


df_models_no_scale_cv_smote


# ### Good Model with Recall of 0.9323

# In[ ]:


df_models_no_scale_oversampled


# ### Recall is same 0.93 with further improvement in f1

# In[ ]:


df_models_no_scale_undersample


# ### Pretty Same 
# 

# In[ ]:


df_models_scale_cv_smote


# ### It is the Best model with Recall of 0.82 and F1 score of 0.41!!!

# In[ ]:



df_models_scale_oversampled


# ### Pretty optimized model

# In[ ]:


df_models_scale_undersample


# ### Best Model for harmonic mean i.e f1score with recall of 0.74
