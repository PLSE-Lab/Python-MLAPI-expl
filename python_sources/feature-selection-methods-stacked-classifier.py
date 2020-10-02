#!/usr/bin/env python
# coding: utf-8

# Author: Jaketuricchi

# # Training a stacked Ensemble Model to recognise human activity

# The ability to continuous track activity through devices such as a smart phone largely inspires two jusxtaposed responses: one of ethical and
# moral uncertainty (should this data be collected? and who has access to it?) and one of recognition of the huge potential such data
# may have when combined with advanced machine learning techniques. Recent work in my PhD lab at The University of Leeds has been 
# concerned with the ability to classify activity and predict energy expenditure resulting in my interest in these kinds of data sets. 

# Already, it seems like other users are classifying activity with extraordinarily high accuracy (~95%) with very basic models. Next,<br>
# I aim to increase this by using a stacked ensemble model. Since model accuracy is the primary goal and EDA has already been done <br>
# extensively on Kaggle, I will skip EDA and move to

# # Set up<br>
# Import, set, read, initial exploration

# Import packages

# In[ ]:


import pandas as pd
import numpy as np
import os
import warnings 
import sklearn
import seaborn as sns
import matplotlib as plt
#%matplotlib inline
#%matplotlib qt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
warnings.simplefilter('ignore')
pd.set_option('display.max_rows', 1000)


# Read data

# In[ ]:


train_df = pd.read_csv('../input/human-activity-recognition-with-smartphones/train.csv')
test_df = pd.read_csv('../input/human-activity-recognition-with-smartphones/test.csv')


# # Is the data set ready for modelling?

# In[ ]:


print(train_df.isna().sum()) 


# We have no missing data in the df. For data of this kind this is unrealistic and implies some kind of previous preprocessing and<br>
# imputation.

# In[ ]:


print(train_df.dtypes)


# All data types are numeric, with categorical for the target, so all is in order here.

# # Modelling - Feature selection

# With >560 features we have a substantial data frame and it's likely that we can do without many of these. However, given that there<br>
# are a series of different processes that can be used to remove features, I will take multiple approaches and consider the agreement<br>
# between these before removal. I will use: 
# RF importance;<br>
# Generic Univariate Selection<br>
# and Model-based selection (i'll use a different algo from RF to differentiate between RF importance)

# Note that I use guidance provided in a very useful kernel from Aleksey Bilogur (https://www.kaggle.com/residentmario/automated-feature-selection-with-sklearn)

# To prepare, lets split the data into features and labels.

# In[ ]:


X = train_df.drop('Activity', axis=1).reset_index(drop=True)
y = train_df['Activity']


# # RF importance

# This involves fitting an RF model and extracting the top n(/%) of the features. It will also give us a good idea of what a basic model<br>
# can produce in terms of classification accuracy.

# In[ ]:


rf = RandomForestClassifier(random_state = 1)
rf_model_basic = rf.fit(X, y)
rf_importance = pd.Series(rf_model_basic.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_importance.nlargest(560).plot(kind='barh')


# Here we can see the way in which there is an exponential drop in importance of variables. At around 75% there is a slight drop where<br>
# importance becomes very limited. For this purpose we will remove 25% of the variables

# In[ ]:


rf_variables = rf_importance.head(420).index.values


# # Generic Univariate Feature Selection<br>
# Here we must define an arbitrary or % of data to remove. Lets go with 50% of variables for now and see how it compares later.

# In[ ]:


from sklearn.feature_selection import GenericUnivariateSelect


# In[ ]:


trans = GenericUnivariateSelect(score_func=lambda X, y: X.mean(axis=0), mode='percentile', param=75)
chars_X_trans = trans.fit_transform(X, y)


# Get columns to keep and create new dataframe with those only

# In[ ]:


cols = trans.get_support(indices=True)
GUS_variables = X.iloc[:,cols].columns.values


# In[ ]:


print("By Generic Univariate Selection we keep {1} of our original {0} features".format(X.shape[1], chars_X_trans.shape[1]))


# # Model Based Selection<br>
# Here we will try a couple of different types of classifiers. In order for the SelectFromModel fn to perform, the classifier algo either<br>
# (a) must have a built in feature importance, or we can conduct permutation feature importance <br>
# (see:https://scikit-learn.org/stable/modules/permutation_importance.html#:~:text=The%20permutation%20feature%20importance%20is,model%20depends%20on%20the%20feature. )<br>
# which will allow us to provide the feature importance from all clf models. Permutation is very computationally expensive so for the<br>
# present purposes I will use only those with a built in output.

# In[ ]:


from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance


# Here we can see that model selection<br>
# We can also do this similarly using an algo such as Decision Tree Classifier:

# In[ ]:


clf = DecisionTreeClassifier()
selection = SelectFromModel(clf, threshold='3*median', max_features=420)
vars_selected = selection.fit_transform(X, y)
cols = selection.get_support(indices=True)
DT_variables = X.iloc[:,cols].columns.values


# # Feature selection comparison<br>
# Lets begin by converting each of these to lists as it makes matching simpler.

# In[ ]:


rf_var_list=list(rf_variables)
gus_var_list=list(GUS_variables)
dt_var_list=list(DT_variables)


# So we have 3 lists of selected features generated by slightly different methods. How many are in common?

# In[ ]:


print(len(set(rf_var_list) & set(gus_var_list) & set (dt_var_list)))
print(len(set(gus_var_list) & set (dt_var_list)))
print(len(set(rf_var_list)  & set (dt_var_list)))
print(len(set(rf_var_list) & set(gus_var_list)))


# There are 284 in common of the ~420 selected. This is not driven down by a single method. The implication is that there is no <br>
# clear agreement on which are most important. One way to decide on the best is to test the selections in models.

# # Data splitting

# In[ ]:


X1 = X.filter(items=rf_variables)
X2 = X.filter(items=GUS_variables)
X3 = X.filter(items=DT_variables)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[ ]:


X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 42)
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state = 42)
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size = 0.2, random_state = 42)


# # Modelling - Algorithm selection.<br>
# We'll test a range of classification (clf) models to see which to use in our stacked model. We will avoid using any models too similar<br>
# (e.g. RF and decision trees). Lets start by listing what are (usually) the top performing clfs.

# In[ ]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf",probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier()]


# Now we'll run a loop over each basic model and store the results for comparison.

# In[ ]:


def fit_basic_clf (train_features, train_labels, test_features, test_labels):
    
    # Log results for performance vis
    log_cols=["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)
    for clf in classifiers:
        clf.fit(train_features, train_labels)
        name = clf.__class__.__name__
        
        print("="*30)
        print(name)
        
        print('****Results****')
        train_predictions = clf.predict(test_features)
        acc = accuracy_score(test_labels, train_predictions)
        
        # calculate score
        precision = precision_score(test_labels, train_predictions, average = 'macro') 
        recall = recall_score(test_labels, train_predictions, average = 'macro') 
        f_score = f1_score(test_labels, train_predictions, average = 'macro')
        
        
        print("Precision: {:.4%}".format(precision))
        print("Recall: {:.4%}".format(recall))
        print("F-score: {:.4%}".format(recall))
        print("Accuracy: {:.4%}".format(acc))
        
        train_predictions = clf.predict_proba(test_features)
        ll = log_loss(test_labels, train_predictions)
        print("Log Loss: {}".format(ll))
        
        log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
        log = log.append(log_entry)
        print("="*30)
        
    # Plot results
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

    sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

        
# First we'll test on the complete data to see how much accuracy we lose when we drop features
fit_basic_clf(X1_train, y_train, X1_test, y_test)   
 
# Now test on subsets
fit_basic_clf(X1_train, y_train, X1_test, y_test)    
fit_basic_clf(X2_train, y_train, X2_test, y_test)    
fit_basic_clf(X3_train, y_train, X3_test, y_test)    


# We are getting extremely well performing models with less features. If the purpose was to have a computationally fast model,<br>
# we could probably reduce the features by ~50%. Reducing by ~75% reduces accuracy of the best models by ~0.2-0.5%. However, since<br>
# I am aiming to get maximal accuracy here every fraction of a % matters. Therefore I will work with the full feature set.<br>
# It is an option to tune some parameters but it would likely require an extensive grid search to get a fraction of a percent extra<br>
# so we'll see how stacking goes.

# # Stacking<br>
# Next, I will pick some different models and stack. I'll go for KNN, RF, NN with XGb as they meta-classifier the 3 top performing models. <br>
# I'll use package mlx, the guidance for which is very useful and can be found at:<br>
# http://rasbt.github.io/mlxtend/user_guide/

# In[ ]:


from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection


# In[ ]:


RANDOM_SEED = 42


# In[ ]:


rf = RandomForestClassifier(n_estimators=500, random_state = 42)
knn=KNeighborsClassifier(3)
NN=MLPClassifier()
xgb= GradientBoostingClassifier()


# In[ ]:


stack = StackingCVClassifier(classifiers=[rf, knn, NN],
                            use_probas=True,
                            meta_classifier=xgb,
                            random_state=42)


# Here we use_probas =T -- this will use probabilities of the 3 classifiers as meta-features in the meta-classifier model

# In[ ]:


print('3-fold cross validation:\n')
for clf, label in zip([rf, knn, NN, stack],
                      ['Random Forest',
                       'KNearestNeighbours',
                       'NeuralNetwork', 
                       'StackingClassifer']):
    scores = model_selection.cross_val_score(clf, X_train, y_train, 
                                              cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
    


# Here we see that the stacked classifier has improved the accuracy to >99% meaning the stacked model has improved estimates.
# We can feed the stack grid searches for each classifier, however this would take a lot of time and we're already at >99%, so I'm happy with the final stacked model.

# # Final predictions

# In[ ]:


stack.fit(X,y) # Fit to train


# In[ ]:


X_test = test_df.drop('Activity', axis=1).reset_index(drop=True)
y_test = test_df['Activity']


# Predict test data

# In[ ]:


final_preds = stack.predict(X_test)
    
# calculate final accuracy
acc = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds, average = 'macro') 
recall = recall_score(y_test, final_preds, average = 'macro') 
f_score = f1_score(y_test, final_preds, average = 'macro')


# In[ ]:


print('The stacked performance on test data:')
print("Precision: {:.4%}".format(precision))
print("Recall: {:.4%}".format(recall))
print("F-score: {:.4%}".format(recall))
print("Accuracy: {:.4%}".format(acc))


# Good scores, but could be better with some additional tuning next time round!
