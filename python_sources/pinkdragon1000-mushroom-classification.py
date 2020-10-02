#!/usr/bin/env python
# coding: utf-8

# ## Mushroom Classification-Is it poisonous or edible?
# 
# ## PinkDragon1000
# 
# #### Date: 8/13/18
# 
# ![](https://img.webmd.com/dtmcms/live/webmd/consumer_assets/site_images/articles/health_tools/all_about_mushrooms_slideshow/493ss_thinkstock_rf_poisonous_mushroom.jpg)
# ---
# 
# 

# #### Abstract
# 

# For this project I used the Mushroom Classification dataset from Kaggle.  It was put on Kaggle by the UCI Machine Learning group.  I first started with regular data preparation techniques and went on to exploring the data.  The data is completely categorical and there are 8124 rows and 23 columns.  To prepare the data for use in machine learning models, I had to map these categorical values to numerical values using the labelEncoder function.  Then I split the dataset into test (25%) and train (75%).  The two main classes of mushrooms are poisonous and not poisonous (edible).  The purpose of these models are to predict that.  I used three types of models using scikit learn which are naive bayes, svm (secure vector machines), and logistic regression. The best model was svm and it gave 100% accuracy, the next best model was logistic regression with around 94% accuracy, and finally naive bayes with 81% accuracy.         

# ### 1. Introduction

# I chose the Mushroom Classification dataset from Kaggle.  The dataset was put on Kaggle by UCI Machine Learning Repository which maintains around 351 datasets.  This dataset is one of those.  The sample set contains around 23 species of mushrooms from the gilled mushroom Agaricus and Lepiota Family.  The mushrooms are either identified as completely poisonous or completely edible.      

# #### Objectives 
# * Preprocessing and exploratory data analysis steps such as: loading the data into the data frame, checking the shape (number of rows/columns), getting the head of data, checking for missing and duplicate values, etc.  
# * Splitting the dataset into test and train.  
# * Finding the best sklearn model that accurately predicts whether a mushroom is poisonous or edible using naive bayes, support vector machines, and logistic regression

# ### 2. Problem Definition
# 

# Poisonous mushrooms can often times confuse people into thinking they are not posionous due to their similar appearance to some non-poisonous mushroom types. Even though most mushrooms seem to be edible, mushroom poisoning can cause discomfort and even in some cases death.  Poisonous mushrooms are found to mostly cause gastrointestinal problems and in the worst case may cause respiratory or kidney failure.  The symptoms appear within twenty minutes to four hours of ingestion. The goal is to find the best model to predict whether a mushroom is poisionous or edible (poisonous or not poisonous).

# ### 3. Data Sets

# The dataset can be found here: https://www.kaggle.com/uciml/mushroom-classification/home. The format of the data is in a CSV format and is in one single file called mushrooms.csv. To prepare the dataset for classification purposes it needs to be split for the test and train.

# ### 4. Preparation

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv('../input/mushrooms.csv')


# There are 8124 rows and 23 columns in this dataset.

# In[ ]:


df.shape


# Showing first 5 rows of the dataset.  This dataset is comprised of completely categorical features.

# In[ ]:


df.head()


# Looking at the columns in dataset:

# In[ ]:


df.columns


# There are no null values in this dataset.

# In[ ]:


df.isnull().sum()


# There are no duplicate values in this dataset.

# In[ ]:


df.duplicated().sum()


# ### 5. Exploration and Visualization
# 

# In[ ]:


df.nunique()


# Shows how many unique values there are for each column.

# In[ ]:


df['class'].unique()


# The two classes of mushrooms are p (poisonous) and e (edible).

# The Kaggle website provides information on what the column data means:
#     
# **class**: p=poisonous,e=edible  
# **cap-shape**: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s                   
# **cap-surface**: fibrous=f,grooves=g,scaly=y,smooth=s                 
# **cap-color**: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y                   
# **bruises**: bruises=t,no=f                      
# **odor**: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s                         
# **gill-attachment**: attached=a, descending=d, free=f, notched=n              
# **gill-spacing**: close=c,crowded=w,distant=d                
# **gill-size**: broad=b,narrow=n                 
# **gill-color**: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y               
# **stalk-shape**: enlarging=e,tapering=t                  
# **stalk-root**: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?                  
# **stalk-surface-above-ring**: fibrous=f,scaly=y,silky=k,smooth=s    
# **stalk-surface-below-ring**: fibrous=f,scaly=y,silky=k,smooth=s     
# **stalk-color-above-ring**: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y        
# **stalk-color-below-ring**: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y       
# **veil-type**: partial=p,universal=u                    
# **veil-color**: brown=n,orange=o,white=w,yellow=y                
# **ring-number**: none=n,one=o,two=t                 
# **ring-type**: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z                    
# **spore-print-color**: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y            
# **population**: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y                   
# **habitat**: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d 

# Let's look at the descriptive statistics for the data:

# In[ ]:


df.describe()


# Looking at the number of poisonous and edible mushrooms in this dataset.

# In[ ]:


df['class'].value_counts()


# There are more edible mushrooms than poisonous mushrooms 4,208 versus 3,916 in this dataset.

# Visualizing number of poisonous and edible mushrooms:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
items=pd.DataFrame(df['class'].value_counts())
items.plot(kind='bar', figsize=(4,6), width=0.3, color=[('#63d363', '#d36363')], legend=False)
plt.title("Number of Edible and Poisonous Mushrooms in this Dataset", fontsize="15")
plt.xlabel("Edible or Poisonous", fontsize="12")
plt.ylabel("Number of Mushrooms", fontsize="12")
plt.xticks(np.arange(2),("Edible", "Poisonous"), rotation=0)
plt.grid()   
plt.show()


# Looking at the cap-color distribution in this dataset.

# In[ ]:


df['cap-color'].value_counts()


# Visualizing number of each cap color:

# In[ ]:


caps=pd.DataFrame(df['cap-color'].value_counts())
caps.plot(kind='bar', figsize=(8,8), width=0.8, color=[('#bf7050', '#A9A9A9', '#d36363', '#f3f6c3', '#DCDCDC', '#bfa850', '#f9d7f7', '#D2691E', '#63d363', '#7050bf')], legend=False)
plt.xlabel("Cap Color",fontsize=12)
plt.ylabel('Number of Mushrooms',fontsize=12)
plt.title('Mushroom Cap Color Types in the Dataset', fontsize=15)
plt.xticks(np.arange(10),('Brown', 'Gray','Red','Yellow','White','Buff','Pink','Cinnamon', 'Green','Purple'))
plt.grid()       
plt.show() 


# Looking at the number of mushrooms there are for each cap shape in this dataset.

# In[ ]:


df['cap-shape'].value_counts()


# Visualizing number of each cap shape:

# In[ ]:


capsh=pd.DataFrame(df['cap-shape'].value_counts())
capsh.plot(kind='bar', figsize=(8,8), width=0.5, color=[('#A9A9A9')], legend=False)
plt.xlabel("Cap Shape",fontsize=12)
plt.ylabel('Number of Mushrooms',fontsize=12)
plt.title('Mushroom Cap Types in the Dataset', fontsize=15)
plt.xticks(np.arange(6),('Convex', 'Flat','Knobbed','Bell','Sunken','Conical'))
plt.grid()       
plt.show() 


# ### 6. Modeling and Evaluation
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# The sklearn naive bayes algorithm cannot directly operate on categorical features that are non-numeric so we will use sklearn's LabelEncoder to convert the categorical features to numeric values.

# In[ ]:


# Encodes labels from 0 to n_classes-1
labelEncoder = preprocessing.LabelEncoder()
for col in df.columns:
    df[col] = labelEncoder.fit_transform(df[col])


# Let's look at how the LabelEncoder transformed our data.

# In[ ]:


df.head()


# Now the categorical variables are shown numerically. 

# Seeing what labelEncoder did to the poisonous (p) and edible (e) labels.  

# In[ ]:


df['class'].value_counts()


# Edible is set to 0 and Poisonous is now set to 1.

# To prepare the model the dataset needs to be split into test and train.  The method train_test_split randomly splits the dataset into 75% train and 25% test.

# In[ ]:


# 75% train, 25% test
train, test = train_test_split(df, test_size = 0.25) 
y_train = train['class']
X_train = train[[x for x in train.columns if 'class' not in x]]
y_test = test['class']
X_test = test[[x for x in test.columns if 'class' not in x]]

from sklearn.feature_extraction.text import TfidfVectorizer
# Vectorize the training and test data 
vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')


# ##### Naive Bayes

# Scikit-Learn Implementation

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
# Creating a MultinomialNB classifier and fit the model
cl = MultinomialNB()
cl.fit(X_train, y_train)


# Now that we have trained our model. Let us predict our labels using the test portion of the data set.

# In[ ]:


y_pred=cl.predict(X_test)


# Now let's evaluate how well the model performs.

# In[ ]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Recall score: ", recall_score(y_test, y_pred, average = 'weighted'))
print("Precision score: ", precision_score(y_test, y_pred, average = 'weighted'))
print("F1 score: ", f1_score(y_test, y_pred, average = 'weighted'))


# Cross validation is a way to check for overfitting.

# In[ ]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'
results = model_selection.cross_val_score(cl, X_train, y_train, cv=kfold, scoring=scoring)
print("Cross validation average accuracy with 10-folds: %f" % (results.mean()))


# Given the cross validation average is close to the accuracy of the naive bayes model we can conclude that our model does not really overfit. 

# Making a confusion matrix:

# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cm, classes=['p','e'], title='Confusion matrix, without normalization')


# The matrix below is the same but is normalized.   

# In[ ]:


plt.figure()
plot_confusion_matrix(cm, classes=['p','e'], normalize=True, title='Confusion matrix, with normalization')


# ##### Support Vector Machines (SVM)

# Scikit-Learn Implementation

# In[ ]:


from sklearn import svm


# In[ ]:


clf = svm.SVC(gamma='auto')
clf.fit(X_train, y_train) 


# In[ ]:


y_pred=clf.predict(X_test)


# Now let's evaluate how well the model performs.

# In[ ]:


print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Recall score: ", recall_score(y_test, y_pred, average = 'weighted'))
print("Precision score: ", precision_score(y_test, y_pred, average = 'weighted'))
print("F1 score: ", f1_score(y_test, y_pred, average = 'weighted'))


# Cross validation is a way to check for overfitting.

# In[ ]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'
results = model_selection.cross_val_score(clf, X_train, y_train, cv=kfold, scoring=scoring)
print("Cross validation average accuracy with 10-folds: %.3f" % (results.mean()))


# Given the cross validation average is close to the accuracy of the SVM model we can conclude that our model generalizes well and is not overfitting.

# In[ ]:


from sklearn.metrics import confusion_matrix


# The two matrices below show that the SVM model performed with 100% accuracy.  There was nothing inaccurately predicted.

# In[ ]:


cm = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cm, classes=['p','e'], title='Confusion matrix, without normalization')


# Normalized matrix:

# In[ ]:


plt.figure()
plot_confusion_matrix(cm, classes=['p','e'], normalize=True, title='Confusion matrix, with normalization')


# ##### Logistic Regression

# Scikit-Learn Implementation

# In[ ]:


from sklearn import linear_model, datasets
logreg = linear_model.LogisticRegression(solver='lbfgs',max_iter=2000)


# In[ ]:


logreg.fit(X_train, y_train)


# Now that we have trained our model. Let us predict our labels using the test portion of the data set.

# In[ ]:


y_pred=logreg.predict(X_test)


# Now let's evaluate how well the logistic model performs.

# In[ ]:


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Recall score: ", recall_score(y_test, y_pred, average = 'weighted'))
print("Precision score: ", precision_score(y_test, y_pred, average = 'weighted'))
print("F1 score: ", f1_score(y_test, y_pred, average = 'weighted'))


# Cross validation:

# In[ ]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'
results = model_selection.cross_val_score(logreg, X_train, y_train, cv=kfold, scoring=scoring)
print("Cross validation average accuracy with 10-folds: %.3f" % (results.mean()))


# Since the cross validation score is close to the logistic regression model score that means it is not overfitting the data. 

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


plt.figure()
plot_confusion_matrix(cm, classes=['p','e'], title='Confusion matrix, without normalization')


# Same matrix with normalization:

# In[ ]:


plt.figure()
plot_confusion_matrix(cm, classes=['p','e'], normalize=True, title='Confusion matrix, with normalization')


# ### 7. Conclusion
# 

# Even though all models did well the best model out of naive bayes, SVM, and logistic regression is SVM.  SVM gave about 100% accuracy while logistic gave about 94% accuracy and naive bayes performed the worst which was around 81% accuracy.  The worst situation seems to be if it is predicted edible but it is actually poisonous.  The other situations are if it is predicted poisonous but is edible, poisonous and is poisonous, edible and is edible.     

# ### 8. References

# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# 
# https://www.namyco.org/mushroom_poisoning_syndromes.php
# 
# http://scikit-learn.org/stable/index.html
