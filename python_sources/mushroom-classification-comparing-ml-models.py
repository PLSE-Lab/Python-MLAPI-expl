#!/usr/bin/env python
# coding: utf-8

# # Poisonous vs. Edible Mushroom Classification 

# ### Table of contents
# 1. [Introduction](#introduction)
# 2. [Data Import/Cleaning](#import)
# 3. [Feature Selection](#feat_select)
#     1. [Data Exploration](#exp)
# 4. [PCA Visualiztion](#pca)
# 5. [Kmeans Visualization](#kmean)
# 6. [Linear Classifiers](#lin)
#     1. [Logistic Regression](#logreg)
#     2. [SVM](#svm)
# 7. [Bagging Classifiers](#bag)
#     1. [GradientBoost](#bag)
#     2. [XGBoost](#xgb)
# 8. [Random Forest Classifier](#trees)

# #### Synopsis <a name="introduction"></a>
# The following analysis will explore the kaggle Mushroom Classification dataset (https://www.kaggle.com/uciml/mushroom-classification). Several ML models will be explored for their ability to classify mushrooms as poisonous or edible based off of the provided data. 
# 
# Both unsupervised and supervised methods will be used, as well as regression and ensemble methods for a rounded look at how different models work with categorical datasets. 

# #### Mushrooms:
# Mushrooms come in all shapes, sizes, colours, and flavours--as the saying goes: every mushroom is edible at least <i>once</i>.
# 
# Mushroom identification is a multifaceted process, where several important features of the fruiting body are taken into account before determining edibility. In addition to physical factors, the time of season and where a mushroom fruits (dirt, grass, manure, on a tree, on a fallen log, etc.) are also important considerations when id'ing fungi. Id'ing should always be done by an experienced mushroom hunter with local knowledge. 
# 
# #### Quick Poisonous Mushroom Identifiers
# * Spore Print: Spores are collected by placing a mushroom cap facedown over a sheet of paper or mirror. Different species' spores will be specific shades/colours, for example genus <a href="https://en.wikipedia.org/wiki/Amanita"><i>Amanita</i></a> will spore print white--poison. 
# * Fruiting Body: Several dispersal mechanisms for spores have evolved; between gilled, porous, sac or puffball fungi poisonous species may all mimic edible look-a-likes. 
# * Bruising/Color: Certain species that bruise dark when handled can sometimes indicate poison or inedibility. 
# * Morphology: 
#    * The genus <a href="https://en.wikipedia.org/wiki/Amanita"><i>Amanita</i></a> carries some of the some deadliest mushrooms in the world. Destroying Angel, Death Cap and Fool's Mushroom are all fatal, however share characteristics of the Amanita class making them easily identifiable. While the cap colour may alter, typically White cap, gills, and spore print, along with a physical structure called the volva are telltale signs of  <a href="https://en.wikipedia.org/wiki/Amanita"><i>Amanita's</i></a>. 
#    <img src="https://atrium.lib.uoguelph.ca/xmlui/bitstream/handle/10214/6850/Amanita_virosa_Destroying_Angel_amanitin_and_phalloidin.jpg?sequence=1&isAllowed=y" alt="Identifying a Destroying angel" width="40%"></img>
#    <div align="center"><small>Source: University of Guelph</small></div>
#    * The genus <a href="https://en.wikipedia.org/wiki/Gyromitra_esculenta"><i>Gyromitra</i></a>, better known as the False Morel, is an example of a poisonous mushroom that looks like the famously delicious Morel. Inexperienced mushroom hunters could potentially mix this type of mushroom up with an edible counterpart and suffer the consequences. However, false morels have a full stem and are tellingly <i>not</i> hollow. 
#    <img src="https://cdn0.wideopenspaces.com/wp-content/uploads/2017/03/morel-mushroom-real-fake.jpg" alt="Two true morels on the left, false morel on the right" width="50%"></img>
#    <div align="center"><small><a href="https://www.wideopenspaces.com/learn-important-difference-real-false-morel-mushrooms/">Source: Wide Open Spaces</a></small></div>

# Let's first import packages and data. <a name="import"></a>

# In[ ]:


#import cleaning and visualization modules
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import rcParams, gridspec

#import analysis modules
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import linear_model
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from xgboost import plot_importance
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

#pandas configuration
import warnings
warnings.filterwarnings('ignore')


# Data is read into a dataframe and displayed. Next dataypes, counts and columns are shown and a check if any null values are present.

# In[ ]:


#read into dataframe, display first 5 values
df = pd.read_csv('../input/mushrooms.csv')
df.head()


# In[ ]:


#Look into df for datatypes, if nulls exist 
null_count = 0
for val in df.isnull().sum():
    null_count += val
print('There are {} null values.\n'.format(null_count))
df.info()


# We've seen that the data is categorical and there are no nulls in our set. Let's check how many classes are in each column. <a name="feat_select"></a>

# In[ ]:


def show_features(df):
    '''Takes a dataframe and outputs the columns, number of classes and category variables.'''
    col_count, col_var = [], []
    for col in df:
        col_count.append(len(df[col].unique()))
        col_var.append(df[col].unique().sum())
    df_dict = {'Count': col_count, 'Variables': col_var}
    df_table = pd.DataFrame(df_dict, index=df.columns)
    print(df_table)
    
show_features(df)


# Right away there are some interesting things to note: Veil-type has only one class and can be removed from our set, several classes are binary and can be reduced to a single feature, and there is a '?' class in the stalk-root class. One-hot encoding will transform our data into a usable format for our models, remove excess feature columns and prepare our independent variable for supervised learning. 
# 
# Next, let's see how many '?' values are present in the stalk-root category.

# In[ ]:


df['stalk-root'].value_counts()


# There are a few options--remove the class entirely but lose potential information in our dataset, delete any row with a '?' but lose data across all variables, or encode the data and treat it as an unknown variable. For the purposes of this study we'll keep the class and use encoding to transform '?' into a feature. 

# In[ ]:


df_dum = pd.get_dummies(df, drop_first=True)
df_dum.head()


# ### Mushroom Hunting & Important Features of Determining Edibility  <a name="exp"></a>
# 
# As discussed earlier, the art of mushroom foraging can be difficult at times when ID'ing unknown species. In the field, we'll collect a spore print, look at morphological features, mark time of year, use smell, test for bruising, note the conditions it was found in: healthy or rotted terrain, neighbouring trees and plant life, near other fungi--all important steps in correctly identifying whether a find is edible or poisonous. 
# 
# We can graph some of the data to determine if there are any features that are more associated with poisonous species at a glance before running our models and testing for important features.
# 

# In[ ]:


plt.figure(figsize=[16,12])

plt.subplot(231)
sns.countplot(x='odor', hue='class', data=df)
plt.title('Odor')
plt.xticks(np.arange(10),('Pungent', 'Almond', 'Anise', 'None', 'Foul', 'Creosote', 'Fish', 'Spicy', 'Musty'), rotation='vertical')
plt.ylabel('Count')

plt.subplot(232)
sns.countplot(x='spore-print-color', hue='class', data=df)
plt.title('Spore Print Color')
plt.xticks(np.arange(10),('Black', 'Brown','Purple','Chocolate','White','Green','Orange','Yellow','Brown'), rotation='vertical')
plt.legend(loc='upper right')

plt.subplot(233)
sns.countplot(x='cap-color', hue='class', data=df)
plt.title('Cap Color')
plt.xticks(np.arange(11),('Brown', 'Yellow','White','Gray','Red','Pink','Buff','Purple','Cinnamon','Green'), rotation='vertical')
plt.legend(loc='upper right')

plt.subplot(234)
sns.countplot(x='bruises', hue='class', data=df)
plt.title('Bruising')
plt.xticks(np.arange(2),('Bruise', 'No Bruise'), rotation='vertical')
plt.legend(loc='upper right')

plt.subplot(235)
sns.countplot(x='habitat', hue='class', data=df)
plt.title('Habitat')
plt.xticks(np.arange(8),('Urban', 'Grasses','Meadows','Woods','Paths','Waste','Leaves'), rotation='vertical')
plt.legend(loc='upper right')

plt.subplot(236)
sns.countplot(x='population', hue='class', data=df)
plt.title('Population')
plt.xticks(np.arange(7),('Scattered', 'Numerous','Abundant','Several','Solitary','Clustered'), rotation='vertical')
plt.legend(loc='upper right')

plt.tight_layout()
sns.despine()


# At first glance, odour, spore print color and bruising are fairly good features to predict edibility, while cap colour, habitat and population-type show a bit more variance between species. These graphs are interesting from a foraging standpoint and reinforce how it's important to use multiple traits to ID a mushroom. 
# 
# Next we'll run some classification models to try and predict whether a mushroom is poison or edible. The data is split into our X-dependent feature columns and y-independent label, then split again into a 70:30 train:test set. 

# In[ ]:


#set features
X = df_dum.drop('class_p', axis=1)
#set independent variable
y = df_dum['class_p']
#split the training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#print shapes of training/testing sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### PCA & KMeans Visualization <a name="pca"></a>

# In[ ]:


#visualize edible vs poison classes
pca = PCA(n_components=2)

x_pca = X.values
x_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=y, s=40, edgecolor='k')
plt.title('Visualizing Edible vs. Poison Classes')


# PCA is used to visualize the dataset by transforming our features into 2 dimensions. Right away we can see a clear separation of classes with some overlap in the left cluster. We can see if an unsupervised KMeans with K=2 clusters is able to classify our data with any accuracy. <a name="kmean"></a>

# ### KMeans

# In[ ]:


from sklearn import metrics
from sklearn.cluster import KMeans

#Specify the model and fit to training set
km = KMeans(n_clusters = 2)
km.fit(X_train)

#PCA X_test for visualization
pca_test = PCA(n_components = 2)
pca_test.fit(X_test)
X_test_pca = X_test.values
X_test_pca = pca_test.fit_transform(X_test)

#KMeans prediction
y_pred_km = km.predict(X_test)

#Plot the data
plt.figure(figsize=(8,6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_km, 
            s=40, edgecolor='k')
plt.title('KMeans: Test Data')
plt.show();


# Visually this looks OK, but not great. The model acheives 89% accuracy. KMeans is fast, and works by measuring the distance from the centroid of a cluster to classify points. Since there is some overlap in the left-most cluster, it groups everything to the same class.   

# In[ ]:


def plot_confusion_matrix(cm, classes, fontsize=15,
                          normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    cm_num = cm
    cm_per = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title.replace('_',' ').title()+'\n', size=fontsize)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=fontsize)
    plt.yticks(tick_marks, classes, size=fontsize)

    fmt = '.5f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Set color parameters
        color = "white" if cm[i, j] > thresh else "black"
        alignment = "center"

        # Plot perentage
        text = format(cm_per[i, j], '.5f')
        text = text + '%'
        plt.text(j, i,
            text,
            fontsize=fontsize,
            verticalalignment='baseline',
            horizontalalignment='center',
            color=color)
        # Plot numeric
        text = format(cm_num[i, j], 'd')
        text = '\n \n' + text
        plt.text(j, i,
            text,
            fontsize=fontsize,
            verticalalignment='center',
            horizontalalignment='center',
            color=color)
        
    plt.tight_layout()
    plt.ylabel('True label'.title(), size=fontsize)
    plt.xlabel('Predicted label'.title(), size=fontsize)

    return None


# In[ ]:


cm_km = metrics.confusion_matrix(y_test, y_pred_km)
plot_confusion_matrix(cm_km, classes=['Edible','Poison'])
print(f'KMeans accuracy: {str(accuracy_score(y_test, y_pred_km)*100)[:5]}%')


# Overall an unsupervised KMeans approach was interesting but by reducing our features into 2 dimensions there was a loss in accuracy. Next, we're going to try some regression models on our training set, starting with a Logistic Regression. Features can be auto-selected for and tuned using Recursive Feature Elimination and Cross-validation. <a name="lin"></a>

# ### Logistic Regression

# In[ ]:


#set the model and fit entire data to RFECV--train/test splits are done automatically and cross-validated.
lm = linear_model.LogisticRegression()
rfecv = RFECV(estimator=lm, step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)

print('Optimal number of features: %d' % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

#plot features vs. validation scores
plt.figure(figsize=(10,6))
plt.xlabel('Number of features selected')
plt.ylabel('Cross validation score')
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# Wow! 14/95 of the features give us the optimal amount for modelling. The rest is noise. This selection tests helps us to avoid overfitting and multicollinearity. Let's move forward with only those 14 features and see how our models perform. <a name="logreg"></a>

# In[ ]:


#set optimal features and assign new X, train/test split
opt_features = ['odor_c', 'odor_f', 'odor_l', 'odor_n', 'odor_p', 
                'gill-spacing_w', 'gill-size_n', 'stalk-surface-above-ring_k', 
                'ring-type_f', 'spore-print-color_k', 'spore-print-color_n', 
                'spore-print-color_r', 'spore-print-color_u', 'population_c']
#new dependent variables
X_opt = X[opt_features] 

#split the training and test data
Xo_train, Xo_test, yo_train, yo_test = train_test_split(X_opt, y, test_size=0.3)
#print shapes of training/testing sets
print(Xo_train.shape, Xo_test.shape, yo_train.shape, yo_test.shape)


# In[ ]:


#logistic regression
lm = linear_model.LogisticRegression()
lm.fit(Xo_train, yo_train)
log_probs = lm.predict_proba(Xo_test)
loss = log_loss(yo_test, log_probs)
print(f'Loss value: {loss}')
print(f'Training accuracy: {str(lm.score(Xo_train, yo_train)*100)[:5]}%')
print(f'Test accuracy: {str(lm.score(Xo_test, yo_test)*100)[:5]}%')


# In[ ]:


y_pred_lm = lm.predict(Xo_test)

cm_lm = metrics.confusion_matrix(yo_test, y_pred_lm)
plot_confusion_matrix(cm_lm, ['Edible','Poison'])
print(f'Logistic Regression accuracy: {str(accuracy_score(yo_test, y_pred_lm)*100)[:5]}%')


# As shown, a logistic regression performs very well on this smaller categorical dataset. With only one false negative and no false positives the model makes quick work of this problem. Now we have our features and benchmarks for performance, let's try an SVM (Support Vector Machine) with different kernals to see what our best fit is. <a name="svm"></a>

# ### SVM

# In[ ]:


#test out different SVMs using the different kernals
kerns = ['linear', 'rbf', 'sigmoid']
for i in kerns:
    #Kernel trick
    svm_kern = SVC(kernel=f'{i}')
    svm_kern.fit(Xo_train,yo_train)
    
    #Get the score
    print(f'{i} kernal SVM score: {str(100*svm_kern.score(Xo_test,yo_test))[:6]}%')


# In[ ]:


#fit SVM model to scaled data
svm = LinearSVC()
svm.fit(Xo_train, yo_train)
print(f'Linear SVM Training accuracy is: {svm.score(Xo_train, yo_train)*100}%')
print(f'Linear SVM Test accuracy is: {svm.score(Xo_test, yo_test)*100}%')


# In[ ]:


y_pred_svm = svm.predict(Xo_test)

cm_svm = metrics.confusion_matrix(yo_test, y_pred_svm)
plot_confusion_matrix(cm_svm, ['Edible','Poison'])
print(f'SVM accuracy: {str(accuracy_score(yo_test, y_pred_svm)*100)[:5]}%')


# Both of our linear models performed spectacularly on the mushroom dataset. Let's see how tree and bagging classifiers handle the data. <a name="bag"></a>
# 
# ### Gradient Boost and XGBoost Classifiers

# In[ ]:


#initialize gradientboost and xgboost
gb = GradientBoostingClassifier()
xgb = XGBClassifier()
#fit models
gb.fit(Xo_train,yo_train)
xgb.fit(Xo_train,yo_train)
#score models
print(f'Gradient Boost score: {(100 * gb.score(Xo_test,yo_test))}%')
print(f'XG Boost score: {(100 * xgb.score(Xo_test,yo_test))}%')


# It's apparent that our binary classification of mushroom toxicity is not a difficult problem for bagging and regression models to solve. We can see how XGBoost weighted the feature columns below. <a name="xgb"></a>

# In[ ]:


#plot feature importance XGBoost
plot_importance(xgb)
plt.show()


# XGBoost puts narrow-gills and odourless as it's top predictors of edibility--this makes sense after exploring our data and knowing how mushrooms are ID'd. Next we'll run the same analysis but with a Random Forest. <a name="trees"></a>
# 
# ### Random Forest Classifier

# In[ ]:


#fitting a random forest
rf = RandomForestClassifier()
rf.fit(Xo_train, yo_train)
print("Default RFR: %3.1f" % (rf.score(Xo_test, yo_test)*100))


# The default Random Forest achieves a 100% accuracy as well, but will assign different weights and importance to features. Next, we'll run a Grid Search with 10-fold cross-validation to observe optimal parameters--this will illustrate how to tune for hyperparameters and to avoid overfitting. 

# In[ ]:


param_grid = { 
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[ ]:


CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 10)
CV_rfc.fit(Xo_train, yo_train)


# In[ ]:


CV_rfc.best_params_


# In[ ]:


rfcv = RandomForestClassifier(criterion= 'gini',
 max_depth= 6,
 max_features= 'auto',
 n_estimators= 50)
rfcv.fit(Xo_train, yo_train)
print(f'GridSearchCV RFR: {(rfcv.score(Xo_test, yo_test)*100)}%')


# The optimized Random Forest classifier still achieves a 100% accuracy, which is to be expected. Let's visualize the difference between feature importance.

# In[ ]:


feature_imp = pd.Series(rfcv.feature_importances_,index=Xo_train.columns).sort_values(ascending=False)
print(feature_imp)


# In[ ]:


#plot feature importance for RFR
plt.figure(figsize=(12,8))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title('Random Forest Feature Importance');


# The features are slightly different than XGBoost, however the first two, odourless and narrow-gilled are still the strongest predictors. 
# 
# That sums up our look at Mushroom Classifications. Happy foraging!

# In[ ]:




