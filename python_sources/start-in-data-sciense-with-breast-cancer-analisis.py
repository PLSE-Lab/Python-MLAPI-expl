#!/usr/bin/env python
# coding: utf-8

# # 1. Loading libraries and reading data

# ## 1.1 Loading libraries
# <br></br>

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
import plotly
import itertools
import warnings
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
import graphviz
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# <br></br>
# ## 1.2 Reading the data
# <br></br>

# In[ ]:


data = pd.read_csv('../input/data.csv')


# <br></br>
# ## 1.3 What about missing values?
# <br></br>

# In[ ]:


data.info()


# <br></br>
# #### As we can see, all features are totally completed, exept 'Unnamed: 32'. Let drop features 'id' and 'Unnamed: 32' from our data set.
# <br></br>

# In[ ]:


list = ['id', 'Unnamed: 32']
data.drop(list, axis = 1, inplace = True)


# <br></br>
# # 2. Exploratory Data Analysis (EDA)

# ## 2.1 Data head and describe

# <br></br>
# #### Let's take a look on our data.
# <br></br>

# In[ ]:


data.head()


# In[ ]:


data.describe()


# <br></br>
# #### Some of our fetatures, like 'area_mean', 'area_worst' etc., have lage max values. Are they outliers? Possible yes. We can check thise by building distribution plots of the features.  For example boxplots, qqplots, histograms plots. Distribution plots can show as, is the distribution of the feature normal and have it outliers.  
# <br></br>

# ## 2.2 Features and target distribution 

# ### 2.2.1 Target distribution
# <br></br>

# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(data['diagnosis'],  palette = "husl")


# <br></br>
# ### 2.2.2 Features distribution
# <br></br>

# #### What is the best way to see the distribution of the feature and find outliers? As discussed earlier, boxplots, qqplots and histograms can help us. As for me, i prefer boxplots and qqplot. First we will look at the distributions of features by plotting qqplots. Next, boxplots will help us to confirm information about distribution and find outliers. Then we will build features boxplots for two classes (M = malignant, B = benign), and compare them.   
# <br></br>

# In[ ]:


fig = plt.figure(figsize = (20,15))
plt.subplot(221)
stats.probplot(data['radius_mean'], dist = 'norm', plot = plt)
plt.title('QQPlot for radius mean')
plt.subplot(222)
stats.probplot(data['texture_mean'], dist = 'norm', plot = plt)
plt.title('QQPlot for texture mean')
plt.subplot(223)
stats.probplot(data['perimeter_mean'], dist = 'norm', plot = plt)
plt.title('QQPlot for perimeter mean')
plt.subplot(224)
stats.probplot(data['area_mean'], dist = 'norm', plot = plt)
plt.title('QQPlot for area mean')
fig.suptitle('Features distribution', fontsize = 20)


# In[ ]:


# To see qqplots for rest features delete #!

#plt.figure(figsize=(15,8))
#stats.probplot(data['smoothness_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for smoothness mean')
#plt.show()
#stats.probplot(data['compactness_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for compactness mean')
#plt.show()
#stats.probplot(data['concavity_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concavity mean')
#plt.show()
#stats.probplot(data['concave points_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave points mean')
#plt.show()
#stats.probplot(data['fractal_dimension_mean'], dist = 'norm', plot = plt)
#plt.title('QQPlot for fractal dimension mean')
#plt.show()
#stats.probplot(data['radius_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for radius se')
#plt.show()
#stats.probplot(data['texture_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for texture se')
#plt.show()
#stats.probplot(data['perimeter_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for perimeter se')
#plt.show()
#stats.probplot(data['area_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave area se')
#plt.show()
#stats.probplot(data['smoothness_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for smoothness se')
#plt.show()
#stats.probplot(data['compactness_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for compactness se')
#plt.show()
#stats.probplot(data['concavity_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concavity se')
#plt.show()
#stats.probplot(data['concave points_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave points se')
#plt.show()
#stats.probplot(data['symmetry_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave symmetry se')
#plt.show()
#stats.probplot(data['fractal_dimension_se'], dist = 'norm', plot = plt)
#plt.title('QQPlot for fractal dimension se')
#plt.show()
#stats.probplot(data['radius_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for radius worst')
#plt.show()
#stats.probplot(data['texture_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for texture worst')
#plt.show()
#stats.probplot(data['perimeter_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for perimeter worst')
#plt.show()
#stats.probplot(data['area_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave area worst')
#plt.show()
#stats.probplot(data['smoothness_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for smoothness worst')
#plt.show()
#stats.probplot(data['compactness_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for compactness worst')
#plt.show()
#stats.probplot(data['concavity_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concavity worst')
#plt.show()
#stats.probplot(data['concave points_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave points worst')
#plt.show()
#stats.probplot(data['symmetry_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for concave symmetry worst')
#plt.show()
#stats.probplot(data['fractal_dimension_worst'], dist = 'norm', plot = plt)
#plt.title('QQPlot for fractal dimension worst')
#plt.show()


# <br></br>
# #### As we can see, the real distribution of each feature is close to it theoretical distribution. But some features have strong deviation from theoretical distribution of the feature outside [-2, 2]. Next we will build boxplots to compare two grops ( M = malignant, B = benign), and will find something intresting.
# <br></br>

# In[ ]:


fig = plt.figure(figsize = (20, 20))
plt.subplot(321)
sns.boxplot(x = data['diagnosis'], y = data['radius_mean'], palette = "husl")
plt.title('Radius mean')
plt.subplot(322)
sns.boxplot(x = data['diagnosis'], y = data['texture_mean'], palette = "husl")
plt.title('Texture mean')
plt.subplot(323)
sns.boxplot(x = data['diagnosis'], y = data['perimeter_mean'], palette = "husl")
plt.title('Perimeter mean')
plt.subplot(324)
sns.boxplot(x = data['diagnosis'], y = data['area_mean'], palette = "husl")
plt.title('Area mean')
plt.subplot(325)
sns.boxplot(x = data['diagnosis'], y = data['smoothness_mean'], palette = "husl")
plt.title('Smoothness mean')
plt.subplot(326)
sns.boxplot(x = data['diagnosis'], y = data['compactness_mean'], palette = "husl")
plt.title('Compactness mean')
fig.suptitle('Features boxplots to compare malignant and benign', fontsize = 20)


# <br></br>
# #### First five boxplots show as that there is strong difference between two classes (M = malignant, B = benign). Also we can see, that some of the features have outliers. But we can find some intresting features. Look at the boxplots of five features (fractal_dimension_mean, texture_se, smoothness_se, symmetry_se, fractal_dimension_se) shown below. The distributions of these features for malignant and benign are almost the same. Thise is new information for as. We can make a guess that these features have no influence on diagnosis. Let's confirme these statement using statistics.
# <br></br>

# In[ ]:


fig = plt.figure(figsize = (20, 20))
plt.subplot(321)
sns.boxplot(x = data['diagnosis'], y = data['fractal_dimension_mean'], palette = "husl")
plt.title('Fractal dimension mean')
plt.subplot(322)
sns.boxplot(x = data['diagnosis'], y = data['texture_se'], palette = "husl")
plt.title('Texture se')
plt.subplot(323)
sns.boxplot(x = data['diagnosis'], y = data['smoothness_se'], palette = "husl")
plt.title('Smoothness se')
plt.subplot(324)
sns.boxplot(x = data['diagnosis'], y = data['symmetry_se'], palette = "husl")
plt.title('Symmetry se')
plt.subplot(325)
sns.boxplot(x = data['diagnosis'], y = data['fractal_dimension_se'], palette = "husl")
plt.title('Fractal dimension se')
fig.suptitle('Features boxplots to compare malignant and benign', fontsize = 20)


# ## 2.3 Statistical analysis

# ### 2.3.1 Prepare dataset

# #### Let's split data set in two sepparate data sets for malignant and for benign and drop diagnosis from datasets.  
# <br></br>

# In[ ]:


df1 = data[data['diagnosis'] == 'M']
df2 = data[data['diagnosis'] == 'B']
df1.drop('diagnosis', axis = 1, inplace = True)
df2.drop('diagnosis', axis = 1, inplace = True)


# <br></br>
# ### 2.3.2 Statistical analysis

# #### To compare two classes we will use Student's t-test. For our convenience I'll show features with p-value > 0.05.

# In[ ]:


feature = []
t_value = []
p_value = []
for column in df1.columns:
    ttest = stats.ttest_ind(df1[column], df2[column])
    feature.append(column)
    t_value.append(ttest[0])
    p_value.append(ttest[1])
ttest_data = {'feature' : feature, 't_value' : t_value, 'p_value' : p_value}
ttest_df = pd.DataFrame(ttest_data)
ttest_df.loc[ttest_df['p_value'] > 0.05]


# <br></br>
# #### Five features have p-value > 0.05. It means that these features have no influence on diagnosis. To confirm thise statement we can build confidence intervals. Let's do thise.

# In[ ]:


fig = plt.figure(figsize = (20,20))
plt.subplot(321)
sns.pointplot(y = data['diagnosis'], x = data['fractal_dimension_mean'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for fractal dimencion mean')
plt.subplot(322)
sns.pointplot(y = data['diagnosis'], x = data['texture_se'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for texture se')
plt.subplot(323)
sns.pointplot(y = data['diagnosis'], x = data['smoothness_se'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for smoothness se')
plt.subplot(324)
sns.pointplot(y = data['diagnosis'], x = data['symmetry_se'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for symmetry se')
plt.subplot(325)
sns.pointplot(y = data['diagnosis'], x = data['fractal_dimension_se'], join= False, capsize= 0.1, palette= 'husl')
plt.title('Confidence interval for fractal dimension se')
fig.suptitle('Confidence intervals', fontsize = 20)


# <br></br>
# #### As we can see feature mean of each class is within confidence interval of opposite class. Thats confirm our statistical conclusion. We can drop thise features from our data.
# <br></br>
# 

# In[ ]:


list = ['fractal_dimension_mean', 'texture_se', 'smoothness_se', 'symmetry_se', 'fractal_dimension_se']
data.drop(list, axis = 1, inplace = True)


# <br></br>
# ## 2.4 Correlation

# <br></br>
# #### Let's check the correlation between features 
# <br></br>

# In[ ]:


plt.figure(figsize=(25,20))
plt.title('Correlation matrix')
sns.heatmap(data.corr(), cmap = "Blues_r", annot = True)


# #### Will find out how look like relationship between features with correlation > 0.9

# In[ ]:


fig = plt.figure(figsize = (20,15))
plt.subplot(231)
sns.scatterplot(x = data['perimeter_mean'], y = data['radius_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Perimeter mean vs radius mean')
plt.subplot(232)
sns.scatterplot(x = data['area_mean'], y = data['radius_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs radius mean')
plt.subplot(233)
sns.scatterplot(x = data['radius_mean'], y = data['radius_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Radius mean vs radius worst')
plt.subplot(234)
sns.scatterplot(x = data['area_mean'], y = data['perimeter_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs perimeter mean')
plt.subplot(235)
sns.scatterplot(x = data['area_se'], y = data['perimeter_se'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area se vs perimeter se')
plt.subplot(236)
sns.scatterplot(x = data['perimeter_mean'], y = data['radius_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Perimeter_mean vs radious worst')
fig.suptitle('Correlation > 0.9', fontsize = 20)


# #### Will find out how look like relationship between features with  0.5 < correlation < 0.9

# In[ ]:


fig = plt.figure(figsize = (20,15))
plt.subplot(231)
sns.scatterplot(x = data['concavity_mean'], y = data['concavity_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Concavity mean vs concavity worst (correlation = 0.88)')
plt.subplot(232)
sns.scatterplot(x = data['concavity_mean'], y = data['concave points_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Concavity mean vs concave points worst (correlation = 0.86)')
plt.subplot(233)
sns.scatterplot(x = data['area_mean'], y = data['concave points_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs concave points mean (correlation = 0.82)')
plt.subplot(234)
sns.scatterplot(x = data['area_mean'], y = data['radius_se'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs radius se (correaltion = 0.73)')
plt.subplot(235)
sns.scatterplot(x = data['compactness_mean'], y = data['symmetry_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Compactness mean vs symmetry mean (correlation = 0.6)')
plt.subplot(236)
sns.scatterplot(x = data['area_mean'], y = data['compactness_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs compactness mean (correlation = 0.5)')
fig.suptitle('0.5 < correlation < 0.9', fontsize = 20)


# #### Correlation < 0.5

# In[ ]:


fig = plt.figure(figsize = (20,15))
plt.subplot(221)
sns.scatterplot(x = data['area_mean'], y = data['texture_mean'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs texture mean (correlation = 0.32)')
plt.subplot(222)
sns.scatterplot(x = data['area_mean'], y = data['compactness_se'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Area mean vs compactness se (correlation = 0.21)')
plt.subplot(223)
sns.scatterplot(x = data['concavity_se'], y = data['texture_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Concavity se vs texture worst (correlation = 0.1)')
plt.subplot(224)
sns.scatterplot(x = data['radius_mean'], y = data['fractal_dimension_worst'], hue = "diagnosis", data = data, palette = "husl")
plt.title('Radius mean vs fractal dimension worst (correaltion = 0.0071)')
fig.suptitle('correlation < 0.5', fontsize = 20)


# In[ ]:


y = data['diagnosis'].map({'M' : 1, 'B' : 0})


# In[ ]:


drop_list = ['diagnosis', 'radius_mean', 'perimeter_mean', 'concavity_mean', 'radius_se', 'perimeter_se', 'radius_worst', 'perimeter_worst', 
             'compactness_mean', 'concave points_mean', 'area_se', 'area_worst', 'smoothness_worst', 'compactness_worst', 'compactness_se', 
             'concavity_worst', 'concavity_se', 'fractal_dimension_worst', 'smoothness_mean']


# In[ ]:


X = data.drop(drop_list, axis = 1)


# In[ ]:


y.shape, X.shape


# # 3. Predictive Modeling
# 

# ## 3.1 Function for visualization and train test splitting

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
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


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 17)


# In[ ]:


acc_score = []


# <br></br>
# ## 3.2 Decision Tree

# In[ ]:


f_tree = DecisionTreeClassifier(random_state = 17)


# In[ ]:


tree_params = {'max_depth' : np.arange(1, 11), 'max_features' : np.arange(1, 8)}


# In[ ]:


tree_grid = GridSearchCV(f_tree, tree_params, cv = 20, n_jobs = -1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tree_grid.fit(X_train, Y_train)')


# In[ ]:


tree_grid.grid_scores_


# In[ ]:


tree_grid.best_estimator_


# In[ ]:


dot_data = tree.export_graphviz(tree_grid.best_estimator_, out_file = None, feature_names = X_test.columns, class_names= ['B', 'M'], filled = True, leaves_parallel = True)
graph = graphviz.Source(dot_data)
graph


# In[ ]:


Y_predict = tree_grid.best_estimator_.predict(X_test)


# In[ ]:


acc_score.append(accuracy_score(Y_test, Y_predict))
accuracy_score(Y_test, Y_predict)


# In[ ]:


cm1 = confusion_matrix(Y_test, Y_predict)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm1, classes = classes_name,  normalize= False, title = 'Confusion matrix for Decision Tree Classifier' )


# <br></br>
# ## 3.3 Logistic regression CV = 20

# In[ ]:


log_reg_cv = LogisticRegressionCV(n_jobs= -1, random_state= 17, cv = 20, solver= 'lbfgs' )


# In[ ]:


get_ipython().run_cell_magic('time', '', 'log_reg_cv.fit(X_train, Y_train)')


# In[ ]:


Y_predict = log_reg_cv.predict(X_test)


# In[ ]:


acc_score.append(accuracy_score(Y_test, Y_predict))
accuracy_score(Y_test, Y_predict)


# In[ ]:


cm2 = confusion_matrix(Y_test, Y_predict)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm2, classes = classes_name,  normalize= False, title = 'Confusion matrix for Logistic Regression CV Classifier')


# <br></br>
# ## 3.4 Logistic regression 

# In[ ]:


log_reg = LogisticRegression()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'log_reg.fit(X_train, Y_train)')


# In[ ]:


Y_predict = log_reg.predict(X_test)


# In[ ]:


acc_score.append(accuracy_score(Y_test, Y_predict))
accuracy_score(Y_test, Y_predict)


# In[ ]:


cm3 = confusion_matrix(Y_test, Y_predict)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm3, classes = classes_name,  normalize= False, title = 'Confusion matrix for Logidtic Regression Classifier')


# <br></br>
# ## 3.5 KNN (weight points by the inverse of their distance)

# #### In this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

# In[ ]:


knn = KNeighborsClassifier(algorithm='ball_tree', weights = 'distance')
knn_params = {'n_neighbors' : np.arange(1, 20)}


# In[ ]:


grid = GridSearchCV(knn, knn_params, cv = 20, n_jobs = -1)


# In[ ]:


grid.fit(X_train, Y_train)


# In[ ]:


grid.grid_scores_


# In[ ]:


grid.best_estimator_


# In[ ]:


Y_predict_knn = grid.best_estimator_.predict(X_test)


# In[ ]:


acc_score.append(accuracy_score(Y_test,Y_predict_knn))
accuracy_score(Y_test, Y_predict_knn)


# In[ ]:


cm4 = confusion_matrix(Y_test, Y_predict_knn)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm4, classes = classes_name,  normalize= False, title = 'Confusion matrix for KNN (weight points by the inverse of their distance)')


# <br></br>
# ## 3.6 KNN

# In[ ]:


knn = KNeighborsClassifier(algorithm='ball_tree')
knn_params = {'n_neighbors' : np.arange(1, 20)}


# In[ ]:


grid = GridSearchCV(knn, knn_params, cv = 20, n_jobs = -1)


# In[ ]:


grid.fit(X_train, Y_train)


# In[ ]:


grid.best_estimator_


# In[ ]:


grid.grid_scores_


# In[ ]:


Y_predict_knn = grid.best_estimator_.predict(X_test)


# In[ ]:


acc_score.append(accuracy_score(Y_test, Y_predict_knn))
accuracy_score(Y_test, Y_predict_knn)


# In[ ]:


cm5 = confusion_matrix(Y_test, Y_predict_knn)
classes_name = ['B', 'M']
plt.figure(figsize = (10, 10))
plot_confusion_matrix(cm5, classes = classes_name,  normalize= False, title = 'Confusion matrix for KNN')


# <br></br>
# ## 4. Conclusion

# ### 4.1 Results

# In[ ]:


fig = plt.figure(figsize = (30,20))
plt.subplot(321)
classes_name = ['B', 'M']
plot_confusion_matrix(cm1, classes = classes_name,  normalize= False, title = 'Confusion matrix for Decision Tree Classifier' )
plt.subplot(322)
classes_name = ['B', 'M']
plot_confusion_matrix(cm2, classes = classes_name,  normalize= False, title = 'Confusion matrix for Logistic Regression CV Classifier')
plt.subplot(323)
classes_name = ['B', 'M']
plot_confusion_matrix(cm3, classes = classes_name,  normalize= False, title = 'Confusion matrix for Logidtic Regression Classifier')
plt.subplot(324)
classes_name = ['B', 'M']
plot_confusion_matrix(cm4, classes = classes_name,  normalize= False, title = 'Confusion matrix for KNN (weight points by the inverse of their distance)')
plt.subplot(325)
classes_name = ['B', 'M']
plot_confusion_matrix(cm5, classes = classes_name,  normalize= False, title = 'Confusion matrix for KNN')


# In[ ]:


model_data = {'model' : ['Decision tree', 'Logic Regression CV', 'Logic Regression', 'KNN (distance)', 'KNN'], 'accuracy' : acc_score}
model_df = pd.DataFrame(model_data)
model_df


# ## 5. P.S.

# ### Thise is my start in data science. I am not a native speaker. If there are mistakes, please tell me in the comments below. Thanks for reading :) 
