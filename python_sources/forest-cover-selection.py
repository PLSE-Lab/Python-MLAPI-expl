#!/usr/bin/env python
# coding: utf-8

# # Forest Cover Type

# In this notebook, I will make use of the Forest Cover dataset to try to predict the different Cover Types given the previded features. I will successivelly try differerent feature elimination techniques to see how this can affect training times and overall model accuracy. <br>
# 
# Reducing the number of features in a dataset, can lead to:
# 
# - Accuracy improvements
# - Overfitting risk reduction
# - Speed up in training
# - Improved Data Visualization

# ## Preprocessing

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn import preprocessing
import time
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/forest-cover-type-dataset/covtype.csv')
pd.options.display.max_columns = None
df.head()


# In[ ]:


percent_missing = df.isnull().sum() * 100 / len(df)
missing_values = pd.DataFrame({'percent_missing': percent_missing})
missing_values.sort_values(by ='percent_missing' , ascending=False)


# In[ ]:


sns.set(style="ticks")
f = sns.countplot(x="Cover_Type", data=df, palette="bwr")
plt.show()


# In[ ]:


df['Cover_Type'].value_counts()


# In[ ]:


X = df.drop(['Cover_Type'], axis = 1)
Y = df['Cover_Type']


# In[ ]:


X.shape


# In[ ]:


figure(num=None, figsize=(15, 20), dpi=80, facecolor='w', edgecolor='k')

corr= X.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


# ## Machine Learning

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X2 = StandardScaler().fit_transform(X)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X2, Y, test_size = 0.30, random_state = 101)


# In[ ]:


start = time.process_time()
trainedsvm = svm.LinearSVC().fit(X_Train, Y_Train)
print(time.process_time() - start)
predictionsvm = trainedsvm.predict(X_Test)
print(confusion_matrix(Y_Test,predictionsvm))
print(classification_report(Y_Test,predictionsvm))


# In[ ]:


start = time.process_time()
trainedtree = tree.DecisionTreeClassifier().fit(X_Train, Y_Train)
print(time.process_time() - start)
predictionstree = trainedtree.predict(X_Test)
print(confusion_matrix(Y_Test,predictionstree))
print(classification_report(Y_Test,predictionstree))


# In[ ]:


import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = export_graphviz(trainedtree,out_file=None,feature_names= X.columns,
                       class_names=['1', '2', '3', '4', '5', '6', '7'],  
                       filled=True, rounded=True,  
                       max_depth=2,
                       special_characters=True)
graph = graphviz.Source(data)
graph


# In[ ]:


start = time.process_time()
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)
print(time.process_time() - start)
predictionforest = trainedforest.predict(X_Test)
print(confusion_matrix(Y_Test,predictionforest))
print(classification_report(Y_Test,predictionforest))


# ### Feature Selection

# There are many different methods which can be applied for Feature Selection. Some of the most important ones are:
# 
# - Filter Method = filtering our dataset and taking only a subset of it containg all the relevant features (eg. correlation matrix using Pearson Correlation)
# - Wrapper Method = follows the same objective of the FIlter Method but uses a Machine Learning model as it's evaluation criteria (eg. Forward/Backward/Bidirectional/Recursive Feature Elimination). We feed some features to our Machine Learning model, evaluate their performance and then decide if add or remove feature to increase accuracy. As a result, this mothod can be more accurate than filtering, is more computationally expensive.
# - Embedded Method = like the FIlter Method also the Embedded Method makes use of a Machine Learning model. The difference between the two different methods is that the Embedded Method examines the different training iterations of our ML model and then ranks the importance of each feature based on how much each of the features contributed to the ML model training (eg. LASSO Regularization).

# #### Feature Importance

# Decision Trees models which are based on ensembles (eg. Extra Trees and Random Forest) can be used to rank the importaqnce of the different features. Knowing which features our model is giving most importance can be of vital importance to understand how our model is making it's predictions (therefore making it more explainable). At the same time, we can get rid of the features which do not bring any benefit to our model (our confuse it to make a wrong decision!).

# In[ ]:


figure(num=None, figsize=(16, 18), dpi=80, facecolor='w', edgecolor='k')

feat_importances = pd.Series(trainedforest.feature_importances_, index= X.columns)
feat_importances.nlargest(20).plot(kind='barh')


# In[ ]:


X_Reduced = df[['Elevation','Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points','Horizontal_Distance_To_Hydrology']]
X_Reduced = StandardScaler().fit_transform(X_Reduced)
X_Train2, X_Test2, Y_Train2, Y_Test2 = train_test_split(X_Reduced, Y, test_size = 0.30, random_state = 101)


# In[ ]:


start = time.process_time()
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train2,Y_Train2)
print(time.process_time() - start)
predictionforest = trainedforest.predict(X_Test2)
print(confusion_matrix(Y_Test2,predictionforest))
print(classification_report(Y_Test2,predictionforest))


# #### SelectFromModel: Meta-transformer for selecting features based on importance weights.

# SelectFromModel is another Scikit-learn method which can be used for Feature Selection. This method can be used with all the different types of Scikit-learn models (after fitting) which have a coef_ or featureimportances attribute. Compared to RFE, SelectFromModel is a less robust solution. In fact, SelectFromModel just removes less important features based on a calculated threshold (no optimization iteration process involved).
# 
# In order to test SelectFromModel efficacy, I decided to use an ExtraTreesClassifier in this example. ExtraTreesClassifier (Extremely Randomized Trees) is tree based ensamble classifier which can yield less variance compared to Random Forest methods (reducing therefore the risk of overfitting). The main difference between Random Forest and Extremely Randomized Trees is that in Extremely Randomized Trees nodes are sampled without replacement.

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

model = ExtraTreesClassifier()
start = time.process_time()
model = model.fit(X_Train,Y_Train)
model = SelectFromModel(model, prefit=True)
print(time.process_time() - start)
Selected_X = model.transform(X_Train)
Selected_X.shape


# In[ ]:


start = time.process_time()
trainedforest = RandomForestClassifier(n_estimators=700).fit(Selected_X, Y_Train)
print(time.process_time() - start)
Selected_X_Test = model.transform(X_Test)
predictionforest = trainedforest.predict(Selected_X_Test)
print(confusion_matrix(Y_Test,predictionforest))
print(classification_report(Y_Test,predictionforest))


# In[ ]:


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
importances = trainedforest.feature_importances_
std = np.std([tree.feature_importances_ for tree in trainedforest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(Selected_X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(Selected_X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(Selected_X.shape[1]), indices)
plt.xlim([-1, Selected_X.shape[1]])
plt.show()


# #### Correlation Matrix Analysis

# Using Seaborn, we can now plot the Pearson correlation heatmap of our dataset. Inspecting this plot, we can then be able to see the correlation of our independent variables (X) with our label (Y). Finally, we can then select just the features which are most correlated with Y and train/test an SVM model to test the results of this approach.
# 
# Using Pearson correlation our returned coefficient values will vary between -1 and 1:
# 
# - If the correlation between two features is 0 this means that changing any of these two features will not affect the other.
# - If the correlation between two features is greater than 0 this means that increating the values in one feature will make increase also the values in the other feature (the closer the correlation coefficient is to 1 and the stronger is going to be this bond between the two different features).
# - If the correlation between two features is less than 0 this means that increating the values in one feature will make decrease the values in the other feature (the closer the correlation coefficient is to -1 and the stronger is going to be this relationship between the two different features).
# 
# Another possible aspect to control in this analysis would be to check if the selected variables are highly correlated each other. If they are, we would then need to keep just one of the correlated ones and drop the others.

# In[ ]:


Numeric_df = pd.DataFrame(X)
Numeric_df['Y'] = Y
Numeric_df.head()


# In[ ]:


figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

corr= Numeric_df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

# Selecting only correlated features
corr_y = abs(corr["Y"])
highest_corr = corr_y[corr_y >0.25]
highest_corr.sort_values(ascending=True)


# In[ ]:


X_Reduced2 = X[['Elevation','Wilderness_Area4']]
X_Reduced2 = StandardScaler().fit_transform(X_Reduced2)
X_Train3, X_Test3, Y_Train3, Y_Test3 = train_test_split(X_Reduced2, Y, test_size = 0.30, random_state = 101)


# In[ ]:


start = time.process_time()
trainedsvm = svm.LinearSVC().fit(X_Train3, Y_Train3)
print(time.process_time() - start)
predictionsvm = trainedsvm.predict(X_Test3)
print(confusion_matrix(Y_Test3,predictionsvm))
print(classification_report(Y_Test3,predictionsvm))


# #### Univariate Feature Selection

# Univariate Feature Selection is a statistical method used to select the features which have the strongest relationship with our corrispondent labels. Using the SelectKBest method we can decide which metrics to use to evaluate our features and the number of K best features we want to keep. Different types of scoring functions are available depending on our needs:
# 
# - Classification: chi2, f_classif, mutual_info_classif
# - Regression: f_regression, mutual_info_regression
# 
# In this example, we will be using chi2. Chi-squared (Chi2) can take as input just non-negative values, therefore, first of all we scale our input data in a range between 0 and 1.

# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
Scaled_X = min_max_scaler.fit_transform(X2)


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_new = SelectKBest(chi2, k=2).fit_transform(Scaled_X, Y)
X_Train3, X_Test3, Y_Train3, Y_Test3 = train_test_split(X_new, Y, test_size = 0.30, random_state = 101)
start = time.process_time()
trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train3,Y_Train3)
print(time.process_time() - start)
predictionforest = trainedforest.predict(X_Test3)
print(confusion_matrix(Y_Test3,predictionforest))
print(classification_report(Y_Test3,predictionforest))


# #### Lasso Regression

# When applying regularization to a Machine Learning model, we add a penalty to the model parameters so that to avoid that our model tries to resemble too closely our input data. In this way, we can make our model less complex and we can avoid overfitting (making learn to our model not just the key data characheteristics but also it's intrinsic noise).
# 
# One of the possible Regularization Methods is Lasso (L1) Regrssion. When using Lasso Regression, the coefficients of the inputs features gets shrinken if they are not positively contributing towards our Machine Learning model training. In this way, some of the features might get automatically discarded assigning them coefficients equal to zero.

# In[ ]:


from sklearn.linear_model import LassoCV

regr = LassoCV(cv=5, random_state=101)
regr.fit(X_Train,Y_Train)
print("LassoCV Best Alpha Scored: ", regr.alpha_)
print("LassoCV Model Accuracy: ", regr.score(X_Test, Y_Test))
model_coef = pd.Series(regr.coef_, index = list(X.columns[:-1]))
print("Variables Eliminated: ", str(sum(model_coef == 0)))
print("Variables Kept: ", str(sum(model_coef != 0))) 


# In[ ]:


figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

top_coef = model_coef.sort_values()
top_coef[top_coef != 0].plot(kind = "barh")
plt.title("Most Important Features Identified using Lasso (!0)")

