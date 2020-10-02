#!/usr/bin/env python
# coding: utf-8

# ## Handwritten digit recognizer using SVM

# A classifier to predict the label of a given digit using PCA and SVM
# 
# This kernel aims at building a classifier to predict the label of a handwritten digits using PCA and SVM
# 
# <b>Dataset</b>
# 
# The dataset consists of images of handwritten numeric digits between 0-9. Each image is of 28 x 28 pixels, i.e. 28 pixels along both length and breadth of the image. Each pixel is an attribute with a numeric value (representing the intensity of the pixel), and thus, there are 784 attributes in the dataset.
# 
# It can be found at the kaggle link
# https://www.kaggle.com/c/digit-recognizer/data
# 
# <b>Problem Statement</b>
# 
# The task is to build a classifier that predicts the label of an image (a digit between 0-9) given the features. Thus, this is a 10-class classification problem.
# Since this dataset has a large number of features (784), we will use PCA to reduce the dimensionality, and then, build a model on the low-dimensional data.
# 
# <b>Solution Overview:</b>
# 
# The following topics are covered in this tutorial
# 
# <b>Understanding and Exploring the Data</b>
#  
# <b>Dimensionality Reduction: </b>
# PCA will be used to reduce the dimensionality of the dataset.
# 
# <b>Model building using SVM on PCA transformed data</b>
# Linear, Poly and rbf kernel will be used to build the model and the comparisons will be done.
# 
# <b>Hyperparameter tuning and pipelining:</b>
# Final model will be built with the help of hyperparameter tuning by cross validation grid search and pipeline functionality.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# ### Reading and understanding Data

# In[ ]:


numbers = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
numbers.head()


# In[ ]:


numbers.shape


# In[ ]:


numbers.info()


# In[ ]:


numbers.describe(percentiles = [0.05,0.10,0.25,0.50,0.75,0.90,0.99])


# ### Data Cleaning and Data exploration

# In[ ]:


#missing value check
sum(numbers.isnull().sum(axis=0))


# #### No null values in the data. Therefore, we can proceed with the data exploration

# In[ ]:


numbers['label'].value_counts()


# In[ ]:


np.unique(numbers['label'])


# In[ ]:


sns.countplot(numbers['label'],palette = 'icefire')


# #### There is no data imbalance and the target values are equally distributed

# #### Next, lets see how the pixel values are distributed

# In[ ]:


#Checking average value of all pixels
#round(numbers.drop('label', axis=1).mean(), 2).sort_values(ascending = False)
y = pd.value_counts(numbers.values.ravel()).sort_index()


# In[ ]:


width = 0.9
plt.figure(figsize=[8,8])
plt.bar(range(len(y)),y,width,color="blue")
plt.title('Pixel Value Frequency (Log Scale)')
plt.yscale('log')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')


# In[ ]:


plt.figure(figsize=[15,15])
plt.subplot(2,3,1)
sns.distplot(numbers['pixel575'],kde=False)
plt.subplot(2,3,2)
sns.distplot(numbers['pixel624'],kde=False)
plt.subplot(2,3,3)
sns.distplot(numbers['pixel572'],kde=False)
plt.subplot(2,3,4)
sns.distplot(numbers['pixel407'],kde=False)
plt.subplot(2,3,5)
sns.distplot(numbers['pixel576'],kde=False)
plt.subplot(2,3,6)
sns.distplot(numbers['pixel580'],kde=False)
plt.show()


# #### Checking distribution of pixel values with repect to labels

# In[ ]:


plt.figure(figsize=[15,12])
plt.subplot(2,3,1)
sns.barplot(x='label', y='pixel575', data=numbers)
plt.subplot(2,3,2)
sns.barplot(x='label', y='pixel624', data=numbers)
plt.subplot(2,3,3)
sns.barplot(x='label', y='pixel572', data=numbers)
plt.subplot(2,3,4)
sns.barplot(x='label', y='pixel683', data=numbers)
plt.subplot(2,3,5)
sns.barplot(x='label', y='pixel576', data=numbers)
plt.subplot(2,3,6)
sns.barplot(x='label', y='pixel580', data=numbers)
plt.show()


# #### Inferences:
# - Label 6 have an average value of 255 for pixel575 and pixel572
# - Label 2 have an average value of 100 for pixel580
# 

# ### Lets visualize how a digit is written in different styles

# In[ ]:


numbers.loc[numbers['label']==1].head(10).index.values


# In[ ]:


plt.figure(figsize=[10,10])
ones_index = numbers.loc[numbers['label']==1].head(10).index.values
for i in range(0,10):
    one = numbers.iloc[ones_index[i], 1:]
    one = one.values.reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(one)
    plt.title("Digit 1")


# In[ ]:


plt.figure(figsize=[10,10])
threes_index = numbers.loc[numbers['label']==3].head(10).index.values
for i in range(0,10):
    one = numbers.iloc[threes_index[i], 1:]
    one = one.values.reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(one)
    plt.title("Digit 3")


# In[ ]:


plt.figure(figsize=[10,10])
fives_index = numbers.loc[numbers['label']==5].head(10).index.values
for i in range(0,10):
    one = numbers.iloc[fives_index[i], 1:]
    one = one.values.reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(one)
    plt.title("Digit 5")


# In[ ]:


plt.figure(figsize=[10,10])
fours_index = numbers.loc[numbers['label']==4].head(10).index.values
for i in range(0,10):
    one = numbers.iloc[fours_index[i], 1:]
    one = one.values.reshape(28,28)
    plt.subplot(2,5,i+1)
    plt.imshow(one)
    plt.title("Digit 4")


# In[ ]:


### Heatmap


# ## Data Preparation
# - Splitting data into train and test
# - Scaling of the data

# In[ ]:


y = numbers['label']
X = numbers.drop('label',axis=1)
X.head()


# In[ ]:


y[:5]


# In[ ]:


#test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2 ,test_size = 0.8, random_state=100)


# In[ ]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#X_train_scaled = pd.DataFrame(X_train_scaled)
#round(X_train_scaled.describe(),2)


# In[ ]:


print('X_train shape:',X_train_scaled.shape)
print('y_train shape:',y_train.shape)
print('X_test shape:',X_test_scaled.shape)
print('y_test shape:',y_test.shape)


# ## Dimensionality Reduction: 
# We see that number of pixels are a quite large. Let us first try to reduce the number of features with the help of PCA

# #### Appying PCA to reduce the number of  features

# In[ ]:


pca = PCA(random_state=42)


# In[ ]:


pca.fit(X_train_scaled)


# In[ ]:


pca.components_.shape


# In[ ]:


#pca.explained_variance_ratio_


# In[ ]:


var_cummu = np.cumsum(pca.explained_variance_ratio_)


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=[12,8])
plt.vlines(x=200, ymax=1, ymin=0, colors="r", linestyles="--")
plt.hlines(y=0.91, xmax=800, xmin=0, colors="g", linestyles="--")
plt.plot(var_cummu)
plt.ylabel("Cumulative variance explained")
plt.show()


# We see that 90% of variance is explained by 200 Principal Components.

# ### Performing PCA with 200 components

# In[ ]:


pca_final = IncrementalPCA(n_components = 200)


# In[ ]:


X_train_pca = pca_final.fit_transform(X_train_scaled)


# In[ ]:


X_train_pca.shape


# In[ ]:


pca_final.components_.shape


# In[ ]:


df_train_pca = pd.DataFrame(X_train_pca)
df_train_pca.head()


# In[ ]:


y_train_df = pd.DataFrame(y_train)
y_train_df['label'].value_counts()


# In[ ]:


new_df = pd.concat([df_train_pca,y_train_df],axis=1)
new_df['label'].value_counts().sort_index()


# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot(x=new_df[1],y=new_df[0],hue=new_df['label'],size=10,legend='full',palette='rainbow')


# In[ ]:


sns.pairplot(data=new_df, x_vars=[0,1,2], y_vars=[0,1,2], hue = "label", size=5)


# In[ ]:


pca_final.explained_variance_ratio_


# ### Redcucing the dimensions of the test data

# In[ ]:


X_test_pca = pca_final.transform(X_test_scaled)
X_test_pca.shape


# #### Till now we have converted our initial scaled data to pca transformed data
# 
#     - X_train --> X_train_scaled --> X_train_pca
#     - X_test  --> X_test_scaled  --> X_test_pca

# ### Model building using SVM on PCA transformed data
# 
# SVM uses following three kernel to buld a model.
# 
# <b>The linear kernel:</b> This gives the linear support vector classifier, or the hyperplane.
# 
# <b>The polynomial kernel:</b> It is capable of creating nonlinear, polynomial decision boundaries 
# 
# <b>The radial basis function (RBF) kernel:</b> This is the most complex one, which is capable of transforming highly nonlinear feature spaces to linear ones. It is even capable of creating elliptical (i.e. enclosed) decision boundaries
# 

# #### Let us first try linear model

# In[ ]:


# linear model
model_linear = SVC(kernel='linear')
model_linear.fit(X_train_pca, y_train)

# predict
y_train_pred = model_linear.predict(X_train_pca)
y_test_pred = model_linear.predict(X_test_pca)


# #### We just built our first SVM model using a `linear` kernel. Its time to evaluate our model using Accuracy as our evauation metric

# In[ ]:


train_accuracy = metrics.accuracy_score(y_train,y_train_pred)
print("Accuracy on training data: {}".format(train_accuracy))
test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
print("Accuracy on testing data: {}".format(test_accuracy))

print("\nClassification report on testing set \n")
print(metrics.classification_report(y_test, y_test_pred))

print("\nConfusion metrics on testing set \n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))


# #### The linear model gives approx. 90.6% testing accuracy. Whereas our training data shows an accuracy of 95%. 
# 
# #### Let's look at a non-linear model with randomly chosen hyperparameters.
# 

# Trying `poly` kernel

# In[ ]:


# non-linear model
# using poly kernel, C=1, default value of gamma

# model
non_linear_model_poly = SVC(kernel='poly')
non_linear_model_poly.fit(X_train_pca, y_train)

# predict
y_train_pred = non_linear_model_poly.predict(X_train_pca)
y_test_pred = non_linear_model_poly.predict(X_test_pca)


# #### Lets evaluate our model using Accuracy as our evauation metric

# In[ ]:


train_accuracy = metrics.accuracy_score(y_train,y_train_pred)
print("Accuracy on training data: {}".format(train_accuracy))
test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
print("Accuracy on testing data: {}".format(test_accuracy))

print("\nClassification report on testing set \n")
print(metrics.classification_report(y_test, y_test_pred))
print("\nConfusion metrics on testing set \n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))


# #### The accuracy increased to 95.66, so obviously no point in going with linear. Let us try 'rbf'.

# #### Trying `rbf` kernel

# In[ ]:


# non-linear model
# using rbf kernel, C=1, default value of gamma

# model
non_linear_model_poly = SVC(kernel='rbf')
non_linear_model_poly.fit(X_train_pca, y_train)

# predict
y_train_pred = non_linear_model_poly.predict(X_train_pca)
y_test_pred = non_linear_model_poly.predict(X_test_pca)


# In[ ]:


train_accuracy = metrics.accuracy_score(y_train,y_train_pred)
print("Accuracy on training data: {}".format(train_accuracy))
test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
print("Accuracy on testing data: {}".format(test_accuracy))

print("\nClassification report on testing set \n")
print(metrics.classification_report(y_test, y_test_pred))
print("\nConfusion metrics on testing set \n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))


# #### Accuracy is descreased by using rbf model. Lets go forward with poly kernel model.

# Till now, after PCA transformation with 200 components, we tried SVM using 3 different kernels and using default hyperparamters (C and gamma)
# We finialized that `PCA transformed` data along with `poly` SVM kernel gives us an accuracy of 96%.
# 
# Therefore, to further fine tune our model we can consider following changes:
# - Trying different number of PCA components. We can connsider using 195 and 200 components.
# - Trying different values of hyperparameter C. The value of C tells us how much you want to avoid misclassifying on training data. We can consider C as 1, 10.
# - Trying different values of hyperparameter gamma. The hyperparameter gamma controls  the amount of non-linearity in the model - as gamma increases, the model becomes more non-linear, and thus model complexity increases. We can consider gamma as 0.1,0.01.
#     
# Now we want to know what should be the best combination of values of number PCA compnents, C and gamma for our model built using `poly` kernel of SVM.
# 
# We will be using grid search cross validation to find the best comnination of the hyperparamters.
# 
# Along with this, we will introduce the pipelining functionality of sklearn.
# 
# Till now the steps we did include:
# 
# 1. Scale the initial data
# 2. Perform PCA to reduce the dimensionality
# 3. Build a model using SVM.
# 
# A pipeline, will schedule all the above steps and create the final model in our gridSearch.

# #### Using Pipeline for performing scaling, PCA and SVM on the Data

# In[ ]:


pipe_steps = [('scaler',StandardScaler()),('pca',PCA()),('SVM',SVC(kernel='poly'))]
check_params = {
    'pca__n_components' : [195,200],
    'SVM__C':[1,10],
    'SVM__gamma':[0.01,0.001]
}

pipeline = Pipeline(pipe_steps)


# Now, gridSearch will be used to find the best combbiantion of parameters. In the above pipeline, for every value of PCA number of components, the grid search will build model 4 models.
# 
# ![image.png](attachment:image.png)
# 
# As we are considering, 2 values for PCA components, therefore, GridSearch will build 8 different models. 
# 
# For the 8 different models, we will be using KFold cross validation with 3 splits. Therefore, the training will be done on 8X3 = 24 different models fits.
# 
# Now lets get our final parameters for the model.

# In[ ]:


folds = KFold(n_splits=3,shuffle=True,random_state=101)


#setting up GridSearchCV()
model_cv = GridSearchCV(estimator = pipeline,
                       param_grid = check_params,
                       scoring = 'accuracy',
                       cv = folds,
                       verbose = 3,
                       return_train_score=True,
                       n_jobs=-1)

#fit the model
model_cv.fit(X_train,y_train) # Considering our initial data as scaling will be handled by the pipeline.
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# In[ ]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# <b>The above table gives the results for 24 different model fits that were done by the grid Search. The best parameteres are selected based on the `mean_train_score` and `mean_test_score` as shown in the above table.
# 

# In[ ]:


# converting C to numeric type for plotting on x-axis
cv_results['param_SVM__C'] = cv_results['param_SVM__C'].astype('int')

# # plotting
plt.figure(figsize=(20,7))

# subplot 1/3
plt.subplot(121)
gamma_01 = cv_results[(cv_results['param_SVM__gamma']==0.01) & (cv_results['param_pca__n_components'] == 195)]

plt.plot(gamma_01["param_SVM__C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_SVM__C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.50, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')


# subplot 2/3
plt.subplot(122)
gamma_001 = cv_results[(cv_results['param_SVM__gamma']==0.001) & (cv_results['param_pca__n_components'] == 195)]

plt.plot(gamma_001["param_SVM__C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_SVM__C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.50, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')


# In[ ]:


# # plotting
plt.figure(figsize=(20,7))

# subplot 1/3
plt.subplot(121)
gamma_01 = cv_results[(cv_results['param_SVM__gamma']==0.01) & (cv_results['param_pca__n_components'] == 200)]

plt.plot(gamma_01["param_SVM__C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_SVM__C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.50, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')


# subplot 2/3
plt.subplot(122)
gamma_001 = cv_results[(cv_results['param_SVM__gamma']==0.001) & (cv_results['param_pca__n_components'] == 200)]

plt.plot(gamma_001["param_SVM__C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_SVM__C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.50, 1.1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')


# #### Now we have the best values of hyper parameters. 
# ### Let us build the final model using these values and evaluate the results.

# In[ ]:


pca_final = IncrementalPCA(n_components = 195)

X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca = pca_final.transform(X_test_scaled)
print(X_test_pca.shape)
print(X_train_pca.shape)


# In[ ]:


# model with optimal hyperparameters

# model
final_model = SVC(C=1, gamma=0.01, kernel="poly")

final_model.fit(X_train_pca, y_train)
# predict
y_train_pred = final_model.predict(X_train_pca)
y_test_pred = final_model.predict(X_test_pca)


# In[ ]:


# metrics
train_accuracy = metrics.accuracy_score(y_train,y_train_pred)
print("Accuracy on training data: {}".format(train_accuracy))
test_accuracy = metrics.accuracy_score(y_test,y_test_pred)
print("Accuracy on testing data: {}".format(test_accuracy))

print("\nClassification report on testing set \n")
print(metrics.classification_report(y_test, y_test_pred))

print("\nConfusion metrics on testing set \n")
print(metrics.confusion_matrix(y_true=y_test, y_pred=y_test_pred))


# ### The accuracy of the final model is 95.68%

# ### Using final model on unseen data (test.csv)

# In[ ]:


#import file and reading few lines
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_df.head(10)


# In[ ]:


test_df.shape


# In[ ]:


test_scaled = scaler.transform(test_df)


# In[ ]:


final_test_pca = pca_final.transform(test_scaled)
final_test_pca.shape


# In[ ]:


#model.predict
test_predict = final_model.predict(final_test_pca)


# In[ ]:


# Plotting the distribution of prediction
a = {'ImageId': np.arange(1,test_predict.shape[0]+1), 'Label': test_predict}
data_to_export = pd.DataFrame(a)
sns.countplot(data_to_export['Label'], palette = 'icefire')


# In[ ]:


# Let us visualize few of predicted test numbers
df = np.random.randint(1,test_predict.shape[0]+1,5)

plt.figure(figsize=(16,4))
for i,j in enumerate(df):
    plt.subplot(150+i+1)
    d = test_scaled[j].reshape(28,28)
    plt.title(f'Predicted Label: {test_predict[j]}')
    plt.imshow(d)
plt.show()


# In[ ]:


# Exporting the predicted values 
data_to_export.to_csv(path_or_buf='submission.csv', index=False)


# In[ ]:


submitted = pd.read_csv('submission.csv')


# In[ ]:


submitted.head()


# In[ ]:




