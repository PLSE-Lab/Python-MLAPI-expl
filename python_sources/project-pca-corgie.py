#!/usr/bin/env python
# coding: utf-8

# The purpose of the case study is to classify a given silhouette as one of three different types of vehicle, using a set of features extracted from the silhouette. The vehicle may be viewed from one of many different angles.
# 
# Four "Corgie" model vehicles were used for the experiment: a double decker bus, Cheverolet van, Saab 9000 and an Opel Manta 400 cars. This particular combination of vehicles was chosen with the expectation that the bus, van and either one of the cars would be readily distinguishable, but it would be more difficult to distinguish between the cars.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import zscore


# # 1. Observe dataset

# In[ ]:


data = pd.read_csv('../input/vehicle.csv')
data.head()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# There are many missing values. We will choose to impute these null values instead of dropping them.

# In[ ]:


data.describe().transpose()


# In[ ]:


data.info()


# There are few null values in almost every column,we will try to impute this with mean value this would slightly bias the dataset but won't impact the model significantly.  

# In[ ]:


data.fillna(data.mean(), axis = 0, inplace = True)
print(data.isnull().sum())
print(data.shape)


# We have imputed all the rows with their mean values.

# In[ ]:


#class is target column
data.groupby('class').count()


# There is a data imbalance as car has a significantly higher count as compared to bus and van. While van has the lowest count.

# In[ ]:


plt.figure(figsize = (15,15))
sns.pairplot(data = data, diag_kind = 'kde', hue = 'class')


# Most of these columns share a strong correlation with each other. We can also notice certain outliers in this data. 

# # 2.EDA

# Let's analyze every column in detail.

# ## 2.1 Compactness

# In[ ]:


fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['compactness'], ax = g1)
g1.set_title('Distribution Plot')

sns.boxplot(data['compactness'], ax = g2)
g2.set_title('Box Plot')


# Compactness column shows an approximately normal distribution curve while having no outliers.

# ## 2.2 Circularity

# In[ ]:


fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['circularity'], ax = g1)
g1.set_title('Distribution Plot')

sns.boxplot(data['circularity'], ax = g2)
g2.set_title('Box Plot')


# The distribution plot has 3 peaks and is right skewed.

# ## 2.3 Distance Circularity 

# In[ ]:


fig, (g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['distance_circularity'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['distance_circularity'], ax = g2)
g2.set_title("Box Plot")


# Here the distribution plot has 2 peaks and is left skewed

# ## 2.4 Radius Ratio

# In[ ]:


fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['radius_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['radius_ratio'], ax = g2)
g2.set_title("Box Plot")


# Here it is right skewed and there is presence of outliers.

# ### Performing outlier analysis

# In[ ]:


q1 = np.quantile(data['radius_ratio'], 0.25)
q2 = np.quantile(data['radius_ratio'], 0.50)
q3 = np.quantile(data['radius_ratio'], 0.75)
IQR = q3 - q1
#outlier = q1 - 1.5*IQR and q3 + 1.5*IQR... here as outliers are in the 4th quartile hence using 2 formula
#Printing the quartile

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("radius_ratio above ", data['radius_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", data[data['radius_ratio'] > 276]['radius_ratio'].shape[0])


# ## 2.5 Pr.Axis aspect ratio

# In[ ]:


fig, (g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['pr.axis_aspect_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['pr.axis_aspect_ratio'], ax = g2)
g2.set_title("Box Plot")


# The distribution is approx normally distributed as well as right skewed and there is presence of outliers.

# In[ ]:


#Performing Outlier analysis

q1 = np.quantile(data['pr.axis_aspect_ratio'], 0.25)
q2 = np.quantile(data['pr.axis_aspect_ratio'], 0.50)
q3 = np.quantile(data['pr.axis_aspect_ratio'], 0.75)
IQR = q3 - q1
#outlier = q1 - 1.5*IQR and q3 + 1.5*IQR... here as outliers are in the 4th quartile hence using 2 formula
#Printing the quartile

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("pr.axis_aspect_ratio above ", data['pr.axis_aspect_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", data[data['pr.axis_aspect_ratio'] > 77.0]['pr.axis_aspect_ratio'].shape[0])


# ## 2.6 Max length aspect ratio

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['max.length_aspect_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['max.length_aspect_ratio'], ax = g2)
g2.set_title("Box Plot")


# There are 2 peaks and significant no. of outliers are observed.

# In[ ]:


#Performing Outlier analysis

q1 = np.quantile(data['max.length_aspect_ratio'], 0.25)
q2 = np.quantile(data['max.length_aspect_ratio'], 0.50)
q3 = np.quantile(data['max.length_aspect_ratio'], 0.75)
IQR = q3 - q1
#outlier = q1 - 1.5*IQR and q3 + 1.5*IQR... here as outliers are in the 4th quartile hence using 2 formula
#Printing the quartile

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("max.length_aspect_ratio above ", data['max.length_aspect_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")
print("max.length_aspect_ratio below ", data['max.length_aspect_ratio'].quantile(0.25) - (1.5*IQR), "are outliers")

print("No. of outliers above  are",data[data['max.length_aspect_ratio']>14.5]['max.length_aspect_ratio'].shape[0])
print("No. of outliers below are",data[data['max.length_aspect_ratio']<2.5]['max.length_aspect_ratio'].shape[0])


# ## 2.7 Scatter ratio

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['scatter_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['scatter_ratio'], ax = g2)
g2.set_title("Box Plot")


# There are 2 peaks observed and the distribution is right skewed.

# ## 2.8 Elongatedness

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['elongatedness'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['elongatedness'], ax = g2)
g2.set_title("Box Plot")


# There are 2 peaks observed and the distribution is left skewed.

# ## 2.9 Pr.Axis rectangularity

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['pr.axis_rectangularity'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['pr.axis_rectangularity'], ax = g2)
g2.set_title("Box Plot")


# There are 2 peaks observed and the distribution is right skewed.

# ## 2.10 Max length rectangularity

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['max.length_rectangularity'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['max.length_rectangularity'], ax = g2)
g2.set_title("Box Plot")


# There are 3 peaks observed and no outliers in the data.

# ## 2.11 Scaled variance

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['scaled_variance'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['scaled_variance'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


#Performing Outlier analysis

q1 = np.quantile(data['scaled_variance'], 0.25)
q2 = np.quantile(data['scaled_variance'], 0.50)
q3 = np.quantile(data['scaled_variance'], 0.75)
IQR = q3 - q1
#outlier = q1 - 1.5*IQR and q3 + 1.5*IQR... here as outliers are in the 4th quartile hence using 2nd formula
#Printing the quartile

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("scaled_variance above ", data['scaled_variance'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", data[data['scaled_variance'] > 292]['scaled_variance'].shape[0])


# ## 2.12 Scaled variance.1

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['scaled_variance.1'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['scaled_variance.1'], ax = g2)
g2.set_title("Box Plot")


# The distribution plot shows 2 peaks with outliers.

# In[ ]:


#Performing Outlier analysis

q1 = np.quantile(data['scaled_variance.1'], 0.25)
q2 = np.quantile(data['scaled_variance.1'], 0.50)
q3 = np.quantile(data['scaled_variance.1'], 0.75)
IQR = q3 - q1
#outlier = q1 - 1.5*IQR and q3 + 1.5*IQR... here as outliers are in the 4th quartile hence using 2 formula
#Printing the quartile

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("scaled variance.1 above ", data['scaled_variance.1'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", data[data['scaled_variance.1'] > 989.5]['scaled_variance.1'].shape[0])


# ## 2.13 Scaled radius of gyration

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['scaled_radius_of_gyration'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['scaled_radius_of_gyration'], ax = g2)
g2.set_title("Box Plot")


# It has an approximate normal distribution with no outliers.

# ## 2.14 Scaled radius of gyration.1

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['scaled_radius_of_gyration.1'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['scaled_radius_of_gyration.1'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


#Performing Outlier analysis

q1 = np.quantile(data['scaled_radius_of_gyration.1'], 0.25)
q2 = np.quantile(data['scaled_radius_of_gyration.1'], 0.50)
q3 = np.quantile(data['scaled_radius_of_gyration.1'], 0.75)
IQR = q3 - q1
#outlier = q1 - 1.5*IQR and q3 + 1.5*IQR... here as outliers are in the 4th quartile hence using 2 formula
#Printing the quartile

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("scaled radius of gyration.1 above ", data['scaled_radius_of_gyration.1'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", data[data['scaled_radius_of_gyration.1'] > 87]['scaled_radius_of_gyration.1'].shape[0])


# ## 2.15 Skewness about

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['skewness_about'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['skewness_about'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


#Performing Outlier analysis

q1 = np.quantile(data['skewness_about'], 0.25)
q2 = np.quantile(data['skewness_about'], 0.50)
q3 = np.quantile(data['skewness_about'], 0.75)
IQR = q3 - q1
#outlier = q1 - 1.5*IQR and q3 + 1.5*IQR... here as outliers are in the 4th quartile hence using 2 formula
#Printing the quartile

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("skewness_about above ", data['skewness_about'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", data[data['skewness_about'] > 19.5]['skewness_about'].shape[0])


# ## 2.16 Skewness about.1

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['skewness_about.1'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['skewness_about.1'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


#Performing Outlier analysis

q1 = np.quantile(data['skewness_about.1'], 0.25)
q2 = np.quantile(data['skewness_about.1'], 0.50)
q3 = np.quantile(data['skewness_about.1'], 0.75)
IQR = q3 - q1
#outlier = q1 - 1.5*IQR and q3 + 1.5*IQR... here as outliers are in the 4th quartile hence using 2 formula
#Printing the quartile

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("skewness about.1 above ", data['skewness_about.1'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", data[data['skewness_about.1'] > 40]['skewness_about.1'].shape[0])


# ## 2.17 Skewness about.2

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['skewness_about.2'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['skewness_about.2'], ax = g2)
g2.set_title("Box Plot")


# ## 2.18 Hollows ratio

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(data['hollows_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(data['hollows_ratio'], ax = g2)
g2.set_title("Box Plot")


# There are 2 peaks observed and no outliers in the data.

# ## 2.19 Class

# In[ ]:


data.groupby('class').count()


# In[ ]:


sns.countplot(data['class'])


# There is some imbalance in terms of class. This will have some impact on model.

# # 3. EDA Summary of Independent and dependent variables

# We can infer,
# 
# * compactness - Approx normal distribution, no outliers
# * circularity - Slightly right skewed
# * distance_circularity - Left skewed
# * radius_ratio - Approx normal distribution, with 3 outliers
# * pr.axis_aspect_ratio - Approx normal distribution, with 8 outliers
# * max.length_aspect_ratio - 2 peak, 13 outliers
# * scatter_ratio - 2 peaks, right skewed
# * elongatedness - 2 peaks, left skewed
# * pr.axis_rectangularity - 2 peaks, right skewed
# * max.length_rectangularity - 3 peaks, no outliers
# * scaled_variance - 2 peaks, 1 outlier
# * scaled_variance.1 - 2 peaks, 2 outliers
# * scaled_radius_of_gyration - Slightly right skewed
# * scaled_radius_of_gyration.1 - 15 Outliers
# * skewness_about - right skewed, 12 outliers
# * skewness_about.1 - 1 outlier
# * skewness_about.2 - No outlier
# * hollows_ratio - 2 peaks, no outlier
# * class - more no. of car > bus > van
# 
# 
# We have chosen to not remove the outliers as the no. of outliers is insignificant and it wouldn't bias the model. However dropping these outliers may give us a better model. Assuming these outliers were artificially introduced during data collection.
# 

# ## Correlation Matrix

# In[ ]:


plt.figure(figsize = (15,10))
sns.heatmap(data.corr(), annot = True)


# Here there are some columns which share high collinearity while some share low collinearity. We can drop the columns whichever share high collinearity as there's no use in considering multiple columns. We will use Principal Component Analysis to determine which columns can be dropped.

# # 4. Principal Component Analysis

# In[ ]:


#Preparing X independent columns, y dependent columns
data_attr = data.drop('class', axis = 1)
data_target = data['class']

print(data_attr.shape)
print(data_target.shape)


# In[ ]:


#Scaling the attribute data

data_attr_s = data_attr.apply(zscore)


# In[ ]:


#Replacing Target column into numbers

data_target.replace({"car": 0, "bus": 1, "van": 2}, inplace = True)

print(data_target.shape)


# In[ ]:


#Applying Covariance matrix

cov_mat = np.cov(data_attr_s, rowvar = False)
print(cov_mat)


# In[ ]:


#Shape of Covariance matrix
print(cov_mat.shape)


# In[ ]:


#Applying Principal Component Analysis for all 18 columns

from sklearn.decomposition import PCA
pca_18 = PCA(n_components = 18)

pca_18.fit(data_attr_s)


# In[ ]:


#Eigen values
print(pca_18.explained_variance_)


# In[ ]:


#Eigen vectors
print(pca_18.components_)


# In[ ]:


#Variance ratio
print(pca_18.explained_variance_ratio_)


# In[ ]:


#Plot Eigen values
plt.bar(list(range(1,19)), pca_18.explained_variance_ratio_, alpha = 0.5, align = 'center')
plt.ylabel('Variation explained')
plt.xlabel('Eigen values')
plt.show()


# In[ ]:


#Plot using step function

plt.step(list(range(1,19)), np.cumsum(pca_18.explained_variance_ratio_), where = 'mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('Eigen value')
plt.show()


# We have 8 components in this model that explain 95% variation in this data.

# In[ ]:


#Applying PCA for 8 components this time
pca_8 = PCA(n_components = 8)
pca_8.fit(data_attr_s)
print(pca_8.components_)
print(pca_8.explained_variance_ratio_)


# In[ ]:


#Transform the raw data with 18 dim into 8 dims
data_attr_s_pca_8 = pca_8.transform(data_attr_s)
data_attr_s_pca_8.shape


# In[ ]:


#Draw pairplot to find correlation
sns.pairplot(pd.DataFrame(data_attr_s_pca_8))


# # 5. Support Vector Machines

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.svm import SVC


# In[ ]:


accuracies = {}
model = SVC()

X_train, X_test, y_train, y_test = train_test_split(data_attr_s_pca_8, data_target, test_size = 0.30, random_state = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_svm = model.score(X_test, y_test) *100

accuracies['SVM'] = acc_svm
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


# In[ ]:


print(classification_report(y_test,y_pred))


# **SVM model gives an accuracy of 93%**

# # Using Grid Search

# In[ ]:


#Finding best parameters for our SVM model

param = {
    'C' : [0.01,0.05,0.5,1],
    'kernel' :['linear','rbf']
}

grid_svm = GridSearchCV(model, param_grid = param, scoring = 'accuracy', cv = 10)


# In[ ]:


grid_svm.fit(X_train,y_train)


# In[ ]:


grid_svm.best_params_


# In[ ]:


#Running our kernel with best parameters


#Kernel = rbf, C = 1
model_svm = SVC(C = 1, kernel = 'rbf', gamma = 1)
X_train, X_test, y_train, y_test = train_test_split(data_attr_s_pca_8, data_target, test_size = 0.30, random_state = 1)
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test)

acc_svm_gs = model_svm.score(X_test, y_test) * 100
accuracies['SVM_GS'] = acc_svm_gs
print(model.score(X_test, y_test))
print(classification_report(y_test, y_pred))


# **Accuracy score is 83% using Grid Search, however SVM gives an accuracy of 93%**

# In[ ]:


#Cross validation score for SVM

svm_eval = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
svm_eval.mean()


# **Using Grid Search we can infer the model accuracy is 83%.**

# # 6. Naive Bayes Algorithm

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
expected = y_test
predicted = nb_model.predict(X_test)

acc_nb = nb_model.score(X_test, y_test) * 100
accuracies['NB'] = acc_nb


# In[ ]:


#Determine Model score

print(metrics.classification_report(expected, predicted))
print('Total accuracy: ', np.round(metrics.accuracy_score(expected, predicted), 2))


# **Naive Bayes model gives an accuracy of 76%**

# # 7. Model Comparison

# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize = (8,5))
plt.yticks(np.arange(0,100,10))
sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()))


# In[ ]:


models = pd.DataFrame({
    'Model': ['SVM', 'SVM_GS','Naive Bayes'],
    
    'Score': [acc_svm, acc_svm_gs, acc_nb]
    })

models.sort_values(by='Score', ascending=False)


# # 8. Confusion Matrix

# In[ ]:


y_cm_svm = model.predict(X_test)
y_cm_svm_gs = model_svm.predict(X_test)
y_cm_nb = nb_model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix

cm_svm = confusion_matrix(y_test, y_cm_svm)
cm_svm_gs = confusion_matrix(y_test, y_cm_svm_gs)
cm_nb = confusion_matrix(y_test, y_cm_nb)


# In[ ]:


plt.figure(figsize = (16,4))
plt.suptitle("Confusion Matrices",fontsize=12)
plt.subplots_adjust(wspace = 0.8, hspace = 0.8)

plt.subplot(1,3,1)
plt.title("SVM Confusion Matrix")
sns.heatmap(cm_svm, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})


plt.subplot(1,3,2)
plt.title("SVM Grid Search Confusion Matrix")
sns.heatmap(cm_svm_gs, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})

plt.subplot(1,3,3)
plt.title("NB Confusion Matrix")
sns.heatmap(cm_nb, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})


# **We can conclude SVM is the best suited algorithm as it gives a good accuracy score of 93% on test data and confusion matrix gives a good classification.**
# 
# 
# **To conclude PCA helps to reduced 18 dimension data into 8 dimension data with SVM score of 93% accuracy**
