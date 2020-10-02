#!/usr/bin/env python
# coding: utf-8

# # The purpose of the case study is to classify a given silhouette as one of three different types of vehicle, using a set of features extracted from the silhouette. The vehicle may be viewed from one of many different angles.
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


myData = pd.read_csv('../input/vehicle.csv')
myData.head()


# In[ ]:


myData.shape


# In[ ]:


myData.isnull().sum()


# We will impute these null values rather to drop them.

# In[ ]:


myData.describe().transpose()


# In[ ]:


myData.info()


# Rows imputation with respective mean values

# In[ ]:


myData.fillna(myData.mean(), axis = 0, inplace = True)
myData.isnull().sum()
print(myData.shape)


# In[ ]:


myData.groupby('class').count()


# In[ ]:


plt.figure(figsize = (20,20))
sns.pairplot(data = myData, hue = 'class')


# # 2.EDA

# Compactness

# In[ ]:


fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['compactness'], ax = g1)
g1.set_title('Distribution Plot')

sns.boxplot(myData['compactness'], ax = g2)
g2.set_title('Box Plot')


# Circularity

# In[ ]:


fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['circularity'], ax = g1)
g1.set_title('Distribution Plot')

sns.boxplot(myData['circularity'], ax = g2)
g2.set_title('Box Plot')


# Distance Circularity 

# In[ ]:


fig, (g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['distance_circularity'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['distance_circularity'], ax = g2)
g2.set_title("Box Plot")


# Radius Ratio

# In[ ]:


fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['radius_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['radius_ratio'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


q1 = np.quantile(myData['radius_ratio'], 0.25)
q2 = np.quantile(myData['radius_ratio'], 0.50)
q3 = np.quantile(myData['radius_ratio'], 0.75)
IQR = q3 - q1

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("radius_ratio above ", myData['radius_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", myData[myData['radius_ratio'] > 276]['radius_ratio'].shape[0])


# Pr.Axis aspect ratio

# In[ ]:


fig, (g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['pr.axis_aspect_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['pr.axis_aspect_ratio'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


q1 = np.quantile(myData['pr.axis_aspect_ratio'], 0.25)
q2 = np.quantile(myData['pr.axis_aspect_ratio'], 0.50)
q3 = np.quantile(myData['pr.axis_aspect_ratio'], 0.75)
IQR = q3 - q1

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("pr.axis_aspect_ratio above ", myData['pr.axis_aspect_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", myData[myData['pr.axis_aspect_ratio'] > 77.0]['pr.axis_aspect_ratio'].shape[0])


# Max length aspect ratio

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['max.length_aspect_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['max.length_aspect_ratio'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


q1 = np.quantile(myData['max.length_aspect_ratio'], 0.25)
q2 = np.quantile(myData['max.length_aspect_ratio'], 0.50)
q3 = np.quantile(myData['max.length_aspect_ratio'], 0.75)
IQR = q3 - q1

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("max.length_aspect_ratio above ", myData['max.length_aspect_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")
print("max.length_aspect_ratio below ", myData['max.length_aspect_ratio'].quantile(0.25) - (1.5*IQR), "are outliers")


# Scatter ratio

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['scatter_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['scatter_ratio'], ax = g2)
g2.set_title("Box Plot")


# Elongatedness

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['elongatedness'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['elongatedness'], ax = g2)
g2.set_title("Box Plot")


# Pr.Axis rectangularity

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['pr.axis_rectangularity'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['pr.axis_rectangularity'], ax = g2)
g2.set_title("Box Plot")


# Max length rectangularity

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['max.length_rectangularity'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['max.length_rectangularity'], ax = g2)
g2.set_title("Box Plot")


# Scaled variance

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['scaled_variance'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['scaled_variance'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


q1 = np.quantile(myData['scaled_variance'], 0.25)
q2 = np.quantile(myData['scaled_variance'], 0.50)
q3 = np.quantile(myData['scaled_variance'], 0.75)
IQR = q3 - q1

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("scaled_variance above ", myData['scaled_variance'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", myData[myData['scaled_variance'] > 292]['scaled_variance'].shape[0])


# Scaled variance.1

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['scaled_variance.1'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['scaled_variance.1'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


q1 = np.quantile(myData['scaled_variance.1'], 0.25)
q2 = np.quantile(myData['scaled_variance.1'], 0.50)
q3 = np.quantile(myData['scaled_variance.1'], 0.75)
IQR = q3 - q1

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("scaled variance.1 above ", myData['scaled_variance.1'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", myData[myData['scaled_variance.1'] > 989.5]['scaled_variance.1'].shape[0])


# Scaled radius of gyration

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['scaled_radius_of_gyration'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['scaled_radius_of_gyration'], ax = g2)
g2.set_title("Box Plot")


# Scaled radius of gyration.1

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['scaled_radius_of_gyration.1'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['scaled_radius_of_gyration.1'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


q1 = np.quantile(myData['scaled_radius_of_gyration.1'], 0.25)
q2 = np.quantile(myData['scaled_radius_of_gyration.1'], 0.50)
q3 = np.quantile(myData['scaled_radius_of_gyration.1'], 0.75)
IQR = q3 - q1

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("scaled radius of gyration.1 above ", myData['scaled_radius_of_gyration.1'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", myData[myData['scaled_radius_of_gyration.1'] > 87]['scaled_radius_of_gyration.1'].shape[0])


# Skewness about

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['skewness_about'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['skewness_about'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


q1 = np.quantile(myData['skewness_about'], 0.25)
q2 = np.quantile(myData['skewness_about'], 0.50)
q3 = np.quantile(myData['skewness_about'], 0.75)
IQR = q3 - q1

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("skewness about above ", myData['skewness_about'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", myData[myData['skewness_about'] > 19.5]['skewness_about'].shape[0])


# Skewness about.1

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['skewness_about.1'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['skewness_about.1'], ax = g2)
g2.set_title("Box Plot")


# In[ ]:


q1 = np.quantile(myData['skewness_about.1'], 0.25)
q2 = np.quantile(myData['skewness_about.1'], 0.50)
q3 = np.quantile(myData['skewness_about.1'], 0.75)
IQR = q3 - q1

print("Quartile q1: ", q1)
print("Quartile q2: ", q2)
print("Quartile q3: ", q3)
print("Inter Quartile Range: ", IQR)

print("skewness about.1 above ", myData['skewness_about.1'].quantile(0.75) + (1.5*IQR), "are outliers")
print("No. of outliers ", myData[myData['skewness_about.1'] > 40]['skewness_about.1'].shape[0])


# Skewness about.2

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['skewness_about.2'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['skewness_about.2'], ax = g2)
g2.set_title("Box Plot")


# Hollows ratio

# In[ ]:


fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)
fig.set_size_inches(15,2)
sns.distplot(myData['hollows_ratio'], ax = g1)
g1.set_title("Distribution Plot")

sns.boxplot(myData['hollows_ratio'], ax = g2)
g2.set_title("Box Plot")


# Class

# In[ ]:


myData.groupby('class').count()


# In[ ]:


sns.countplot(myData['class'])


# 3. EDA Summary of Independent and dependent variables

# * compactness - Approx normal distribution, no outliers
# * circularity - Slightly right skewed
# * distance_circularity - Left skewed
# * radius_ratio - Approx normal distribution, with 3 outliers
# * pr.axis_aspect_ratio - Approx normal distribution, with 8 outliers
# * max.length_aspect_ratio - 2 peak, 12 outliers
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

# ## Correlation Matrix

# In[ ]:


plt.figure(figsize = (20,10))
sns.heatmap(myData.corr(), annot = True)


# # 4. Principal Component Analysis

# In[ ]:


myData_attr = myData.drop('class', axis = 1)
myData_target = myData['class']

print(myData_attr.shape)
print(myData_target.shape)


# In[ ]:


myData_attr_s = myData_attr.apply(zscore)


# In[ ]:


myData_target.replace({"car": 0, "bus": 1, "van": 2}, inplace = True)
print(myData_target.shape)


# In[ ]:


cov_mat = np.cov(myData_attr_s, rowvar = False)
print(cov_mat)


# In[ ]:


print(cov_mat.shape)


# In[ ]:


from sklearn.decomposition import PCA
pca_18 = PCA(n_components = 18)
pca_18.fit(myData_attr_s)


# In[ ]:


print(pca_18.explained_variance_)


# In[ ]:


print(pca_18.components_)


# In[ ]:


print(pca_18.explained_variance_ratio_)


# In[ ]:


plt.bar(list(range(1,19)), pca_18.explained_variance_ratio_, alpha = 0.5)
plt.ylabel('Variation explained')
plt.xlabel('Eigen values')
plt.show()


# In[ ]:


plt.step(list(range(1,19)), np.cumsum(pca_18.explained_variance_ratio_), where = 'mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('Eigen value')
plt.show()


# 95% variation observed in this data.

# In[ ]:


pca_8 = PCA(n_components = 8)
pca_8.fit(myData_attr_s)
print(pca_8.components_)
print(pca_8.explained_variance_ratio_)


# In[ ]:


myData_attr_s_pca_8 = pca_8.transform(myData_attr_s)
myData_attr_s_pca_8.shape


# In[ ]:


sns.pairplot(pd.DataFrame(myData_attr_s_pca_8))


# # 5. Support Vector Machines

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.svm import SVC


# In[ ]:


accuracies = {}
model = SVC()

X_train, X_test, y_train, y_test = train_test_split(myData_attr_s_pca_8, myData_target, test_size = 0.30, random_state = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc_svm = model.score(X_test, y_test) *100

accuracies['SVM'] = acc_svm
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


# In[ ]:


print(classification_report(y_test,y_pred))


# # Grid Search

# In[ ]:


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


model_svm = SVC(C = 1, kernel = 'rbf', gamma = 1)
X_train, X_test, y_train, y_test = train_test_split(myData_attr_s_pca_8, myData_target, test_size = 0.30, random_state = 1)
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test)

acc_svm_gs = model_svm.score(X_test, y_test) * 100
accuracies['SVM_GS'] = acc_svm_gs
print(model.score(X_test, y_test))
print(classification_report(y_test, y_pred))


# Accuracy score = 83%

# In[ ]:


svm_eval = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
svm_eval.mean()


# # Feature Importances

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(myData_attr_s, myData_target)
rf.score(myData_attr_s, myData_target)

feature_importances = pd.DataFrame(rf.feature_importances_, index = myData_attr_s.columns,
                                  columns = ['importance']).sort_values('importance', ascending = False) * 100

feature_importances


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
expected = y_test
predicted = nb_model.predict(X_test)

acc_nb = nb_model.score(X_test, y_test) * 100
accuracies['NB'] = acc_nb

print(metrics.classification_report(expected, predicted))
print('Total accuracy: ', np.round(metrics.accuracy_score(expected, predicted), 2))


# # 7. Model Comparison

# In[ ]:


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

from sklearn.metrics import confusion_matrix

cm_svm = confusion_matrix(y_test, y_cm_svm)
cm_svm_gs = confusion_matrix(y_test, y_cm_svm_gs)
cm_nb = confusion_matrix(y_test, y_cm_nb)

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


# Thus PCA helps to reduce dimension data from 18 to 8 with 93% accuracy
