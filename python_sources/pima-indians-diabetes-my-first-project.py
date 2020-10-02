#!/usr/bin/env python
# coding: utf-8

# Aljo Jose - 04 June 2017
# # Database - Pima Indians Diabetes
# **Motivation - Help medical professionals to make diagnosis easier by bridging gap between huge datasets and human knowledge. Apply machine learning techniques for given classification in a dataset that describes a population that is under a high risk of the onset of diabetes.**

# In[ ]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ##1.1 Load dataset and  summarize 

# In[ ]:


# Loading dataset and view a few records.
dataset = pd.read_csv('../input/diabetes.csv')
dataset.shape


# In[ ]:


dataset.head()


# Some of the columns ( SkinThickness, Insulin) having incorrect 0. Let's try to find columns having 0 values.

# In[ ]:


dataset.describe()


# Attributes Glucose,  BloodPressure, SkinThickness, Insulin, BMI having non-realistic value (0). We can try to expose dataset with and without handling 0 value and observe performance.

# In[ ]:


dataset.info()


# Observed columns DiabetesPedigreeFunction and BMI having float datatype. All others are of integer type.

# ## 1.2 Visualize dataset

# In[ ]:


# Histogram
dataset.hist(figsize=(10,8))


# Attributes BMI, BloodPressure, Glucose are found to be normally distributed.
# <br>BMI and BloodPressure nearly have Gaussian distribution.
# <br>Age, DiabetesPedigreeFunction, Insulin, Pregnancies found to be exponentially distributed.

# In[ ]:


# Box plot
dataset.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))


# Observed that spread of attributes is quite different. Attributes Age, Insulin appear to be quite skewed towards smaller values. <br>Scaling on dataset can be applied during data pre-processing.

# In[ ]:


# Correlation plot
Corr=dataset[dataset.columns].corr() 
sns.heatmap(Corr, annot=True)


# Observed that attributes BloodPressure, SkinThickness are not much related to outcome.  <br> Feature extraction can be tried to observe performance.

# ## 1.3 Data Pre-processing

# Note : Replaced 0 values by mean, but no performance improvement was observed while evaluating models. <br> Dropped rows with 0 values, performance seems to be improved. But dataset reduces to half. <br> Hence commented below lines.

# In[ ]:


# Data preprocessing - replace zeroes with mean or drop records with 0 values.
#attributes_to_replace_zero =list(dataset.columns[1:6])      # list all column names. 
#dataset[attributes_to_replace_zero] = dataset[attributes_to_replace_zero].replace(0, np.NaN)
#dataset.fillna(dataset.mean(), inplace=True) 
#dataset.dropna(inplace=True)


# In[ ]:


# Split into Input and Output.
attributes = list(dataset.columns[:8])
X = dataset[attributes].values 
y= dataset['Outcome'].values


# In[ ]:


# Scale input dataset.
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X = sc_X.fit_transform(X) 


# Note : Normalization reduced performance while evaluating models. Hence code disabled.

# In[ ]:


#Performacne reduces with normalizer
#from sklearn import preprocessing
#X = preprocessing.normalize(X)


# Split into train and test sets.

# In[ ]:


# Split into train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =0)


# Note : Applied feature selection, but not much change in performance. So code lines disabled.

# In[ ]:


# Applying Kernel PCA ( Not much change in performance)
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 6)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_


# # 1.4  Evaluate models.

# In[ ]:



# Import suite of algorithms.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


# In[ ]:


# Create objects of required models.
models = []
models.append(("LR",LogisticRegression()))
models.append(("GNB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DecisionTree",DecisionTreeClassifier()))
models.append(("LDA",  LinearDiscriminantAnalysis()))
models.append(("QDA",  QuadraticDiscriminantAnalysis()))
models.append(("AdaBoost", AdaBoostClassifier()))
models.append(("SVM Linear",SVC(kernel="linear")))
models.append(("SVM RBF",SVC(kernel="rbf")))
models.append(("Random Forest",  RandomForestClassifier()))
models.append(("Bagging",BaggingClassifier()))
models.append(("Calibrated",CalibratedClassifierCV()))
models.append(("GradientBoosting",GradientBoostingClassifier()))
models.append(("LinearSVC",LinearSVC()))
models.append(("Ridge",RidgeClassifier()))


# In[ ]:



# Find accuracy of models.
results = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=0)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    results.append(tuple([name,cv_result.mean(), cv_result.std()]))
  
results.sort(key=lambda x: x[1], reverse = True)    
for i in range(len(results)):
    print('{:20s} {:2.2f} (+/-) {:2.2f} '.format(results[i][0] , results[i][1] * 100, results[i][2] * 100))


# #1.5 Optimize peformance of best model.

# Linear classifiers performs well. So I have added all possible classifiers to suite of algorithms.
# <br> SVM Linear seems performs best. Now let us try to find the optimistic parameters for SVM using GridSearchCV. 

# In[ ]:


from sklearn.model_selection import GridSearchCV
model = SVC()
paramaters = [
             {'C' : [0.01, 0.1, 1, 10, 100, 1000], 'kernel' : ['linear']}                                       
             ]
grid_search = GridSearchCV(estimator = model, 
                           param_grid = paramaters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_ 
best_parameters = grid_search.best_params_  

print('Best accuracy : ', grid_search.best_score_)
print('Best parameters :', grid_search.best_params_  )


# #1.6 Finalize model 

# **Selected SVM  model with parameters C= 0.01 and kernel = 'linear'.**

# In[ ]:


# Predict output for test set. 
final_model = SVC(C = 0.1, kernel = 'linear')
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cf = confusion_matrix(y_test, y_pred)
print(cf)
print(accuracy_score(y_test, y_pred) * 100) 


# **Conclusion :- Observed accuracy of 82.46% on test set using SVM linear model.**

# ***Thank you very much for reading the post. This work is just an attempt to apply the concepts I learned. Please suggest improvements if any.***
