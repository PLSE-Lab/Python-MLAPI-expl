#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Diagnostic Classification.Breast Cancer Diagnostic Classification of Malignant or Benign

# This NoteBook is the Practice for the K Nearest Neighbors Classifications Algorithm.

# Importing all the Required Libraries for data analysis and data visualization.

# In[ ]:


import numpy as numpyInstance
import pandas as pandasInstance
import matplotlib.pyplot as matplotlibInstance
import seaborn as seabornInstance


# Now Setting Inline Data Visualization Settings.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Now Importing the Data of Breast Cancer.

# In[ ]:


breastCancerData = pandasInstance.read_csv('../input/data.csv')


# Checking the Header of the Data.

# In[ ]:


breastCancerData.head()


# Now Checking the Info of the columns.

# In[ ]:


breastCancerData.info()


# Now Let's Standarize the Data for the Convinent Use of The KNN Algorithm.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
breastCancerData.columns


# In[ ]:


scaler.fit(breastCancerData[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']])


# Now Transforming it into more understandable form.

# In[ ]:


transformed = scaler.transform(breastCancerData[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']])


# In[ ]:


toMakeNewDataFrame = breastCancerData[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]


# In[ ]:


newDataFrameWithFeatures = pandasInstance.DataFrame(transformed,columns=toMakeNewDataFrame.columns)


# Now Making The Test and Train Data.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(newDataFrameWithFeatures, breastCancerData['diagnosis'], test_size=0.33, random_state=42)


# Now Training the KNN Model.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knModel = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knModel.fit(X_train,y_train)


# Now let's make predictions.

# In[ ]:


predictions = knModel.predict(X_test)


# Now Let's Check the Accuracy of the K = 1.

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(predictions,y_test))


# Now let's Try to Make it Even More Accurate.

# In[ ]:


#Here we are Calculating predictions for each Value of K from 1 to 40 and Calculating the Average Error Value and Storing it in ERROR.
errors = []
for number in range(1,50):
    anOtherModel = KNeighborsClassifier(n_neighbors=number)
    anOtherModel.fit(X_train,y_train)
    anOtherpredictions = anOtherModel.predict(X_test)
    errors.append(numpyInstance.mean(predictions!=y_test))


# Now Let's Check Where we have the Least Error.

# In[ ]:


matplotlibInstance.figure(figsize=(10,6))
matplotlibInstance.plot(range(1,50),errors,color='green', linestyle='dashed', marker='o')
matplotlibInstance.title('Error Rate vs. K Value')
matplotlibInstance.xlabel('K')
matplotlibInstance.ylabel('Error Rate')


# ## This Means that we are already at the Best Accuracy so Value of K=1 is already perfect.

# In[ ]:




