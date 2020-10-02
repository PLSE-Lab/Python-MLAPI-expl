#!/usr/bin/env python
# coding: utf-8

# # Mushroom Classify Machine Learning Decision Tree

# ### Used Libraries
# 1. NumPy (Numerical Python)
# 2. Pandas
# 3. Matplotlib
# 4. Seaborn
# 5. Sckit learn
# 6. Missingno

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Content:
# 1. Pandas Profiling Report
# 2. Missingo - Missing Data
# 3. Separating Features and Labels
# 4. Converting String Value To int Type for Labels
# 5. Data Standardisation
# 6. Splitting Dataset into Training Set and Testing Set
# 7. Build Decision Tree Model with Default Hyperparameter
# 8. Accuracy Score
# 9. Confusion Matrix with Seaborn - Heatmap
# 10. F1 Score

# ### Reading Data

# In[ ]:


df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")


# In[ ]:


df.sample(5)


# In[ ]:


df.info()


# # Pandas Profiling Report

# In[ ]:


report = pp.ProfileReport(df)

report.to_file("report.html")

report


# # Missingno - Missing Data

# In[ ]:


import missingno as msno
msno.matrix(df)
plt.show()


# # Separating Features and Labels

# In[ ]:


X=df.iloc[:, 1:23]
X.head()
y=df.iloc[:,0]


# # Converting String Value To int Type for Labels
# ### Encode label category
# * Poisonous = p -> 1
# * Edible = e -> 0 

# In[ ]:


df["class"].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)


# In[ ]:


X = pd.get_dummies(X)
X.sample(5)


# # Data Standardisation

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# # Splitting Dataset into Training Set and Testing Set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# # Build Decision Tree Model with Default Hyperparameter

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)


# # Accuracy Score

# In[ ]:


print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# # Confusion Matrix with Seaborn - Heatmap
# * Poisonous -> 1
# * Edible -> 0 

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, linewidths=0.5,linecolor="red", fmt= '.0f',ax=ax)
plt.show()
plt.savefig('ConfusionMatrix.png')


# # F1 Score

# In[ ]:


from sklearn.metrics import  f1_score
f1_score = f1_score(y_test, y_pred)
print("F1 Score:")
print(f1_score)


# # Thank You
# 
# If you have any suggestion or advice or feedback, I will be very appreciated to hear them.
# ### Also there are other kernels
# * [FIFA 19 Player Data Analysis and Visualization EDA](https://www.kaggle.com/ismailsefa/f-fa-19-player-data-analysis-and-visualization-eda)
# * [Crimes Data Analysis and Visualzation (EDA)](https://www.kaggle.com/ismailsefa/crimes-data-analysis-and-visualzation-eda)
# * [Google Play Store Apps Data Analysis (EDA)](https://www.kaggle.com/ismailsefa/google-play-store-apps-data-analysis-eda)
# * [World Happiness Data Analysis and Visualization](https://www.kaggle.com/ismailsefa/world-happiness-data-analysis-and-visualization)
# * [Used Cars Data Analysis and Visualization (EDA)](https://www.kaggle.com/ismailsefa/used-cars-data-analysis-and-visualization-eda)
# * [Gender Recognition by Voice Machine Learning SVM](https://www.kaggle.com/ismailsefa/gender-recognition-by-voice-machine-learning-svm)
# * [Iris Species Classify Machine Learning KNN](https://www.kaggle.com/ismailsefa/iris-species-classify-machine-learning-knn)
# * [Breast Cancer Diagnostic Machine Learning R-Forest](https://www.kaggle.com/ismailsefa/breast-cancer-diagnostic-machine-learning-r-forest)
# * [Heart Disease Predic Machine Learning Naive Bayes](https://www.kaggle.com/ismailsefa/heart-disease-predic-machine-learning-naive-bayes)
# * [Mushroom Classify Machine Learning Decision Tree](https://www.kaggle.com/ismailsefa/mushroom-classify-machine-learning-decision-tree)
