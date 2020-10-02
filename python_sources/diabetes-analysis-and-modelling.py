#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# <h1 style="text-align :center;"> Pima Indian Diabetes Dataset</h1> 
# <img src ="https://image.freepik.com/free-vector/diabetes-elements-collection_1212-405.jpg"/>

# <h2>Diabetes In India</h2>
# Over 30 million have now been diagnosed with diabetes in India. The CPR (Crude prevalence rate) in the urban areas of India is thought to be 9 per cent.
# 
# In rural areas, the prevalence is approximately 3 per cent of the total population.
# 
# The population of India is now more than 1000 million: this helps to give an idea of the scale of the problem.
# 
# The estimate of the actual number of diabetics in India is around 40 million.
# 
# This means that India actually has the highest number of diabetics of any one country in the entire world. IGT (Impaired Glucose Tolerance) is also a mounting problem in India.
# 
# The prevalence of IGT is thought to be around 8.7 per cent in urban areas and 7.9 per cent in rural areas, although this estimate may be too high. It is thought that around 35 per cent of IGT sufferers go on to develop type 2 diabetes, so India is genuinely facing a healthcare crisis.
# 
# In India, the type of diabetes differs considerably from that in the Western world.
# 
# Type 1 is considerably more rare, and only about 1/3 of type II diabetics are overweight or obese.
# 
# Diabetes is also beginning to appear much earlier in life in India, meaning that chronic long-term complications are becoming more common. The implications for the Indian healthcare system are enormous.
# 
# 

# In[ ]:


# importing libraries
# imports 
import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from IPython.display import Image
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import roc_curve,accuracy_score,auc,roc_auc_score,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score,cross_validate,cross_val_predict
from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[ ]:


# reading file
dataframe = pd.read_csv('../input/diabetes.csv')
dataframe.head()


# ## Now lets analyze which age people are more likely to be diagnosed to be diabetic
# 

# In[ ]:


# dataframe['Age'].hist()
plt.hist(dataframe['Age'][dataframe['Outcome'] == 1],label="Patient vs Age")


# **Its seems like people lying in age groups of 20-30 and 40-50 are more prone to diabetes**

# **Lets visualize the correlation map to visualize which feauture are most contributing to diagnosing diabetes**

# In[ ]:


sns.heatmap(dataframe.corr(),annot=True)


# **It seems like high glucose levels in the Blood is the most important factor for diabetic patient with correlation value of 0.47**

# # Now Lets check cleaning data removing Nan and missing values 

# In[ ]:


print(dataframe.isnull().sum())
print("Minimum Bloodpressure",dataframe['BloodPressure'].min())
print("Minimum BMI",dataframe['BMI'].min())


# # BMI and BloodPressure can't be Null Values

# In[ ]:


dataframe['BMI'] = dataframe['BMI'].replace(0,dataframe['BMI'].mean())
dataframe['BloodPressure'] = dataframe['BloodPressure'].replace(0,dataframe['BloodPressure'].mean())


# In[ ]:


dataframe['BloodPressure'].min()


# In[ ]:


Y = dataframe.Outcome
X = dataframe.drop('Outcome',axis=1)


# ##  Now Lets analyze Various Classifiers and train our model  
# 1. Random Forest
# <br>
# 2. Support Vector Machines
# <br>
# 3. Logistic Regression
# <br>
# 4. Schotastistic Gradient Classifiers
# <br>
# 5. naive Bayes
# <br>
# 6.  AdaBoostClassifier
# <br>
# 7. ExtraForestClassifier
# <br>
# 8. Decision Tree Classifier
# <br>
# 9.  Multilayer Perceptron
# <br>
# 10. Voting Classifier
# 

# In[ ]:


# Now without feauture enginnering and data normalization lets check how our model performs on test and train data
classifier = SVC()
HPoptimizerSVC = GridSearchCV(classifier,param_grid={'C': [1,10],'gamma': [0.0001,0.001,0.01,0.1]})
classifiers = {'Random_Forest':RandomForestClassifier(),
               'Logistic_Reggression':LogisticRegression(),
               'Decision Tree Classifier' : DecisionTreeClassifier(),
               'SGDClassifier':SGDClassifier(),
               'naive_bayes':GaussianNB(),
               "Support_vector_Machine": HPoptimizerSVC,
               "AdaBoost" : AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10),
               "ExtraForestClassifier" : GradientBoostingClassifier(), 
               'Multilayer Perceptron' : MLPClassifier(hidden_layer_sizes=(100,),momentum=0.9,solver='sgd'),
               'Voting Classifier' : VotingClassifier(estimators=[('log',LogisticRegression()),('SVM',SVC(C=1000)),('MLP',MLPClassifier(hidden_layer_sizes=(100,)))],voting='hard')

              }
#Holds accuracy for various models
Acc= {}
Acc_Train = {}
Acc_Test = {}
Predictions = {}
ROC = {}
AUC = {}
Confusion_Matrix = {}
Gmean = {}
Precision = {}
Recall = {}
F1_score = {}
mats = pd.DataFrame(Confusion_Matrix)


# # **Train Test Split**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# # **Data Normalization**

# In[ ]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[ ]:


for clf in classifiers:
    Acc[clf] = cross_validate(classifiers[clf],X,Y,cv=10,n_jobs=-1,scoring='accuracy',return_train_score=True)
    Acc_Train[clf] =  Acc[clf]['train_score'].mean()
    Acc_Test[clf] = Acc[clf]['test_score'].mean()
    classifiers[clf].fit(scaler.transform(X_train),y_train)
    pred =  classifiers[clf].predict(scaler.transform(X_test))
    ROC[clf] = roc_auc_score(y_test,pred)
    AUC[clf] = auc(y_test,pred,reorder=True)
    Confusion_Matrix[clf] = confusion_matrix(y_test,pred)
    Gmean[clf] = fowlkes_mallows_score(y_test,pred)
    Precision[clf] = precision_score(y_test,pred)
    Recall[clf] = recall_score(y_test,pred)    
    F1_score[clf] = f1_score(y_test,pred)


# In[ ]:


Accuracy_train = pd.DataFrame([Acc_Train[vals]*100 for vals in Acc_Train],columns=['Accuracy_Train'],index=[vals for vals in Acc_Train])
Accuracy_pred = pd.DataFrame([Acc_Test[vals]*100 for vals in Acc_Test],columns=['Accuracy_Test'],index=[vals for vals in Acc_Test])


# In[ ]:


ROC_Area = pd.DataFrame([ROC[vals] for vals in ROC],columns=['ROC(area)'],index=[vals for vals in ROC])
AUC_Area = pd.DataFrame([AUC[vals] for vals in AUC],columns=['AUC(area)'],index=[vals for vals in AUC])
Gmean = pd.DataFrame([Gmean[vals] for vals in Gmean],columns=['Gmean'],index=[vals for vals in Gmean])
Prec = pd.DataFrame([Precision[vals] for vals in Precision],columns=['precision'],index=[vals for vals in Precision])
Rec = pd.DataFrame([Recall[vals] for vals in Recall],columns=['recall'],index=[vals for vals in Recall])
Prec = pd.DataFrame([Precision[vals] for vals in Precision],columns=['precision'],index=[vals for vals in Precision])
f1 =  pd.DataFrame([F1_score[vals] for vals in F1_score],columns=['f1_score'],index=[vals for vals in F1_score])


# ## Accuracy | ROC  | Area under Curve | Gmean  | Precision | Recall 
# ## for various Classification
#  
# <hr>

# In[ ]:


pd.concat([Accuracy_train,Accuracy_pred,ROC_Area,AUC_Area,Gmean,Prec,Rec,f1], axis=1)


# **It seems for some classifiers(Ensembles like Random Forest,Decision Tree,ExtraForestClassifiers ) need to tuned as some are overfitting **

# # **Clearly Logistic Regression proves to be better than other classifiers**

# In[ ]:


CF = {}
for mat in Confusion_Matrix:
    CF[mat] = Confusion_Matrix[mat]
sns.heatmap(CF['Logistic_Reggression'],annot=True)    


# **Thanks !! please upvote if you find this kernel useful.**

# 
