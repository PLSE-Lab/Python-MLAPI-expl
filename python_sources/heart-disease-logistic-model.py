#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression On Heart Disease dataset

# Data description-

# **id:** patient identification number 
# <br>**age:** age in years
# <br>**sex:**(1 = male; 0 = female)
# <br>**cpchest:** pain type
# <br>**trestbps:** resting blood pressure (in mm Hg on admission to the hospital)
# <br>**chol:** serum cholestoral in mg/dl
# <br>**fbs:** (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# <br>**restecg:** resting electrocardiographic results
# <br>**thalach:** maximum heart rate achieved
# <br>**exang:** exercise induced angina (1 = yes; 0 = no)
# <br>**oldpeak:** ST depression induced by exercise relative to rest
# <br>**slope:** the slope of the peak exercise ST segment
# <br>**ca:** number of major vessels (0-3) colored by flourosopy
# <br>**thal:** 3 = normal; 6 = fixed defect; 7 = reversable defect
# <br>**target:**  1= Yes. Has heart disease, 0=No. Does not have heart disease.

# Importing the required libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# Checking the dimension of the dataset

# In[ ]:


heart=pd.read_csv("../input/heart.csv")
heart.shape


# In[ ]:


heart.head(5)


# In[ ]:


heart.dtypes


# In[ ]:


heart['sex'][heart['sex'] == 0] = 'Female'
heart['sex'][heart['sex'] == 1] = 'Male'

heart['cp'][heart['cp'] == 1] = 'Typical angina'
heart['cp'][heart['cp'] == 2] = 'Atypical angina'
heart['cp'][heart['cp'] == 3] = 'Non-anginal pain'
heart['cp'][heart['cp'] == 4] = 'Asymptomatic'

heart['fbs'][heart['fbs'] == 0] = 'Lower than 120mg/ml'
heart['fbs'][heart['fbs'] == 1] = 'Greater than 120mg/ml'

heart['restecg'][heart['restecg'] == 0] = 'Normal'
heart['restecg'][heart['restecg'] == 1] = 'ST-T wave abnormality'
heart['restecg'][heart['restecg'] == 2] = 'Left ventricular hypertrophy'

heart['exang'][heart['exang'] == 0] = 'No'
heart['exang'][heart['exang'] == 1] = 'Yes'

heart['slope'][heart['slope'] == 1] = 'Upsloping'
heart['slope'][heart['slope'] == 2] = 'Flat'
heart['slope'][heart['slope'] == 3] = 'Downsloping'

heart['thal'][heart['thal'] == 1] = 'Normal'
heart['thal'][heart['thal'] == 2] = 'Fixed defect'
heart['thal'][heart['thal'] == 3] = 'Reversable defect'


# In[ ]:


heart['sex']=heart['sex'].astype('object')
heart['fbs']=heart['fbs'].astype('object')
heart['restecg']=heart['restecg'].astype('object')
heart['exang']=heart['exang'].astype('object')
heart['thal']=heart['thal'].astype('object')
heart['target']=heart['target'].astype('object')


# In[ ]:


heart.dtypes


# In[ ]:


heart.describe()


# In[ ]:


heart['ca'].unique()


# In[ ]:


heart['thal'].unique()


# In[ ]:


heart['target'].value_counts()


# In[ ]:


sns.countplot(x='target',data=heart,palette='hls')
plt.show()


# In[ ]:


sns.distplot(heart['age'])


# In[ ]:


sns.distplot(heart['chol'])


# In[ ]:


sns.distplot(heart['trestbps'])


# In[ ]:


sns.distplot(heart['thalach'])


# In[ ]:


sns.distplot(heart['ca'])


# In[ ]:


count_no = len(heart[heart['target']==0])
count_yes = len(heart[heart['target']==1])
pct_of_no = count_no/(count_no+count_yes)
print("Percentage of people not having heart disease is", pct_of_no*100)
pct_of_yes = count_yes/(count_no+count_yes)
print("Percentage of people having heart disease is", pct_of_yes*100)


# In[ ]:


heart.groupby('sex').mean()


# In[ ]:


heart.groupby('cp').mean()


# In[ ]:


heart.groupby('fbs').mean()


# In[ ]:


heart.groupby('thal').mean()


# In[ ]:


heart.groupby('ca').mean()


# In[ ]:


pd.crosstab(heart.sex,heart.target).plot(kind='bar')
plt.title('Heart disease frequency in gender')
plt.xlabel('Sex')
plt.ylabel('Frequency of Heart Disease')


# From the output we can see that the frequency of heart disease in Females(sex=0) is less as compared to Males(sex=1).

# In[ ]:


heart.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')


# In[ ]:


correlation=heart.corr()
correlation


# ### Creating Dummy Variables 

# In[ ]:


heart= pd.get_dummies(heart, drop_first=True)


# In[ ]:


heart.head(4)


# Splitting the data into train and test dataset

# In[ ]:


x=heart.iloc[:,0:13]
y=heart['target_1']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .2, random_state=10) 


# In[ ]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


x_train.head(3)


# In[ ]:


y_train.head()


# # Building the logistic model

# In[ ]:


import statsmodels.api as sm
logit_model = sm.Logit(y, x)
result=logit_model.fit()
print(result.summary())


# Predicting the test result and calculating the accuracy

# In[ ]:


logreg=LogisticRegression()
logreg.fit(x_train,y_train)
pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, pred))


# **Precision:** is the ratio TP / (TP + FP) where TP is number of true positives and FP number of false positives. The precision is intuitively the ability of the classifier to not label a sample as positive if it is negative.
# 
# <br>**Recall:** is the ratio TP / (TP + FN) where TP is the number of true positives and FN number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
# 
# <br>**F-score:** can be interpreted as a weighted harmonic mean of the precision and recall, where an F score reaches its best value at 1 and worst score at 0. The F score weights the recall more than the precision by a factor of beta. beta = 1.0 means recall.
# 
# <br>**Support:** is the number of occurrences of each class in y_test.

# ### Removing variables having p-values>0.05 and rebuilding the model

# In[ ]:


cols=['thalach', 'oldpeak', 'ca', 'sex_Male', 'cp_Atypical angina', 
      'cp_Non-anginal pain', 'cp_Typical angina'] 
X=x_train[cols]
Y=y_train
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())


# In[ ]:


logreg=LogisticRegression()
logreg.fit(X,Y)
pred = logreg.predict(X)


# In[ ]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X, Y)))


# ### Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y, pred)
print(confusion_matrix)


# The output tells that there are 79 + 125 correct predictions and 14 + 24 incorrect predictions.

# ### Sensitivity and Specificity

# In[ ]:


total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)


# In[ ]:


fpr, tpr, thresholds = roc_curve(Y, pred)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# ### Identifying the area under curve

# In[ ]:


auc(fpr, tpr)


# ### Precision, Recall, F-score and Support

# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(Y, pred))

