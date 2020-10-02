#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the necessary packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skew
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[ ]:


import os
print(os.listdir("../input"))


# ### Read the dataset and display top 5

# In[ ]:


filename = '../input/diabetes.csv'
data = pd.read_csv(filename)
data.head()


# In[ ]:


data.info()


# ### Well, the data set has no null value

# In[ ]:


data.describe()


# ### From this result, we can see that the Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI all have a value of 0. But actually, can they be 0?  
# ### I think Glucose,BloodPressure,SkinThickness,Insulin,BMI cannot be 0.  
# ### fill the dataset with a median
# ### First, the result of Outcome is 0 and 1 are calculated separately, because the results are different, maybe their values are different.
# ### Second, the value of 0 is removed to calculate the median or average.

# In[ ]:


glucose_0 = data[data['Glucose'] > 0][data['Outcome'] == 0]['Glucose'].median()
glucose_1 = data[data['Glucose'] > 0][data['Outcome'] == 1]['Glucose'].median()
bloodPressure_0 = data[data['BloodPressure'] > 0][data['Outcome'] == 0]['BloodPressure'].median()
bloodPressure_1 = data[data['BloodPressure'] > 0][data['Outcome'] == 1]['BloodPressure'].median()
SkinThickness_0 = data[data['SkinThickness'] > 0][data['Outcome'] == 0]['SkinThickness'].median()
SkinThickness_1 = data[data['SkinThickness'] > 0][data['Outcome'] == 1]['SkinThickness'].median()
insulin_0 = data[data['Insulin'] > 0][data['Outcome'] == 0]['Insulin'].median()
insulin_1 = data[data['Insulin'] > 0][data['Outcome'] == 1]['Insulin'].median()
bmi_0 = data[data['BMI'] > 0][data['Outcome'] == 0]['BMI'].median()
bmi_1 = data[data['BMI'] > 0][data['Outcome'] == 1]['BMI'].median()

print(glucose_0,glucose_1,bloodPressure_0,bloodPressure_1,SkinThickness_0,SkinThickness_1,insulin_0,insulin_1,bmi_0,bmi_1)


# In[ ]:


# fill median to the data set
data.loc[(data["Glucose"]==0) & (data['Outcome']==0),'Glucose'] = glucose_0
data.loc[(data["Glucose"]==0) & (data['Outcome']==1),'Glucose'] = glucose_1

data.loc[(data["BloodPressure"]==0) & (data['Outcome']==0),'BloodPressure'] = bloodPressure_0
data.loc[(data["BloodPressure"]==0) & (data['Outcome']==1),'BloodPressure'] = bloodPressure_1

data.loc[(data["SkinThickness"]==0) & (data['Outcome']==0),'SkinThickness'] = SkinThickness_0
data.loc[(data["SkinThickness"]==0) & (data['Outcome']==1),'SkinThickness'] = SkinThickness_1

data.loc[(data["Insulin"]==0) & (data['Outcome']==0),'Insulin'] = insulin_0
data.loc[(data["Insulin"]==0) & (data['Outcome']==1),'Insulin'] = insulin_1

data.loc[(data["BMI"]==0) & (data['Outcome']==0),'BMI'] = bmi_0
data.loc[(data["BMI"]==0) & (data['Outcome']==1),'BMI'] = bmi_1


# ### Let's look at the distribution of features

# In[ ]:


fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(3,3,1)
ax2 = fig.add_subplot(3,3,2)
ax3 = fig.add_subplot(3,3,3)
ax4 = fig.add_subplot(3,3,4)
ax5 = fig.add_subplot(3,3,5)
ax6 = fig.add_subplot(3,3,6)
ax7 = fig.add_subplot(3,3,7)
ax8 = fig.add_subplot(3,3,8)

ax1.hist(data['Pregnancies'])
ax1.set_title('Distribution of Pregnancies')

ax2.hist(data['Glucose'])
ax2.set_title('Distribution of Glucose')

ax3.hist(data['BloodPressure'])
ax3.set_title('Distribution of BloodPressure')

ax4.hist(data['SkinThickness'])
ax4.set_title('Distribution of SkinThickness')

ax5.hist(data['Insulin'])
ax5.set_title('Distribution of Insulin')

ax6.hist(data['BMI'])
ax6.set_title('Distribution of BMI')

ax7.hist(data['DiabetesPedigreeFunction'])
ax7.set_title('Distribution of DiabetesPedigreeFunction')

ax8.hist(data['Age'])
ax8.set_title('Distribution of Age')

plt.show()


# ### From the above figure, we can see that some of the data is not normally distributed. 
# ### Let us look at the skewness value of the data.

# In[ ]:


skewed_feats = data.drop('Outcome', axis = 1).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness


# ### Filter out features with a skewness greater than 0.75

# In[ ]:


skewness = skewness[abs(skewness) > 0.75]
skewness


# ### The Box-Cox transformation is a generalized power transformation method proposed by Box and Cox in 1964. It is a kind of data transformation commonly used in statistical modeling for the case where continuous response variables do not satisfy the normal distribution.  
# ###  Here we use the scipy boxcox1p function for Box-Cox transformation

# In[ ]:


from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.245
for feat in skewed_features:
    data[feat] = boxcox1p(data[feat], lam)


# ### Separate X and y

# In[ ]:


y = data['Outcome'].values
X = data.drop('Outcome', axis = 1)


# In[ ]:


X.describe()


# ### Compared with before doing data processing, the variance is much smaller.
# ### Label the data

# In[ ]:


# use pandas.cut() function to label the data set
X1=pd.DataFrame()
X1["Pregnancies"] =pd.cut(X["Pregnancies"],3,labels=[1,2,3]).astype(int)
X1["Glucose"] =pd.cut(X["Glucose"],3,labels=[1,2,3]).astype(int)
X1["BloodPressure"] =pd.cut(X["BloodPressure"],3,labels=[1,2,3]).astype(int)
X1["SkinThickness"] =pd.cut(X["SkinThickness"],3,labels=[1,2,3]).astype(int)
X1["Insulin"] =pd.cut(X["Insulin"],3,labels=[1,2,3]).astype(int)
X1["BMI"] =pd.cut(X["BMI"],3,labels=[1,2,3]).astype(int)
X1["DiabetesPedigreeFunction"] =pd.cut(X["DiabetesPedigreeFunction"],3,labels=[1,2,3]).astype(int)
X1["Age"] = pd.cut(X["Age"],3,labels=[1,2,3]).astype(int)


# In[ ]:


X1.head()


# ### Vectorization of categorical variables

# In[ ]:


categorical_features = list(X1.columns)

#The data type becomes object before it can be processed by get_dummies
for col in categorical_features:
    X1[col] = X1[col].astype('object')
    
X1 = pd.get_dummies(X1[categorical_features])
X1.head()


# In[ ]:


X_train = pd.concat([X, X1], axis = 1)


# In[ ]:


X_train.head()


# ### Randomly sample 20% of the data to build test samples, and the rest as training samples

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y, random_state=64, test_size=0.2)


# ### Use the LogisticRegression to build the model
# ### Use cross_val_score to calculate the accuracy

# In[ ]:


lr = LogisticRegression(solver = 'liblinear')
loss = cross_val_score(lr, X_train, y_train, cv=10, scoring='accuracy')
print( 'accuracy of each fold is:\n',loss)
print('cv accuracy is:', loss.mean())


# ### Parameter tuning on LogisticRegression

# In[ ]:


penaltys = ['l1','l2']
Cs = [0.001, 0.01, 0.1,0.2, 1, 1.01, 4, 10, 100]
#Cs = [x for x in range(1,100)]
tuned_parameters = dict(penalty = penaltys, C = Cs)

lr_penalty= LogisticRegression(solver = 'liblinear')
grid= GridSearchCV(lr_penalty, tuned_parameters,cv=10, scoring='accuracy')
grid.fit(X_train,y_train)
print('accuracy on train data set:',grid.score(X_train,y_train))
print('best_score_:',grid.best_score_)
print('best_params_:',grid.best_params_)
print('accuracy on test data set',grid.score(X_test,y_test))
print('confusion_matrix:\n',metrics.confusion_matrix(y_test,grid.predict(X_test), labels=[1, 0]))
print('classification_report:\n',metrics.classification_report(y_test, grid.predict(X_test), labels=[1,0], digits=4))


# ### Use the LinearSVC to build the model

# In[ ]:


SVC1 = LinearSVC().fit(X_train, y_train)

#Test on the check set to estimate model performance
y_predict = SVC1.predict(X_test)

y_train_predict = SVC1.predict(X_train)
print('*' * 40,'train data set','*' * 40)
print("Classification report for classifier %s:\n%s\n"
      % (SVC1, metrics.classification_report(y_train,y_train_predict)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_train, y_train_predict))
print('*' * 40,'test data set','*' * 40 )
print("Classification report for classifier %s:\n%s\n"
      % (SVC1, metrics.classification_report(y_test, y_predict)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_predict))


# ### Parameter tuning on LinearSVC

# In[ ]:


def fit_grid_point_Linear(C, X_train, y_train, X_val, y_val):
    # Train the model on the training data set
    SVC2 = LinearSVC(C=C)
    SVC2.fit(X_train, y_train)

    # Return accuracy on the checksum
    accuracy = SVC2.score(X_val, y_val)
    return accuracy

# Parameters that need to be optimized
# C_s = [0.001, 0.01, 0.1,0.2, 1,1.01, 10, 100, 1000]
C_s = [x/10 for x in range(1,100)]

accuracy_s = []
for i, oneC in enumerate(C_s):
    tmp = fit_grid_point_Linear(oneC, X_train, y_train, X_test, y_test)
    accuracy_s.append(tmp)

x_axis = np.log10(C_s)

plt.plot(x_axis, np.array(accuracy_s), 'b-')

plt.legend()
plt.xlabel('log(C)')
plt.ylabel('accuracy')

plt.show()
print(max(accuracy_s))

