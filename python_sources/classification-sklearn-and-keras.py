#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# We have a data which classified if Bank customer leave the bank or not according to features in it. We will try different classifiers to predict if a customer will leave or not.

# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
sns.set(color_codes=True)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler


# In[ ]:


import os
print(os.listdir("../input"))


# ### Read Data
# $ $

# In[ ]:


df=pd.read_csv('../input/Churn_Modelling.csv')
df.head(5)


# 
# $ $
# 
# Data contains; 
# 
# 
# - RowNumber
# - CustomerId
# - Surname
# - Creditscore 
# - Geography - (France, Spain, Germany)
# - Gender - (Famale; Male)
# - age - age in years 
# - Tenure - (from 1 to 10 years)
# - Balance
# - NumberOfProducts
# - HasCrCard - (1=yes, 0=no)
# - IsActiveMember - (1=yes, 0=no)
# - EstimatedSalary
# - Exited - Did they leave the bank (1=yes, 0=no)
# 
# $ $ 

# ## Feature engineering

# We notice that features RowNumber, CustomerId and Surname are useless for our peorpuse. So we can drop them from the data.

# In[ ]:


df=df.drop(['RowNumber','CustomerId','Surname'], axis=1)


# In[ ]:


df.Exited.value_counts()


# In[ ]:


sns.set(color_codes=True)
sns.countplot(x="Exited", data=df, palette="bwr")
plt.title('Class Distributions \n (0: No Exited || 1: Exited)', fontsize=15)
plt.show()


# In[ ]:


countnotleave = len(df[df.Exited == 0])
countleave = len(df[df.Exited == 1])
print("Percentage of Costumers who didn't leave: {:.2f}%".format((countnotleave / (len(df.Exited))*100)))
print("Percentage of Costumers who did leave: {:.2f}%".format((countleave / (len(df.Exited))*100)))


# In[ ]:


sns.countplot(x='Gender', data=df, palette="mako_r")
plt.xlabel("Gender")
plt.show()


# In[ ]:


df.drop(['Geography','Gender', 'Exited'], axis=1).describe()


# We have imbalanced data. Let's plot pairwise relationships of certain features in a dataset with different levels of our target variable.

# In[ ]:


#sns.pairplot(df.drop(['Geography','Gender','NumOfProducts','HasCrCard','IsActiveMember'], axis=1), hue="Exited")
#plt.show()


# In[ ]:


f, axes = plt.subplots(ncols=3, figsize=(18,3))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Exited", y="Age", data=df, ax=axes[0], palette="Set2")
axes[0].set_title('Age vs Exited')

sns.boxplot(x="Exited", y="EstimatedSalary", data=df, ax=axes[1], palette="Set2")
axes[1].set_title('Salary vs Exited')

sns.boxplot(x="Exited", y="Tenure", data=df, ax=axes[2], palette="Set2")
axes[2].set_title('Tenure vs Exited')

plt.show()


# In[ ]:


sns.set(color_codes=True)
plt.figure(figsize=(20,12))

A=['Gender','Geography','NumOfProducts','HasCrCard',"IsActiveMember"]
B=["muted","husl","dark","RdBu_r","BrBG"]

for i in range(5):
    plt.subplot(2,3,i+1)
    sns.countplot(x=A[i], hue='Exited', data=df, palette=B[i])
    #plt.title('Exited Frequency for Gender')
    #plt.xlabel('Gender')
    plt.xticks(rotation=0)
    plt.legend(["Not Exited", "Exited"])
    #plt.ylabel('Frequency')
    #plt.show()

plt.show()


# In[ ]:


#plt.figure(figsize=(15,4))
#new_Balance=pd.cut(round(df.Balance/1000,1),8)
#sns.countplot(new_Balance, hue=df.Exited, palette='PuRd')
#plt.xlabel('Balance ($K)')
#plt.title('Exited Frequency According to Balance')
#dfb=pd.get_dummies(new_Balance)


# 
# ### Creating Dummy Variables
# 

# Since 'Gender' and 'Geography' are categorical variables we'll turn them into dummy variables.

# In[ ]:


sex=pd.get_dummies(df['Gender'])
country=pd.get_dummies(df['Geography'])


# In[ ]:


df=pd.concat([df,country],axis=1)


# In[ ]:


df = df.drop(columns = ['Gender', 'Geography', 'Spain'])
df['Gender']=sex['Female']
df.head()


# $ $
# 
# ## Classifiers
# 
# $ $
# 
# We can use sklearn library or we can write functions ourselves. We will try both. Firstly we will write our functions after that we'll use sklearn library to calculate score.
# 
# $ $

# In[ ]:


y = df.Exited.values
x_data = df.drop(['Exited'], axis = 1)


# #### Normalize Data
# 
# 
# $$X_{\text{changed}}=\frac{X-X_{\min}}{X_{\max}-X_{\min}}$$
# 
# $ $

# In[ ]:


# Normalize
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# $ $
# 
# We will split our data into train and test set.
# 
# $ $

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)


# In[ ]:


#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# $ $
# 
# Let's say weight = 0.01 and bias = 0.0
# 
# $ $

# In[ ]:


#initialize
def initialize(dimension):
    
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias


# #### Sigmoid Function
# 
# 
# $$\sigma(x)=\frac{1}{1+e^{-x}}$$
# 
# $ $
# 

# In[ ]:


def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head


# Logistic Regression model estimated probability:
# $$\hat{p} =h_{\theta}(x)=\sigma(\theta^T\cdot x)$$
# 
# 

# #### Cost Function
# 
# $$J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log (h_{\theta}(x^{(i)}))+(1-y^{(i)})\log (1-h_{\theta}(x^{(i)})]$$
# 
# $ $

# #### Gradient Descent
# 
# 
# $$\theta_{j}:= \theta_j-\alpha\frac{\partial}{\partial \theta_{j}}J(\theta)$$
# 
# $$\theta_{j}:= \theta_j-\frac{\alpha}{m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}$$
# 
# $ $

# By the way in formulas; 
# 
# - $h_{\theta}(x^{(i)})$= y_head
# - $y^{(i)}$ = y_train
# - $x^{(i)}$ = x_train

# In[ ]:


def forwardBackward(weight,bias,x_train,y_train):
    # Forward
    
    y_head = sigmoid(np.dot(weight.T,x_train) + bias)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    
    # Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    
    return cost,gradients


# In[ ]:


def update(weight,bias,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight and bias values
    for i in range(iteration):
        cost,gradients = forwardBackward(weight,bias,x_train,y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]
        
        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight,"bias": bias}
    
    #print("iteration:",iteration)
    #print("cost:",cost)

    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients


# In[ ]:


def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction


# In[ ]:


def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
    dimension = x_train.shape[0]
    weight,bias = initialize(dimension)
    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)
    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    print("Manuel Test Accuracy: {:.2f}%".format((1 - np.mean(np.abs(y_prediction - y_test)))*100))


# In[ ]:


logistic_regression(x_train,y_train,x_test,y_test,1,200)


# ### <font color='blue'>Manuel Test Accuracy is **81.25%** <\font>
# 

# Let's find out sklearn's score.

# In[ ]:


X=df.drop("Exited", axis = 1)
y=df['Exited']

pipeline = Pipeline([('std_scaler', StandardScaler()),
        ])

pipeline.fit_transform(X)

X_scaled = pd.DataFrame(pipeline.fit_transform(X), columns=X.columns)

data_scaled=pd.concat([X_scaled,y],axis=1)


# In[ ]:


data_scaled.head()


# In[ ]:


X=data_scaled.drop('Exited',axis=1)
y=data_scaled['Exited']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


#import imblearn
#from imblearn.under_sampling import RandomUnderSampler
#rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
#x_train, y_train = rus.fit_resample(x_train, y_train)


# In[ ]:


classifiers = {
    "LogisiticRegression": LogisticRegression(solver='lbfgs'),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(gamma='scale'),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100)
}


# In[ ]:


for key, classifier in classifiers.items():
    classifier.fit(x_train, y_train)
    training_score = cross_val_score(classifier, x_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of"
 , round(training_score.mean(), 3) * 100, "% accuracy score")


# In[ ]:


# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV


# Logistic Regression 
log_reg_params = {"penalty": ['l2'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(solver='lbfgs'), log_reg_params, cv=5)
grid_log_reg.fit(x_train, y_train)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_


# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(gamma='scale'), svc_params,cv=5)
grid_svc.fit(x_train, y_train)

# SVC best estimator
svc = grid_svc.best_estimator_

# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params,cv=5)
grid_tree.fit(x_train, y_train)

# tree best estimator
tree_clf = grid_tree.best_estimator_

#Random forest
param_grid = parameters = {'n_estimators': [100, 200, 400], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [10],
              'min_samples_split': [30],
              'min_samples_leaf': [1]
             }

grid_forest = GridSearchCV(RandomForestClassifier(n_estimators=100), param_grid, cv=5)
grid_forest.fit(x_train, y_train)

forest = grid_forest.best_estimator_


# In[ ]:


# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, x_train, y_train, cv=5,
                             method="decision_function")

svc_pred = cross_val_predict(svc, x_train, y_train, cv=5,
                             method="decision_function")

tree_pred = cross_val_predict(tree_clf, x_train, y_train, cv=5)

forest_pred = cross_val_predict(forest, x_train, y_train, cv=5)


# In[ ]:


from sklearn.metrics import roc_auc_score

d = {'Classifier': ['Logistic Regression', 'Support Vector Classifier', 'Decision Tree Classifier', 'Random Forest Classifier'], 
     '(ROC AUC) Score': [roc_auc_score(y_train, log_reg_pred), roc_auc_score(y_train, svc_pred), roc_auc_score(y_train, tree_pred), roc_auc_score(y_train, forest_pred)]}
df = pd.DataFrame(data=d)
 
df


# In[ ]:


sns.set_style("white")

log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)
forest_fpr, forest_tpr, forest_threshold = roc_curve(y_train, forest_pred)

def graph_roc_curve_multiple(log_fpr, log_tpr, forest_fpr, forest_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(10,5))
    plt.title('ROC Curve \n 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot(forest_fpr, forest_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_train, forest_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, forest_fpr, forest_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)
sns.set_style("white")
plt.show()


# ### Test Data with classifiers

# In[ ]:


y_pred_log_reg = log_reg.predict(x_test)
y_pred_forest = forest.predict(x_test)
y_pred_svc = svc.predict(x_test)
y_pred_tree = tree_clf.predict(x_test)


log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
forest_cf = confusion_matrix(y_test, y_pred_forest)
svc_cf = confusion_matrix(y_test, y_pred_svc)
tree_cf = confusion_matrix(y_test, y_pred_tree)

plt.figure(figsize=(20,10))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,2,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(log_reg_cf,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,2,2)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(svc_cf,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,2,3)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(tree_cf,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,2,4)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(forest_cf,annot=True,cmap="Blues",fmt="d",cbar=False)


plt.show()


# In[ ]:


from sklearn.metrics import roc_auc_score

d = {'Classifier': ['Logistic Regression', 'Support Vector Classifier', 'Decision Tree Classifier', 'Random Forest Classifier'], 
     '(ROC AUC) Score': [roc_auc_score(y_test, y_pred_log_reg), roc_auc_score(y_test, y_pred_svc), roc_auc_score(y_test, y_pred_tree), roc_auc_score(y_test, y_pred_forest)]}
roc_test = pd.DataFrame(data=d)
 
roc_test


# ## Principle Component Analysis (PCA) for Data Visualization

# In[ ]:


from sklearn.decomposition import PCA

pca=PCA(n_components=2)
principalComponents = pca.fit_transform(data_scaled.drop('Exited', axis=1))
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[ ]:


finalDf = pd.concat([principalDf, data_scaled[['Exited']]], axis = 1)
finalDf.head(5)


# In[ ]:


sample=finalDf[:1000]

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = [0,1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = sample['Exited'] == target
    ax.scatter(sample.loc[indicesToKeep, 'principal component 1']
               , sample.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# ## Neural Network with Keras

# In[ ]:


from keras import models
from keras import layers
from keras.optimizers import Adam, RMSprop
from keras.metrics import categorical_crossentropy
from keras import regularizers

n_inputs = x_train.shape[1]

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(n_inputs,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])


# In[ ]:


history = model.fit(x_train, y_train, validation_split=0.2, batch_size=512, epochs=100, shuffle=True, verbose=2)


# In[ ]:


keras_predictions = model.predict_classes(x_test, batch_size=200, verbose=0)


# In[ ]:




cm = confusion_matrix(y_test, keras_predictions)
actual_cm = confusion_matrix(y_test, y_test)
labels = ['No Exited', 'Exited']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plt.title("Confusion Matrix \n keras")
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)

#fig.add_subplot(222)
#plt.title("Confusion Matrix \n (with 100% accuracy)")
#sns.heatmap(actual_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.show()


# In[ ]:


acc = history.history['acc']
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

