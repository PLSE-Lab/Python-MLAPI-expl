#!/usr/bin/env python
# coding: utf-8

# # Tour of Machine Learning  Classification Algorithms
# - Decision Tree 
# - Random Forest 
# - Linear SVM
# - RBF SVM
# - Navie Bayes
# - Logistic Regression
# - MLPClassifier
# 
# As a Newbie to Machine Learning , so pls tell any correction or error in the dataset!
# 
# -- Ganesh Kasturi 22/06/2020

# # Attribute Information:
# - Age (age in years)
# - Sex (1 = male; 0 = female)
# - CP (chest pain type)
# - TRESTBPS (resting blood pressure (in mm Hg on admission to the hospital))
# - CHOL (serum cholestoral in mg/dl)
# - FPS (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# - RESTECH (resting electrocardiographic results)
# - THALACH (maximum heart rate achieved)
# - EXANG (exercise induced angina (1 = yes; 0 = no))
# - OLDPEAK (ST depression induced by exercise relative to rest)
# - SLOPE (the slope of the peak exercise ST segment)
# - CA (number of major vessels (0-3) colored by flourosopy)
# - THAL (3 = normal; 6 = fixed defect; 7 = reversable defect)
# - TARGET (1 or 0)

# In[ ]:


# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,roc_curve,auc


# In[ ]:


data = pd.read_csv(r"../input/heart-disease-uci/heart.csv")


# In[ ]:


print("Data First 5 Rows\n")
data.head()


# In[ ]:


print("Data Last 5 Rows\n")
data.tail()


# In[ ]:


data.describe()


# In[ ]:


print("Data Info \n ")
data.info()


# In[ ]:


print('Data columns \n')
data.columns


# In[ ]:


data.sample(frac=0.5)   # its like to take the sample from your dataset only!!


# In[ ]:


data.sample(5)


# In[ ]:


data=data.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})


# In[ ]:


data.columns


# In[ ]:


data.shape


# # Missing values

# In[ ]:


data.isnull().sum()


# In[ ]:


data.isnull().values.any()


# In[ ]:


sns.pairplot(data)
plt.show()


# ### Age Analysis

# In[ ]:


data.Age.value_counts()[:10]


# In[ ]:


sns.barplot(x = data.Age.value_counts()[:10].index, y = data.Age.value_counts()[:10].values)
plt.xlabel('AGE')
plt.ylabel('AGE counter')
plt.title("Age Analysis System")
plt.show()


# In[ ]:


minAge = min(data.Age)
maxAge = max(data.Age)
meanAge = data.Age.mean()
print('Min Age :',minAge)
print('Max Age :',maxAge)
print('Mean of Age: ',meanAge)


# In[ ]:


young_age = data[(data.Age >=29) & (data.Age < 40)]
middle_age = data[(data.Age >=40) & (data.Age < 55)]
elderly_age = data[(data.Age > 55)]
print("Young Age Group :",len(young_age))
print("Middle Age Group :",len(middle_age))
print("Elderly Age Group :",len(elderly_age))


# In[ ]:


sns.barplot(x=['young_age','middle_age','elderly_age'],y=[len(young_age),len(middle_age),len(elderly_age)])
plt.xlabel("Age")
plt.ylabel('Age COUNT')
plt.title("Age Stats in dataset")
plt.show()


# In[ ]:


colors = ['blue','green','yellow']
explode = [0,0,0.1]
plt.figure(figsize = (5,5))
#plt.pie([target_0_agerang_0,target_1_agerang_0], explode=explode, labels=['Target 0 Age Range 0','Target 1 Age Range 0'], colors=colors, autopct='%1.1f%%')
plt.pie([len(young_age),len(middle_age),len(elderly_age)],labels=['young ages','middle ages','elderly ages'],explode=explode,colors=colors, autopct='%1.1f%%')
plt.title('Age States',color = 'blue',fontsize = 15)
plt.show()


# # Sex (Gender)

# In[ ]:


data.Sex.value_counts()


# In[ ]:


sns.countplot(data.Sex)
plt.show()


# In[ ]:


sns.countplot(data.Sex,hue =data.Slope)
plt.title("Slope & Age Rates Show")
plt.show()


# In[ ]:


total_gender_count = len(data.Sex)
male_count = len(data[data['Sex']==1])
female_count = len(data[data['Sex']==0])
print('Total Genders :',total_gender_count)
print('Male Count    :',male_count)
print('Female Count  :',female_count)


# In[ ]:


# # Plot miles per gallon against horsepower with other semantics
sns.relplot(x="Trestbps", y="Age",
            sizes=(40, 400), alpha=.5, palette="muted",
            height=6, data=data)


# # Chest Pain Type

# In[ ]:


data.Cp.value_counts()


# In[ ]:


sns.countplot(data.Cp)
plt.xlabel("Chest Type")
plt.ylabel("Count")
plt.title("Chest tye VS Count")
plt.show()
#0 status at least
#1 condition slightly distressed
#2 condition medium problem
#3 condition too bad


# In[ ]:


# Show the results of a linear regression within each dataset
sns.lmplot(x="Trestbps", y="Chol",data=data,hue="Cp")
plt.show()


# In[ ]:


data.Thalach.value_counts()[:10]


# In[ ]:


sns.barplot(x = data.Thalach.value_counts()[:10].index,y= data.Thalach.value_counts()[:10].values)
plt.xlabel('Thalach')
plt.ylabel('Count')
plt.title('Thalach Counts')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


sns.swarmplot(x =data.Age)
plt.title('Age Rates')
plt.show()


# # Thal Analysis
# 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[ ]:


data.Thal.value_counts()


# In[ ]:


sns.countplot(data.Thal)
plt.show()


# # Target Analysis
# 

# In[ ]:


data.Target.unique()


# In[ ]:


sns.countplot(data.Target)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Counter 1 & 0')
plt.show()


# In[ ]:


sns.countplot(data.Target,hue =data.Sex)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target & Sex Counter 1 & 0')
plt.show()


# In[ ]:


sns.lineplot(x= 'Sex',y='Oldpeak',hue= 'Target',data =data)
plt.show()


# # Data Preparation (Train/test Split)

# In[ ]:


X=data.drop('Target',axis=1)
y=data['Target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,stratify = y,random_state =123)

print('X_train Shape : ',X_train.shape)
print('X_test Shape : ',X_test.shape)


# # Data Standardization

# In[ ]:


sc = StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,auc
from matplotlib.colors import ListedColormap
import warnings; warnings.filterwarnings('ignore')


# # Function IMP 

# In[ ]:


def run_classifier(clf,param_grid,title):
    # ------------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=123)
    # Randomized grid Search
    n_iter_search = 10
    gs = RandomizedSearchCV(clf,
                           param_distributions=param_grid,
                           n_iter= n_iter_search,
                           cv = cv,
                           iid= False,
                           scoring ='accuracy')
    # Train model
    gs.fit(X_train, y_train)  
    print("The best parameters are %s" % (gs.best_params_)) 
    # Predict on test set
    y_pred = gs.best_estimator_.predict(X_test)
    # Get Probability estimates
    y_prob = gs.best_estimator_.predict_proba(X_test)[:, 1]
    # -----------------------------------------------------
    print('Accuracy score: %.2f%%' %(accuracy_score(y_test, y_pred)*100))  
    print('Precision score: %.2f%%' % (precision_score(y_test, y_pred)*100))
    print('Recall score: %.2f%%' % (recall_score(y_test, y_pred)*100))
    # ----------------------------------------------------- 
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(21, 7))
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot = True, cbar = False, fmt = "d", linewidths = .5, cmap = "Blues", ax = ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted class")
    ax1.set_ylabel("Actual class")
    fig.tight_layout()
    # -----------------------------------------------------
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, lw = 2, label = 'AUC: {:.2f}'.format(auc(fpr, tpr)))
    ax2.plot([0, 1], [0, 1],
             linestyle = '--',
             color = (0.6, 0.6, 0.6),
             label = 'Random guessing')
    ax2.plot([0, 0, 1], [0, 1, 1],
             linestyle = ':',
             color = 'black', 
             label = 'Perfect performance')
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('True Positive Rate (TPR)')
    ax2.set_title('Receiver Operator Characteristic (ROC) Curve')
    ax2.legend(loc = "lower right")
    fig.tight_layout()      
    # -----------------------------------------------------
    


# # Linear SVM

# In[ ]:


from sklearn.svm import SVC
svm_linear = SVC(kernel="linear",probability=True)
param_grid = {'gamma':np.logspace(-2,2,5),
             'C':np.logspace(-2,2,5)}
run_classifier(svm_linear,param_grid,'Linear SVM')


# # DecisionTree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

param_grid = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': np.arange(1, 20, 2),
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4, 10],
              'max_features': ['auto', 'sqrt', 'log2', None]}

run_classifier(dtree, param_grid, "Decision Tree")


# # RBF SVM

# In[ ]:


svm_rbf = SVC(kernel="rbf", probability=True)

param_grid = {'gamma': np.logspace(-2, 2, 5),
              'C': np.logspace(-2, 2, 5)}

run_classifier(svm_rbf, param_grid, "RBF SVM")


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

param_grid = {'n_estimators': [100, 200],
              'max_depth': [10, 20, 100, None],
              'max_features': ['auto', 'sqrt', None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4, 10],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}

run_classifier(rf, param_grid, 'Random Forest')


#  # Navie Bayes 

# In[ ]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

param_grid = {'priors': [None]}

run_classifier(nb, param_grid, 'Naive Bayes')


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

param_grid = {'penalty': ['l2'],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

run_classifier(lr, param_grid, 'Logistic Regression')


# # MLPClassifier

# In[ ]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()

param_grid = {'hidden_layer_sizes': [(10,), (50,), (10, 10), (50, 50)],
             'activation': ['identity', 'logistic', 'tanh', 'relu'],
             'solver': ['lbfgs', 'sgd', 'adam'],
             'alpha': np.logspace(-5, 3, 5),
             'learning_rate': ['constant', 'invscaling','adaptive'],
             'max_iter': [100, 500, 1000]}

run_classifier(mlp, param_grid, 'Neural Net')


# In[ ]:




