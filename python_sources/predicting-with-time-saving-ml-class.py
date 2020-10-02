#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ### 01/01/2020

# 
# Hello,
# This is my first kernel in kaggle. In this notebook, I will share a small AutoML class that I wrote last week. I am going to improve it for sure, but for now, I think you might also use and extend it for your own purposes. 

# Now what this class will do is:
# 1. It will **Import the data** and explore it. It will give you info about **missing values** and the **distribution** of variables. 
# 2. It has some utility functions, such as **removing variables**. 
# 3. It has a **outlier detection** function. Given a number, for example 2, it will determine the rows that has more than 2 outliers. Removing them is optional. This function uses Tukey method and is taken from https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# 4. It has **null value filling** option. For a given threshold, if a column has more than %thresh null values, it fills "None" for categoric variables, and it drops the numeric feature. For this dataset, that is suitable. Below that threshold, it fills numeric rows with similar values by performing groupby and median. For the categoric that are below the threshold, it fills with mode.
# 5. It **encodes the categoric variables** by pandas' get dummies.
# 6. **Splits the train-test sets**. Thus, all functions mentioned above are suitable with train-test option. Set use_train_set = True in above function when calling them. 
# 7. Finally, it **fits some models** with default parameters. Choses the best one. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
from collections import Counter
from sklearn.model_selection import train_test_split
import operator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


class AutoML:
    def __init__(self, target_name, data_path, extension):
        #self.numeric_nan = numeric_nan
        #self.categoric_nan = categoric_nan
        self.target_name = target_name
        self.data_path= data_path
        self.extension = extension
        
        global df
        if self.extension == 'csv':
            df = pd.read_csv(self.data_path)
        elif self.extension == 'xls':
            df = pd.read_excel(self.data_path)
        elif self.extension == 'xlsx':
            df = pd.read_excel(self.data_path)
             
            
    def explore(self, corrmap=False, pairplot=False):

        #target exploration
        target_vals = list(pd.unique(df[self.target_name]))
        target_vals.sort()
        
        
        if df[self.target_name].dtypes == "object":
            target_type = "Categoric"
        else:
            target_type = "Numeric"
            
        if target_vals == [0,1]:
            target_def = "Binary"
        else:
            target_def = "Non-binary"
        
        
        #features
        cat_variables = list(df.columns[df.dtypes == "object"])
        num_variables = list(df.columns[df.dtypes != "object"])
        
        
        #output
        print("Numeric variables: ", num_variables)
        print("Categoric variables: ", cat_variables)
        print("Target is ", target_type, " and ",target_def)
        print()
        print(df.describe())
        print()
        print(pd.DataFrame({'count': df.isnull().sum(), 'Percent':df.isnull().sum()*100/len(df)}))
        if corrmap == True:
            print(sns.heatmap(df.corr(),annot=True,cmap = 'bwr',vmin=-1, vmax=1, square=True, linewidths=0.5))
        if pairplot ==True:
            #grid = sns.PairGrid(data=df, size = 4)
            #grid = grid.map_upper(plt.scatter, color = 'darkred')
            #grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', edgecolor = 'k')
            #grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
            sns.pairplot(df,kind='reg',markers='+',diag_kind='kde')
            
    def remove_features(self, features):
        df.drop(columns=features, inplace=True)
        
    
    def detect_outliers(self,n,features,use_train_set=False):
        """
        Takes a dataframe df of features and returns a list of the indices
        corresponding to the observations containing more than n outliers according
        to the Tukey method.
        """
        global df
        global x_train
        
        if use_train_set==True:
            df_local = x_train
        else:
            df_local = df
            
        outlier_indices = []
        
        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(df_local[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(df_local[col],75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1
            
            # outlier step
            outlier_step = 1.5 * IQR
            
            # Determine a list of indices of outliers for feature col
            outlier_list_col = df_local[(df_local[col] < Q1 - outlier_step) | (df_local[col] > Q3 + outlier_step )].index
            
            # append the found outlier indices for col to the list of outlier indices 
            outlier_indices.extend(outlier_list_col)
            
        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)        
        multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
        
        if len(multiple_outliers)>0:
            print("There are indices that has more than ",n," outliers in the data.\n")
        else:
            print("There isn't any observations that has more than ",n," outliers.\n")
        return multiple_outliers   
    
 


    def drop_outliers(self,outliers_to_drop,use_train_set=False):
        global df, x_train, y_train
        if use_train_set==True:
            x_train = x_train.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
            y_train = y_train.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
        else:
            df = df.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
        
    
    
    def fillnull(self, thresh, use_train_set=False):
        global df, x_train, x_test
        
        if use_train_set==True:
            sets = [x_train,x_test]
        else:
            sets = [df]
            
        for local_df in sets:
            
            cat_variables = list(local_df.columns[local_df.dtypes == "object"])
            num_variables = list(local_df.columns[local_df.dtypes != "object"])
            #categoric
            null_df = pd.DataFrame({'Percent':local_df.isnull().sum()*100/len(local_df)})
            
            features_to_handle = np.array(null_df[null_df["Percent"]>=thresh].index.values)
            features_to_fix = np.array(null_df[(null_df["Percent"]<thresh) & (null_df["Percent"]!=0)].index.values)
            
            for feature in features_to_handle:
                if feature in cat_variables:
                    local_df[feature].fillna("None",inplace=True)
                else:
                    local_df.drop(columns=feature,inplace=True)
                    
            for feature in features_to_fix:
                if feature in cat_variables:
                    local_df[feature].fillna(local_df[feature].mode()[0],inplace=True)
                else:
                    corr_local_df = pd.DataFrame(local_df.corr())
                    top_corr = pd.DataFrame(corr_local_df[feature].abs().sort_values(ascending=False))
                    neighbor_variables = list(top_corr[1:4].index.values)
                    local_df[feature] = (local_df.groupby(neighbor_variables)[feature].transform(lambda x: x.fillna(x.median())))
                    #if it cant be filled with a similar value
                    local_df[feature].fillna(local_df[feature].median(), inplace=True)
            

            

    def encode_cat(self,use_train_set=False):
        global df, x_train, x_test
        if use_train_set==True:
            x_train = pd.get_dummies(x_train)
            x_test = pd.get_dummies(x_test)
        else:
            df = pd.get_dummies(df)
        
    def train_test(self, test_size=0.25,random_state=1):
        global x_train,y_train,x_test,y_test,target_name
        y = df[self.target_name]
        X = df.drop(columns=self.target_name)
        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
        
    def fit_model(self,use_train_set=False):
        #target exploration
        target_vals = list(pd.unique(df[self.target_name]))
        target_vals.sort()
        
        if df[self.target_name].dtypes == "object":
            target_type = "Categoric"
        else:
            target_type = "Numeric"
            
        if target_vals == [0,1]:
            target_def = "Binary"
        else:
            target_def = "Non-binary"
            
            
        #binary classification
        if (target_type=="Numeric")&(target_def=="Binary"):
            print("This is a binary classification problem.\n")
            print("Preparing your model...\n")
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
            classifiers = {
                            'Logistic Regression': LogisticRegression(),
                            'KNeighbors':KNeighborsClassifier(),
                            'SVC':SVC(),
                            'Decision Tree Classifier':DecisionTreeClassifier(max_depth=5),
                            'Random Forest Classifier':RandomForestClassifier(max_depth=5),
                            'AdaBoost Classifier':AdaBoostClassifier()
                           }
            accuracys = {}
            for c in classifiers:
                classifier = classifiers.get(c)
                if use_train_set==True:
                    classifier.fit(x_train,y_train)
                    pred = classifier.predict(x_test)
                    print("Results of ",c)
                    print()
                    print("Accuracy is: ",accuracy_score(y_test,pred))
                    print()
                    print("Confusion matrix: \n")
                    cm = confusion_matrix(y_test,pred)
                    ax= plt.subplot()
                    sns.heatmap(cm, annot=True, ax = ax,cmap='YlGnBu',fmt='g')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.title('Confusion Matrix')
                    plt.show()
                    print()
                    print(classification_report(y_test,pred))
                    print()
                    score = accuracy_score(y_test,pred)
                    accuracys[c] = score
                else:
                    y = df[self.target_name]
                    X = df.drop(columns=self.target_name)
                    classifier.fit(X,y)
            print(accuracys)
            best_model = max(accuracys.items(), key=operator.itemgetter(1))[0]
            print("Your best performing model is: ",best_model)
            print("Preparing for usage.\n")
            classifier = classifiers.get(best_model)
            classifier.fit(x_train,y_train)
            pred = classifier.predict(x_test)
            print("With score: ",accuracy_score(y_test,pred))
            print()


# Now we are going use it like the following. First, we create our AutoML object.
# 

# In[ ]:


auto1 = AutoML("Survived","/kaggle/input/titanic/train.csv","csv")


# Now I'm going to remove the Id, Ticket, Name and Cabin variables. Although these features include information, we are not going to dive into feature engineering in this topic. But know that titles can be acquired from Names column and Cabin column can be simplified to a one letter feature.

# In[ ]:


auto1.remove_features(features=["PassengerId","Ticket","Name","Cabin"])


# In[ ]:


auto1.explore(corrmap=True,pairplot=True)

Lets do the train-test split to evaluate our model later.
# In[ ]:


auto1.train_test()


# Now we will detect and drop outliers in the given columns. Note that we are going to do detect and remove outliers only in the train set. For the titanic data, we can safely do this. But for another data sets, removing outliers can affect our model negatively because test set can also include outliers. So be cautious while using it.

# In[ ]:


outliers = auto1.detect_outliers(2,features=["Pclass","Age","SibSp","Fare","Parch"],use_train_set=True)
print(outliers)
auto1.drop_outliers(outliers,use_train_set=True)


# Handle missing values and categorical features:

# In[ ]:


auto1.fillnull(thresh=50,use_train_set=True)
auto1.encode_cat(use_train_set=True)


# We are ready to fit our models.

# In[ ]:


auto1.fit_model(use_train_set = True)


# After this step, you can tune your models parameters and predict an unseen data. 
# Feel free to use this class in any of your work!
# Happy new year!

# In[ ]:




