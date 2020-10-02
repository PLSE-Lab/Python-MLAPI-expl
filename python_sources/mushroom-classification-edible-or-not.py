#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from IPython.display import display
pd.options.display.max_columns = 500
pd.options.display.max_rows = 200

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import math


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        fname = (os.path.join(dirname, filename))
        df = pd.read_csv(fname)
# Any results you write to the current directory are saved as output.


# In[ ]:


df.describe()


# # Checking for missing Values

# In[ ]:


sum(df['stalk-root'] == '?')


# In[ ]:


df.isna().sum()


# In[ ]:


df.head()


# In[ ]:


df.columns


# # Categorical Variable Conversion - One hot encoding

# In[ ]:


cols_convert = set(df.columns) - set(['class'])
df_new = pd.concat([df,pd.get_dummies(df[cols_convert],prefix = cols_convert)],axis=1)


# # Target Variable conversion to binary

# In[ ]:


df_new['Class'] = 1 # 1 means poisonous
mask_edible = (df_new['class'] == 'e')
df_new.loc[mask_edible,'Class'] = 0


# In[ ]:


df_new


# In[ ]:


df_final = df_new.drop(columns= df.columns)


# In[ ]:


df_final


# # Correlation

# In[ ]:


df_final.corr()
correlation_with_target = df_final[df_final.columns[:]].corr()[['Class']]


# In[ ]:


correlation_with_target = correlation_with_target.reset_index()
correlation_with_target.columns = ['Variable','Correlation']


# In[ ]:


correlation_with_target


# # Modeling

# In[ ]:


y_col = ['Class']
modelling_cols = [col for col in df_final.columns if not col in y_col ]


# In[ ]:


X_train, X_test, y_train, y_test =  train_test_split(df_final, df_final[y_col], random_state=1)


# In[ ]:


y_test.Class.value_counts()


# ## Naive Bayes

# In[ ]:


nb_model = GaussianNB()


# In[ ]:


nb_model.fit(X_train[modelling_cols], np.ravel(y_train))


# In[ ]:


nb_y_predict = nb_model.predict(X_test[modelling_cols])


# ## Test Data Metrics - Accuracy and ROC (Naive Bayes)

# In[ ]:


print(accuracy_score(y_test, nb_y_predict))
print(roc_auc_score(y_test, nb_y_predict))
print(confusion_matrix(y_test, nb_y_predict))
print(classification_report(y_test,nb_y_predict))


# ## KNN (K-Nearest Neigbours)

# In[ ]:


knn_model = KNeighborsClassifier(n_neighbors=int(math.sqrt(len(X_train))))


# In[ ]:


knn_model.fit(X_train[modelling_cols], np.ravel(y_train))


# ## Test Data Metrics - Accuracy and ROC (KNN)

# In[ ]:


# By default 0.5
# y_predict = model.predict(X_test[modelling_cols])
knn_y_pred_proba = knn_model.predict_proba(X_test[modelling_cols])[:, 1]
thresh = .5
knn_y_predict = [1 if value > thresh else 0 for value in knn_y_pred_proba]


# In[ ]:


print(accuracy_score(y_test, knn_y_predict))
print(roc_auc_score(y_test, knn_y_pred_proba))
print(confusion_matrix(y_test, knn_y_predict))
print(classification_report(y_test,knn_y_predict))


# ## Logistic Regression

# In[ ]:


lr_model = LogisticRegression(random_state=1, solver='lbfgs')


# In[ ]:


lr_model.fit(X_train[modelling_cols], np.ravel(y_train))


# ## Test Data Metrics - Accuracy and ROC (Logistic Regression)

# In[ ]:


# By default 0.5
# y_predict = model.predict(X_test[modelling_cols])
lr_y_pred_proba = lr_model.predict_proba(X_test[modelling_cols])[:, 1]
thresh = .5
lr_y_predict = [1 if value > thresh else 0 for value in lr_y_pred_proba]


# In[ ]:


print(accuracy_score(y_test, lr_y_predict))
print(roc_auc_score(y_test, lr_y_pred_proba))
print(confusion_matrix(y_test, lr_y_predict))


# In[ ]:


print(classification_report(y_test,lr_y_predict))


# ## SVM - Support Vector Machine

# In[ ]:


svm_model = SVC(C=1.0, kernel='rbf', gamma = 'auto' ,random_state=1, probability= True)


# In[ ]:


svm_model.fit(X_train[modelling_cols], np.ravel(y_train))


# In[ ]:


svm_y_pred_proba = svm_model.predict_proba(X_test[modelling_cols])[:, 1]
thresh = .5
svm_y_predict = [1 if value > thresh else 0 for value in svm_y_pred_proba]


# ## Test Data Metrics - Accuracy and ROC (SVM Classifier)

# In[ ]:


print(accuracy_score(y_test, svm_y_predict))
print(roc_auc_score(y_test, svm_y_pred_proba))
print(confusion_matrix(y_test, svm_y_predict))


# In[ ]:


print(classification_report(y_test,dt_y_predict))


# ## Decision Tree

# In[ ]:


dt_model = DecisionTreeClassifier(random_state= 1)


# In[ ]:


dt_model.fit(X_train[modelling_cols], np.ravel(y_train))


# In[ ]:


dt_y_pred_proba = dt_model.predict_proba(X_test[modelling_cols])[:, 1]
thresh = .5
dt_y_predict = [1 if value > thresh else 0 for value in dt_y_pred_proba]


# ## Test Data Metrics - Accuracy and ROC (Decision Tree)

# In[ ]:


print(accuracy_score(y_test, dt_y_predict))
print(roc_auc_score(y_test, dt_y_pred_proba))
print(confusion_matrix(y_test, dt_y_predict))


# In[ ]:


print(classification_report(y_test,dt_y_predict))


# ## Gradient Boosting

# In[ ]:


gb_model = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 100, random_state=1)


# In[ ]:


gb_model.fit(X_train[modelling_cols], np.ravel(y_train))


# In[ ]:


gb_y_pred_proba = gb_model.predict_proba(X_test[modelling_cols])[:, 1]
thresh = .5
gb_y_predict = [1 if value > thresh else 0 for value in gb_y_pred_proba]


# ## Test Data Metrics - Accuracy and ROC (Gradient Boost)

# In[ ]:


print(accuracy_score(y_test, gb_y_predict))
print(roc_auc_score(y_test, gb_y_pred_proba))
print(confusion_matrix(y_test, gb_y_predict))


# In[ ]:


print(classification_report(y_test,gb_y_predict))


# ## Random Forest

# In[ ]:


model = RandomForestClassifier(n_estimators =  100, n_jobs=-1, random_state=1)


# In[ ]:


model.fit(X_train[modelling_cols], np.ravel(y_train))


# ## Training Data Metrics - Accuracy and ROC (Random Forest)

# In[ ]:


# y_predict_train = model.predict(X_train[modelling_cols])
y_pred_proba_train = model.predict_proba(X_train[modelling_cols])[:,1]
y_predict_train = [1 if value >0.5 else 0 for value in y_pred_proba_train]
accuracy_score(y_train, y_predict_train)
roc_auc_score(y_train, y_pred_proba_train)
confusion_matrix(y_train, y_predict_train)


# ## Test Data Metrics - Accuracy and ROC (Random Forest)

# In[ ]:


# By default 0.5
# y_predict = model.predict(X_test[modelling_cols])
y_pred_proba = model.predict_proba(X_test[modelling_cols])[:, 1]
thresh = .5
y_predict = [1 if value > thresh else 0 for value in y_pred_proba]


# In[ ]:


print(accuracy_score(y_test, y_predict))
print(roc_auc_score(y_test, y_pred_proba))
print(confusion_matrix(y_test, y_predict))


# In[ ]:


print(classification_report(y_test,y_predict))


# # Feature Importance

# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 12, 9


# In[ ]:


feat_importances = pd.Series(model.feature_importances_, index=modelling_cols)
feat_importances.nlargest(15).plot(kind='barh', color = 'b')
plt.title("Feature Importance Plot")
plt.xlabel("Feature Importance Score (Larger value indicates more important)")
plt.plot()


# # Feature Importance with Correlation Plot - Diverging Bar Plot

# In[ ]:


correlation_with_target


# In[ ]:


df_feature_imp = feat_importances.to_frame()
df_feature_imp = df_feature_imp.reset_index()
df_feature_imp.columns = ['Variable', 'Feature_Importance']
df_featuresImp_correlation = df_feature_imp.merge(correlation_with_target, how ='left')
df_featuresImp_correlation['Feature_Importance_New'] = np.where(df_featuresImp_correlation['Correlation']<0,-1*df_featuresImp_correlation['Feature_Importance'],df_featuresImp_correlation['Feature_Importance'])


# In[ ]:


df_featuresImp_correlation


# In[ ]:


def diverging_bar_plot(df,variable_col, f_imp_col, mod_f_imp_col, corr_col, n_features):
    # Prepare Data
    print([x for x in df])
    #x = df.loc[:, ['mpg']]
    #df['mpg_z'] = (x - x.mean())/x.std()
    
    df['colors'] = ['red' if x[corr_col] < 0 else 'green' for i,x in df.iterrows()]
    df.sort_values(f_imp_col, inplace=True)
    df = df.reset_index(drop=True)
    #df.reindex(df[mod_f_imp_col].sort_values().index)
    #SUBSET
    df = df[-1*n_features:]
    # Draw plot
    plt.figure(figsize=(14,10), dpi= 80)
    plt.hlines(y=df.index, xmin=0, xmax=df[mod_f_imp_col], color=df.colors, alpha=0.4, linewidth=5)
    
    # Decorations
    plt.gca().set(ylabel='$Feature$', xlabel='$Feature Importance$')
    plt.yticks(df.index, df[variable_col], fontsize=12)
    plt.title('Feature Importance-Correlation Plot', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    #plt.savefig(plot_dir + "feat_importances_correlation_plot.png", bbox_inches='tight')


# In[ ]:


fig = diverging_bar_plot(df_featuresImp_correlation,'Variable','Feature_Importance','Feature_Importance_New','Correlation', 30)


# # Inference

# ## Having odour (more) suggests that the mushroom is not poisonous (low) and vice versa, lets validate it with some quick EDA.

# In[ ]:


df_final[['odor_n','Class']].describe()
odor_check = pd.crosstab(df_final.odor_n, df_final.Class)
print(odor_check)
print("Around " + str(odor_check[0][1]/(odor_check[0][0] + odor_check[0][1])) +" percent of odorless mushroom are poisonous")
print("Around " + str(odor_check[1][0]/(odor_check[1][0] + odor_check[1][1])) +" percent of mushrooms with Odor are edible")
print("Combined %age " + str((odor_check[0][1] + odor_check[1][0])/(odor_check.values.sum())))
print("This proves our hypothesis true.")

