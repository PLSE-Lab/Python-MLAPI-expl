#!/usr/bin/env python
# coding: utf-8

# ## Diagnose and Predict Heart Disease
# ![](https://cld.activebeat.com/image/upload/t_tn,f_auto,q_auto,$h_360,$w_580/ab/2016/06/shutterstock_259237739-580x360.jpg)
# 
# Target:
# 1. Fast Exploration 
# 2. Find the key features
# 3. Modeling an Predicting
# 
# 

# In[ ]:


## It is cleaned data ,which is normally not gonna happen in real world. 
## But thanks to UCI, we can skip the data-cleaning job in this report.

# Basic Part
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

# Modeling Part
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz 
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import eli5 
from eli5.sklearn import PermutationImportance
import shap 
from pdpbox import pdp, info_plots 

# Set a seed for tracing back and reproducing
np.random.seed(101) #ensure reproducibility


# In[ ]:


df = pd.read_csv('../input/heart.csv')
df.head()


# As what we see, the data is very clean and transformatted.
# According to the describe, it means as follow:
# 1. age: age in years
# 2. sex: (1 = male; 0 = female)
# 3. cp: chest pain type
# 4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 5. chol: serum cholestoral in mg/dl
# 6. fbs:(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7. restecg: resting electrocardiographic results
# 8. thalach: maximum heart rate achieved
# 9. exang: exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak: ST depression induced by exercise relative to rest
# 11. slope: the slope of the peak exercise ST segment
# 12. ca: number of major vessels (0-3) colored by flourosopy
# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. target: 1 or 0
# 
# Now we just need to check how many unique value in each observation and continue transformatting based on our demand.

# 

# In[ ]:


# or, a fast and easy way
df.describe()


# In[ ]:


# Age Distribution
sns.violinplot(df.age,palette = 'Set2',bw = .1, cut =1)
plt.title('Age Distribution')


# In[ ]:


# Chest Pain Type Distribution
sns.countplot(x = 'cp', data = df)
plt.title('Chest Pain Type Distribution')


# In[ ]:


# if you want to quickly take a glance of regression relationship, it is a fast way
## sns.pairplot(hue = 'target',data = df)


# In[ ]:


df = pd.get_dummies(df,drop_first = True)


# In[ ]:


y = df.target
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', 1), y, test_size = .2, random_state=101)
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)


# In[ ]:


estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'Neg'
y_train_str[y_train_str == '1'] = 'Pos'
y_train_str = y_train_str.values


# ### What happened in machine but out of our sight ?

# In[ ]:


export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)

from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')


# In[ ]:


# The very manually way: (but it is good to understand the concept and the function)
def plot_feature_importances(n):
    n_features = X_train.shape[1]
    plt.figure(figsize = (10,10))
    plt.barh(range(n_features), n.feature_importances_, align='center',color = 'm',alpha =0.6)
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


# In[ ]:


plot_feature_importances(model)


# In[ ]:


# The faster way:
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


y_predict = model.predict(X_test)
y_pred_quant = model.predict_proba(X_test)[:, 1]
y_pred_bin = model.predict(X_test)


# ### Key Point: Sensitivity and Specificity 
# ![](https://www.researchgate.net/profile/Diana_Carvajal5/publication/49650721/figure/fig1/AS:277352501268483@1443137397711/Calculation-of-sensitivity-specificity-and-positive-and-negative-predictive.png)
# ![](https://i2.wp.com/emcrit.org/wp-content/uploads/2017/06/12.png)

# In[ ]:


confusion_matrix = confusion_matrix(y_test, y_pred_bin)

total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)


# ### Key Point: ROC Graph
# ![](https://upload.wikimedia.org/wikipedia/commons/3/36/ROC_space-2.png)

# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='r')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for RandomForest classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[ ]:


auc(fpr, tpr)
# See, it works quite well.


# ### Note:
# ROC works well in binary classification and the data is not asysmmetric. 
# The "credit card fraud detection" is inapproprite due to the extremely skewed distribution. It looks like
# ![](http://nycdatascience.com/blog/wp-content/uploads/2018/11/pic1.png)
# Even you do nothing the ROC performance still looks perfect, even higher than 95% but make non-sense. We know in real world only a few people will defaut or cheat. If that is not true, most of the banks already bankrupted and we had to suffer extremely high interest.
# 
# Be sure to check the distribution before choosing models

# In[ ]:


sns.countplot(x = 'target',data = df)


# Now Let's dig deeper. The follows are the important features we knew from above. 
# I'd like to explore each influence on target.
# 
# * ca: number of major vessels (0-3) colored by flourosopy
# * oldpeak: ST depression induced by exercise relative to rest
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# * thalach: maximum heart rate achieved
# 

# In[ ]:


features = df.columns.values.tolist()
features.remove('target')
feat_name = 'ca'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features = features, feature = feat_name)
pdp.pdp_plot(pdp_dist, feat_name)


# In[ ]:


# how about the maximum heart beat?
feat_name = 'thalach'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features = features, feature = feat_name)
pdp.pdp_plot(pdp_dist, feat_name)


# The higher max heart beat, the higher risk. 
# 
# 

# 

# In[ ]:


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)


# Let's try to predict any individual patient and see how the different variables are affecting their outcomes.

# In[ ]:


def heart_disease_risk_predict(model, patient_Id):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_Id)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_Id)


# In[ ]:


p1 = X_test.iloc[1,:].astype(float)
heart_disease_risk_predict(model, p1)


# In[ ]:


p5 = X_test.iloc[5,:].astype(float)
heart_disease_risk_predict(model, p5)


# In[ ]:


p10 = X_test.iloc[10,:].astype(float)
heart_disease_risk_predict(model, p10)


# There is a marvelous function which shows the predictions andfactors for amounts of  patients all together.

# In[ ]:



shap_values = explainer.shap_values(X_train.iloc[20:50])
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test.iloc[20:50])


# ### The End
# 
# The dataset might be a little outdated and small but it is very good source to learn and practice.
# The more important thing is to understand concept and to know how.
# 
# Data science is a life journey. It is awesome to learn and try things new everyday.
