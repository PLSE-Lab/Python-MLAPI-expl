#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


# In[ ]:


df.head()


# In[ ]:


sns.countplot(data=df,x='sex',)


# Attribute Information: 
# > 1. age 
# > 2. sex 
# > 3. chest pain type (4 values) 
# > 4. resting blood pressure 
# > 5. serum cholestoral in mg/dl 
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved 
# > 9. exercise induced angina 
# > 10. oldpeak = ST depression induced by exercise relative to rest 
# > 11. the slope of the peak exercise ST segment 
# > 12. number of major vessels (0-3) colored by flourosopy 
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## EDA

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.distplot(df[(df['target']==1) & (df['sex']==1)]['age'],label='Male with heart disease',color='#3498DB',bins=20)
plt.legend()
plt.subplot(2,2,3)
sns.distplot(df[(df['target']==0) & (df['sex']==1)]['age'],label='Male without heart disease',color='#3498DB',bins=20)
plt.legend()
plt.subplot(2,2,2)
sns.distplot(df[(df['target']==1) & (df['sex']==0)]['age'],label='Female with heart disease',color='green',bins=20)
plt.subplot(2,2,4)
sns.distplot(df[(df['target']==0) & (df['sex']==0)]['age'],label='Female without heart disease',color='green',bins=20)
plt.legend()


# In[ ]:


sns.countplot(data=df, x='cp', hue = 'target', palette= 'Set2')
plt.xlabel('Chest pain in people with and without heart disease')
plt.legend( loc='upper right', labels=['Without heart disease', 'With heart disease'])


# In[ ]:


sns.pairplot(df,hue='target',palette='Set1')


# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(df[df['target']==1].drop('target',axis=1).corr(),cmap='Blues',annot=True)
plt.title('People with heart disease correlation heatmap')


# In[ ]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='target', y='thalach', hue="slope")


# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop('target',axis=1)
y = df['target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree = DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train,y_train)


# ## Prediction and Evaluation 

# In[ ]:


# Defining a function to store evaluation metrics
def evaluate(prediction,y_test): 
    result = classification_report(y_test,prediction,output_dict=True)
    f1 = result['1']['f1-score']
    accuracy = result['accuracy']
    performance_data= {'f1-score':round(f1, 2),
                      'accuracy':round(accuracy, 2)}
    return performance_data


# In[ ]:


dt_prediction = dtree.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score


# In[ ]:


print(classification_report(y_test,dt_prediction))

dtree_pr= evaluate(dt_prediction,y_test)
dtree_pr


# ## Tree Visualization

# In[ ]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 


# In[ ]:


features = list(df.columns[:-1])
features


# In[ ]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)


# ## Prediction and Evaluation 

# In[ ]:


rf_prediction = rf.predict(X_test)


# In[ ]:


print(classification_report(y_test,rf_prediction))
rf_pr = evaluate(rf_prediction,y_test)
rf_pr


# Random Forest performed better that desicion tree and 15 points mislabeled.
# But what if we tuning the Random Forest parameters?  

# ## Randomized Search
# Instead of using grid search which takes for ever to produce the best hyperparameters, we can use Random Search where uses random combinations of hyperparameters in order to find the best ones.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


param_rand = {'n_estimators': np.arange(100,800,100),
              'max_features' : ['auto', 'sqrt'],
              'min_samples_leaf': [1,2,4],
              'min_samples_split': [2, 5, 10],
              'max_depth' : np.arange(10,100,10),
              'max_leaf_nodes': np.arange(2,5,10),
             }


# In[ ]:


rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = param_rand, n_iter = 100,
                               cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[ ]:


# takes so much time, you can see the rsults for best parameters
#rf_random.fit(X_train,y_train)


# In[ ]:


rf_randomized = RandomForestClassifier(n_estimators= 600,min_samples_split= 5, min_samples_leaf = 1,
                                       max_leaf_nodes= 2,max_features= 'auto',max_depth= 70)
rf_randomized.fit(X_train,y_train)


# In[ ]:


rf_rand_prediction = rf_randomized.predict(X_test)


# In[ ]:


print(classification_report(y_test,rf_rand_prediction))
rf_rand_pr = evaluate(rf_rand_prediction,y_test)
rf_rand_pr


# ## Support Vector Machine (SVM)

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model = SVC()


# In[ ]:


model.fit(X_train,y_train)


# ## Prediction and Evaluation

# In[ ]:


svm_prediction = model.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,svm_prediction))
svm_pr = evaluate(svm_prediction,y_test)
svm_pr


# ## Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV
svm_param_grid= {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001,0.00001],
           'kernel': ['rbf','sigmoid']}


# In[ ]:


svm_grid = GridSearchCV(SVC(),svm_param_grid,refit=True,verbose=2)
svm_grid.fit(X_train,y_train)


# In[ ]:


svm_grid.best_params_


# In[ ]:


svm_grid_prediction = svm_grid.predict(X_test)


# In[ ]:


print(classification_report(y_test,svm_grid_prediction))
svm_grid_pr = evaluate(svm_grid_prediction,y_test)
svm_grid_pr


# ## PCA (dimension reduction)

# In[ ]:


## PCA (dimension reduction)

# Scale Data
from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
scale.fit(df.drop('target',axis=1))
scaled_data = scale.transform(df.drop('target',axis=1))


# ## 2D Visualization

# In[ ]:


# 2 compoemts
from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)

pca2.fit(scaled_data)

transformed_pca2 = pca2.transform(scaled_data)

transformed_pca2.shape


# In[ ]:


transformed_df_2 = pd.DataFrame(transformed_pca2,columns=['component1', 'component2'])
transformed_df_2['target'] = df['target']


# In[ ]:


sns.scatterplot(x=transformed_df_2['component1'],y=transformed_df_2['component2'],hue=transformed_df_2['target'])


# ## 3D Visualization

# In[ ]:


# 3 compoemts
pca3 = PCA(n_components=3)

pca3.fit(scaled_data)

transformed_pca = pca3.transform(scaled_data)

transformed_pca.shape


# In[ ]:


import plotly.io as pio
import plotly.express as px
transformed_df = pd.DataFrame(transformed_pca,columns=['component1', 'component2','component3'])
transformed_df['target'] = df['target']
transformed_df.head()

fig = px.scatter_3d(df, x=transformed_df['component1'], y=transformed_df['component2'], 
                    z=transformed_df['component3'],color='target')
fig.show(renderer='kaggle')


# In[ ]:


X2 = transformed_df.drop('target',axis=1)
y2= transformed_df['target']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3,random_state=101)


# In[ ]:


from sklearn.svm import SVC
model2 = SVC(C= 1000, gamma= 0.00001, kernel= 'rbf')
model2.fit(X_train2,y_train2)


# ## Evaluate model

# In[ ]:


prediction2 = model2.predict(X_test2)


# In[ ]:


print(classification_report(prediction2,y_test2))
svm_pca3_pr = evaluate(prediction2,y_test2)
svm_pca3_pr


# ## K Nearest Neighbors

# In[ ]:


df_scaled = pd.DataFrame(scaled_data,columns=df.drop('target',axis=1).columns)
df_scaled.head()


# In[ ]:


X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test = train_test_split(scaled_data,df['target'],
                                                    test_size=0.30)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)


# In[ ]:


knn.fit(X_scaled_train,y_scaled_train)


# In[ ]:


knn_prediction = knn.predict(X_scaled_test)


# In[ ]:


print(classification_report(knn_prediction,y_scaled_test))
knn1_pr = evaluate(knn_prediction,y_scaled_test)
knn1_pr


# ## Comparison

# In[ ]:


prediction_data={'Model':['Decision Tree',
                          'Random Forest',
                          'Random Forest using Randomized Search',
                          'SVM',
                          'SVM using Grid Search',
                          'SVM after PCA',
                          'KNN'
                         ],
                   'F1-score':[dtree_pr['f1-score'],
                              rf_pr['f1-score'],
                              rf_rand_pr['f1-score'],
                              svm_pr['f1-score'],
                              svm_grid_pr['f1-score'],
                              svm_pca3_pr['f1-score'],
                              knn1_pr['f1-score']],
                 
                   'Accuracy':[dtree_pr['accuracy'],
                              rf_pr['accuracy'],
                              rf_rand_pr['accuracy'],
                              svm_pr['accuracy'],
                              svm_grid_pr['accuracy'],
                              svm_pca3_pr['accuracy'],
                              knn1_pr['accuracy']]
                    }
 
# Create DataFrame
prediction_table = pd.DataFrame(prediction_data)
prediction_table


# In[ ]:


import plotly.express as px
fig = px.bar(x=prediction_table['Model'], y=prediction_table['F1-score'])
fig.update_layout( title={
        'text': "Comparison of defferent ML models",
        'y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
    xaxis_title=" ML models",
    yaxis_title="F1-score")

fig.show(renderer='kaggle')


# ## Sources:
# https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
# 
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# 
# https://plot.ly/python/bar-charts/
