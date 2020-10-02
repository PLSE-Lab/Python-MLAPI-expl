#!/usr/bin/env python
# coding: utf-8

# # Load Libraries

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


# # Load Dataset 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()


# # Divide data into Dependent and Independent Variables

# In[ ]:


#Get Target data 
y = data['target']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['target'], axis = 1)


# # Divide Data into Train and test

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[ ]:


#Export Train DataFarme
export_Train = pd.concat([X_train, y_train], axis = 1)
export_Train.to_csv('train_df.csv', index = False)


# # Build Decision Tree Model with Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


#Using max_depth, criterion will suffice for DT Models, rest all will remain constant 
parameters = {'max_depth' : (3,5,7,9,10,15,20,25)
              , 'criterion' : ('gini', 'entropy')
              , 'max_features' : ('auto', 'sqrt', 'log2')
              , 'min_samples_split' : (2,4,6)
             }


# In[ ]:


DT_grid  = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions = parameters, cv = 5, verbose = True)


# In[ ]:


DT_grid.fit(X_train,y_train)


# ## Check Accuracy (Not Overfitting anymore)

# In[ ]:


DT_grid.best_estimator_


# In[ ]:


#Re Build Model with Best Estimators
DT_Model = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=3, max_features='log2', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')

DT_Model.fit(X_train,y_train)


# In[ ]:


print (f'Train Accuracy - : {DT_Model.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {DT_Model.score(X_test,y_test):.3f}')


# # Plot Tree
# export_graphviz documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

# In[ ]:


dot_data = export_graphviz(DT_Model,   
                      out_file=None, 
                      feature_names=X.columns,       #Provide X Variables Column Names 
                      class_names=['Yes','No'],      # Provide Target Variable Column Name
                      filled=True, rounded=True,     # Controls the look of the nodes and colours it
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# # Export Decision Tree PDF

# In[ ]:


dot_data = export_graphviz(DT_Model, out_file=None, 
                      feature_names=X.columns,  
                      class_names=['Yes','No'],
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render("Heart_Diesease") 


# # END
