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


# # Build Basic Decision Tree Model

# In[ ]:


DT_Model = DecisionTreeClassifier()


# In[ ]:


DT_Model.fit(X_train,y_train)


# ## Check Accuracy (Obvioulsy Overfitting)

# In[ ]:


print (f'Train Accuracy - : {DT_Model.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {DT_Model.score(X_test,y_test):.3f}')


# # Plot Tree
# export_graphviz documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html

# In[ ]:


dot_data = export_graphviz(DT_Model, max_depth = 3,  #Limit to a Depth of 3 only
                      out_file=None, 
                      feature_names=X.columns,       #Provide X Variables Column Names 
                      class_names=['Yes','No'],          # Provide Target Variable Column Name
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
