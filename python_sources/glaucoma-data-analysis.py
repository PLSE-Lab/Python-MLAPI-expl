#!/usr/bin/env python
# coding: utf-8

# Glaucoma is a gathering of eye conditions that harm the optic nerve, the soundness of which is indispensable for acceptable vision. This harm is frequently brought about by a strangely high weight in your eye. 
# 
# Glaucoma is one of the main sources of visual impairment for individuals beyond 60 years old. It can happen at any age yet is progressively normal in more established grown-ups. 
# 
# Numerous types of glaucoma have no admonition signs. The impact is continuous to such an extent that you may not see an adjustment in vision until the condition is at a propelled arrange. 
# 
# Since vision misfortune because of glaucoma can't be recouped, it's essential to have customary eye tests that incorporate estimations of your eye pressure so a conclusion can be made in its beginning times and treated properly. On the off chance that glaucoma is perceived early, vision misfortune can be eased back or forestalled. In the event that you have the condition, you'll for the most part need treatment for a mind-blowing remainder.
# 

# ![](https://www.kaggleusercontent.com/kf/16917358/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..J3SdYQETDQfIPyjVfxzsRg.FPl9uKD_qD6aZF1a9XMJOByjUFC19WEsNJtnN_WFC9tChP0PXbEbCNuCGm1LKfEVKbCSPH77yIE4Y2MkbBCYTwC36le-U-tKxCXNkNuhYj473KKFxEnxwYr5wla1KmS4gdyz7wvJJFjU3GbeE6gr9tVgqQxqCawNxkomYcUVncY.2AX-ccvCI-T4X0ILiroMvA/__results___files/__results___11_0.png)

# In[ ]:


#library
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go


# In[ ]:


#dataset
data_load = pd.read_csv('/kaggle/input/glaucoma-dataset/GlaucomaM.csv')


# In[ ]:


data_load.head()


# In[ ]:


data_load.isnull().sum()


# In[ ]:


le = LabelEncoder()


# In[ ]:


data_load.Class = le.fit_transform(data_load.Class)


# In[ ]:


data_load['Class']


# In[ ]:


model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}   


# In[ ]:


pd.DataFrame(model_params)


# In[ ]:


scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False)
    clf.fit(data_load.drop('Class',axis='columns'), data_load.Class)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = data_load.drop('Class', axis='columns')
y = data_load.Class


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


# In[ ]:



model = SVC(C=1.0,kernel='linear')


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


model.score(X_test, y_test)


# In[ ]:


classes1 = {
    0:'Normal',
    1:'Gulcoma',
}


# In[ ]:


y_predicted = model.predict(X_test)


# In[ ]:


y_predicted


# In[ ]:


classes1[y_predicted[3]]


# In[ ]:


cm = confusion_matrix(y_test, y_predicted)
cm


# In[ ]:


fig = go.Figure(data=go.Heatmap(
                   z=cm,
                   x=['Normal','Glucoma'],
                   y=['Normal','Glucoma'],
                   hoverongaps = False))
fig.show()


# In[ ]:





# In[ ]:




