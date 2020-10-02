#!/usr/bin/env python
# coding: utf-8

# # **Churn Prediction**

# Keeping clients is much cheaper than attracting new ones. Therefore, it is extremely important to know which clients are about to stop buying your products.
# 
# In this notebook the focus is on both predicting as accurate as possible whether a person is going to churn and on determining important factors that influence churners.
# 
# Suggestions to improve this notebook are highly appreciated!

# In[ ]:


import pandas as pd
import numpy as np
import csv
import gc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import seaborn as sns
import graphviz 
from sklearn import *
from xgboost import XGBClassifier
from xgboost import plot_tree
from datetime import date
from datetime import datetime
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
warnings.filterwarnings('ignore')


# ## Data Retrieval

# In[ ]:


churned_users = pd.read_csv("churned_users_23_12_2017.csv", sep = ",")
churned_users.columns=['user_id','orders','amount','clicks','days_member','is_unsub','was_unsub','helpdesk','helpdesk_filtered','last_month','last_2_months','last_3_months','trend','amount_weighted','amount_weighted_2','amount_weighted_3','orders_weighted','orders_weighted_2','orders_weighted_3','days_since_last_click','days_since_last_order','cashrefund?','churned']
#del churned_users['helpdesk_filtered']
del churned_users['user_id']
churned_users = churned_users.loc[churned_users['is_unsub'] == 0]
del churned_users['is_unsub']
churned_users = churned_users.reset_index(drop=True) #the indices are not contiguous after deleting some rows
#del churned_users['helpdesk']
#churned_users = pd.read_csv("../churned_users_august.csv", sep = ",")
#churned_users.columns=['orders','amount','clicks','opens','mails','helpdesk','last_month','last_2_months','last_3_months','amount_weighted','days_since_last_click','days_since_last_order','cashrefund?','churned']
#churned_users.drop(churned_users.index[:5000], inplace=True)
churned_users.head(5)


# The dataset looks like this. All the features are numerical, so this makes the preprocessing easy. If churned equals 1, this corresponds to a user that has not bought any product the last two months.

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(churned_users.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# One thing that that the Pearson Correlation plot can tell us is that there are not too many features strongly correlated with one another. This is good from a point of view of feeding these features into your learning model because this means that there isn't much redundant or superfluous data in our training set and we are happy that each feature carries with it some unique information. Here the most correlated features are last_3_months and last_2_months, which is normal since last_3_months consists of the last_2_months plus one month.

# ## Preprocessing (normalization + undersampling)

# In[ ]:


label_column = 20
churned_users.dropna(inplace=True)
churned_users = shuffle(churned_users)
samp = churned_users[churned_users.iloc[:,label_column] == 1].sample(len(churned_users)-sum(churned_users[churned_users.columns[label_column]]))
churned_users.drop((churned_users[churned_users.iloc[:,label_column] == 1]).index, inplace=True)
churned_users = pd.concat([churned_users,samp]) #adding the subsample
churned_users = shuffle(churned_users)
churned_users = churned_users.reset_index(drop=True) #the indices are not contiguous after deleting some rows
y = churned_users[churned_users.columns[label_column]]
x = churned_users
x.drop(x.columns[label_column], axis=1,inplace=True)
#X = x.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
#X.head(5)


# In[ ]:


print(len(y[y==1]))
print(len(y[y==0]))
print(len(x))
print(len(y))


# The number of elements in class 0 now equals the number of elements in class 1. In the preprocessing step, we maked the problem balanced in order to avoid trivial models.

# ## Model Selection

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, shuffle=True)
X_train.head()


# In the first phase, we simply loop over some available models in the sklearn module. The best models in this step we will use later on.

# In[ ]:


names = [ #"KNeighbors",
        #"Linear SVM",
         #"Gaussian",
         #"Decision Tree", 
         #"Random Forest",
         #"Neural Net" ,
         #"AdaBoost",
         "Naive Bayes",
         #"QDA",
         #"XGBoost",
         #"GradientBoostClassifier",
         #"ExtraTreesClassifier"
        ]

classifiers = [
    #KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=5),
    #RandomForestClassifier(),
    #MLPClassifier(alpha=1),
    #AdaBoostClassifier(n_estimators=500,learning_rate=0.75),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis(),
    XGBClassifier(),
    #GradientBoostingClassifier(n_estimators=500,max_features=0.2,max_depth=5,min_samples_leaf= 2,verbose=0),
    #ExtraTreesClassifier(n_jobs=-1,n_estimators=500,max_depth=8,min_samples_leaf=2,verbose=0)
    ]

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    print(clf.feature_importances_)
    predicted = clf.predict(X_test)
    print(clf.score(X_test, y_test)*100,"% van de users worden juist geclassifieerd door ",name, sep="")


# In[ ]:


estlist = [
    ('gb',XGBClassifier()),
    ('lr',LogisticRegression()),
    ('gd',GradientBoostingClassifier()),
    ('meta',NonNegativeLinearClassification())
]
sm = StackedClassifier(estlist)
sm.fit(X_train,y_train)
print(sm.score(X_test, y_test)*100)


# In[ ]:


plt.scatter(y_test,predicted)


# In[ ]:


def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)         # Initialize a classifier with key word arguments
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)     # NumPy interprets True and False as 1. and 0.


# In[ ]:


def baseline_predictor(X, y):
    y_preds = [0] * len(y)
    for i in range(0,len(X)):
        if(i<20270):
            y_preds[i] = 0
        else:
            y_preds[i] = 1
    return accuracy(y_preds, y)
print(baseline_predictor(X,y))


# In[ ]:


from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(XGBClassifier(), X, y, cv=10)
print(scores)
print(scores.mean())
#clf.score(X_test,y_test)
#plot_tree(clf,num_trees = 0)
#fig = plt.gcf()
#fig.set_size_inches(150, 100)
#fig.savefig('tree0.png')
#plot_tree(clf,num_trees = 1)
#fig = plt.gcf()
#fig.set_size_inches(150, 100)
#plot_tree(clf,num_trees = 2)
#fig = plt.gcf()
#fig.set_size_inches(150, 100)
#plt.show()


# ## Neural Network Model using TensorFlow and Keras

# In[ ]:


model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(21,)))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, batch_size=50, epochs=15, verbose=2, validation_split=0.2)


# In[ ]:


model.evaluate(X_test,y_test)[1] #the accuracy of the model regarding the test set


# In[ ]:


max_features = 20000
maxlen = 100
def get_model():
    embed_size = 128
    inp = Input(shape=(21, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="relu")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# In[ ]:


model = get_model()
model.fit(X_train, y_train, batch_size=50, epochs=15, verbose=2, validation_split=0.2)


# ## Ensembling / Stacking Model

# In[ ]:


# Some useful parameters which will come in handy later on
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


gb = GradientBoostingClassifier(n_estimators=500,max_features=0.2,max_depth=5,min_samples_leaf= 2,verbose=0)
xgb = XGBClassifier()
ab = AdaBoostClassifier(n_estimators=500,learning_rate=0.75)
dt = DecisionTreeClassifier(max_depth=5)

# Create our OOF train and test predictions. These base results will be used as new features
gb_oof_train, gb_oof_test = get_oof(gb, X_train.values, y_train.ravel(), X_test.values) # Extra Trees
xgb_oof_train, xgb_oof_test = get_oof(xgb,X_train.values, y_train.ravel(), X_test.values) # Random Forest
ab_oof_train, ab_oof_test = get_oof(ab, X_train.values, y_train.ravel(), X_test.values) # AdaBoost 
dt_oof_train, dt_oof_test = get_oof(dt,X_train.values, y_train.ravel(), X_test.values) # Gradient Boost


gb_features = gb.feature_importances_
xgb_features = xgb.feature_importances_
ab_features = ab.feature_importances_
dt_features = dt.feature_importances_


# In[ ]:


cols = X.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Gradient Boost feature importances': gb_features,
     'XGB feature importances': xgb_features,
      'AdaBoost feature importances': ab_features,
    'Decision Tree feature importances': dt_features
    })


# In[ ]:


# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boost Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

trace = go.Scatter(
    y = feature_dataframe['XGB feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['XGB feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'XGB Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Decision Tree feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Decision Tree feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Decision Tree Feature Importance',
    hovermode= 'closest',
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# In[ ]:


feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head()


# In[ ]:


y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')


# In[ ]:


base_predictions_train = pd.DataFrame( {'GradientBoost': gb_oof_train.ravel(),
     'XGB': xgb_oof_train.ravel(),
     'AdaBoost': ab_oof_train.ravel(),
      'DecisionTree': dt_oof_train.ravel()
    })
base_predictions_train.head()


# In[ ]:


data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')


# In[ ]:


x_train = np.concatenate(( gb_oof_train, xgb_oof_train, ab_oof_train, dt_oof_train), axis=1)
x_test = np.concatenate(( gb_oof_test, xgb_oof_test, ab_oof_test, dt_oof_test), axis=1)


# In[ ]:


gbm = XGBClassifier(
    learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 5,
 min_child_weight= 2,
 gamma=1,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
#predictions = gbm.predict(x_test)


# In[ ]:


gbm.score(x_test, y_test)*100


# In[ ]:


predictions = gbm.predict_proba(x_test)


# In[ ]:


y_test.reset_index(drop=True,inplace=True)


# In[ ]:


gbm.score(x_test[indices], y_test[indices])*100


# In[ ]:


indices = []
for 


# In[ ]:


len(res)/len(x_test)


# In[ ]:


y_test_2 = y_test.reset_index(name='index')


# In[ ]:


y_test


# In[ ]:


gbm.score(x_test[res], y_test[res])*100


# In[ ]:


from sklearn.base import ClassifierMixin
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


results = []
names = []
models = []
scoring = 'accuracy'
for est in all_estimators():
     if issubclass(est[1], ClassifierMixin):
        models.append((est[0],est[1]))

for name, model in models:
    kfold = KFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model(), X, y, cv=kfold, scoring=scoring)
    results.append(cv_results) 
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:


from keras.models import Sequential

model = Sequential()


# In[ ]:




