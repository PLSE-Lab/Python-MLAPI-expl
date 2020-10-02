#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

def add_features(df):
    df = df.drop(['Name', 'Ticket'], axis=1)
    df['hasSibSp'] = df['SibSp'] > 0
    df['hasParch'] = df['Parch'] > 0
    df['isChild']  = df['Age'] < 16
    #df['isElder']  = df['Age'] >= 55
    df.loc[df['Age'].isna(),'Age'] = df.groupby(['Pclass', 'SibSp'])['Age'].transform('mean')
    df.loc[df['Age'].isna(),'Age'] = df.groupby(['Pclass'])['Age'].transform('mean')
    df.loc[df['Fare'].isna(), 'Fare'] = df.groupby('Pclass').transform('mean')
    df.loc[df['Fare'] == 0, 'Fare'] = df.groupby('Pclass')['Fare'].transform('mean')
    df.loc[df['Embarked'].isna(), 'Embarked'] = df['Embarked'].value_counts().index[0]
    df.loc[:,'Cabin'] = df['Cabin'].replace('\d+', ' ', regex=True)
    #df.loc[df['Cabin'].eq(''), 'Cabin'] = 'None '
    text_cols = df.dtypes[df.dtypes.eq('object')].index
    num_cols = df.dtypes[~df.dtypes.eq('object')].index
    df.loc[:,text_cols] = df.loc[:,text_cols].fillna('')
    df.loc[:,num_cols] = df.loc[:,num_cols]
    df.loc[:,'FamilySize'] = df['SibSp'] + df['Parch']
    df.loc[df['FamilySize'] > 3, 'FamilySize'] = 4
    df.loc[:,'FamilySize'] = df.loc[:,'FamilySize'].map({0: 'Solo', 1: 'Small', 2: 'Small', 3: 'Medium', 4: 'Big'})
    #df = pd.concat([df, pd.get_dummies(df['Sex'], drop_first=True), pd.get_dummies(df['Embarked']).drop('Q', axis=1)], axis=1).drop(['Sex', 'Embarked'], axis=1).set_index('PassengerId')
    df = pd.concat([df,
                    pd.get_dummies(df['Sex']),
                    pd.get_dummies(df['Embarked']),
                    pd.get_dummies(df['FamilySize']),
                    #pd.get_dummies(pd.cut(df['Age'], 10))
                   ], axis=1) \
                .drop(['Sex',
                       'Embarked',
                       #'Age',
                       'FamilySize'], axis=1).set_index('PassengerId')
    df['possibleMother'] = (df['isChild'].eq(False) & df['hasParch'].eq(True) & df['male'].eq(0) & df['hasSibSp'].eq(True))
    return df

df = add_features(train)
test = add_features(test)

df.head(3)


# In[ ]:





# In[ ]:


#df.groupby([df['FamilySize'], 'Survived'])['Fare'].count()


# In[ ]:


import pandas_profiling as pdpf
#pdpf.ProfileReport(df)


# In[ ]:


CountVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(df['Cabin'])


# In[ ]:


from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

get_text_data = FunctionTransformer(lambda df: df[df.dtypes[df.dtypes.eq('object')].index].apply(lambda s: " ".join(s), axis=1), validate=False)
get_cabin_data = FunctionTransformer(lambda df: df[df.dtypes[df.dtypes.eq('object')].index].apply(lambda s: " ".join(s), axis=1), validate=False)
get_num_data = FunctionTransformer(lambda df: df[df.dtypes[df.dtypes != 'object'].index], validate=False)

pl = Pipeline([
    ('union', FeatureUnion([
        ('numeric_features', Pipeline([
            ('selector', get_num_data),
            ('scaler', MinMaxScaler())
        ])),
        ('cabin_feature', Pipeline([
            ('selector', get_cabin_data),
            ('vectorizer', CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
        ])),
    ])),
    #('poly', PolynomialFeatures(degree=2))
    ])

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    random_state=101)


X_train = pl.fit_transform(X_train)
X_test = pl.transform(X_test)
X = pl.transform(X)
sub_test = pl.transform(test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(sub_test)
submission = pd.DataFrame({'PassengerId':test.index,'Survived':y_pred})
submission.to_csv('LogReg.csv',index=False)


# In[ ]:


print('Train score:',lr.score(X_train, y_train))
print('Test score:',lr.score(X_test, y_test))
print('DF score:',lr.score(X, y))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nimport tensorflow as tf\nfrom keras.models     import Sequential\nfrom keras.layers     import Dense, Activation, Dropout\nfrom keras.utils      import to_categorical\nfrom keras.optimizers import SGD, Adam\nfrom keras.callbacks  import EarlyStopping\nnp.random.seed(101)\n\ndef get_keras_model(input_shape):\n    model = Sequential()\n    model.add(Dense(X_train.shape[1], activation='relu', input_shape=input_shape))\n    model.add(Dropout(.5))\n    for i in range(0,3):\n        model.add(Dense(input_shape[0], activation='relu'))\n        model.add(Dropout(.3))\n    model.add(Dense(2, activation='softmax'))\n    return model\n\npredictors  = X\ntarget      = to_categorical(y)\ninput_shape = (predictors.shape[1],)\nesm = EarlyStopping(monitor='loss', mode='min',  patience=10, verbose=1, restore_best_weights=True)\nmodel = get_keras_model(input_shape)\nmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n\nmodel.fit(predictors, target,\n          #callbacks=[esm],\n          epochs=250, batch_size=32,\n          validation_data=[X_test, to_categorical(y_test)]\n         )\n\n#from sklearn.model_selection import GridSearchCV\n#params = dict(batch_size = [16,32,64],\n#              epochs=[15, 50],\n#              \n#    )\n#cv = GridSearchCV()")


# In[ ]:


metrics = pd.DataFrame(model.history.history)             .rename(columns={'loss': 'Training Loss', 'val_loss': 'Validation Loss', 'accuracy': 'Training Accuracy', 'val_accuracy': 'Validation Accuracy'})

fig, ax = plt.subplots(2,1,figsize=(16,5), sharex=True)
sns.lineplot(data=metrics.iloc[:,[0,2]], ax=ax[0])
ax[0].set_title('Loss')
sns.lineplot(data=metrics.iloc[:,[1,3]], ax=ax[1])
ax[1].set_title('Accuracy')
plt.tight_layout()


# In[ ]:


y_pred = pd.DataFrame({'PassengerId':test.index,'Survived':model.predict_classes(sub_test)})
y_pred.to_csv('NeuralNetwork.csv', index=False)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV

param_grid = {'base_estimator__max_depth': np.arange(4,9),
              'base_estimator__min_samples_leaf': np.linspace(.05, .2, 10)}

dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=.0666666, random_state=101)
clf = BaggingClassifier(base_estimator=dt, n_estimators=200, n_jobs=-1, random_state=101, oob_score=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'Accuracy on Training Set: {accuracy_score(y_train, clf.predict(X_train)) * 100:.2f}%')
print('Log Loss on Training Set:', log_loss(y_train, clf.predict_proba(X_train)))
print(f'Accuracy on Test Set: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print('Log Loss on Holdout Set: ', log_loss(y_test, clf.predict_proba(X_test)))


# In[ ]:


submission = pd.DataFrame({'PassengerId':test.index,'Survived':clf.predict(sub_test)})
submission.to_csv(f'BaggingClassifierWithGridSearch.csv',index=False)


# In[ ]:


print(f'OOB accuracy: {clf.oob_score_ * 100:.2f}%')


# In[ ]:


#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import BaggingClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import log_loss, accuracy_score
#from sklearn.model_selection import GridSearchCV
#
#param_grid = {'base_estimator__max_depth': np.arange(4,9),
#              'base_estimator__min_samples_leaf': np.linspace(.05, .2, 10)}
#
#dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=.1, random_state=101)
#clf = GridSearchCV(BaggingClassifier(base_estimator=dt, n_estimators=100, n_jobs=-1, random_state=101), param_grid, cv=3, scoring='neg_log_loss')
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#
#print(f'Accuracy on Training Set: {accuracy_score(y_train, clf.predict(X_train)) * 100:.2f}%')
#print('Log Loss on Training Set:', log_loss(y_train, clf.predict(X_train)))
#print(f'Accuracy on Test Set: {accuracy_score(y_test, y_pred) * 100:.2f}%')
#print('Log Loss on Holdout Set: ', log_loss(y_test, y_pred))


# In[ ]:


#sns.heatmap(pd.pivot_table(pd.DataFrame(clf.cv_results_), values='mean_test_score', index='param_base_estimator__min_samples_leaf', columns='param_base_estimator__max_depth'), annot=True)


# In[ ]:


#%%time
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import StandardScaler, PolynomialFeatures
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import log_loss
#
#classifiers = [('LogReg', LogisticRegression(solver='lbfgs', max_iter=2000, n_jobs=-1)),
#               ('RndFst', RandomForestClassifier(n_estimators=100, n_jobs=-1))]
#
#for clf_name, clf in classifiers:
#    clf.fit(X_train, y_train)
#    print(clf_name, '- Holdout set score:',clf.score(X_test, y_test))
#    #print(clf_name, '- Holdout Log Loss:',log_loss(X_test, y_test))
#    print(clf_name, '- Log Loss on Training Set:', log_loss(y_train, clf.predict_proba(X_train)))
#    print(clf_name, '- Log Loss on Holdout Set: ', log_loss(y_test, clf.predict_proba(X_test)))
#
#    clf.fit(X, y)
#    print(clf_name, '- Full set score:', clf.score(X_test, y_test))
#    print(clf_name, '- Log Loss on Training Set:', log_loss(y_train, clf.predict_proba(X_train)))
#    print(clf_name, '- Log Loss on Holdout Set: ', log_loss(y_test, clf.predict_proba(X_test)))
#    
#    submission = pd.DataFrame({'PassengerId':test.index,'Survived':clf.predict(sub_test)})
#    submission.to_csv(f'{clf_name}.csv',index=False)
##print('Features Shape', )


# In[ ]:


#from sklearn.tree import DecisionTreeClassifier
#dtree = DecisionTreeClassifier()
#dtree.fit(get_num_data.fit_transform(X_train),y_train)
#y_pred = dtree.predict(X_test)
#print(classification_report(y_test,y_pred))
#export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)


# In[ ]:





# In[ ]:


#from IPython.display import Image  
#from sklearn.externals.six import StringIO  
#from sklearn.tree import export_graphviz
#import pydot 

#dot_data = StringIO()  
#export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)
#
#graph = pydot.graph_from_dot_data(dot_data.getvalue())  
#Image(graph[0].create_png())  


# In[ ]:





# In[ ]:





# In[ ]:


#lr = LogisticRegression()
#lr.fit(X, y)
#y_pred = lr.predict(test[features])
#submission = pd.DataFrame({'PassengerId':test.index,'Survived':y_pred})
#submission.to_csv('Titanic Predictions.csv',index=False)


# In[ ]:


#from IPython.display import FileLink
#FileLink('Titanic Predictions.csv')


# In[ ]:




