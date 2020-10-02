#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import cufflinks as cf
cf.set_config_file(offline=True)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from vecstack import stacking
df = pd.read_csv("../input/Iris.csv")
df.sample(5)
import seaborn as sns
import matplotlib.pyplot as plt


# https://towardsdatascience.com/sweetviz-automated-eda-in-python-a97e4cabacde

# In[ ]:



get_ipython().system('pip install sweetviz')
import sweetviz as sv


# In[ ]:


iris_report = sv.analyze(df)
iris_report.show_html('iris.html')


# In[ ]:


get_ipython().system('ls')


# https://pypi.org/project/dtale/

# In[ ]:


get_ipython().system('conda install dtale -c conda-forge -y')


# In[ ]:


get_ipython().system('pip install flask_ngrok')


# In[ ]:


#https://www.youtube.com/watch?v=8Il-2HHs2Mg
import pandas as pd

import dtale
import dtale.app as dtale_app

dtale_app.USE_NGROK = True
d=dtale.show(df)
d.main_url()


# In[ ]:


dtale.instances()


# In[ ]:


dtale.get_instance(1).kill()


# In[ ]:


#https://github.com/santosjorge/cufflinks/issues/185
get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')


# In[ ]:


df.info()


# In[ ]:


df.groupby(by='Species').describe().T


# In[ ]:


# https://seaborn.pydata.org/examples/scatterplot_matrix.html
ax = sns.pairplot(df, hue="Species")


# In[ ]:


# https://seaborn.pydata.org/generated/seaborn.boxplot.html
_ = sns.boxplot(x="Species", y="PetalLengthCm", data=df)


# In[ ]:


# https://seaborn.pydata.org/generated/seaborn.boxplot.html
_ = sns.boxplot(x="Species", y="PetalWidthCm", data=df)


# In[ ]:


# https://seaborn.pydata.org/generated/seaborn.boxplot.html
_ = sns.boxplot(x="Species", y="SepalLengthCm", data=df)


# In[ ]:


# https://seaborn.pydata.org/generated/seaborn.boxplot.html
_ = sns.boxplot(x="Species", y="SepalWidthCm", data=df)


# In[ ]:


# https://seaborn.pydata.org/tutorial/axis_grids.html
g = sns.FacetGrid(df, col="Species", hue="Species")
_=g.map(sns.kdeplot, "PetalLengthCm", "PetalWidthCm", alpha=.7)
_=g.add_legend()


# In[ ]:


# https://seaborn.pydata.org/tutorial/axis_grids.html
g = sns.FacetGrid(df, col="Species", hue="Species")
_=g.map(sns.kdeplot, "SepalLengthCm", "SepalWidthCm", alpha=.7)
_=g.add_legend()


# In[ ]:


# https://seaborn.pydata.org/tutorial/axis_grids.html
g = sns.FacetGrid(df, col="Species", hue="Species")
_=g.map(sns.kdeplot, "PetalLengthCm", "SepalWidthCm", alpha=.7)
_=g.add_legend()


# In[ ]:


# https://seaborn.pydata.org/tutorial/axis_grids.html
g = sns.FacetGrid(df, col="Species", hue="Species")
_=g.map(sns.kdeplot, "PetalWidthCm", "SepalLengthCm", alpha=.7)
_=g.add_legend()


# In[ ]:


df.Species.value_counts()


# In[ ]:


df[df.columns[1:5]].iplot(kind='hist')


# In[ ]:


for s in df.Species.unique():
    df.loc[df.Species==s, df.columns[1:5]].iplot(kind='hist', title=s)


# In[ ]:


df.Species.value_counts().iplot(kind='bar')


# In[ ]:


df[df.columns[1:5]].scatter_matrix()


# In[ ]:


df[df.columns[1:5]].iplot(kind='box')


# In[ ]:


for s in df.Species.unique():
    df.loc[df.Species==s, df.columns[1:5]].iplot(kind='box', title=s)


# In[ ]:


#df['colors'] = df['Species']
#https://stackoverflow.com/questions/21131707/multiple-data-in-scatter-matrix?rq=1
#df.colors.replace({'Iris-versicolor' : '#0392cf', 'Iris-virginica' : '#7bc043', 'Iris-setosa' : '#ee4035' }, inplace=True)


# In[ ]:


# color_wheel = {'Iris-versicolor': "#0392cf", 
#                'Iris-virginica': "#7bc043", 
#                'Iris-setosa': "#ee4035"}
# colors = df["Species"].map(lambda x: color_wheel.get(x))
#https://stackoverflow.com/questions/22943894/class-labels-in-pandas-scattermatrix
#df[df.columns[1:5]].scatter_matrix(color=colors)


# In[ ]:


df.Species.replace({'Iris-versicolor' : 3, 'Iris-virginica' : 2, 'Iris-setosa' : 1 }, inplace=True)
df.head()


# In[ ]:


y = df[['Species']]
X = df.loc[:,df.columns[1:5]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


clf = RandomForestClassifier(max_depth=4, n_estimators=100, random_state=0)
clf.fit(X_train, np.ravel(y_train))
y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


X.columns


# In[ ]:


from sklearn.inspection import plot_partial_dependence
features = [2, 3, (0, 3), (1,2)]
plot_partial_dependence(clf, X, features, target=1, n_cols=4) 
# fig.set_figwidth(8)
# fig.set_figheight(15)
# fig.tight_layout()
plt.gcf().set_figwidth(8)


# In[ ]:


from sklearn.inspection import plot_partial_dependence
features = [2, 3, (0, 3), (1,2)]
plot_partial_dependence(clf, X, features, target=2, n_cols=4) 
# fig.set_figwidth(8)
# fig.set_figheight(15)
# fig.tight_layout()
plt.gcf().set_figwidth(8)


# In[ ]:


from sklearn.inspection import plot_partial_dependence
features = [2, 3, (0, 3), (1,2)]
plot_partial_dependence(clf, X, features, target=3, n_cols=4) 
# fig.set_figwidth(8)
# fig.set_figheight(15)
# fig.tight_layout()
plt.gcf().set_figwidth(8)


# In[ ]:


clf.classes_


# In[ ]:


y_test[:5]


# In[ ]:


import shap

# load JS visualization code to notebook
shap.initjs()

explainer = shap.KernelExplainer(clf.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)


# In[ ]:


# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:])


# In[ ]:


shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:])


# In[ ]:


shap.force_plot(explainer.expected_value[2], shap_values[2][0,:], X_test.iloc[0,:])


# In[ ]:


shap_values = explainer.shap_values(X_test)


# In[ ]:



shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)


# In[ ]:


shap.force_plot(explainer.expected_value[1], shap_values[1], X_test)


# In[ ]:


shap.force_plot(explainer.expected_value[2], shap_values[2], X_test)


# In[ ]:


shap.dependence_plot("PetalLengthCm", shap_values[0], X_test)


# In[ ]:


shap.dependence_plot("PetalWidthCm", shap_values[0], X_test)


# In[ ]:


shap.dependence_plot("PetalLengthCm", shap_values[1], X_test)


# In[ ]:


shap.dependence_plot("PetalWidthCm", shap_values[1], X_test)


# In[ ]:


shap.dependence_plot("PetalLengthCm", shap_values[2], X_test)


# In[ ]:


shap.dependence_plot("PetalWidthCm", shap_values[2], X_test)


# In[ ]:


shap.summary_plot(shap_values[0], X_test)


# In[ ]:


shap.summary_plot(shap_values[1], X_test)


# In[ ]:


shap.summary_plot(shap_values[2], X_test)


# In[ ]:


shap.summary_plot(shap_values, X_test, plot_type="bar")


# In[ ]:


sgd_clf = SGDClassifier(random_state=0)
sgd_clf.fit(X_train, np.ravel(y_train))
y_pred = sgd_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


log_clf = LogisticRegression(multi_class='ovr', solver='lbfgs')
log_clf.fit(X_train, np.ravel(y_train))
y_pred = log_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:


xgb_clf = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
                   n_estimators=100, max_depth=3)
xgb_clf.fit(X_train, np.ravel(y_train))
y_pred = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e
# 
# https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb

# In[ ]:


models = [
#     KNeighborsClassifier(n_neighbors=5,
#                         n_jobs=-1),
    SGDClassifier(random_state=0),
        
#     RandomForestClassifier(random_state=0, n_jobs=-1, 
#                            n_estimators=100, max_depth=3),
     RandomForestClassifier(random_state=0),    
#     XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
#                   n_estimators=100, max_depth=3)
    LogisticRegression(random_state=0,multi_class='ovr', solver='lbfgs')
]


# In[ ]:


S_train, S_test = stacking(models,                   
                           X_train, np.ravel(y_train), X_test,   
                           regression=False, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=4, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)


# In[ ]:


S_train


# In[ ]:


S_train.shape


# In[ ]:


S_test


# In[ ]:


S_test.shape


# In[ ]:


# model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 
#                       n_estimators=100, max_depth=3)
model = LogisticRegression(multi_class='ovr', solver='lbfgs')    
model = model.fit(S_train, np.ravel(y_train))
y_pred = model.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(y_test.values, y_pred))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

