#!/usr/bin/env python
# coding: utf-8

# __Kevin.F__  
# _17/03/2019_

# # Predicting Heart Disease

# ## Content

# <ul>
#   <li style="color: #539DC2;"><span style="color: #539DC2">Introduction</li>
#   <li style="color: #539DC2;"><span style="color: #539DC2">Exploratory Data Analysis (EDA)</li>
#   <li style="color: #539DC2;"><span style="color: #539DC2">Model</li>
# </ul> 

# <h1 style="color: #539DC2;">Introduction</h1>

# In this Notebook we will predict if a person present heart disease or not.
# We will first perform exploratory data analysis on the dataset and identify
# relationship between heart disease and the others variables.
# We will finish by using ML algorithms to make future prediction on unlabeled dataset.

# #### Loading packages

# In[ ]:


# Data manipulation
import numpy as np
import pandas as pd

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')

# command for work offline
plotly.offline.init_notebook_mode(connected=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "df = pd.read_csv('../input/heart.csv')")


# ##### Display 5 rows randomly

# In[ ]:


df.sample(5)


# In[ ]:


df.columns


# #### Variables :

# <ol>
#   <li>age</li>
#   <li>sex</li>
#   <li>chest pain type (4 values)</li>
#   <li>resting blood pressure</li>
#   <li>serum cholestoral in mg/dl</li>
#   <li>fasting blood sugar > 120 mg/dl</li>
#   <li>resting electrocardiographic results (values 0,1,2)</li>
#   <li>maximum heart rate achieved</li>
#   <li>exercise induced angina</li>
#   <li>oldpeak = ST depression induced by exercise relative to rest</li>
#   <li>the slope of the peak exercise ST segment</li>
#   <li>number of major vessels (0-3) colored by flourosopy</li>
#   <li>thal: 3 = normal; 6 = fixed defect; 7 = reversable defect</li>   
# </ol> 

# In[ ]:


df.info()


# In[ ]:


# Basic descriptive statistics for each column
df.describe()


# In[ ]:


df.shape


# In[ ]:


def missing_data(data):
    null_columns = data.columns[data.isnull().any()]
    return data[null_columns].isnull().sum()

missing_data(df)


# ##### There are no missing values in the dataset

# <br>

# <h1 style="color: #539DC2;">EDA</h1>

# In[ ]:


# Correlation heatmap beetween the columns
plt.rcParams['figure.figsize']=(35,16)
hm=sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap='Blues')
hm.set_title(label='Heatmap of dataset', fontsize=20)
hm;


# In[ ]:


sns.pairplot(df[['age','trestbps', 'cp', 'thalach','chol','target']],hue='target',
             palette = sns.color_palette("GnBu_d"), size=2.5)


# ### Target analysis

# In[ ]:


sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=[7, 5])
sns.countplot(x = "target", data = df)


# In[ ]:


df['target'].value_counts()
# There is approximately the same number of people sick of hearts as of non-sick 


# In[ ]:


fig, ax = plt.subplots(figsize=[7, 5])
palette = sns.color_palette("RdBu", n_colors=2)
sns.countplot(x = "target", hue = "sex", data = df, palette = palette)


# ### Age

# In[ ]:


df['age'].describe()


# In[ ]:


df['age'].value_counts()[ : 10]


# In[ ]:


fig, ax = plt.subplots(figsize=[14, 5])
sns.countplot(x = "age", data = df)


# In[ ]:


fig, axe = plt.subplots(figsize = [7, 5])
sns.distplot(df['age'], color = 'r');


# In[ ]:


trace = go.Histogram(x = df['age'], name = 'age', marker=dict(color='darkcyan'))

layout = go.Layout(
    title="Histogram Frequency Counts of Age"
)


fig = go.Figure(data=go.Data([trace]), layout=layout)
plotly.offline.iplot(fig, filename='histogram-freq-counts of ')


# In[ ]:


Adults = df[(df['age'] >= 29) & (df['age'] <= 33)]
Middle_Age = df[(df['age'] > 33) & (df['age'] <= 40)]
Senior = df[(df['age'] > 40) & (df['age'] <= 66)]
Retired = df[df['age'] > 66]


# In[ ]:


x_ = ['Adults', 'Middle_Age', 'Senior', 'Retired']
y_ = [len(Adults), len(Middle_Age), len(Senior), len(Retired)]


# In[ ]:


trace = go.Bar(
    x=x_,
    y=y_,
    textposition = 'auto',
    name='target 0',
    marker=dict(
        color='rgba(255, 135, 141,0.7)',
        line=dict(
            color='rgba(255, 135, 141,1)',
            width=1.5),
        ),
    opacity=1
)


data = [trace]

plotly.offline.iplot(data, filename='bar-chart')


# In[ ]:


colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

trace = go.Pie(labels=x_, values=y_,
               hoverinfo='label+percent',
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

plotly.offline.iplot([trace], filename='pie-chart')


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))
women = df[df['sex'] == 0]
men = df[df['sex'] == 1]

ax = sns.distplot(women[women['target'] == 1].age, bins=18, label = 'sick', ax = axes[0], kde =False, color="green")
ax = sns.distplot(women[women['target'] == 0].age, bins=40, label = 'not_sick', ax = axes[0], kde =False, color="red")
ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['target']==1].age, bins=18, label = 'sick', ax = axes[1], kde = False, color="green")
ax = sns.distplot(men[men['target']==0].age, bins=40, label = 'not_sick', ax = axes[1], kde = False, color="red")
ax.legend()
ax.set_title('Male');


# ##### For both Female and Male peoples are more likely to be sick if they are older

# In[ ]:


df.groupby('target')['age'].mean()


# ### Sex analysis

# In[ ]:


df['sex'].value_counts()/len(df)


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.countplot(x = "sex", data = df, palette = sns.cubehelix_palette(8))


# In[ ]:


# We will select randomly the same number of Male than Female and plot the distribution of sick people in function of their sex
nb_0 = df.loc[df['sex'] == 0, ['sex']].count()[0]
nb_1 = df.loc[df['sex'] == 1, ['sex']].count()[0]

print('Male   : ', nb_1)
print('Female : ', nb_0)


# In[ ]:


nb_0 = df.loc[df['sex'] == 0, ['sex']].count()[0]
nb_1 = df.loc[df['sex'] == 1, ['sex']].count()[0]

print('Male   : ', nb_1)
print('Female : ', nb_0)


# In[ ]:


df0 = df[df['sex'] == 0].sample(nb_0)
df1 = df[df['sex'] == 1].sample(nb_0)


# In[ ]:


print(df0.shape)
print(df1.shape)


# In[ ]:


# We concatenate df1 and df2
dfBis = pd.concat([df0,df1])


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.countplot(x = "target", hue = "sex", data = dfBis, palette = sns.cubehelix_palette(8))


# ##### We can see that female are more likely to be sick than male

# ### Cholesterol analysis

# In[ ]:


df['chol'].describe()


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.distplot(df['chol'], color = 'b');


# In[ ]:


ax, figure = plt.subplots(figsize = [9,5])
sns.distplot(df[df['target'] == 0].chol, label = "not sick", color = 'r');
sns.distplot(df[df['target'] == 1].chol, label = "sick", color = 'b');
ax.legend()


# In[ ]:


df[df['target'] == 0]['chol'].describe()


# In[ ]:


df[df['target'] == 1]['chol'].describe()


# In[ ]:


trace0 = go.Box(
    x=df[df['target'] == 0].target,
    y=df['chol'],
    marker=dict(
        color='#FF851B'
    ),
    name='not_sick'
)

trace1 = go.Box(
    x=df[df['target'] == 1].target,
    y=df['chol'],
    marker=dict(
        color='#FF4136'
    ),
    name = 'sick'
)


plotly.offline.iplot([trace0, trace1], filename='pie-chart')


# In[ ]:


palette2 = sns.color_palette("RdBu", n_colors=2)
fig, ax2 = plt.subplots(figsize=[5, 5])
ax = sns.stripplot(x = "target", y = "chol", data = df, jitter=True, linewidth=1, palette = palette2);


# In[ ]:


ax, figure = plt.subplots(figsize = [7, 5])
sns.violinplot(x = "sex", y = "chol", hue="target", data = df, palette = "muted", split=True)


# In[ ]:


ax, figure = plt.subplots(figsize = [9,5])
sns.regplot(x="age", y="chol", data=df);


# In[ ]:


trace = go.Scatter(
    x = df['age'],
    y = df['chol'],
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgba(255, 182, 193, .9)',
        line = dict(
            width = 2,
        )
    )
)

data = [trace]

# Plot and embed in ipython notebook!
plotly.offline.iplot(data, filename='scatter')


# In[ ]:


df.groupby('age')['chol'].mean()


# In[ ]:


ax, figure = plt.subplots(figsize = [12,5])
sns.pointplot(x="age", y="chol", data=df, color = "#feda6a")


# ### Cp (Chest pain type) analysis

# In[ ]:


df['cp'].value_counts()/len(df)


# In[ ]:


ax, figure = plt.subplots(figsize = [7, 5])
palette2 = sns.color_palette("GnBu_d")
sns.countplot(x = "cp", data = df, palette = palette2)


# In[ ]:


ax, figure = plt.subplots(figsize = [7, 5])
palette2 = sns.color_palette("GnBu_d")
sns.countplot(x = "cp", hue = 'target', data = df, palette = palette2)


# ##### We can see that people with chest pain are more likely to be sick

# In[ ]:


fig, ax = plt.subplots(figsize=[7, 5])
sns.pointplot(x = "cp", y = "chol", hue = "target", data = df)


# In[ ]:


palette2 = sns.color_palette("RdBu", n_colors=3)
fig, ax2 = plt.subplots(figsize=[7, 5])
ax = sns.stripplot(x = "cp", y = "chol", data = df, jitter=True, linewidth=1, palette = palette2);


# In[ ]:


fig, ax = plt.subplots(figsize=[7, 5])
sns.pointplot(x = "cp", y = "chol", hue = "sex", data = df)


# In[ ]:


ax, figure = plt.subplots(figsize = [12,7])
sns.boxplot(x = "sex", y = "age", hue = "cp", data = df)


# ### Trestbps (resting blood pressure) analysis

# In[ ]:


ax1, figure = plt.subplots(figsize = [7, 5])
ax1 = sns.distplot(df[df['target'] == 0].trestbps, color = 'yellow')


# In[ ]:


ax2, figure = plt.subplots(figsize = [7, 5])
ax2 = sns.distplot(df[df['target'] == 1].trestbps, color = 'green')


# In[ ]:


ax, figure = plt.subplots(figsize = [7, 5])
sns.boxplot(x = 'target', y = 'trestbps', data = df)


# In[ ]:


ax, figure = plt.subplots(figsize = [7, 5])
sns.violinplot(x = "target", y = "trestbps", hue="sex", data = df, palette = "muted", split=True)


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.boxplot(x = 'sex', y = 'trestbps', data = df)


# ### Fbs (Fasting blood sugar) analysis

# In[ ]:


df['fbs'].value_counts()/len(df)


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.countplot(x = "fbs", hue = 'target', data = df, palette = sns.color_palette("cubehelix", 8))


# ### Age/fbs

# In[ ]:


ax, figure = plt.subplots(figsize = [15,7])
sns.countplot(x = "age", hue = 'fbs', data = df, palette = sns.cubehelix_palette(8, start=.5, rot=-.75))


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.violinplot(x = "fbs", y = "age", data = df[df['fbs'] == 1], palette = "Set2", split=True)


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.stripplot(x = "fbs", y = "age", data = df)


# In[ ]:


ax, figure = plt.subplots(figsize = [9,5])
sns.regplot(x="age", y="thalach", data=df, color = "red");


# In[ ]:


sns.jointplot(x="age", y="thalach", data=df)


# ### Thalach (maximum heart rate achieved)

# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.distplot(df['thalach'], color = "pink")


# ### Thal analysis

# In[ ]:


df['thal'].value_counts()


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.countplot(x = "thal", data = df)


# In[ ]:


ax, figure = plt.subplots(figsize = [7,5])
sns.countplot(x = "thal", hue = "target", data = df)


# ##### We can see that most people with a thal of 2 are sick

# <h1 style="color: #539DC2;">Model</h1>

# In[ ]:


cols = df.shape[1]


# In[ ]:


X = df.iloc[:, : cols - 1].values
y = df.iloc[:, cols - 1].values


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_ = scaler.fit_transform(X)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# ### Logistic Regression

# In[ ]:


lrc = LogisticRegression()
lrc.fit(X_train, y_train)
print("Logistics Regression accurary - training: ", lrc.score(X_train,y_train))
print("Logistics Regression accurary - test: ", lrc.score(X_test,y_test))


# In[ ]:


y_scores_lr = lrc.decision_function(X_test)
y_score_list = list(zip(y_test[0:20], y_scores_lr[0:20]))

# show the decision_function scores for first 20 instances
y_score_list


# In[ ]:


y_proba_lr = lrc.predict_proba(X_test)
y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))

# show the probability of positive class for first 20 instances
y_proba_list


# In[ ]:


precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.figure(figsize = [12,7])
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()


# In[ ]:


y_score_lr = lrc.decision_function(X_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure(figsize = [12,7])
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve ', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()


# In[ ]:


cm_lrc = confusion_matrix(y_test,lrc.predict(X_test))


f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_lrc,annot = True, linewidths=.5, cmap='Spectral')
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()

# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall) 
print("precision_score: ", precision_score(y_test,lrc.predict(X_test)))
print("recall_score: ", recall_score(y_test,lrc.predict(X_test)))
print("f1_score: ",f1_score(y_test,lrc.predict(X_test)))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'param_grid = {\'C\': [1,0.01,0.1,10,100],\n              \'penalty\' : ["l1", "l2"],\n              \'class_weight\' : [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]\n          }\n\nlr = LogisticRegression()\n\nlr_cv = GridSearchCV(\n    estimator = LogisticRegression(random_state=12,solver="liblinear"),\n    param_grid = param_grid, \n     scoring=\'roc_auc\',\n    cv = 5\n   )\n\nlr_cv.fit(X_train, y_train)\n\nprint("tuned hyperparameters :(best parameters) ",lr_cv.best_params_)\nprint("accuracy on test set:",lr_cv.best_score_)\nprint(\'accuracy on training set : \',lr_cv.score(X_train, y_train))\nprint(\'accuracy on test set : \',lr_cv.score(X_test, y_test))')


# ### Linear SVM

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ngrid = {\'C\':[0.01, 0.1, 1, 10, 100]} \nsvm = LinearSVC(max_iter = 10000).fit(X_train, y_train)\nsvm_cv = GridSearchCV(svm, grid, cv = 10, n_jobs = -1)\nsvm_cv.fit(X_train, y_train)\n\nprint("tuned hyperparameters :(best parameters) ",svm_cv.best_params_)\nprint("accuracy on test set:",svm_cv.best_score_)\nprint(\'accuracy on training set : \',svm.score(X_train, y_train))')


# In[ ]:


from matplotlib import cm


plt.figure(figsize = [7,15])
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
for c in [0.01, 0.1, 1, 10, 100]:
    svm = LinearSVC(max_iter = 10000).fit(X_train, y_train)
    y_score_svm = svm.decision_function(X_test)
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    accuracy_svm = svm.score(X_test, y_test)
    print("C = {:.2f}  accuracy = {:.2f}   AUC = {:.2f}".format(c, accuracy_svm, 
                                                                    roc_auc_svm))
    plt.plot(fpr_svm, tpr_svm, lw=3, alpha=0.7, 
             label='SVM (C = {:0.2f}, area = {:0.2f})'.format(c, roc_auc_svm))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate (Recall)', fontsize=16)
plt.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')
plt.legend(loc="lower right", fontsize=11)
plt.title('ROC curve: ', fontsize=16)
plt.axes().set_aspect('equal')

plt.show()


# ### Kernelized SVM

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nSVMC = SVC(probability=True)\nsvc_param_grid = {\'kernel\': [\'rbf\'], \n                  \'gamma\': [ 0.001, 0.01, 0.1, 1],\n                  \'C\': [1, 10, 50, 100,200,300, 1000]}\n\ngsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=5, scoring="accuracy", n_jobs = -1, verbose = 1)\n\ngsSVMC.fit(X_train,y_train)\n\nSVMC_best = gsSVMC.best_estimator_\n\n# Best score\nprint(gsSVMC.best_score_)')


# ### Naives bayes

# In[ ]:


get_ipython().run_cell_magic('time', '', "nb = GaussianNB()\nnb.fit(X_train, y_train)\nprint('Naives bayes accuracy - training : ', nb.score(X_train, y_train))\nprint('Naives bayes accuracy - test : ', nb.score(X_test, y_test))")


# ### Knn

# In[ ]:


get_ipython().run_cell_magic('time', '', "knn = KNeighborsClassifier()\nknn.fit(X_train, y_train)\nprint('Knn accuracy - training : ', knn.score(X_train, y_train))\nprint('Knn accuracy - test : ', knn.score(X_test, y_test))")


# In[ ]:


acc_knn_train = []
acc_knn_test = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    score_train = knn.score(X_train, y_train)
    score_test = knn.score(X_test, y_test)
    acc_knn_train.append(score_train)
    acc_knn_test.append(score_test)


# In[ ]:


fig, ax = plt.subplots(figsize = [7, 5])
ax.plot(range(1,21), acc_knn_train, label='train_score')
ax.plot(range(1,21), acc_knn_test, label='test_score')
plt.xticks(np.arange(1,21,1))
plt.xlabel("K value")
plt.ylabel("Score")
legend = ax.legend(loc='top right', shadow=True)
plt.show()


# ### Decision Tree

# In[ ]:


get_ipython().run_cell_magic('time', '', "tr = DecisionTreeClassifier()\ntr.fit(X_train, y_train)\nprint('Decision Tree accuracy - training', tr.score(X_train, y_train))\nprint('Decision Tree accuracy - test', tr.score(X_test, y_test))")


# ### RandomForest

# In[ ]:


get_ipython().run_cell_magic('time', '', "rfc = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features=0.5)\nrfc.fit(X_train, y_train)\nrfc.predict(X_test)\nprint('Random Forest accuracy - training : ', rfc.score(X_train, y_train))\nprint('Random Forest accuracy  - test :', rfc.score(X_test, y_test))")


# ### Extra Tree Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "etc = ExtraTreesClassifier()\netc.fit(X_train, y_train)\netc.predict(X_test)\nprint('Extra Tree Classifier accuracy - training', etc.score(X_train, y_train))\nprint('Extra Tree Classifier accuracy - test', etc.score(X_test, y_test))")


# ### AdaBoostClassifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "adaboost = AdaBoostClassifier()\nadaboost.fit(X_train, y_train)\nprint('AdaBoost accuracy - training', adaboost.score(X_train, y_train))\nprint('AdaBoost accuracy - test', adaboost.score(X_test, y_test))")


# ### BaggingClassifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "bag = BaggingClassifier()\nbag.fit(X_train, y_train)\nprint('Bagging accuracy - training', bag.score(X_train, y_train))\nprint('Bagging accuracy - test', bag.score(X_test, y_test))")


# ### Artificial Neural Network

# In[ ]:


mlp = MLPClassifier()
mlp.fit(X_train, y_train)
print('Neural Networks accuracy - training', etc.score(X_train, y_train))
print('Neural Networks accuracy - test', etc.score(X_test, y_test))


# <br>

# ## Neural Network with Keras

# In[ ]:


get_ipython().run_cell_magic('time', '', "from keras.wrappers.scikit_learn import KerasClassifier\nfrom sklearn.model_selection import cross_val_score\nfrom keras.models import Sequential\nfrom keras.layers import Dense\ndef build_classifier():\n    classifier = Sequential()\n    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))\n    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))\n    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))\n    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])\n    return classifier\nclassifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)\naccuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train)\nmean = accuracies.mean()\nvariance = accuracies.std()\nprint('Accuracy : ', mean)\nprint('Variance : ', variance)")

