#!/usr/bin/env python
# coding: utf-8

# # Loading modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, math
from datetime import datetime
import seaborn as sns
import matplotlib.cm as cm

from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split


# # Cleaning of data

# In[ ]:


dat = pd.read_csv("../input/ATP.csv")


# Converting the date of tourney as datatime

# In[ ]:


dat['date'] = dat.tourney_date.apply(lambda t: datetime.strptime(str(t), '%Y%m%d'))


# Changing the format of initial dataframe.<br>
# I would like to create a binary target which means the win of the player j0 or j1.<br>
# For example, target = 0 means that the player j0 won, otherwise target = 1 means that the player j1 won.<br>
# Objective : Predict this target based on several variables from this dataset.

# In[ ]:


colnames = dict()
colnames['type1'] = ['1stIn', '1stWon', '2ndWon', 'SvGms', 'ace', 'bpFaced', 'bpSaved', 'df', 'svpt']
colnames['type2'] = ['age', 'entry', 'hand', 'ht', 'id', 'ioc', 'name', 'rank', 'rank_points', 'seed']
colnames['type3'] = ['best_of', 'draw_size', 'match_num', 'minutes', 'round', 'score', 'surface', 'tourney_date',
                     'tourney_id', 'tourney_level', 'tourney_name', 'date']


# In[ ]:


df = pd.DataFrame()
mat = []
for i in dat.index:
    row = []
    for col in colnames['type3']:
        row.append(dat[col][i])
    if i % 2 == 0: #j0
        # j0=loser, j1=winner
        for col in colnames['type1']:
            row.append(dat['l_'+col][i])
        for col in colnames['type2']:
            row.append(dat['loser_'+col][i])
        for col in colnames['type1']:
            row.append(dat['w_'+col][i])
        for col in colnames['type2']:
            row.append(dat['winner_'+col][i])
        row.append(1) #target winner --> j1
    else: #j1
        # j0=winner, j1=loser
        for col in colnames['type1']:
            row.append(dat['w_'+col][i])
        for col in colnames['type2']:
            row.append(dat['winner_'+col][i])
        for col in colnames['type1']:
            row.append(dat['l_'+col][i])
        for col in colnames['type2']:
            row.append(dat['loser_'+col][i])
        row.append(0) #target winner --> j0
    mat.append(row)


# In[ ]:


colDataFrame = colnames['type3']
for col in colnames['type1']:
    colDataFrame.append('j0_'+col)
for col in colnames['type2']:
    colDataFrame.append('j0_'+col)
for col in colnames['type1']:
    colDataFrame.append('j1_'+col)
for col in colnames['type2']:
    colDataFrame.append('j1_'+col)
colDataFrame.append("target")


# In[ ]:


df = pd.DataFrame(columns=colDataFrame, data=mat)


# In[ ]:


df.head()


# # Exploration of data

# In[ ]:


print("nRows : {}, nCols : {}".format(df.shape[0], df.shape[1]))


# In[ ]:


dfe = df.copy()


# The first rankings were published in August, 23rd 1973, so we can delete all row without rank as this value seems to be important for the bookmakers !

# In[ ]:


dfe = dfe.loc[np.invert(dfe.j0_rank.isna()) & np.invert(dfe.j1_rank.isna())]
dfe = dfe.loc[np.invert(dfe.j0_rank_points.isna()) & np.invert(dfe.j1_rank_points.isna())]


# Rows with 'None' or NaN for the surface will be removed as it is a pertinent variable for the prediction.

# In[ ]:


dfe = dfe.loc[np.invert(dfe.surface.isna())]
dfe = dfe.loc[dfe.surface != "None"]


# In[ ]:


print("nRows : {}, nCols : {}".format(dfe.shape[0], dfe.shape[1]))


# In[ ]:


print("There are {} matches.".format(dfe.shape[0]))
print("There are {} different players.".format(len(list(set(dfe.j0_name + dfe.j1_name)))))


# There are 4 types of surface in the Tennis : Hard, Grass, Carpet, Clay.

# In[ ]:


# pie chart of surface
count_surface = dfe[["tourney_id", "surface"]]
count_surface = count_surface.groupby(["surface"]).agg('count')
count_surface.reset_index(inplace=True)
count_surface.columns=["surface","Count"]
count_surface.sort_values("Count", inplace=True)

x = np.arange(count_surface.shape[0])
ys = [i+x+(i*x)**2 for i in range(count_surface.shape[0])]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(5,5), dpi=120)
labels=count_surface.surface.values

sizes=count_surface.Count.values

explode = [0.9 if sizes[i] < 1000 else 0.0 for i in range(len(sizes))]
ax.pie(sizes, explode = explode, labels=labels, colors = colors,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow = False, startangle=0, textprops={'fontsize': 7})
ax.axis('equal')
ax.set_title('Surface', bbox={'facecolor':'blue', 'pad':3}, color = 'w', fontsize=10)
_ = ax


# In[ ]:


# bpFaced barplot
dfe_bpFaced = dfe.groupby(["target"])["j0_bpFaced","j1_bpFaced"].agg('mean')
dfe_bpFaced.reset_index(inplace=True)

players=['player 0', 'player 1']
lw=[0,1]
pos = np.arange(len(players))
bar_width = 0.35
index_loser=dfe_bpFaced.iloc[0,1:].values
index_winner=dfe_bpFaced.iloc[1,1:].values

plt.bar(pos,index_loser,bar_width,color='red',edgecolor='black')
plt.bar(pos+bar_width,index_winner,bar_width,color='green',edgecolor='black')
plt.xticks(pos+0.1, players)
plt.xlabel('', fontsize=16)
plt.ylabel('Breakpoints faced', fontsize=15)
plt.title('Barchart',fontsize=18)
plt.legend(lw,loc=2)
plt.show()


# The breakpoints faced is an important variable to predict a win of a match.<br>
# As I said before, the surface seems to be naturally a good predictor.<br>
# For example, Raphael Nadal may win Roger Federer in clay surface (Roland Garros) with a high probability.<br>
# That is why we binarize the surface variable.

# In[ ]:


# Binarize surface
df_surface = dfe.surface.str.get_dummies()
df_surface.head()


# In[ ]:


ax = sns.boxplot(x="surface", y="j1_rank_points", hue="target", data=dfe, palette=['blue','red'])
ax.set_ylim([0, 5000])
ax.set_ylabel("rank points")
_ = ax


# The rank points is an important data because it allows to justify the level of the player. So, if the player has a huge rank, he could win the match very easily.

# In[ ]:


ax = sns.boxplot(x="target", y="j1_rank_points", data=dfe, palette=["blue","red"])
ax.set_ylim([0,4000])
ax.set_ylabel('rank points')
_ = ax


# # Prediction
# In this section, I would suggest 4 different algorithms (supervised): Logistic regression (lr), Gradient boosting (gb), k-nearest neighbors (knn), Random forest (rf). For each model, the metric will be the AUC (ROC Curve) using a gridsearchcv to determine the best estimator for the prediction.
# ## Class implementation

# In[ ]:


class Model:
    def __init__(self,data,seed,random_sample):
        self.random_sample = random_sample
        self.seed = seed
        
        self.data = data.sample(frac=self.random_sample, replace=False, random_state=self.seed)
        
        self.lr=None
        self.pred_train=None
        self.pred_test=None
    def split(self, test_size):
        self.test_size = test_size
        train_X, test_X, train_y, test_y = train_test_split(self.data,self.data['target'], test_size = test_size, random_state=self.seed)
        self.train_X = train_X.drop(columns=['target'])
        self.test_X = test_X.drop(columns=['target'])
        self.train_y = train_y
        self.test_y = test_y
    def model_LR(self,n_jobs,cv,regul):
        self.regul = regul
        if regul=='none':
            n_iters = np.array([50, 200])
            model = linear_model.SGDClassifier(loss='log', random_state=0, penalty=self.regul)
            grid = GridSearchCV(estimator=model, param_grid=dict(n_iter_no_change=n_iters), scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)
            grid.fit(self.train_X,self.train_y)
            self.grid = grid
        elif regul=='elasticnet':
            n_iters = np.array([50, 200])
            alphas = np.logspace(-5, 1, 5)
            l1_ratios = np.array([0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.85, 1])
            model = linear_model.SGDClassifier(loss='log', random_state=0, penalty=self.regul,n_iter_no_change=100,max_iter=100)
            grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas,l1_ratio=l1_ratios), scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)
            grid.fit(self.train_X,self.train_y)
            self.grid = grid
        return self.lr
    def model_GB(self,n_jobs,cv):
        param_grid = {'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,150,200],
                      'learning_rate':[0.1,0.2,0.5,0.7,0.9,1]}
        model = ensemble.GradientBoostingClassifier()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)
        grid.fit(self.train_X,self.train_y)
        self.grid = grid
    def model_KNN(self,n_jobs,cv):
        param_grid = {'n_neighbors': np.arange(1,310,10)}
        model = neighbors.KNeighborsClassifier()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)
        grid.fit(self.train_X,self.train_y)
        self.grid = grid
    def model_RF(self,n_jobs,cv):
        param_grid = {'criterion' : ['entropy', 'gini'],
                      'n_estimators' : [20, 40, 60, 80, 100, 120, 160, 200, 250, 300],
                      'max_features' :['sqrt', 'log2']}
        model = ensemble.RandomForestClassifier()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', n_jobs=n_jobs, cv=cv, verbose=1)
        grid.fit(self.train_X,self.train_y)
        self.grid = grid
    def predict(self):
        self.pred_train = self.grid.best_estimator_.predict_proba(X=self.train_X)
        self.pred_test = self.grid.best_estimator_.predict_proba(X=self.test_X)
    def get_AUC(self):
        self.train_auc=metrics.roc_auc_score(y_score=self.grid.best_estimator_.predict_proba(X=self.train_X)[:,1], y_true=self.train_y)
        self.test_auc=metrics.roc_auc_score(y_score=self.grid.best_estimator_.predict_proba(X=self.test_X)[:,1], y_true=self.test_y)
        return (self.train_auc,self.test_auc)
    ### get contingency table + recall precision + roc curve !!!
    def boxplot(self):
        plt.figure()
        plt.subplot(1,2,1)
        sns.boxplot(x=self.train_y.values, y=self.grid.best_estimator_.predict_proba(X=self.train_X.values)[:,1])
        plt.title('Train')
        plt.subplot(1,2,2)
        sns.boxplot(x=self.test_y.values, y=self.grid.best_estimator_.predict_proba(X=self.test_X.values)[:,1])
        plt.title('Test')
        return plt
    def rocCurve(self):
        plt.figure()
        plt.subplot(1,2,1)
        fpr, tpr, thresholds = metrics.roc_curve(y_score=self.grid.best_estimator_.predict_proba(X=self.train_X)[:,1], y_true=self.train_y)
        plt.plot(fpr, tpr,'r')
        plt.plot([0,1],[0,1],'b')
        plt.title('Train, AUC: {}'.format(round(metrics.auc(fpr,tpr),3)))
        
        plt.subplot(1,2,2)
        fpr, tpr, thresholds = metrics.roc_curve(y_score=self.grid.best_estimator_.predict_proba(X=self.test_X)[:,1], y_true=self.test_y)
        plt.plot(fpr, tpr,'r')
        plt.plot([0,1],[0,1],'b')
        plt.title('Test, AUC: {}'.format(round(metrics.auc(fpr,tpr),3)))
        return plt
    def confusion(self,set_):
        if set_ == "train":
            res = metrics.confusion_matrix(y_true=self.train_y,y_pred=self.pred_train)
        elif set_ == "test":
            res = metrics.confusion_matrix(y_true=self.test_y,y_pred=self.pred_test)
        return res
    def getAccuracy(self):
        res=(metrics.accuracy_score(y_true=self.train_y,y_pred=self.pred_train),
            metrics.accuracy_score(y_true=self.test_y,y_pred=self.pred_test))
        return res
    def getClassificationReport(self,set_):
        if set_ == "train":
            res = metrics.classification_report(self.train_y, self.pred_train)
        elif set_ == "test":
            res = metrics.classification_report(self.test_y, self.pred_test)
        return res


# The model contains the rank points (rank_points) and the breakpoints faced (bpFaced) for each player as predictors, the surface (binarized).<br>
# The rank points will be replaced by its log-transformation.

# In[ ]:


dfm = dfe[["target","j0_rank_points","j1_rank_points","j0_bpFaced","j1_bpFaced"]]
dfm[df_surface.columns] = df_surface
dfm.dropna(inplace=True)

dfm.j0_rank_points = dfm.j0_rank_points.apply(lambda x: math.log(x))
dfm.j1_rank_points = dfm.j1_rank_points.apply(lambda x: math.log(x))


# In[ ]:


dfm.head()


# ## Logistic regression (lr)

# In[ ]:


lr = Model(data=dfm,seed=123,random_sample=1)
lr.split(0.35)
lr.model_LR(cv=4,n_jobs=8,regul="elasticnet")
lr.predict()


# In[ ]:


_ = lr.rocCurve()


# In[ ]:


_ = lr.boxplot()


# In[ ]:


lr.grid.best_params_


# ## Gradient boosting (gb)

# In[ ]:


gb = Model(data=dfm,seed=123,random_sample=1)
gb.split(0.35)
gb.model_GB(cv=4,n_jobs=8)
gb.predict()


# In[ ]:


_ = gb.rocCurve()


# In[ ]:


_ = gb.boxplot()


# In[ ]:


gb.grid.best_params_


# ## K nearest neighbors (knn)

# In[ ]:


knn = Model(data=dfm,seed=123,random_sample=1)
knn.split(0.35)
knn.model_KNN(cv=4,n_jobs=8)
knn.predict()


# In[ ]:


_ = knn.rocCurve()


# In[ ]:


_ = knn.boxplot()


# In[ ]:


knn.grid.best_params_


# ## Random forest (rf)

# In[ ]:


rf = Model(data=dfm,seed=123,random_sample=1)
rf.split(0.35)
rf.model_RF(cv=4,n_jobs=4)
rf.predict()


# In[ ]:


_ = rf.rocCurve()


# There is overfitting of data using the random forest algorithm.

# In[ ]:


_ = rf.boxplot()


# In[ ]:


rf.grid.best_params_


# # Conclusion/Discussion

# As a conclusion, all models developped have a good diagnostic performance in train/test set (AUC = 0.90) except for the random forest because of the overfitting. Nevertheless, if I had to choose one model, I choose the one based on the gradient boosting because there are a few outliers compared to the others.<br>
# 
# Note that is not easy to collect some data like the breakpoints faced for further predictions. As a matter of fact, if we want to create an api which is going to give the probability of the winner. We have to collect the rank points and surface used which are very easy. But the breakpoints faced will be missed. To pass this obstacle, the solution will be to develop a knn model based on rank points and surface in order to predict the breakpoints faced (as a continuous variable), for each player. Then, after predicting the value of breakpoints faced, the prediction of the winner can be done.<br>
# 
# Another suggestion : use the module "VotingClassifier" which can combine multiple different models into a single model (which is stronger than any of the individual models alone).
