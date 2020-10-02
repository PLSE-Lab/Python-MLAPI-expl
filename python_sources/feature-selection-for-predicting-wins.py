# Feature selection for predicting wins


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import pylab as pl
from sklearn import (cluster, datasets, decomposition, ensemble, lda, manifold, random_projection)
import csv



df = pd.read_csv('../input/team.csv')

names = ['r','ab','h','double','triple','hr','bb','so','sb','ra','er','era','cg','sho','sv','ipouts','ha','hra','bba','soa','e','dp','fp','bpf', 'ppf']
cols_to_keep = names
scores = []
df = df.dropna(axis=0, how='any')
X = df[names]
Y = df['w']
ranks = {}

fout = open('results.txt','w')
#Convert the target to numpy array
arr_target = Y.as_matrix()

#Convert the dataframe to numpy array
arr_X = X.as_matrix()

arr_X_train, arr_X_val, arr_target_train, arr_target_val = train_test_split(arr_X, arr_target, test_size=0.1, random_state=20)

#PCA
num_comp = []
per_variance = []
fout.write('Variance vs No. Of Features\n')
fout.write('-'*30 + '\n')
for n in (0.99, 0.95, 0.90, 0.75, 0.65, 0.5):
	p=PCA(n_components=n).fit(arr_X_train)
	#print(p.explained_variance_)
	print(n*100, len(p.explained_variance_))
	fout.write(str(n*100) + '\t' + str(len(p.explained_variance_)) + '\n')
	num_comp.append(len(p.explained_variance_))
	per_variance.append(n*100)

#num_comp.append(11) #the entire feature set
#hyperparameters
pca_val=num_comp
alpha_val = np.logspace(-5,5, num=11, base=2)
c_val = np.logspace(-5,5, num=11, base=2) #c for SVL and SVG
g_val = np.logspace(-5,5, num=11, base=2) #gamma for SVG

def runModel(model):
	if model == 'lr':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('lr',linear_model.LinearRegression())])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val), cv=10)
	elif model == 'rr':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('rr',linear_model.Ridge())])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val,rr__alpha=alpha_val), cv=10)
	elif model == 'rf':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('rf', RandomForestRegressor(n_estimators=20, max_depth=4,random_state =5))])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val), cv=10)
	elif model == 'svrl':
		pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('svr_lin',SVR(kernel='linear',C=1))])
		gs=GridSearchCV(pipe, dict(pca__n_components=pca_val,svr_lin__C=c_val), cv=10)
	#elif model == 'svrg':
		#pipe=Pipeline([('pca',PCA()), ('scaled',StandardScaler()), ('svr_gaussian',SVR(kernel='rbf',C=1,gamma=1))])
		#gs=GridSearchCV(pipe, dict(pca__n_components=pca_val,svr_gaussian__C=c_val,svr_gaussian__gamma=g_val), cv=10)
	#elif model == 'adaboostRF':
		#pipe=Pipeline([('pca',PCA()),('scaled',StandardScaler()), ('adaboost', AdaBoostRegressor(RandomForestRegressor(), random_state=0))])
		#gs=GridSearchCV(pipe, dict(pca__n_components=pca_val), cv=10)
	#elif model == 'adaboostSVR':
		#pipe=Pipeline([('pca',PCA()),('scaled',StandardScaler()), ('svr_adaboost',AdaBoostRegressor(SVR(), random_state=0))])
		#gs=GridSearchCV(pipe, dict(pca__n_components=pca_val), cv=10)
		
	gs.fit(arr_X_train, arr_target_train)

	#print gs.predict(arr_X_val)
	predictions = gs.predict(arr_X_val)
	print(gs.score(arr_X_val,arr_target_val))
	print('Best score')
	print(gs.best_score_)
	fout.write(str(gs.best_score_) + '\n')
	print ('Best estimator')
	print (gs.best_estimator_)
	print ('Best params')
	print (gs.best_params_)

fout.write('\n\ Model Accuracy\n')
fout.write('-'*30 + '\n')
models = ['lr','rr','rf','svrl']
for model in models:
	print ('*'*30)
	print ('Running : ' , model)
	print ('*'*30) 
	fout.write(model + '\t')
	runModel(model)

fout.write('\n\nImportant Features :\n')
fout.write('-'*30 + '\n')
#Find the k best features
k_val = list(set(num_comp))
for j in range(0,len(k_val)):
	#if k_val[j] != 9:
	contributingFeatures = []
	skb = SelectKBest(f_regression, k = k_val[j])
	arr_X_train_reshape = skb.fit(arr_X_train, arr_target_train)
	#arr_patrons_sales_events_val_reshape = skb.transform(arr_patrons_sales_events_val)
	print ('The top ', k_val[j], ' features are: ')
	fout.write('The top ' + str(k_val[j]) + ' features are: \n')
	get_features = skb.get_support() #print True or False for the features depending on whether it matters for predicting the category or not
	for i in range(0,len(get_features)):
		if get_features[i]:
			contributingFeatures.append(cols_to_keep[i])
			print(i, cols_to_keep[i])
			fout.write(cols_to_keep[i] + '\n')
	fout.write('\n')

X_centered = X - X.mean()
    
#print("Computing PCA projection..."),
pca = decomposition.PCA(n_components=8)
X_pca = pca.fit_transform(X_centered)
#print("done.")

print(pca.explained_variance_ratio_)

N = 8
ind = np.arange(N)  # the x locations for the groups

vals = [0.43506288,
        0.20271212,
        0.134597,
        0.07245318,
        0.05455227,
        0.03206135,
        0.02104035,
        0.0109561]

pl.figure(figsize=(10, 6), dpi=250)
ax = pl.subplot(111)
ax.bar(ind, pca.explained_variance_ratio_, 0.35, 
       color=[(0.949, 0.718, 0.004),
              (0.898, 0.49, 0.016),
              (0.863, 0, 0.188),
              (0.694, 0, 0.345),
              (0.486, 0.216, 0.541),
              (0.204, 0.396, 0.667),
              (0.035, 0.635, 0.459),
              (0.486, 0.722, 0.329),
             ])

ax.annotate(r"%d%%" % (int(vals[0]*100)), (ind[0]+0.2, vals[0]), va="bottom", ha="center", fontsize=12)
ax.annotate(r"%d%%" % (int(vals[1]*100)), (ind[1]+0.2, vals[1]), va="bottom", ha="center", fontsize=12)
ax.annotate(r"%d%%" % (int(vals[2]*100)), (ind[2]+0.2, vals[2]), va="bottom", ha="center", fontsize=12)
ax.annotate(r"%d%%" % (int(vals[3]*100)), (ind[3]+0.2, vals[3]), va="bottom", ha="center", fontsize=12)
ax.annotate(r"%d%%" % (int(vals[4]*100)), (ind[4]+0.2, vals[4]), va="bottom", ha="center", fontsize=12)
ax.annotate(r"%d%%" % (int(vals[5]*100)), (ind[5]+0.2, vals[5]), va="bottom", ha="center", fontsize=12)
ax.annotate(r"%s%%" % ((str(vals[6]*100)[:4 + (0-1)])), (ind[6]+0.2, vals[6]), va="bottom", ha="center", fontsize=12)
ax.annotate(r"%s%%" % ((str(vals[7]*100)[:4 + (0-1)])), (ind[7]+0.2, vals[7]), va="bottom", ha="center", fontsize=12)

ax.set_xticklabels(('       0',
                    '       1',
                    '       2',
                    '       3',
                    '       4',
                    '       5',
                    '       6',
                    '       7',
                    '       8'), 
                   fontsize=12)
ax.set_yticklabels(('0.00', '0.1', '0.2', '0.3', '0.4', '0.5'), fontsize=12)
ax.set_ylim(0, .50)
ax.set_xlim(0-0.45, 8+0.45)

ax.xaxis.set_tick_params(width=0)
ax.yaxis.set_tick_params(width=2, length=12)

ax.set_xlabel("Principal Component", fontsize=12)
ax.set_ylabel("Variance Explained (%)", fontsize=12)

pl.title("Scree Plot for the Team Wins Dataset", fontsize=16)
pl.savefig(r"results.png")