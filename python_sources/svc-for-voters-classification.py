"""
In this script, I use la inear SVC classifier to separate voters between two candidates,based on
a couple of features. I tried some combinations of features like population's age, education, race..

A linear separation is not always possible as we can see, but some data shapes are worth noting. 


"""


import pandas as pd
import numpy as np

from scipy import stats

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from sklearn import svm, preprocessing


###########
# load data

df_county_facts = pd.read_csv('../input/county_facts.csv')

dict_county_facts = pd.read_csv('../input/county_facts_dictionary.csv')
dict_county_facts = {k: v for k, v in zip(dict_county_facts['column_name'].values, dict_county_facts['description'].values)}

df_prim_res = pd.read_csv('../input/primary_results.csv')


candidates = ['Hillary Clinton', 'Bernie Sanders']

df_candidates = df_prim_res[df_prim_res['candidate'].isin(candidates)]

df_all = df_candidates.merge(df_county_facts, on=['fips'])

# reduce to only winner row
def winner(x):

	if all(i in x['candidate'].unique() for i in candidates):

		x['win'] = candidates[0] if x[x['candidate'] == candidates[0]]['fraction_votes'].values[0] > \
									x[x['candidate'] == candidates[1]]['fraction_votes'].values[0] \
								 else candidates[1]

	else:
		x['win'] = 'X'

	x = x.iloc[0]

	return x

df_all = df_all.groupby('fips').apply(winner)

# this is useful for missing data
df_all = df_all[df_all['win'] != 'X']


all_cols = [
		[ 'AGE295214', 'AGE775214'],  # young vs old
	    [ 'EDU635213', 'EDU685213'],  # high school vs bachelor
	    [ 'RHI125214', 'RHI225214'],  # white vs black
	    [ 'POP645213', 'POP715213'],  # foreign born vs 1 year in same house
	    ]  

for ii, cols in enumerate(all_cols):

	X = df_all[cols].values
	Y = df_all['win'].apply(lambda x: 0 if x == candidates[0] else 1).values

    # normalize to faster computation
	scaler = preprocessing.StandardScaler()

	X = scaler.fit_transform(X)

	clf = svm.SVC(kernel='linear')
	clf.fit(X, Y)

	w = clf.coef_[0]
	a = -w[0] / w[1]

	X = scaler.inverse_transform(X)
	xx = np.linspace(min(X[:, 0]), max(X[:, 0]))

	b = clf.support_vectors_[0]
	b = scaler.inverse_transform(b)
	yy_down = a * xx + (b[1] - a * b[0])
	b = clf.support_vectors_[-1]
	b = scaler.inverse_transform(b)
	yy_up = a * xx + (b[1] - a * b[0])

	yy = (yy_up + yy_down) / 2
	
	# plot

	fig = plt.figure()

	plt.plot(xx, yy, 'k-')
	plt.plot(xx, yy_down, 'k--')
	plt.plot(xx, yy_up, 'k--')

	l1 = plt.scatter(X[[ii for ii, k in enumerate(Y) if k == 0], 0], X[[ii for ii, k in enumerate(Y) if k == 0], 1], c='r')
	l2 = plt.scatter(X[[ii for ii, k in enumerate(Y) if k == 1], 0], X[[ii for ii, k in enumerate(Y) if k == 1], 1], c='b')

	plt.legend((l1, l2), (candidates[0], candidates[1]))

	plt.axis('tight')

	plt.xlabel(dict_county_facts[cols[0]])
	plt.ylabel(dict_county_facts[cols[1]])

	fig.savefig("{}_svm.png".format(ii))