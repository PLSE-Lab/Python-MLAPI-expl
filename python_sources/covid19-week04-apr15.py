#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! /usr/bin/python
# -*- coding: utf-8 -*-
#
#	covid19-global-forecasting.py
#
#					Apr/15/2020 PM 18:00
#
#
#	train.csv
#		78 days
#		313 regions
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import sys
from dateutil.parser import parse
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

np.seterr(all=None, divide=None, over=None, under=None, invalid=None)
# ------------------------------------------------------------------
def curve_fit_proc(offset,np_train):
	sys.stderr.write("*** curve_fit_proc *** start ***\n")
	sys.stderr.write("np_train.shape[0] = %d\n" % np_train.shape[0])
	sys.stderr.write("np_train.shape[1] = %d\n" % np_train.shape[1])
	coefs = []

	XX = np.arange(offset, np_train.shape[0], 1)

	for it in range(np_train.shape[1]):
#		sys.stderr.write("%d " % it)
		coefs.append(curve_fit(exp,  XX, np_train[offset:, it],p0 = (0.5, XX[0], 2, 0), maxfev=100000)[0])
#
	sys.stderr.write("*** curve_fit_proc *** end ***\n")
#
	return coefs
# ------------------------------------------------------------------
def exp(xx, aa, bb, dd, pp):
	rvalue = 0.0
	try:
		rvalue = dd * np.exp(aa * xx - bb) + pp
	except Exception as ee:
		sys.stderr.write("*** error *** in exp ***\n")
		sys.stderr.write(str(ee) + "\n")
	return rvalue
#
# ------------------------------------------------------------------
def new_epx(xx, aa, bb, dd, pp):
	eex =  exp(xx, aa, bb, dd, pp)
	rvalue =  eex * (eex >=0)
	return rvalue

# ------------------------------------------------------------------
def plot_exp(offset,np_train,coefs):
#
	XX = np.arange(offset, np_train.shape[0], 1)
	it = 148
	jt = 152
	plt.plot(np_train[offset:,it], label = "Japan")
	plt.plot(exp(XX, *coefs[it]))
	plt.plot(np_train[offset:,jt], label = "Korea")
	plt.plot(exp(XX, *coefs[jt]))
	plt.legend()
	plt.show()

# ------------------------------------------------------------------
def date_format_convert_proc(date_in):
	date_in = pd.to_datetime(date_in, format = '%Y-%m-%d')
	min_max_date_check_proc(date_in)
	f_replace = lambda x: (x - parse("2020-01-01")).days
	date_aa = date_in.apply(f_replace)
	date_bb  = date_aa.astype(int)
	min_max_date_check_proc(date_bb)
#
	return date_bb
# ------------------------------------------------------------------
def min_max_date_check_proc(df_in):
	in_date_min = df_in.min()
	in_date_max = df_in.max()
	print('Minimum date from set: {}'.format(in_date_min))
	print('Maximum date from set: {}'.format(in_date_max))
#
# ------------------------------------------------------------------

def key_define_proc(data_in):
	key = data_in['Country_Region'].astype('str') 		+ " " + data_in['Province_State'].astype('str')
#
	return key
# ------------------------------------------------------------------
def pivot_calc_proc(df_train):
#lets create pivot tables
	pivot_train = pd.pivot_table(df_train, index='Date', 		columns = 'key', values = 'ConfirmedCases')
	pivot_train_death = pd.pivot_table(df_train, index='Date', 		columns = 'key', values = 'Fatalities')
	np_train = pivot_train.to_numpy()
	np_train_death = pivot_train_death.to_numpy()

	sys.stderr.write("*** pivot_calc_proc ***\n")

	pivot_train.head(10)
#
	return np_train,np_train_death
# ------------------------------------------------------------------
def mask_mesh_proc(np_train,test_days):
	nn = 20
	nn = 30
	nn = 19
	mask_deaths = np.zeros_like(np_train[0])
	for it in range(1,nn + 1):
		mask_deaths += np_train_death[-it]/(np_train[-it]+0.0001)
	mask_deaths = mask_deaths/nn   

	mask_deaths[(mask_deaths < 0.5) & (mask_deaths!=0)].mean()

	mask_deaths[(mask_deaths> 0.5)|(mask_deaths<0.005)] = mask_deaths[(mask_deaths < 0.5) & (mask_deaths!=0)].mean()

	sys.stderr.write("*** check *** hhh ***\n")

	mask_mesh = np.meshgrid(mask_deaths, test_days)[0].T.flatten()

#assert mask_mesh.shape[0] == df_test.shape[0]

	return mask_mesh
# ------------------------------------------------------------------
# %%

sys.stderr.write("*** start ***\n")
offset = 50
offset = 45
offset = 48

folder_src='../input/covid19-global-forecasting-week-4'
df_train = pd.read_csv(folder_src + '/train.csv')
df_test = pd.read_csv(folder_src + '/test.csv')
submission = pd.read_csv(folder_src + '/submission.csv')
print(df_train.shape)
print(df_test.shape)
print(submission.shape)


# In[ ]:


df_train.loc[df_train['Country_Region'] == 'Japan']


# In[ ]:


# ------------------------------------------------------------------
#
df_train["Date"] = date_format_convert_proc(df_train["Date"])
df_test["Date"] = date_format_convert_proc(df_test["Date"])
#
# %%
# ------------------------------------------------------------------
# %%

df_train['key'] = key_define_proc(df_train)
df_test['key'] = key_define_proc(df_test)

# ------------------------------------------------------------------
# date_xx = 20200408
# date_xx = 20200326
# date_xx = 20200402
date_xx = (parse("2020-04-02") - parse("2020-01-01")).days
nn_unique = df_train['key'].nunique()
#last day in test data
test_last_day = int(df_test.shape[0]/nn_unique) 	+ int(df_train[df_train.Date< date_xx].shape[0]/nn_unique)

sys.stderr.write("nn_unique = %d\n" % nn_unique)
dtemp = df_train['key'].unique()
print(len(dtemp))
for it in range(nn_unique):
	if dtemp[it][:5] == "Japan":
		print("dtemp[%d] = %s" % (it, dtemp[it]))
	elif dtemp[it][:5] == "Korea":
		print("dtemp[%d] = %s" % (it, dtemp[it]))

print("nn_unique = ",nn_unique)
print("test_last_day = ",test_last_day)


# In[ ]:


# ------------------------------------------------------------------

#days in test data
test_days = np.arange(int(df_train[df_train.Date< date_xx].shape[0]/nn_unique), test_last_day, 1)

print("test_days.shape",test_days.shape)
print(test_days)

# ------------------------------------------------------------------
sys.stderr.write("*** check *** ccc ***\n")

np_train,np_train_death = pivot_calc_proc(df_train)


# df_train[['ConfirmedCases', 'Fatalities']].corr()


# In[ ]:


# ------------------------------------------------------------------

mask_mesh =  mask_mesh_proc(np_train,test_days)


# ------------------------------------------------------------------
# np_train.shape[0]

coefs = curve_fit_proc(offset,np_train)


# In[ ]:


# ------------------------------------------------------------------
# plot_exp(offset,np_train,coefs)


# In[ ]:


# ------------------------------------------------------------------

sys.stderr.write("*** check *** jjj ***\n")
sys.stderr.write("np_train.shape[1] = %d\n" % np_train.shape[1])

ConfirmedCases_test = np.zeros((nn_unique, test_days.shape[0]))

for it in range(np_train.shape[1]):
    ConfirmedCases_test[it] = new_epx(test_days, *coefs[it])


sys.stderr.write("*** check *** kkk ***\n")


# ------------------------------------------------------------------

submission['ConfirmedCases'] = ConfirmedCases_test.flatten()
submission['Fatalities'] = ConfirmedCases_test.flatten()*mask_mesh
#
submission.to_csv('submission.csv', index=False)

sys.stderr.write("*** end ***\n")
# ------------------------------------------------------------------

