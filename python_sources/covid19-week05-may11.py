#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! /usr/bin/python
# -*- coding: utf-8 -*-
#
#	covid19-global-forecasting.py
#
#					May/11/2020 PM 13:20 
#
#	train.csv
#		104 days
#		3463 keys
#
#	test.csv
#		45 days
#		3463 keys
#
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import sys
from dateutil.parser import parse
import matplotlib.pyplot as plt
from dateutil.parser import parse

np.seterr(all=None, divide=None, over=None, under=None, invalid=None)
# ------------------------------------------------------------------
# [8]:
def quantiles_calc_proc(df_train,key):
	unit_aa = {}
	data_confirm = df_train.loc[(df_train['key'] == key) 		& (df_train['Target'] == 'ConfirmedCases')]
	unit_aa['c0.05'] = data_confirm['TargetValue'].quantile(0.05)
	unit_aa['c0.5'] = data_confirm['TargetValue'].quantile(0.5)
	unit_aa['c0.95'] = data_confirm['TargetValue'].quantile(0.95)
#
	data_confirm = df_train.loc[(df_train['key'] == key) 		& (df_train['Target'] == 'Fatalities')]
	unit_aa['f0.05'] = data_confirm['TargetValue'].quantile(0.05)
	unit_aa['f0.5'] = data_confirm['TargetValue'].quantile(0.5)
	unit_aa['f0.95'] = data_confirm['TargetValue'].quantile(0.95)
#
	return unit_aa
#
# ------------------------------------------------------------------
# [6]:
def plot_three_proc(df_in,item_target):
	try:
		jp_data = df_in.loc[(df_in['Country_Region'] == 'Japan') 			& (df_in['Target'] == item_target)]
		print("jp_data.shape ",jp_data.shape)
		jp_data.to_csv("jp_data.csv",index=False)
		pivot_aa = pd.pivot_table(jp_data,index='Date',values= 'q0.05')
		pivot_bb = pd.pivot_table(jp_data,index='Date',values= 'q0.5')
		pivot_cc = pd.pivot_table(jp_data,index='Date',values= 'q0.95')
		print("pivot_aa.shape ",pivot_aa.shape)
		plt.plot(pivot_aa,label = 'q0.05')
		plt.plot(pivot_bb,label = 'q0.5')
		plt.plot(pivot_cc,label = 'q0.95')
		plt.legend()
		plt.show()
	except Exception as ee:
		sys.stderr.write("*** error *** in plot_three_proc bbb ***\n")
		sys.stderr.write(str(ee) + "\n")
		sys.exit(1)
#
# ------------------------------------------------------------------
# [4]:
def plot_proc(df_train,item_target):
	try:
		jp_data = df_train.loc[(df_train['Country_Region'] == 'Japan') 		& (df_train['Target'] == item_target)]
		print("min = ",jp_data['TargetValue'].min())
		print("mean = ",jp_data['TargetValue'].mean())
		print("quantile 0.05 = ",jp_data['TargetValue'].quantile(0.05))
		print("quantile 0.5 = ",jp_data['TargetValue'].quantile(0.5))
		print("quantile 0.95 = ",jp_data['TargetValue'].quantile(0.95))
		print("max = ",jp_data['TargetValue'].max())
		pivot_data = pd.pivot_table(jp_data,index='Date',                               values= 'TargetValue')
		print(pivot_data.shape)
		plt.plot(pivot_data,label = 'Japan')
		plt.legend()
		plt.show()
	except Exception as ee:
		sys.stderr.write("*** error *** in plot_three_proc bbb ***\n")
		sys.stderr.write(str(ee) + "\n")
		sys.exit(1)
#
# ------------------------------------------------------------------
# [2-4-6]:
def key_define_proc(data_in):
	key = data_in['Country_Region'].astype('str') 		+ " " + data_in['Province_State'].astype('str') 		+ " " + data_in['County'].astype('str')
#
	return key
# ------------------------------------------------------------------
# [2-4-4]:
def min_max_date_check_proc(df_in):
	in_date_min = df_in.min()
	in_date_max = df_in.max()
	print('Minimum date from set: {}'.format(in_date_min))
	print('Maximum date from set: {}'.format(in_date_max))
#
# ------------------------------------------------------------------
# [2-4]:
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
# [2]:
def data_read_proc():
	folder_src='../input/covid19-global-forecasting-week-5'
#	folder_src='../input_short/covid19-global-forecasting-week-5'
	df_train = pd.read_csv(folder_src + '/train.csv')
	df_test = pd.read_csv(folder_src + '/test.csv')
	submission = pd.read_csv(folder_src + '/submission.csv')
	print("df_train.shape ",df_train.shape)
	print("df_test.shape ",df_test.shape)
	print("submission.shape ",submission.shape)

	df_train["Date"] = date_format_convert_proc(df_train["Date"])
	df_test["Date"] = date_format_convert_proc(df_test["Date"])
#
	df_train['County']=df_train['County'].fillna("NR")
	df_train['Province_State']=df_train['Province_State'].fillna("NR")
	df_train.head()
#
	df_test['County']=df_test['County'].fillna("NR")
	df_test['Province_State']=df_test['Province_State'].fillna("NR")
	df_test.head()
#
	df_train['key'] = key_define_proc(df_train)
	df_test['key'] = key_define_proc(df_test)
#
	return df_train,df_test,submission
# ------------------------------------------------------------------
# %%
sys.stderr.write("*** start ***\n")

df_train,df_test,submission = data_read_proc()


# In[ ]:


df_train.loc[df_train['Country_Region'] == 'Japan']

plot_proc(df_train,'ConfirmedCases')
plot_proc(df_train,'Fatalities')


# In[ ]:


# ------------------------------------------------------------------
nn_unique_train = df_train['key'].nunique()
nn_unique_test = df_test['key'].nunique()

sys.stderr.write("nn_unique_train = %d\n" % nn_unique_train)
sys.stderr.write("nn_unique_test = %d\n" % nn_unique_test)
dtemp = df_train['key'].unique()
print(len(dtemp))
for it in range(nn_unique_train):
	if dtemp[it][:5] == "Japan":
		print("dtemp[%d] = %s" % (it, dtemp[it]))
	elif dtemp[it][:5] == "Korea":
		print("dtemp[%d] = %s" % (it, dtemp[it]))


quantiles = {}
for key in dtemp:
	quantiles[key] = quantiles_calc_proc(df_train,key)


# In[ ]:


# ------------------------------------------------------------------

sys.stderr.write("np_unique_train = %d\n" % nn_unique_train)
sys.stderr.write("np_unique_test = %d\n" % nn_unique_test)
#
#
test_test_days = np.arange(150,195,1)
data_out = np.zeros((nn_unique_test, test_test_days.shape[0] *6))
print("data_out.shape ",data_out.shape)
#
test_keys = df_test['key'].unique()
it = 0
for key in test_keys:
    rr = np.zeros(test_test_days.shape[0] *6)
    pp = 10.0
    qq = 20.0
    unit_aa = quantiles[key]
    for jt in range(test_test_days.shape[0]):
        kt = jt * 6
        rr[kt] = unit_aa['c0.05']
        rr[kt+1] = unit_aa['c0.5']
        rr[kt+2] = unit_aa['c0.95']
        rr[kt+3] = unit_aa['f0.05']
        rr[kt+4] = unit_aa['f0.5']
        rr[kt+5] = unit_aa['f0.95']
    data_out[it] = rr
    it += 1


sys.stderr.write("*** check *** kkk ***\n")

print("data_out.shape ",data_out.shape)
print("df_test.shape", df_test.shape)
print("submission.shape ",submission.shape)


# In[ ]:


# ------------------------------------------------------------------
llx = len(data_out.flatten())
print("len(data_out.flattern()) ",llx)
if llx != submission.shape[0]:
	sys.stderr.write("*** error *** llx != submission.shape[0] ***\n")
	sys.stderr.write("llx = %d\n" % llx)
	sys.stderr.write("submission.shape[0] = %d\n" % submission.shape[0])
submission['TargetValue'] = data_out.flatten()
#
submission = submission.replace([np.inf, -np.inf], 0)
#
submission.to_csv('submission.csv', index=False)

sys.stderr.write("*** end ***\n")
# ------------------------------------------------------------------

