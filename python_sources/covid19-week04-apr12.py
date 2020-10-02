#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! /usr/bin/python
# -*- coding: utf-8 -*-
#
#	covid19-global-forecasting.py
#
#					Apr/12/2020 AM 09:15 
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
	coefs = []

	XX = np.arange(offset, np_train.shape[0], 1)

	for it in range(np_train.shape[1]):
#		sys.stderr.write("%d " % it)
		if it in as_exponent:
			coefs.append(curve_fit(exp,  XX, np_train[offset:, it],p0 = (0.5, XX[0], 2, 0), maxfev=100000)[0])
#
	sys.stderr.write("*** curve_fit_proc *** end ***\n")
#
	return coefs
# ------------------------------------------------------------------
def pattern_proc():
	as_exponent = range(313)
#
	return as_exponent
# ------------------------------------------------------------------
def exp(x, a, b, d, p):
	rvalue = 0.0
	try:
		rvalue = d * np.exp(a * x - b) + p
	except Exception as ee:
		sys.stderr.write("*** error *** in exp ***\n")
		sys.stderr.write(str(ee) + "\n")
	return rvalue
#
# ------------------------------------------------------------------
def new_epx(x, a, b, d, p):
	return exp(x, a, b, d, p)*(exp(x, a, b, d, p)>=0)

# ------------------------------------------------------------------
def RMSLE(pred,actual):
	return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))
#
# ------------------------------------------------------------------
def plot_exp(offset,np_train,coefs):

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
#	f_replace = lambda x: x.replace("-","")
	f_replace = lambda x: (parse(x) - parse("2020-01-01")).days
	date_aa = date_in.apply(f_replace)
	date_bb  = date_aa.astype(int)
#
	return date_bb
# ------------------------------------------------------------------
def key_define_proc(data_in):
	key = data_in['Country_Region'].astype('str') 		+ " " + data_in['Province_State'].astype('str')
#
	return key
# ------------------------------------------------------------------
def abs_convert_proc(value_in):
	f_replace = lambda x: abs(x)
	value_aa = value_in.apply(f_replace)
#
	return value_aa
# ------------------------------------------------------------------


# In[ ]:



sys.stderr.write("*** start ***\n")
offset = 50
offset = 45
offset = 48

folder_src='../input/covid19-global-forecasting-week-4'
data = pd.read_csv(folder_src + '/train.csv')
test_data = pd.read_csv(folder_src + '/test.csv')
submission = pd.read_csv(folder_src + '/submission.csv')
print(data.shape)
print(test_data.shape)
print(submission.shape)


# In[ ]:


# ------------------------------------------------------------------
datap = {"Date": ["2020-01-30","2020-01-31","2020-02-01","2020-02-02"]}
df = pd.DataFrame(datap, columns = ["Date"])
ee =  date_format_convert_proc(df["Date"])
print(ee)


# In[ ]:


# ------------------------------------------------------------------

data.loc[data['Country_Region'] == 'Japan']


# In[ ]:



data["Date"] = date_format_convert_proc(data["Date"])
test_data["Date"] = date_format_convert_proc(test_data["Date"])

data['key'] = key_define_proc(data)
test_data['key'] = key_define_proc(test_data)


data_train = data


# ------------------------------------------------------------------
# date_xx = 20200408
# date_xx = 20200326
# date_xx = 20200402
date_xx = (parse("20200402") - parse("2020-01-01")).days
nn_unique = data_train['key'].nunique()
#last day in test data
test_last_day = int(test_data.shape[0]/nn_unique) 	+ int(data_train[data_train.Date< date_xx].shape[0]/nn_unique)

sys.stderr.write("nn_unique = %d\n" % nn_unique)
dtemp = data_train['key'].unique()
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
test_days = np.arange(int(data_train[data_train.Date< date_xx].shape[0]/nn_unique), test_last_day, 1)

print(test_days)

# ------------------------------------------------------------------
sys.stderr.write("*** check *** ccc ***\n")

#lets create pivot tables
pivot_train = pd.pivot_table(data_train, index='Date', columns = 'key', values = 'ConfirmedCases')
pivot_train_death = pd.pivot_table(data_train, index='Date', columns = 'key', values = 'Fatalities')
np_train = pivot_train.to_numpy()
np_train_death = pivot_train_death.to_numpy()

sys.stderr.write("*** check *** eee ***\n")

data_train[['ConfirmedCases', 'Fatalities']].corr()

pivot_train.head(10)

#it = 148
#jt = 152
#plt.plot(np_train[offset:,it], label = "Japan")
#plt.plot(np_train[offset:,jt], label = "Korea")
#plt.legend()
#plt.show()
#
#print(np_train[offset:,it])
#print(np_train[offset:,jt])
# ------------------------------------------------------------------

#shift = [0,1,2,3,4,5,6,7]
#for s in shift:
#    sum = 0
#    for i in range(1,20):
#        sum += np.abs((np_train_death[-i][:]/(np_train[-i-s][:]+0.0001)-np_train_death[-i-1][:]/(np_train[-i-1-s][:]+0.0001)).mean())
#    print(sum, s)

sys.stderr.write("*** check *** ggg ***\n")

# ------------------------------------------------------------------
mask_deaths = np.zeros_like(np_train[0])
for i in range(1,21):
    mask_deaths += np_train_death[-i]/(np_train[-i]+0.0001)
mask_deaths = mask_deaths/20   

mask_deaths[(mask_deaths < 0.5) & (mask_deaths!=0)].mean()

mask_deaths[(mask_deaths> 0.5)|(mask_deaths<0.005)] = mask_deaths[(mask_deaths < 0.5) & (mask_deaths!=0)].mean()

sys.stderr.write("*** check *** hhh ***\n")

mask_mesh = np.meshgrid(mask_deaths, test_days)[0].T.flatten()

assert mask_mesh.shape[0] == test_data.shape[0]

# ------------------------------------------------------------------
as_exponent = pattern_proc()

# ------------------------------------------------------------------

# set(as_sigmoid)&set(as_linear) | set(as_sigmoid)&set(as_exponent) | set(as_exponent)&set(as_linear)

sys.stderr.write("*** check *** iii ***\n")

# ------------------------------------------------------------------
np_train.shape[0]

coefs = curve_fit_proc(offset,np_train)


# In[ ]:


# ------------------------------------------------------------------
plot_exp(offset,np_train,coefs)


# In[ ]:


# ------------------------------------------------------------------

ConfirmedCases_test = np.zeros((nn_unique, test_days.shape[0]))

# ------------------------------------------------------------------

sys.stderr.write("*** check *** jjj ***\n")
sys.stderr.write("np_train.shape[1] = %d\n" % np_train.shape[1])

for it in range(np_train.shape[1]):
    if it in as_exponent:
        function = new_epx
    ConfirmedCases_test[it] = function(test_days, *coefs[it])

ConfirmedCases_test.flatten().shape[0]

sys.stderr.write("*** check *** kkk ***\n")

assert ConfirmedCases_test.flatten().shape[0] == test_data.shape[0]

# ------------------------------------------------------------------
test_data['predict'] = ConfirmedCases_test.flatten()

# test_data[test_data['Country_Region']=='Japan']

submission['ConfirmedCases'] = ConfirmedCases_test.flatten()
submission['Fatalities'] = ConfirmedCases_test.flatten()*mask_mesh
#
submission['ConfirmedCases'] = abs_convert_proc(submission['ConfirmedCases'])
submission['Fatalities'] = abs_convert_proc(submission['Fatalities'])
submission.to_csv('submission.csv', index=False)

sys.stderr.write("*** end ***\n")
# ------------------------------------------------------------------

