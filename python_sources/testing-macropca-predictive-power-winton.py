#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import utilfn as fn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

winpath = "../input/the-winton-stock-market-challenge/"
pcpath  = "../input/wintonpca/"
print(os.listdir(winpath))


# In[ ]:


fn.test()


# # Motivation
# 
# *In Progress*
# 
# In [this previous kernel](https://www.kaggle.com/rlagrois/macropca-with-unknown-features) I ran MacroPCA on the 25 unknown given features.  Since what the features and specific rows represent are unknown, interpreting and evaluating the PCA is a little more tricky than normal.  I showed that there were a number of potentially problematic high-leverage rows though they were outnumbered by "good" highleverage points.  At this point it's unclear if this PCA will provide any value in making predictions.  
# 
# In order to try to answer this question I will run a regression on one of the total return target columns using only the PCA data and compare that to doing the same with the original data.  These checks will be done using a Random Forest Regressor and Suport Vector Regressor; these were chosen as I plan to use either extensions or similar methods for the full multi-output problem.  They also benefit from being easy and quick to run and train.

# # Read and Prep

# In[ ]:


# Rows excluded from MacroPCA, drop from train/test data
drp_rows = "1     40    393   522   840   844   1573  1611  1853  2102  3050  3458             3885  4330  4701  5171  5331  5881  6709  6720  6856  7305  7528  8572             8710  9511  9796  10654 10661 11213 12608 12806 13206 13273 13334 14148             14161 14255 14510 14626 15914 15973 16329 16549 16811 17120 17782 19173             19630 19807 19955 21771 22198 22207 22225 22243 23210 23625 23699 23851             24041 24429 24525 24568 24907 24941 25264 25441 25536 25756 26945 27892             28074 28110 28321 28720 30545 31963 32118 32764 32828 32978 33188 33227             33258 33268 33469 33609 33613 33766 33769 33982 34372 34776 35095 35112             35403 35419 35668 36034 36506 36539 36869 37251 37809 37899 38010 38520             39301 39560 39748 3985"

drp_rows = list(map(lambda x: int(x) - 1, list(filter(lambda x: x.isdigit(), drp_rows.split(' ')))))
drp_rows


# In[ ]:


# Read
dfw = pd.read_csv(winpath + 'train.csv')
pc = pd.read_csv(pcpath + 'winton_pca_final.csv')
pcns = pd.read_csv(pcpath + 'winton_pca_finalNS.csv')

# Prep unmodified features
dfw = dfw.drop(index=drp_rows)
x_std = dfw.iloc[:,1:26]
x_std.fillna(x_std.mean(), inplace=True)

# create array containing y and evaluation weights (if use of weights is desired)
y = np.array(dfw['Ret_PlusOne']).reshape((-1,1))
w = np.array(dfw['Weight_Daily']).reshape((-1,1))
yw = np.concatenate([y,w], axis=1)


dfw.head()


# In[ ]:


# Train/Test Split (use for RFR)
x_tr, x_tt, y_tr, y_tt = train_test_split(x_std, yw, test_size=0.33,
                                          random_state=22391) # Features with no PCA

xPC_tr, xPC_tt, yPC_tr, yPC_tt = train_test_split(pc, yw, test_size=0.33,
                                                  random_state=22391) # Scaled PCA

xNS_tr, xNS_tt, yNS_tr, yNS_tt = train_test_split(pcns, yw, test_size=0.33,
                                                  random_state=22391) # Not Scaled PCA


# In[ ]:


# Standardize and Train/Test Split (use for SVR)

std1, std2, std3, = StandardScaler(), StandardScaler(), StandardScaler()
scales = [std1,std2,std3]
frames = [x_std, pc, pcns]

for i,k in enumerate(scales):
    frames[i] = k.fit_transform(frames[i])

x_trs, x_tts, y_trs, y_tts = train_test_split(x_std, yw, test_size=0.33,
                                          random_state=22391) # Features with no PCA

xPC_trs, xPC_tts, yPC_trs, yPC_tts = train_test_split(pc, yw, test_size=0.33,
                                                  random_state=22391) # Scaled PCA

xNS_trs, xNS_tts, yNS_trs, yNS_tts = train_test_split(pcns, yw, test_size=0.33,
                                                  random_state=22391) # Not Scaled PCA


# # Random Forest Models

# In[ ]:


# Fit
std_rf = RandomForestRegressor(max_depth=10, random_state=323, criterion='mse',
                               n_estimators=5, verbose=1, bootstrap=True)

pc_rf = RandomForestRegressor(max_depth=10, random_state=323, criterion='mse',
                               n_estimators=5, bootstrap=True)

ns_rf = RandomForestRegressor(max_depth=10, random_state=323, criterion='mse',
                               n_estimators=5, bootstrap=True)

std_rf.fit(x_tr, y_tr[:,0])
pc_rf.fit(xPC_tr, yPC_tr[:,0])
ns_rf.fit(xNS_tr, yNS_tr[:,0])


# In[ ]:


# Scores
std_tr_score, std_tt_score = std_rf.score(x_tr,y_tr[:,0]), std_rf.score(x_tt,y_tt[:,0])

pc_tr_score, pc_tt_score = pc_rf.score(xPC_tr,yPC_tr[:,0]), pc_rf.score(xPC_tt,yPC_tt[:,0])

ns_tr_score, ns_tt_score = ns_rf.score(xNS_tr,yNS_tr[:,0]), ns_rf.score(xNS_tt,yNS_tt[:,0])

print("No PCA Train Score: {:.3f}  Test Score: {:.3f}".format(std_tr_score,std_tt_score),'\nScaled PCA Train Score: {:.3f}  Test Score: {:.3f}'.format(pc_tr_score,pc_tt_score),'\nNo Scale PCA Train Score: {:.3f}  Test Score: {:.3f}'.format(ns_tr_score,ns_tt_score))


# Somewhat dissapointingly there's a rather large gap in the performance between the unmodified features (that is except for replacing NA with mean) and the MacroPCA scores using RFR.  The scaled PCA version does much better than the Not Scaled though still quite a bit worse than the raw features. 
# 
# Overfitting is also clearly and issue, specially for the PCA models.  Even though the scaled PCA does seem to have some predictive value it overfits the worst.  The unmodified features overfit as well but not to the same degree. Another issue that presented itself was how slowly the models train when using Mean Absolute Error as the objective function.  Frankly I doubt it would have made much, if any, of a difference but the actual evaluation will be done using WMAE as the metric.  It is also possible that comparing these models using the WMAE would have some effect on the relative performance as opposed the R-square above.  Again, however, I'm struck as this being not especially likely.  Furthermore, there is still the issue of overfitting which should lead one to believe that the specifc metric won't ultimately matter here.
# 
# The overfitting problem and relative predictive ability seems to stay consistent through changes to target variable and hyperparameters.

# # Support Vector Regression

# In[ ]:


# Fit
'''std_sv = SVR(kernel='poly', gamma='auto')
pc_sv = SVR(kernel='poly', gamma='auto')
ns_sv = SVR(kernel='poly', gamma='auto')

std_sv.fit(x_trs, y_trs[:,0])
pc_sv.fit(xPC_trs, yPC_trs[:,0])
ns_sv.fit(xNS_trs, yNS_trs[:,0])'''


# In[ ]:


"""std_tr_score, std_tt_score = std_sv.score(x_tr,y_tr[:,0]), std_sv.score(x_tt,y_tt[:,0])

pc_tr_score, pc_tt_score = pc_sv.score(xPC_tr,yPC_tr[:,0]), pc_sv.score(xPC_tt,yPC_tt[:,0])

ns_tr_score, ns_tt_score = ns_sv.score(xNS_tr,yNS_tr[:,0]), ns_sv.score(xNS_tt,yNS_tt[:,0])

print("No PCA Train Score: {:.3f}  Test Score: {:.3f}".format(std_tr_score,std_tt_score),'\n\
Scaled PCA Train Score: {:.3f}  Test Score: {:.3f}'.format(pc_tr_score,pc_tt_score),'\n\
No Scale PCA Train Score: {:.3f}  Test Score: {:.3f}'.format(ns_tr_score,ns_tt_score))"""

