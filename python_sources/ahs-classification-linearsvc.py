# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from matplotlib import style
import numpy as np
style.use('ggplot')
#from sklearn.cluster import KMeans
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#from sklearn.cluster import SpectralClustering
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.cluster import DBSCAN

from sklearn import preprocessing
#cross_validation deprecated
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

import os
import pickle

print(os.listdir("../input"))

#df = pd.read_csv('../input/ahs-woman-1/AHS_Woman_21_Odisha.csv', delimiter='|')
#bigger datasets do not seem to help..might try a full data set if I have an hour to run...
df = pd.read_csv('../input/ahstestonly/small.csv')
#print(df.head)
print(df.shape)
#columns to drop:



def scrubData(df):

    dfStill = df[df.outcome_pregnancy == 2 ]
    print(dfStill.shape)
    dfLve = df[df.outcome_pregnancy == 1 ]
    
    dfSpontArb = df[df.outcome_pregnancy == 4 ]
    
    totalStill = len(dfStill.index)
    totalSpontArb = len(dfSpontArb.index)
    
    totalOK = len(dfLve.index)
    print("Result should be greater than: ", totalStill/(totalStill+totalOK+totalSpontArb))
    
    
    #df3 = df[df.outcome_pregnancy == 4 ]
    #print(df3.shape)
    dfStillAndLive = dfStill.append(dfLve)
    dAll = dfStillAndLive.append(dfSpontArb)
    print(dAll.shape)
    df = dAll
    #df.outcome_pregnancy.astype('int32')
    
    
    
    df = df.sample(frac=1.0)

    df.drop(['w_id','hl_id', 'client_w_id','state','district'], 1, inplace=True)
    df.drop(['psu_id', 'house_no','house_hold_no','year_of_intr', 'month_of_intr', 
             'date_of_intr'], 1, inplace=True)
        
    df.drop(['other_int_code','identifcation_code', 'w_expall_status','w_status','twsi_id', 'client_twsi_id'], 1, inplace=True)
    df.drop(['fid','hh_id', 'client_hh_id','member_identity','father_serial_no', 'mother_serial_no'], 1, inplace=True)
    df.drop(['client_hl_id','building_no', 'hl_expall_status','sn'], 1, inplace=True)
    df.drop(['headname','ever_born', 'wt'], 1, inplace=True)
    df.drop(['fidx','as', 'as_binned'], 1, inplace=True)
    df.drop(['fidh','cdoi', 'anym','catage1','respondentname', 'rtelephoneno'], 1, inplace=True)
    
    df.drop(['healthscheme_1','healthscheme_2'], 1, inplace=True)
    df.drop(['new_born_alive_female','new_born_alive_male', 'new_born_alive_total','new_surviving_female'], 1, inplace=True)
        
    df.drop(['new_surviving_male','new_surviving_total'], 1, inplace=True)
        
    df.drop(['isdeadmigrated'], 1, inplace=True)
    
    df.drop(['isnewrecord','recordupdatedcount', 'recordupdatedcount','schedule_id','year', 'id'], 1, inplace=True)
    
    df.drop(['date_of_marriage', 'month_of_marriage', 'year_of_marriage'],1, inplace=True)
    
    df.drop(['year_of_birth', 'month_of_birth', 'date_of_birth'],1, inplace=True)
    df.drop(['compensation_after_ster', 'received_compensation_after_ster', 'received_compensation_ster_rs'],1, inplace=True)
    
    df.drop(['is_tubectomy','delivered_any_baby','born_alive_female','born_alive_male','born_alive_total'],1, inplace=True)
    df.drop(['surviving_female','surviving_male','surviving_total'],1, inplace=True)
    df.drop(['last_preg_no','previous_last_preg_no','second_last_preg_no', 'third_last_preg_no'],1, inplace=True)
    
    df.drop(['edt','occupation','marital', 'modern', 'traditional'],1, inplace=True)
    df.drop(['recordstatus'],1, inplace=True)
    #hh_serial_no
    df.drop(['hh_serial_no'],1, inplace=True)
    #sex
    df.drop(['sex'],1, inplace=True)
    #religon --sp?
    df.drop(['religion'],1, inplace=True)
    #currently_attending_school 
    df.drop(['currently_attending_school'],1, inplace=True)
    #reason_for_not_attending_school 
    df.drop(['reason_for_not_attending_school'],1, inplace=True)
    #highest_qualification
    df.drop(['highest_qualification'],1, inplace=True)
    #serial_no
    df.drop(['serial_no'],1, inplace=True)
    ##disability_status
    df.drop(['disability_status'],1, inplace=True)
    #occupation_status
    df.drop(['occupation_status'],1, inplace=True)
    ### all of these above single column drops were done experimentally
    #no real change in otcome even for factors that "should" be significant
    #house_status
    df.drop(['house_status'],1, inplace=True)
    #HouseholdStatus
    df.drop(['householdstatus'],1, inplace=True)
    #IsHeadChanged
    df.drop(['isheadchanged'],1, inplace=True)
    
    #acid test 0.9806792271690867 and  0.8754137862661607
    #no_of_times_conceived -- about 1/2 percent cnage in initial -- and a full 1% drop in big set
    #df.drop(['no_of_times_conceived'],1, inplace=True)
    
    #land_possessed 0.9806392255690227 and 0.884334766112039 with 0.8822998618752037
    df.drop(['land_possessed'],1, inplace=True)
    
    #cart
    df.drop(['cart'],1, inplace=True)
    
    #try dripping even mre

    #current_mar_status
    df.drop(['current_mar_status'],1, inplace=True)
    #counselled_for_menstrual_hyg
    df.drop(['counselled_for_menstrual_hyg'],1, inplace=True)
    
  
    df.drop(['aware_abt_haf'],1, inplace=True)
    df.drop(['aware_abt_ort_ors'],1, inplace=True)
    df.drop(['aware_abt_ort_ors_zinc'],1, inplace=True)
    df.drop(['aware_abt_danger_signs_new_born'],1, inplace=True)  
    
    #residancial_status
    df.drop(['residancial_status'],1, inplace=True)  
    #IsCoveredByHealthScheme
    df.drop(['iscoveredbyhealthscheme'],1, inplace=True)  
    #housestatus -- hot tehre is also house_status
    df.drop(['housestatus'],1, inplace=True)  
    #health_prob_afters_fp_use
    df.drop(['health_prob_afters_fp_use'],1, inplace=True)  
    #is_husband_living_with_you
    df.drop(['is_husband_living_with_you'],1, inplace=True)  
    #months_of_preg_first_anc
    df.drop(['months_of_preg_first_anc'],1, inplace=True)
    #age_at_first_conception
    df.drop(['age_at_first_conception'],1, inplace=True)
    
    #regular_treatment_source
    df.drop(['regular_treatment_source'],1, inplace=True) 
    #regular_treatment
    df.drop(['regular_treatment'],1, inplace=True) 
    #diagnosis_source
    df.drop(['diagnosis_source'],1, inplace=True)
    #diagnosed_for
    df.drop(['diagnosed_for'],1, inplace=True)
    #sought_medical_care
    df.drop(['sought_medical_care'],1, inplace=True)
    #symptoms_pertaining_illness
    df.drop(['symptoms_pertaining_illness'],1, inplace=True)
    #treatment_source
    df.drop(['treatment_source'],1, inplace=True)
    #illness_type
    df.drop(['illness_type'],1, inplace=True)
    #injury_treatment_type
    df.drop(['injury_treatment_type'],1, inplace=True)
    #currently_dead_or_out_migrated
    df.drop(['currently_dead_or_out_migrated'],1, inplace=True)
    #twsi_expall_status
    df.drop(['twsi_expall_status'],1, inplace=True)
    #currently_widow
    df.drop(['currently_widow'],1, inplace=True)
    
    #is_currently_pregnant -- statistically sign but also an smoking gun (?)
    #df.drop(['is_currently_pregnant'],1, inplace=True)
    #pregnant_month -- ditto to above
    #df.drop(['pregnant_month'],1, inplace=True)
    
    
    #############
    #df.convert_objects(convert_numeric=True)
    df.replace('', np.nan)
    #'AHILYA BAI'
    #df.replace("AHILYA BAI", np.nan)
    df.apply(pd.to_numeric)
    df.fillna(0, inplace=True)
    
    #print(df.head())
    #result_of_interview need to logic only = 1? drop all others?
    df = df[df.result_of_interview == 1]
    df.drop(['result_of_interview'],1, inplace=True)
    
    print(df.shape)
    return df
    
print("starting data pruning from csv list")
df = scrubData(df)
#df.to_csv("scubbed.csv")
print(df.shape)
print("start analysis")

X = np.array(df.drop(['outcome_pregnancy'], 1))
print(X.shape)
X = preprocessing.scale(X)
print(X.shape)
y = np.array(df['outcome_pregnancy'])


#clf = KMeans(n_clusters=2)
#clf.fit(X)
clf = LinearSVC(max_iter=90000)
clf.fit(X,y)
print("end analysis")
print("test")
correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
   
    if prediction[0] == y[i]:
        correct += 1

print("result: ", correct/len(X))
print("pckle model")
s = pickle.dumps(clf)
pickle.dump( s, open( "AHS_Woman_LinearSVC.p", "wb" ) )
####
print("get bigger dataset")
df2 = pd.read_csv('../input/ahs-woman-1/AHS_Woman_21_Odisha.csv', delimiter='|')
print("clean bigger set")
df2 = scrubData(df2)
print("test against pickled")
clf2 = pickle.loads(s)

X2 = np.array(df2.drop(['outcome_pregnancy'], 1))
print(X2.shape)
X2 = preprocessing.scale(X2)
y2 = np.array(df2['outcome_pregnancy'])

correct = 0
for i in range(len(X2)):
    predict_me = np.array(X2[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
   
    if prediction[0] == y2[i]:
        correct += 1

print("2nd result: ", correct/len(X2))
print("pickle file saved to output")
