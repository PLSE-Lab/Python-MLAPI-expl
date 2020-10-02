#!/usr/bin/env python
# coding: utf-8

# # Data Science and Analytics Final Project
# #### Tim DeSAhhhh
# ###### Hema Mitta Kalyani - Kelly William - Kristianto - Marco Kenata
# ###### DSA-C Semester Genap 2018/2019
# ---

# In[ ]:


get_ipython().system('pip install biosppy')

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import biosppy
import biosppy.signals.tools as st

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_validate

import warnings
warnings.filterwarnings('ignore')


# ### VERSI 1: Sleep Stage berdasarkan csv yang tersedia dari data

# In[ ]:


fixed_dataset = pd.read_csv("../input/datasetcleaned-dsa/new_data.csv").drop(columns=['Unnamed: 0', 'Time', 'EOG-L', 'EOG-R', 'Subm', 'Tib', 'EKG'])
fixed_dataset.head()


# In[ ]:


fixed_dataset.describe()


# In[ ]:


for col in ['O2-A1', 'C4-A1', 'O1-A2', 'C3-A2']:
    # high pass filter
    temp = fixed_dataset[col]
    b, a = st.get_filter(ftype='butter', band='highpass', order=8, frequency=4, sampling_rate=1000)
    aux, _ = st._filter_signal(b, a, signal=temp, check_phase=True, axis=0)
    # low pass filter
    b, a = st.get_filter(ftype='butter', band='lowpass', order=16, frequency=40, sampling_rate=1000)
    filtered, _ = st._filter_signal(b, a, signal=aux, check_phase=True, axis=0)
    fixed_dataset[col] = filtered


# In[ ]:


fixed_dataset.describe()


# In[ ]:


y_all = fixed_dataset['Sleep_Stage']
x_all = fixed_dataset.drop(['Sleep_Stage'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=32)


# In[ ]:


classifiers = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier()
]

for classifier in classifiers:
    model = Pipeline([("Normalize", StandardScaler()), ("Classifier", RandomForestClassifier())])
    model.fit(x_train, y_train)   
    y_pred = model.predict(x_test)
    cf_name = type(classifier).__name__
    print(cf_name)
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("\n\n")


# ---

# ### VERSI 2: Sleep Stage berdasarkan paper yang dapat diakses di http://iaast.iieng.org/upload/2718A1014055.pdf

# In[ ]:


#Get the EEG datas
eeg_dataset = fixed_dataset.copy()[['O2-A1', 'C4-A1', 'O1-A2', 'C3-A2']]
eeg_dataset.describe()


# In[ ]:


result = biosppy.signals.eeg.eeg(eeg_dataset, sampling_rate=1000, show=True)


# In[ ]:


theta_temp = pd.DataFrame(result['theta'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
theta_temp['Theta_Energy'] = theta_temp['O2-A1'] + theta_temp['C4-A1'] + theta_temp['O1-A2'] + theta_temp['C3-A2']

alpha_low_temp = pd.DataFrame(result['alpha_low'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
alpha_low_temp['Alpha_Low_Energy'] = alpha_low_temp['O2-A1'] + alpha_low_temp['C4-A1'] + alpha_low_temp['O1-A2'] + alpha_low_temp['C3-A2']

alpha_high_temp = pd.DataFrame(result['alpha_high'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
alpha_high_temp['Alpha_high_Energy'] = alpha_high_temp['O2-A1'] + alpha_high_temp['C4-A1'] + alpha_high_temp['O1-A2'] + alpha_high_temp['C3-A2']

beta_temp = pd.DataFrame(result['beta'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
beta_temp['Beta_Energy'] = beta_temp['O2-A1'] + beta_temp['C4-A1'] + beta_temp['O1-A2'] + beta_temp['C3-A2']

gamma_temp = pd.DataFrame(result['gamma'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
gamma_temp['Gamma_Energy'] = gamma_temp['O2-A1'] + gamma_temp['C4-A1'] + gamma_temp['O1-A2'] + gamma_temp['C3-A2']

filter_eeg = pd.concat([theta_temp, alpha_low_temp, alpha_high_temp, beta_temp, gamma_temp], ignore_index=True, axis=1)

energy_eeg = filter_eeg[[4, 9, 14, 19, 24]]
energy_eeg.columns = ['Theta_Energy', 'Alpha_Low_Energy', 'Alpha_High_Energy', 'Beta_Energy', 'Gamma_Energy']

energy_eeg['Total_Energy'] = energy_eeg['Theta_Energy'] + energy_eeg['Alpha_Low_Energy'] + energy_eeg['Alpha_High_Energy'] + energy_eeg['Beta_Energy'] + energy_eeg['Gamma_Energy']

energy_eeg.head()


# In[ ]:


#Adding several feature
energy_eeg['Ratio Gamma-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Gamma_Energy']
energy_eeg['Ratio Beta-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Beta_Energy']
energy_eeg['Ratio L_Alpha-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Alpha_Low_Energy']
energy_eeg['Ratio H_Alpha-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Alpha_High_Energy']
energy_eeg['Ratio Theta-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Theta_Energy']

energy_eeg.head()


# In[ ]:


#Classifications
#Based on http://iaast.iieng.org/upload/2718A1014055.pdf

#6: Gamma
#7: Beta
#8: Low Aplha
#9: High Alpha
#10: Theta
#11: Sleep Stage

energy_eeg['Sleep Stage'] = -1

for i in range(len(energy_eeg)):
    if energy_eeg.iloc[i,6] >= max(energy_eeg.iloc[i,7], energy_eeg.iloc[i,8], energy_eeg.iloc[i,9], energy_eeg.iloc[i,10]) * 4:
        energy_eeg.iloc[i,11] = 3 #S3
    elif energy_eeg.iloc[i,8] > energy_eeg.iloc[i,10] or energy_eeg.iloc[i,9] > energy_eeg.iloc[i,10] or energy_eeg.iloc[i,7] > energy_eeg.iloc[i,10]:
        energy_eeg.iloc[i,11] = 0 #Wake
    elif energy_eeg.iloc[i,6] > max(energy_eeg.iloc[i,7], energy_eeg.iloc[i,8], energy_eeg.iloc[i,9], energy_eeg.iloc[i,10]):
        if energy_eeg.iloc[i,10] < energy_eeg.iloc[i,6] // 2:
            energy_eeg.iloc[i,11] = 1 #S1
        elif energy_eeg.iloc[i,10] > energy_eeg.iloc[i,6] // 2:
            energy_eeg.iloc[i,11] = 4 #REM
    elif energy_eeg.iloc[i,10] > max(energy_eeg.iloc[i,7], energy_eeg.iloc[i,8], energy_eeg.iloc[i,9], energy_eeg.iloc[i,6]):
        energy_eeg.iloc[i,11] = 4 #REM
    else:
        energy_eeg.iloc[i,11] = 2 #S2

energy_eeg.head()


# In[ ]:


y2_all = energy_eeg['Sleep Stage']
x2_all = energy_eeg.drop(['Sleep Stage', 'Ratio Gamma-Total', 'Ratio Beta-Total', 'Ratio L_Alpha-Total', 'Ratio H_Alpha-Total', 'Ratio Theta-Total', 'Total_Energy'], axis=1)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2_all, y2_all, test_size=0.3, random_state=32)


# In[ ]:


classifiers = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier()
]

for classifier in classifiers:
    model2 = Pipeline([("Normalize", StandardScaler()), ("Classifier", RandomForestClassifier())])
    model2.fit(x2_train, y2_train)   
    y2_pred = model2.predict(x2_test)
    cf_name = type(classifier).__name__
    print(cf_name)
    print("Accuracy score: ", accuracy_score(y2_test, y2_pred))
    print("Classification report:")
    print(classification_report(y2_test, y2_pred))
    print("\n\n")


# ### VERSI 2 (Extended): Train, Test, and Validation using EEG Energy

# In[ ]:


data_alphabet = ['A', 'B', 'C', 'D', 'E', 'F']
all_datas_dict = {}
for i in data_alphabet:
    df_temp = pd.read_csv('../input/eeg-dataset-mk-ui/' + i + '_data.csv').drop(columns=['Unnamed: 0', 'Time'])
    df_temp = df_temp[(df_temp[['O2-A1', 'C4-A1', 'O1-A2', 'C3-A2']] != 0).all(axis=1)]
    all_datas_dict[i] = df_temp


# In[ ]:


def sleep_stage_calculation(result):    
    theta_temp = pd.DataFrame(result['theta'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
    theta_temp['Theta_Energy'] = theta_temp['O2-A1'] + theta_temp['C4-A1'] + theta_temp['O1-A2'] + theta_temp['C3-A2']

    alpha_low_temp = pd.DataFrame(result['alpha_low'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
    alpha_low_temp['Alpha_Low_Energy'] = alpha_low_temp['O2-A1'] + alpha_low_temp['C4-A1'] + alpha_low_temp['O1-A2'] + alpha_low_temp['C3-A2']

    alpha_high_temp = pd.DataFrame(result['alpha_high'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
    alpha_high_temp['Alpha_high_Energy'] = alpha_high_temp['O2-A1'] + alpha_high_temp['C4-A1'] + alpha_high_temp['O1-A2'] + alpha_high_temp['C3-A2']

    beta_temp = pd.DataFrame(result['beta'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
    beta_temp['Beta_Energy'] = beta_temp['O2-A1'] + beta_temp['C4-A1'] + beta_temp['O1-A2'] + beta_temp['C3-A2']

    gamma_temp = pd.DataFrame(result['gamma'], columns={'O2-A1', 'C4-A1', 'O1-A2', 'C3-A2'})
    gamma_temp['Gamma_Energy'] = gamma_temp['O2-A1'] + gamma_temp['C4-A1'] + gamma_temp['O1-A2'] + gamma_temp['C3-A2']

    filter_eeg = pd.concat([theta_temp, alpha_low_temp, alpha_high_temp, beta_temp, gamma_temp], ignore_index=True, axis=1)

    energy_eeg = filter_eeg[[4, 9, 14, 19, 24]]
    energy_eeg.columns = ['Theta_Energy', 'Alpha_Low_Energy', 'Alpha_High_Energy', 'Beta_Energy', 'Gamma_Energy']

    energy_eeg['Total_Energy'] = energy_eeg['Theta_Energy'] + energy_eeg['Alpha_Low_Energy'] + energy_eeg['Alpha_High_Energy'] + energy_eeg['Beta_Energy'] + energy_eeg['Gamma_Energy']
    
    energy_eeg['Ratio Gamma-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Gamma_Energy']
    energy_eeg['Ratio Beta-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Beta_Energy']
    energy_eeg['Ratio L_Alpha-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Alpha_Low_Energy']
    energy_eeg['Ratio H_Alpha-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Alpha_High_Energy']
    energy_eeg['Ratio Theta-Total'] = energy_eeg['Total_Energy'] // energy_eeg['Theta_Energy']
    
    #Classifications
    #Based on http://iaast.iieng.org/upload/2718A1014055.pdf

    #6: Gamma
    #7: Beta
    #8: Low Aplha
    #9: High Alpha
    #10: Theta
    #11: Sleep Stage

    energy_eeg['Sleep Stage'] = -1

    for i in range(len(energy_eeg)):
        if energy_eeg.iloc[i,6] >= max(energy_eeg.iloc[i,7], energy_eeg.iloc[i,8], energy_eeg.iloc[i,9], energy_eeg.iloc[i,10]) * 4:
            energy_eeg.iloc[i,11] = 3 #S3
        elif energy_eeg.iloc[i,8] > energy_eeg.iloc[i,10] or energy_eeg.iloc[i,9] > energy_eeg.iloc[i,10] or energy_eeg.iloc[i,7] > energy_eeg.iloc[i,10]:
            energy_eeg.iloc[i,11] = 0 #Wake
        elif energy_eeg.iloc[i,6] > max(energy_eeg.iloc[i,7], energy_eeg.iloc[i,8], energy_eeg.iloc[i,9], energy_eeg.iloc[i,10]):
            if energy_eeg.iloc[i,10] < energy_eeg.iloc[i,6] // 2:
                energy_eeg.iloc[i,11] = 1 #S1
            elif energy_eeg.iloc[i,10] > energy_eeg.iloc[i,6] // 2:
                energy_eeg.iloc[i,11] = 4 #REM
        elif energy_eeg.iloc[i,10] > max(energy_eeg.iloc[i,7], energy_eeg.iloc[i,8], energy_eeg.iloc[i,9], energy_eeg.iloc[i,6]):
            energy_eeg.iloc[i,11] = 4 #REM
        else:
            energy_eeg.iloc[i,11] = 2 #S2

    # energy_eeg.head()
    return energy_eeg


# In[ ]:


result = biosppy.signals.eeg.eeg(all_datas_dict['A'], sampling_rate=1000, show=True)
energy_eeg = sleep_stage_calculation(result)
energy_eeg['Sleep Stage'].value_counts()


# In[ ]:


y2_all = energy_eeg['Sleep Stage']
x2_all = energy_eeg.drop(['Sleep Stage', 'Ratio Gamma-Total', 'Ratio Beta-Total', 'Ratio L_Alpha-Total', 'Ratio H_Alpha-Total', 'Ratio Theta-Total', 'Total_Energy'], axis=1)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2_all, y2_all, test_size=0.3, random_state=32)


# In[ ]:


validation_df = biosppy.signals.eeg.eeg(all_datas_dict['E'], sampling_rate=1000, show=True)
energy_eeg_valid = sleep_stage_calculation(validation_df)
energy_eeg['Sleep Stage'].value_counts()


# In[ ]:


y_validation = energy_eeg_valid['Sleep Stage']
x_validation = energy_eeg_valid.drop(['Sleep Stage', 'Ratio Gamma-Total', 'Ratio Beta-Total', 'Ratio L_Alpha-Total', 'Ratio H_Alpha-Total', 'Ratio Theta-Total', 'Total_Energy'], axis=1)


# In[ ]:


classifiers = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier()
]

for classifier in classifiers:
    model2 = Pipeline([("Normalize", StandardScaler()), ("Classifier", RandomForestClassifier())])
    model2.fit(x2_train, y2_train)   
    y2_pred = model2.predict(x2_test)
    cf_name = type(classifier).__name__
    print(cf_name)
    print("Accuracy score:", accuracy_score(y2_test, y2_pred))
    print("Classification report:")
    print(classification_report(y2_test, y2_pred))    
    cv_results = cross_validate(model2, x_validation, y_validation, cv=10)
    plt.plot(cv_results["test_score"])
    plt.title("Test Score with Cross Validation (CV=10)")
    plt.ylabel("Accuracy")
    plt.show()

