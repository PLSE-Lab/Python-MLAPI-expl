#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd

df = pd.read_csv('../input/TestData.csv',
                         sep = ',',
                         comment = '#',
                         low_memory = False
                         )
df.head()


#Insight1 Data
test1    = df.loc[:,['1ail_Order_B0r_U324069','175_R1n75_U324055','Gender_U324057',
                     '913e_11324058','Length_of_residnce_U324061','1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
data1   = test1.dropna(subset=['1ail_Order_B0r_U324069'])
test2    = df.loc[:,['Vehicle_1_Make_U324580','175_R1n75_U324055','Gender_U324057',
                     '913e_11324058','Length_of_residnce_U324061','1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
data2   = test2.dropna(subset=['Vehicle_1_Make_U324580'])
test3    = df.loc[:,['B1r_Cat_Appar_Women_U324437','175_R1n75_U324055',
                     '913e_11324058','Length_of_residnce_U324061','1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
data3   = test3.dropna(subset=['B1r_Cat_Appar_Women_U324437'])



#Insight1
from numpy import *

import statsmodels.formula.api as sm

where_are_NaNs = isnan(data1)
data1[where_are_NaNs] = 0

y1 = data1.loc[:,['1ail_Order_B0r_U324069']]
X_opt1 = data1.loc[:,['175_R1n75_U324055','Gender_U324057',
                     '913e_11324058','Length_of_residnce_U324061','1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
"""Running the OLS method on X_opt and storing results in regressor_OLS"""
regressor_OLS1 = sm.OLS(endog = y1, exog = X_opt1).fit()
regressor_OLS1.summary()

X_opt11 = data1.loc[:,['175_R1n75_U324055','Gender_U324057',
                     '1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
regressor_OLS11 = sm.OLS(endog = y1, exog = X_opt11).fit()
regressor_OLS11.summary()

#Insight2
where_are_NaNs = isnan(data2)
data2[where_are_NaNs] = 0

y2    = data2.loc[:,['Vehicle_1_Make_U324580']]
X_opt2   = data2.loc[:,['175_R1n75_U324055','Gender_U324057',
                     '913e_11324058','Length_of_residnce_U324061','1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
"""Running the OLS method on X_opt and storing results in regressor_OLS"""
regressor_OLS2 = sm.OLS(endog = y2, exog = X_opt2).fit()
regressor_OLS2.summary()

X_opt21   = data2.loc[:,['175_R1n75_U324055',
                     '913e_11324058','1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
"""Running the OLS method on X_opt and storing results in regressor_OLS"""
regressor_OLS21 = sm.OLS(endog = y2, exog = X_opt21).fit()
regressor_OLS21.summary()


#Insight3
where_are_NaNs = isnan(data3)
data3[where_are_NaNs] = 0

y3    = data3.loc[:,['B1r_Cat_Appar_Women_U324437']]
X_opt3   = data3.loc[:,['175_R1n75_U324055',
                     '913e_11324058','Length_of_residnce_U324061','1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
"""Running the OLS method on X_opt and storing results in regressor_OLS"""
regressor_OLS3 = sm.OLS(endog = y3, exog = X_opt3).fit()
regressor_OLS3.summary()

X_opt31   = data3.loc[:,['175_R1n75_U324055',
                     '1omeowne3_S4a45s_5324062',
                     'D9elli6g_8y7e_U324063','5st_88_914315135_3_U324064']]
regressor_OLS31 = sm.OLS(endog = y3, exog = X_opt31).fit()
regressor_OLS31.summary()
