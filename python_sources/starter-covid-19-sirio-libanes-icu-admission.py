#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

from IPython.display import display_html


# In[ ]:


import os
print(os.listdir("../"))


# ### Introduction

# **Patient Encounter -> ICU admission -> Target (Yes | No)**

# Presentation of COVID-19 range from asymptomatic/mild symptoms, which can be treated at home, to severe illness that requires hospitalization. Patients with severe COVID-19 may become critically ill (e.g. acute respiratory distress syndrome) in few days (or even in few hours) demanding lengthy mechanical ventilation and intensive care in ICU beds. Despite the efforts in the clinical management, those patients have increased risk of mortality during follow-up. On the other hand, the early identification of those who will develop an adverse illness course may potentially guide more appropriate therapies and care. Moreover, the prediction of patients who may need ICU beds helps healthcare system providers in the capacity need (human resources, PPE and professionals are available) based in the predicted COVID influx.

# **Population selection**:
# 
# * Only COVID cases (A COVID patient was defined as rt-PCR positive).
# 
# * Only admitted inpatients.
# 
# * Only known outcome (dispatched, death).

# ### Questions  

# * Can we predict which inpatient will need intensive care unit (ICU)?
# 
# * Can we predict which inpatient will need intensive care unit (ICU) using only (Vital signs + Demographics)?
# * Can we predict which inpatient will need intensive care unit (ICU) using only (Laboratory exams + Demographics)?
# * Can we predict which inpatient will need intensive care unit (ICU) using only (Comorbities + Demographics)?
# 
# * What is the best time window for each previous question. Justify. 

# ### Loading data 

# In[ ]:


data = pd.read_excel("../input/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")

comorb_lst = [i for i in data.columns if "DISEASE" in i]
comorb_lst.extend(["HTN", "IMMUNOCOMPROMISED", "OTHER"])

demo_lst = [i for i in data.columns if "AGE_" in i]
demo_lst.append("GENDER")

vitalSigns_lst = data.iloc[:,193:-2].columns.tolist()

lab_lst = data.iloc[:,13:193].columns.tolist()


# ### Features/Outcome

# There are 4 groups of features:
# * **Demographics** ($N^{\underline{o}}: 3$)
#     * Percentil Age.
#     * Above 65 years old.
#     * Gender.
# * **Comorbities**  ($N^{\underline{o}}: 9$)
#   * The features were created based on the historical ICD-10 codes of each patient using the Charlson and Elixhauser range of comorbid conditions (https://pubmed.ncbi.nlm.nih.gov/16224307/
# https://pubmed.ncbi.nlm.nih.gov/9431328/). Here, we have chosen the comorbid groups related to serious adverse outcomes in COVID-19.
# 
# * **Vital Signs**  ($N^{\underline{o}}: 36$) 
#   * Diastolic blood pressure.
#   * Systolic blood pressure.
#   * Heart rate.
#   * Respiratory rate.
#   * Temperature.
#   * Oxygen saturation.
# 
# * **Laboratory**   ($N^{\underline{o}}: 180$)
#   * There are 36 laboratorys types.
# 
# The outcome variable is ICU admission.

# The following features were created for each **vital signs** and time window:
# 
# * mean
# * median
# * min 
# * max
# * amplitude (diff): max-min
# * relative amplitude (rel): amplitude/median
# 
# The following features were created for each **laboratory exam** and time window:
# 
# * mean
# * median
# * min 
# * max
# * amplitude (diff): max-min

# In[ ]:


print(f"Number of Comorbities features: {len(comorb_lst)}") 
print(f"Number of Demographics features: {len(demo_lst)}") 
print(f"Number of Vital Signs features: {len(vitalSigns_lst)}") 
print(f"Number of Laboratory features: {len(lab_lst)}") 


# ### Dataset structure

# In[ ]:


# ID is a identification number for each patient.
print(f"Number of lines in the dataset: {len(data)}")
print(f"Number of inpatients: {len(data.PATIENT_VISIT_IDENTIFIER.unique())}")


# From hospital admission, there are five time windows. Please, observe that there is a flag pointing out whether the patient was or not admitted in the ICU as well as his/her vital signs and laboratory results in that window.

# In[ ]:


print(data.WINDOW.unique())


# In[ ]:


data.groupby("PATIENT_VISIT_IDENTIFIER", as_index = False).agg({"ICU":(list), "WINDOW":list}).iloc[[13,14,15,41,0,2]]


# 0 stands for negative and 1 for positive. 
# Please refer to the bellow example
# 
# * Patient with ID 13 experienced ICU admission in the first 2 hours from Hospital admission (0-2). 
# 
# * Patient with ID 14 experienced ICU admission between 2 and 4 hours from Hospital admission (2-4). 
# 
# * Patient with ID 15 experienced ICU admission between 4  and 6 hours from Hospital admission (4-6). 
# 
# * Patient with ID 41 experienced ICU admission between 6  and 12 hours from Hospital admission (6-12). 
# 
# * Patient with ID 0 experienced ICU admission after 12 hours from Hospital admission (ABOVE_12). 
# * Patient with ID 2 did not experienced ICU admission.

# In[ ]:


aux = abs(data.groupby("PATIENT_VISIT_IDENTIFIER")["ICU"].sum()-5)
aux = aux.value_counts().reset_index()
aux.sort_values(by = "index", inplace = True)
aux.reset_index(drop = True, inplace = True)


# In[ ]:


tot_icu_inpatients = aux.ICU[0:5].sum()
y = aux.ICU[0:5].cumsum()/tot_icu_inpatients
plt.plot(y, marker = ".")

plt.ylabel
plt.xlabel("Window")
plt.yticks(round(y,2) )
plt.xticks([0,1,2,3,4], ["0-2", "2-4", "4-6", "6-12", "Above-12"])
plt.show()


# Of the 249 inpatiens 124 $(49\%)$ experienced ICU admission,in which:
# 
# * 28 patients in the 0-2 window  $(23\%)$ 
# 
# * 14 patients in the 2-4 window  $(28+14 \approx 34\%)$     
# 
# * 20 patients in the 4-6 window  $(28+14+20 \approx 50\%)$   
# 
# * 13 patients in the 6-12 window $(28+14+20+13 \approx 60\%)$ 
# 
# * 49 patients in the ABOVE_12 window $(28+14+20+13+49 \approx 100\%)$ 
# 
# * $249 - 124 = 125$ did not experienced ICU admission.

# ### Missing Data

# In[ ]:


missing_df = data.groupby("WINDOW").count()/249


# In[ ]:


missing_df[vitalSigns_lst]


# Example (BLOODPRESSURE_DIASTOLIC_MEAN).
# Have at least one vital sign. 
# 
# * 0-2 = $36\%$
# * 2-4 = $45\%$
# * 4-6 = $53\%$
# * 6-12 = $82\%$
# * Above-12 = $100\%$

# In[ ]:


df1_styler = missing_df[demo_lst].style.set_table_attributes("style='display:inline'").set_caption('Demographics')
df2_styler = missing_df[comorb_lst].style.set_table_attributes("style='display:inline'").set_caption('Comorbities')

display_html(df1_styler._repr_html_()+df2_styler._repr_html_(), raw=True)


# There is no missing values for comorbities and demographics features

# ### Window example 

# In[ ]:


data[data["PATIENT_VISIT_IDENTIFIER"] == 1]


# For instance, the inpatient 1 was ICU addmited at window "0-2". So, if a bigger window was selected lines 6 to 9 should **not** be used.

# In[ ]:


data[data["PATIENT_VISIT_IDENTIFIER"] == 0]


# The inpatient 0 was addmited at window "Above-12". So, it could be used any window size.
