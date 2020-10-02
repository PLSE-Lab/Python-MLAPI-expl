#!/usr/bin/env python
# coding: utf-8

# These insights would be hard to reproduce using SQLite instance that is provided. We spent many days of calculation testing this data. 
# 
# In the last 2 weeks or so we started to extract "likelihood" features. 
# We realized that there exists many interactions in the data but they were present only looking at the head tables.
# After the data was aggregated at a patient level those interactions were already lost so we worked on the solution to
# check which combinations of features gave the best results. This resulted in some interesting findings.
# 
# How to calculate the likelihood of screening for the patient.
# ------
# 
# Let's say you want to calculate features on `diagnosis_head.diagnosis_code`.  
# 
# 1. Extract disting combinations of `patient_id` + `diagnosis_head.diagnosis_code`. 
# 
# 
#     DROP TABLE IF EXISTS interaction_finder_distincts;
#     CREATE TABLE interaction_finder_distincts
#     as 
#     SELECT DISTINCT
#         patients_all.patient_id,
#         diagnosis_head.diagnosis_code
#     FROM
#       patients_all
#       INNER JOIN diagnosis_head 
#           ON diagnosis_head.patient_id = patients_all.patient_id
#       LEFT JOIN physicians
#           ON physicians.practitioner_id = diagnosis_head.primary_practitioner_id
#     GROUP BY 1,2; 
# 
# 
# 2. Calculate averages `is_screener` response for patients with this feature.
# 
# 
#     DROP TABLE IF EXISTS interaction_finder_avg_target;
#     CREATE TABLE interaction_finder_avg_target as
#     SELECT
#         d.dist,
#         AVG(CAST(patients_all.is_screener as float)) as avg_is_screener,
#         COUNT(*) as cnt,
#         COUNT(DISTINCT patients_all.patient_id) as cnt_patients
#     FROM
#         patients_train
#         INNER JOIN interaction_finder_distincts as d
#           ON patients_all.patient_id = d.patient_id
#     GROUP BY 1;
# 
# 3. Convert back to the patient level by aggreagating those averages.
# 
# 
#     SELECT
#          patients_all.patient_id,
#          MIN(CASE WHEN davg.cnt_patients >= 100 THEN davg.avg_is_screener ELSE NULL END) as risk_min_100,
#          MAX(CASE WHEN davg.cnt_patients >= 100 THEN davg.avg_is_screener ELSE NULL END) as risk_max_100
#     FROM 
#          patients_all
#          INNER JOIN patients_obs_types po
#             ON patients_all.patient_id = po.patient_id
#          LEFT JOIN interaction_finder_distincts as d
#             ON patients_all.patient_id = d.patient_id
#          LEFT JOIN interaction_finder_avg_target as davg
#             ON d.dist = davg.dist 
#     GROUP BY 1;
# 
#     
# Apart from `MAX` for values with at least 100 patients you can do the same with 5,10,25,50,100,250 patients. 
# When taking fewer patients one must do it in a cross-validated fashion not to leak the target into the features.
# Many of those `MAX` variables had 0.88+ AUC.
# 
# Checking the interactions
# ------
# 
# We tested more than 120 interactions and below is the list of the interactions that were selected using a simple forward selection mechanism.
# If an interaction worked it was kept in the model. The model used for evaluating the interactions was very simple XGBoost Classifier with `max_depth=6` and `n_estimators=100`. 
# The focus was on the speed of checking the interactions.
# 
# ![Interactions tested](http://i.imgur.com/lrE88k5.jpg)
# 
# Additional tables:
# 
# `diag_proc_by_ym` - diagnosis_code + procedure_code in the same month
# 
# `diag_diag_by_ym` - diagnosis_code + diagnosis_code in the same month (pairs of diagnoses)
# 
# Strong interactions
# -----
# 
# **Diagnosis code + primary_practitioner_id**
# 
# We had many insights just looking at this table. For example the interaction between `diagnosis_code` and `primary_practitioner_id` is particulary strong. It improved the result by a full point in AUC. 
# Here we introduced it without `primary_practitioner_id`. We noted that even if we introduce `diagnosis_code` and `primary_practitioner_id` before introducing interaction between `diagnosis_code` and `primary_practitioner_id` it still improves the auc by 0.5 point.
# 
# This indirectly proves that there is dependency between practitioners and diagnoses. Some practitioners during the diagnosis seem to overproduce screenings and some underproduce:
# 
# `P(is_screener | primary_practitioner_id) * P(is_screener | diagnosis_code) ~= P(is_screener | primary_practitioner & diagnosis_code)`
# 
# This can be used to target practitioners with low screening rates for particular diagnoses. For example some information campaign can be easily started based on this analysis.
# 
# **diagnosis_code + claim_type + primary_physician_role***
# 
# This is self explanatory. Knowing the average screening rate for this 3-way interaction is a very strong predictor.
# 
# Similar strong interaction exists for procedure code.
# 
# Knowing a few of those interactions can result in a very simple solution reaching 0.92 AUC.
# It could be probably calculated in pure SQL just with simple averages.
# 
# **Interactions with patient features**
# 
# We tested many interactions between patients and features above but it didn't result in many increases. This could mean that there are no strong dependencies between patient characterstics and diagnoses/procedures.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




